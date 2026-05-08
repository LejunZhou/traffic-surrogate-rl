"""
Training loop for the DeepONet surrogate model.

Supports:
- Batched gradient descent with configurable optimizer and scheduler
- Periodic validation with loss logging
- Checkpointing (saves model state + config + epoch + best val loss)
- TensorBoard logging
- Deterministic seeding for reproducibility

Every run is saved to a timestamped directory under the configured output root
so prior results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from surrogate.datasets import TrafficDataset, compute_density_stats
from surrogate.deeponet import BranchNet, DeepONet, TrunkNet
from surrogate.losses import mse_loss
from utils.config import load_config
from utils.logging import ExperimentLogger, make_run_dir


def _build_model(config: dict) -> DeepONet:
    model_cfg = config["model"]
    branch = BranchNet(
        input_dim=int(model_cfg.get("branch_input_dim", 120)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=int(model_cfg.get("latent_dim", 128)),
    )
    trunk = TrunkNet(
        input_dim=int(model_cfg.get("trunk_input_dim", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=int(model_cfg.get("latent_dim", 128)),
    )
    return DeepONet(branch, trunk)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict) -> None:
    """Train the DeepONet surrogate from a config dict.

    Args:
        config: Training config. Expected keys include:
            data.train_file, data.val_file, data.split_info_file,
            model (branch/trunk architecture params),
            training (lr, batch_size, n_epochs, seed),
            output.run_dir
    """
    training_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed)

    device_name = training_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    run_dir = make_run_dir(
        output_cfg.get("base_dir", "runs/surrogate"),
        output_cfg.get("run_name", "deeponet"),
    )
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    split_index_file = data_cfg["split_index_file"]
    raw_dir = data_cfg.get("raw_dir")
    constant_demand = data_cfg.get("constant_mainline_demand_vph")
    stats = compute_density_stats(
        split_file=split_index_file,
        split_name="train",
        raw_dir=raw_dir,
        constant_mainline_demand_vph=constant_demand,
    )
    with (run_dir / "normalization.json").open("w") as f:
        json.dump(stats, f, indent=2)

    train_ds = TrafficDataset(
        split_file=split_index_file,
        split_info_file=data_cfg.get("metadata_file", split_index_file),
        split_name="train",
        raw_dir=raw_dir,
        n_query_points=data_cfg.get("n_query_points"),
        density_mean=stats["mean_density"],
        density_std=stats["std_density"],
        constant_mainline_demand_vph=constant_demand,
        highway_length_m=float(data_cfg.get("highway_length_m", 2000.0)),
        duration_s=float(data_cfg.get("duration_s", 3600.0)),
    )
    val_ds = TrafficDataset(
        split_file=split_index_file,
        split_info_file=data_cfg.get("metadata_file", split_index_file),
        split_name="val",
        raw_dir=raw_dir,
        n_query_points=None,
        density_mean=stats["mean_density"],
        density_std=stats["std_density"],
        constant_mainline_demand_vph=constant_demand,
        highway_length_m=float(data_cfg.get("highway_length_m", 2000.0)),
        duration_s=float(data_cfg.get("duration_s", 3600.0)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-6)),
    )

    n_epochs = int(training_cfg.get("n_epochs", 100))
    eval_every = int(training_cfg.get("eval_every", 1))
    wandb_cfg = output_cfg.get("wandb", {})
    logger = ExperimentLogger(
        run_dir,
        use_wandb=bool(wandb_cfg.get("enabled", False)),
        wandb_project=wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_run_name=wandb_cfg.get("run_name", run_dir.name),
        wandb_config=config,
    )
    best_val = float("inf")

    print(f"[train_surrogate] Run dir: {run_dir}")
    print(f"[train_surrogate] Device : {device}")
    print(f"[train_surrogate] Samples: train={len(train_ds)}, val={len(val_ds)}")
    print(
        "[train_surrogate] Density stats: "
        f"mean={stats['mean_density']:.4f}, std={stats['std_density']:.4f}"
    )

    try:
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_losses = []
            for branch_input, trunk_input, target in train_loader:
                branch_input = branch_input.to(device)
                trunk_input = trunk_input.to(device)
                target = target.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(branch_input, trunk_input)
                loss = mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_cfg.get("grad_clip_norm", 1.0))
                )
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            metrics = {"train_mse": float(np.mean(train_losses))}

            if epoch % eval_every == 0 or epoch == n_epochs:
                val_mse = _evaluate_mse(model, val_loader, device)
                metrics["val_mse"] = val_mse
                if val_mse < best_val:
                    best_val = val_mse
                    _save_checkpoint(run_dir / "best.pt", model, optimizer, config, epoch, stats, best_val)

            logger.log(metrics, epoch)

        _save_checkpoint(run_dir / "final.pt", model, optimizer, config, n_epochs, stats, best_val)
    finally:
        logger.close()

    print(f"[train_surrogate] Done. Best val_mse={best_val:.6g}")


@torch.no_grad()
def _evaluate_mse(
    model: DeepONet, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    losses = []
    for branch_input, trunk_input, target in loader:
        branch_input = branch_input.to(device)
        trunk_input = trunk_input.to(device)
        target = target.to(device)
        pred = model(branch_input, trunk_input)
        losses.append(float(mse_loss(pred, target).cpu()))
    return float(np.mean(losses))


def _save_checkpoint(
    path: Path,
    model: DeepONet,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    normalization: dict[str, float],
    best_val_mse: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "normalization": normalization,
            "best_val_mse": best_val_mse,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepONet surrogate")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    train(cfg)


if __name__ == "__main__":
    main()
