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
import copy
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


def apply_sumo_config_defaults(config: dict) -> dict:
    """Resolve surrogate config values from data.sumo_config when present.

    The SUMO config is the source of truth for geometry and rollout timing.
    This keeps DeepONet training aligned with the exact scenario used to
    generate SUMO trajectories.
    """
    cfg = copy.deepcopy(config)
    project_root = Path(cfg.get("project_root", Path.cwd())).resolve()
    data_cfg = cfg.setdefault("data", {})
    sumo_config_path = data_cfg.get("sumo_config")

    if sumo_config_path:
        sumo_cfg = load_config(str(_resolve_path(sumo_config_path, project_root)))
        net_cfg = sumo_cfg["network"]
        sim_cfg = sumo_cfg["simulation"]
        det_cfg = sumo_cfg["detectors"]
        demand_cfg = sumo_cfg["demand"]

        data_cfg.setdefault("highway_length_m", float(net_cfg["highway_length_m"]))
        data_cfg.setdefault("duration_s", float(sim_cfg["duration_s"]))
        data_cfg.setdefault("dt_ctrl_s", float(sim_cfg["dt_ctrl_s"]))
        data_cfg.setdefault("n_detectors", int(det_cfg["n_detectors"]))
        if data_cfg.get("constant_mainline_demand_vph") is None:
            data_cfg["constant_mainline_demand_vph"] = float(
                demand_cfg["mainline_demand_vph"]
            )

    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("branch_input_dim", "auto") == "auto":
        model_cfg["branch_input_dim"] = resolve_branch_input_dim(cfg)

    return cfg


def resolve_branch_input_dim(config: dict) -> int:
    """Return DeepONet branch input length T_ctrl."""
    model_cfg = config.get("model", {})
    value = model_cfg.get("branch_input_dim", 120)
    if value != "auto":
        return int(value)

    data_cfg = config.get("data", {})
    if "duration_s" not in data_cfg or "dt_ctrl_s" not in data_cfg:
        raise KeyError(
            "model.branch_input_dim='auto' requires data.duration_s and data.dt_ctrl_s. "
            "Set data.sumo_config or provide those fields explicitly."
        )
    return int(float(data_cfg["duration_s"]) / float(data_cfg["dt_ctrl_s"]))


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def _dataset_augmentation_kwargs(data_cfg: dict, split_name: str) -> dict:
    aug_cfg = data_cfg.get("control_augmentation", {})
    if not aug_cfg:
        return {}

    padded_samples = _split_config_value(
        aug_cfg.get(
            "padded_samples_per_rollout",
            aug_cfg.get("padded_control_samples_per_rollout", 0),
        ),
        split_name,
        0,
    )
    include_full = _split_config_value(
        aug_cfg.get("include_full_control", True),
        split_name,
        True,
    )

    return {
        "include_full_control": _as_bool(include_full),
        "padded_control_samples_per_rollout": padded_samples,
        "padded_control_min_prefix_steps": int(
            aug_cfg.get(
                "min_prefix_steps",
                aug_cfg.get("padded_control_min_prefix_steps", 1),
            )
        ),
        "padded_control_max_prefix_steps": aug_cfg.get(
            "max_prefix_steps", aug_cfg.get("padded_control_max_prefix_steps")
        ),
        "padded_control_seed": int(
            aug_cfg.get("seed", aug_cfg.get("padded_control_seed", 0))
        ),
    }


def _split_config_value(value, split_name: str, default):
    if isinstance(value, dict):
        return value.get(split_name, default)
    return value


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _build_model(config: dict) -> DeepONet:
    model_cfg = config["model"]
    branch = BranchNet(
        input_dim=resolve_branch_input_dim(config),
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
    config = apply_sumo_config_defaults(config)
    training_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    project_root = Path(config.get("project_root", Path.cwd())).resolve()
    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed)

    device_name = training_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    resume_checkpoint = None
    resume_checkpoint_path = training_cfg.get("resume_checkpoint")
    if resume_checkpoint_path:
        resume_path = _resolve_path(resume_checkpoint_path, project_root)
        resume_checkpoint = torch.load(resume_path, map_location=device)
        print(f"[train_surrogate] Resuming from: {resume_path}")

    run_dir = make_run_dir(
        output_cfg.get("base_dir", "runs/surrogate"),
        output_cfg.get("run_name", "deeponet"),
    )
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    split_index_file = data_cfg["split_index_file"]
    raw_dir = data_cfg.get("raw_dir")
    constant_demand = data_cfg.get("constant_mainline_demand_vph")
    if resume_checkpoint is not None and "normalization" in resume_checkpoint:
        stats = resume_checkpoint["normalization"]
    else:
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
        **_dataset_augmentation_kwargs(data_cfg, "train"),
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
        **_dataset_augmentation_kwargs(data_cfg, "val"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(
            training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))
        ),
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
    start_epoch = 1
    best_val = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        checkpoint_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        best_val = float(resume_checkpoint.get("best_val_mse", best_val))

        extra_epochs = training_cfg.get("extra_epochs")
        if extra_epochs is not None:
            n_epochs = checkpoint_epoch + int(extra_epochs)
        if n_epochs < start_epoch:
            raise ValueError(
                "Resume checkpoint is already at epoch "
                f"{checkpoint_epoch}, but training.n_epochs={n_epochs}. "
                "Increase training.n_epochs or pass --extra-epochs."
            )

    wandb_cfg = output_cfg.get("wandb", {})
    logger = ExperimentLogger(
        run_dir,
        use_wandb=bool(wandb_cfg.get("enabled", False)),
        wandb_project=wandb_cfg.get("project"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_run_name=wandb_cfg.get("run_name", run_dir.name),
        wandb_config=config,
    )

    print(f"[train_surrogate] Run dir: {run_dir}")
    print(f"[train_surrogate] Device : {device}")
    if resume_checkpoint is not None:
        print(
            "[train_surrogate] Resume: "
            f"start_epoch={start_epoch}, target_epoch={n_epochs}, "
            f"best_val_mse={best_val:.6g}"
        )
    print(
        "[train_surrogate] Samples: "
        f"train={len(train_ds)} "
        f"(full={train_ds.n_full_control_views}, "
        f"padded={train_ds.n_padded_control_views}), "
        f"val={len(val_ds)} "
        f"(full={val_ds.n_full_control_views}, "
        f"padded={val_ds.n_padded_control_views})"
    )
    print(
        "[train_surrogate] Density stats: "
        f"mean={stats['mean_density']:.4f}, std={stats['std_density']:.4f}"
    )

    try:
        for epoch in range(start_epoch, n_epochs + 1):
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
                    _save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        config,
                        epoch,
                        stats,
                        best_val,
                    )

            logger.log(metrics, epoch)

        _save_checkpoint(
            run_dir / "final.pt",
            model,
            optimizer,
            config,
            n_epochs,
            stats,
            best_val,
        )
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
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pt file to resume from.",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=None,
        help=(
            "When resuming, train this many additional epochs beyond the "
            "checkpoint epoch."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config))
    cfg["project_root"] = str(project_root)
    if args.resume is not None:
        cfg.setdefault("training", {})["resume_checkpoint"] = args.resume
    if args.extra_epochs is not None:
        cfg.setdefault("training", {})["extra_epochs"] = args.extra_epochs
    train(cfg)


if __name__ == "__main__":
    main()
