"""
Evaluation of a trained DeepONet surrogate.

Produces:
- Per-sample L2 error and relative L2 error on the test split
- Predicted-vs-true density heatmaps (x vs t) for a selection of test samples
- Summary metrics saved to eval_metrics.json in the run directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from surrogate.datasets import TrafficDataset
from surrogate.deeponet import BranchNet, DeepONet, TrunkNet
from utils.config import load_config
from utils.plotting import plot_density_heatmap


def _build_model(config: dict) -> DeepONet:
    model_cfg = config["model"]
    return DeepONet(
        BranchNet(
            input_dim=int(model_cfg.get("branch_input_dim", 120)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            output_dim=int(model_cfg.get("latent_dim", 128)),
        ),
        TrunkNet(
            input_dim=int(model_cfg.get("trunk_input_dim", 2)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            output_dim=int(model_cfg.get("latent_dim", 128)),
        ),
    )


def evaluate(checkpoint_path: str, config: dict) -> dict:
    """Evaluate a trained DeepONet on the test split.

    Args:
        checkpoint_path: Path to a saved model checkpoint (.pt file).
        config: Evaluation config (data paths, plot settings, output dir).

    Returns:
        Dict of scalar metrics: {"mean_l2": ..., "rel_l2": ..., ...}
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", config)
    data_cfg = config.get("data", model_config["data"])
    eval_cfg = config.get("evaluation", {})
    output_dir = Path(eval_cfg.get("output_dir", Path(checkpoint_path).parent / "eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization = checkpoint["normalization"]
    dataset = TrafficDataset(
        split_file=data_cfg["split_index_file"],
        split_info_file=data_cfg.get("metadata_file", data_cfg["split_index_file"]),
        split_name=data_cfg.get("split_name", "test"),
        raw_dir=data_cfg.get("raw_dir"),
        n_query_points=None,
        density_mean=normalization["mean_density"],
        density_std=normalization["std_density"],
        constant_mainline_demand_vph=data_cfg.get("constant_mainline_demand_vph"),
        highway_length_m=float(data_cfg.get("highway_length_m", 2000.0)),
        duration_s=float(data_cfg.get("duration_s", 3600.0)),
    )

    device_name = eval_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = _build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=int(eval_cfg.get("batch_size", 8)), shuffle=False)
    l2_values = []
    rel_l2_values = []
    mse_values = []

    sample_offset = 0
    max_plots = int(eval_cfg.get("max_plots", 3))
    with torch.no_grad():
        for branch_input, trunk_input, target in loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target = target.to(device)
            pred_norm = model(branch_input, trunk_input)

            pred = (
                pred_norm.cpu().numpy() * normalization["std_density"]
                + normalization["mean_density"]
            )
            true = (
                target.cpu().numpy() * normalization["std_density"]
                + normalization["mean_density"]
            )

            diff = pred - true
            l2 = np.linalg.norm(diff, axis=1)
            rel_l2 = l2 / np.maximum(np.linalg.norm(true, axis=1), 1e-8)
            mse = np.mean(diff**2, axis=1)
            l2_values.extend(l2.tolist())
            rel_l2_values.extend(rel_l2.tolist())
            mse_values.extend(mse.tolist())

            for batch_idx in range(pred.shape[0]):
                sample_idx = sample_offset + batch_idx
                if sample_idx >= max_plots:
                    break
                sample = dataset.samples[sample_idx]
                density_shape = sample["density"].shape
                plot_density_heatmap(
                    predicted=pred[batch_idx].reshape(density_shape),
                    true=true[batch_idx].reshape(density_shape),
                    x_grid=sample["x_grid"],
                    t_grid=sample["t_grid"],
                    output_path=output_dir / f"sample_{sample_idx:03d}.png",
                )
            sample_offset += pred.shape[0]

    metrics = {
        "mean_l2": float(np.mean(l2_values)),
        "mean_rel_l2": float(np.mean(rel_l2_values)),
        "mean_mse_physical": float(np.mean(mse_values)),
        "n_samples": len(dataset),
    }
    with (output_dir / "eval_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[surrogate_eval] Saved metrics and plots to {output_dir}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepONet surrogate")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", default=None, help="Optional eval YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config)) if args.config else {}
    evaluate(str(project_root / args.checkpoint), cfg)


if __name__ == "__main__":
    main()
