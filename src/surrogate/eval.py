"""
Evaluation of a trained DeepONet surrogate.

Produces:
- Per-sample L2 error and relative L2 error on the test split
- Predicted-vs-true density heatmaps (x vs t) for eval samples/views
- Per-plotted-sample .npz files containing true density, predicted density,
  prediction error, x/t grids, and input ramp control u
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
from surrogate.train import apply_sumo_config_defaults, resolve_branch_input_dim
from utils.config import load_config
from utils.plotting import plot_density_heatmap


def _build_model(config: dict) -> DeepONet:
    model_cfg = config["model"]
    return DeepONet(
        BranchNet(
            input_dim=resolve_branch_input_dim(config),
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
    model_config = apply_sumo_config_defaults(checkpoint.get("config", config))
    eval_config = apply_sumo_config_defaults(config) if config else {}
    data_cfg = eval_config.get("data") or model_config["data"]
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
        **_eval_augmentation_kwargs(eval_cfg),
    )

    device_name = eval_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = _build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batch_size = int(eval_cfg.get("batch_size", 8))
    if dataset.n_padded_control_views > 0 and batch_size != 1:
        print(
            "[surrogate_eval] Padded eval views have variable time lengths; "
            "using batch_size=1."
        )
        batch_size = 1

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    l2_values = []
    rel_l2_values = []
    mse_values = []
    full_mse_values = []
    padded_mse_values = []

    sample_offset = 0
    max_plots = _resolve_max_plots(eval_cfg.get("max_plots", "all"), len(dataset))
    n_plotted = 0
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

            for batch_idx, mse_value in enumerate(mse.tolist()):
                view_idx = sample_offset + batch_idx
                _, prefix_len = dataset.views[view_idx]
                if prefix_len is None:
                    full_mse_values.append(float(mse_value))
                else:
                    padded_mse_values.append(float(mse_value))

            for batch_idx in range(pred.shape[0]):
                view_idx = sample_offset + batch_idx
                if n_plotted >= max_plots:
                    break
                sample_idx, prefix_len = dataset.views[view_idx]
                sample = dataset.samples[sample_idx]
                density_shape, t_grid, view_label = _view_metadata(sample, prefix_len)
                ramp_control_input = branch_input[batch_idx].cpu().numpy()
                predicted_density = pred[batch_idx].reshape(density_shape)
                true_density = true[batch_idx].reshape(density_shape)
                output_stem = (
                    f"view_{view_idx:04d}_sample_{sample_idx:04d}_{view_label}"
                )
                plot_density_heatmap(
                    predicted=predicted_density,
                    true=true_density,
                    x_grid=sample["x_grid"],
                    t_grid=t_grid,
                    output_path=output_dir / f"{output_stem}.png",
                )
                np.savez(
                    output_dir / f"{output_stem}.npz",
                    predicted_density=predicted_density.astype(np.float32),
                    true_density=true_density.astype(np.float32),
                    density_error=(predicted_density - true_density).astype(np.float32),
                    ramp_control_input=ramp_control_input.astype(np.float32),
                    ramp_control_full=sample["ramp_control"].astype(np.float32),
                    prefix_len=np.array(-1 if prefix_len is None else prefix_len),
                    x_grid=sample["x_grid"].astype(np.float32),
                    t_grid=t_grid.astype(np.float32),
                )
                n_plotted += 1
            sample_offset += pred.shape[0]

    metrics = {
        "mean_l2": float(np.mean(l2_values)),
        "mean_rel_l2": float(np.mean(rel_l2_values)),
        "mean_mse_physical": float(np.mean(mse_values)),
        "mean_mse_physical_full": _mean_or_none(full_mse_values),
        "mean_mse_physical_padded": _mean_or_none(padded_mse_values),
        "n_samples": len(dataset),
        "n_full_control_views": dataset.n_full_control_views,
        "n_padded_control_views": dataset.n_padded_control_views,
        "n_plots": n_plotted,
    }
    with (output_dir / "eval_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[surrogate_eval] Saved metrics and plots to {output_dir}")
    print(json.dumps(metrics, indent=2))
    return metrics


def _eval_augmentation_kwargs(eval_cfg: dict) -> dict:
    padded_samples = eval_cfg.get("padded_samples_per_rollout", 0)
    include_full = eval_cfg.get("include_full_control", True)
    return {
        "include_full_control": _as_bool(include_full),
        "padded_control_samples_per_rollout": padded_samples,
        "padded_control_min_prefix_steps": int(eval_cfg.get("min_prefix_steps", 1)),
        "padded_control_max_prefix_steps": eval_cfg.get("max_prefix_steps"),
        "padded_control_seed": int(eval_cfg.get("seed", 0)),
    }


def _resolve_max_plots(value, n_samples: int) -> int:
    if value is None:
        return n_samples
    if isinstance(value, str):
        if value.lower() == "all":
            return n_samples
        return int(value)
    return int(value)


def _view_metadata(
    sample: dict[str, np.ndarray], prefix_len: int | None
) -> tuple[tuple[int, int], np.ndarray, str]:
    if prefix_len is None:
        return sample["density"].shape, sample["t_grid"], "full"
    return (
        (sample["density"].shape[0], prefix_len),
        sample["t_grid"][:prefix_len],
        f"padded_k{prefix_len:03d}",
    )


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepONet surrogate")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", default=None, help="Optional eval YAML config")
    parser.add_argument(
        "--max-plots",
        default=None,
        help="Number of eval plots to save, or 'all'. Defaults to config/all.",
    )
    parser.add_argument(
        "--padded-samples-per-rollout",
        default=None,
        help="Number of padded-prefix eval views per rollout, or 'all'.",
    )
    parser.add_argument(
        "--padded-only",
        action="store_true",
        help="Evaluate only padded-prefix views, without full-control views.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    cfg = load_config(str(project_root / args.config)) if args.config else {}
    if cfg:
        cfg["project_root"] = str(project_root)
    eval_cfg = cfg.setdefault("evaluation", {})
    if args.max_plots is not None:
        eval_cfg["max_plots"] = args.max_plots
    if args.padded_samples_per_rollout is not None:
        eval_cfg["padded_samples_per_rollout"] = args.padded_samples_per_rollout
    if args.padded_only:
        eval_cfg["include_full_control"] = False
    evaluate(str(project_root / args.checkpoint), cfg)


if __name__ == "__main__":
    main()
