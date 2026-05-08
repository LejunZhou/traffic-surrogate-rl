"""
PyTorch Dataset for loading DeepONet training data.

Loads processed split files (train.npz, val.npz, test.npz) and returns
(branch_input, trunk_input, target) tuples ready for batched training.

Branch input: ramp_control, shape (T_ctrl,) = (120,) for the constant-inflow MVP
Trunk input:  query coordinates (x, t) normalized, shape (N_query, 2)
Target:       density ρ(x, t), shape (N_query,) — z-score normalized

Query point sampling:
- During training, N_query points may be randomly sub-sampled from the full
  (N_x × T_ctrl) grid per batch to reduce memory usage.
- At evaluation time, the full grid is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """Dataset of (branch_input, trunk_input, target) tuples for DeepONet training."""

    def __init__(
        self,
        split_file: str,
        split_info_file: str,
        n_query_points: int | None = None,
        split_name: str = "train",
        raw_dir: str | None = None,
        density_mean: float | None = None,
        density_std: float | None = None,
        constant_mainline_demand_vph: float | None = None,
        highway_length_m: float = 2000.0,
        duration_s: float = 3600.0,
    ) -> None:
        """
        Args:
            split_file: Path to split_index.json produced by make_splits().
            split_info_file: Path to metadata.json or split_index.json.
            n_query_points: If set, randomly sub-sample this many query points per sample.
                            If None, use the full (N_x × T_ctrl) grid.
            split_name: Which split to load: "train", "val", or "test".
            raw_dir: Directory containing raw sim_*.npz files. If None, inferred
                     as ../raw relative to the split index directory.
            density_mean: Optional fixed train-set density mean.
            density_std: Optional fixed train-set density std.
            constant_mainline_demand_vph: If set, keep only samples with this
                                          mainline demand.
            highway_length_m: Used to normalize x_grid to [0, 1].
            duration_s: Used to normalize t_grid to [0, 1].
        """
        self.split_file = Path(split_file)
        self.split_name = split_name
        self.n_query_points = n_query_points
        self.highway_length_m = float(highway_length_m)
        self.duration_s = float(duration_s)

        with self.split_file.open("r") as f:
            split_index = json.load(f)
        if split_name not in split_index:
            raise KeyError(f"Split {split_name!r} not found in {self.split_file}")

        self.raw_dir = (
            Path(raw_dir)
            if raw_dir is not None
            else self.split_file.resolve().parent.parent / "raw"
        )

        self.samples: list[dict[str, np.ndarray]] = []
        for filename in split_index[split_name]:
            path = self.raw_dir / filename
            with np.load(str(path)) as data:
                demand = float(data["mainline_demand_vph"])
                if (
                    constant_mainline_demand_vph is not None
                    and not np.isclose(demand, constant_mainline_demand_vph)
                ):
                    continue
                self.samples.append(
                    {
                        "ramp_control": data["ramp_control"].astype(np.float32),
                        "density": data["density"].astype(np.float32),
                        "x_grid": data["x_grid"].astype(np.float32),
                        "t_grid": data["t_grid"].astype(np.float32),
                    }
                )

        if not self.samples:
            demand_msg = (
                f" with mainline_demand_vph={constant_mainline_demand_vph}"
                if constant_mainline_demand_vph is not None
                else ""
            )
            raise ValueError(f"No {split_name} samples found{demand_msg}.")

        if density_mean is None or density_std is None:
            metadata = _load_metadata(split_info_file)
            density_mean = float(metadata["mean_density"])
            density_std = float(metadata["std_density"])

        self.density_mean = float(density_mean)
        self.density_std = max(float(density_std), 1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (branch_input, trunk_input, target) for sample idx.

        Returns:
            branch_input: shape (T_ctrl,)
            trunk_input:  shape (N_query, 2)
            target:       shape (N_query,)
        """
        sample = self.samples[idx]
        density = sample["density"]
        x_grid = sample["x_grid"] / self.highway_length_m
        t_grid = sample["t_grid"] / self.duration_s

        n_x, t_ctrl = density.shape
        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        coords = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
        target = density.ravel().astype(np.float32)

        if self.n_query_points is not None and self.n_query_points < len(target):
            query_idx = np.random.choice(
                len(target), size=self.n_query_points, replace=False
            )
            coords = coords[query_idx]
            target = target[query_idx]

        target = (target - self.density_mean) / self.density_std

        branch_input = torch.from_numpy(sample["ramp_control"])
        trunk_input = torch.from_numpy(coords)
        target_tensor = torch.from_numpy(target.astype(np.float32))
        return branch_input, trunk_input, target_tensor


def compute_density_stats(
    split_file: str,
    split_name: str = "train",
    raw_dir: str | None = None,
    constant_mainline_demand_vph: float | None = None,
) -> dict[str, float]:
    """Compute fixed z-score stats from one split, usually training only."""
    split_path = Path(split_file)
    with split_path.open("r") as f:
        split_index = json.load(f)

    raw_path = (
        Path(raw_dir)
        if raw_dir is not None
        else split_path.resolve().parent.parent / "raw"
    )

    arrays = []
    for filename in split_index[split_name]:
        with np.load(str(raw_path / filename)) as data:
            demand = float(data["mainline_demand_vph"])
            if (
                constant_mainline_demand_vph is not None
                and not np.isclose(demand, constant_mainline_demand_vph)
            ):
                continue
            arrays.append(data["density"].astype(np.float32).ravel())

    if not arrays:
        raise ValueError(f"No samples available to compute stats for {split_name}.")

    values = np.concatenate(arrays)
    return {
        "mean_density": float(np.mean(values)),
        "std_density": float(np.std(values)),
    }


def _load_metadata(path: str) -> dict:
    with Path(path).open("r") as f:
        data = json.load(f)
    return data.get("metadata", data)
