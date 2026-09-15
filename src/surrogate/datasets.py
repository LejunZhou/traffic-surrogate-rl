"""
PyTorch Dataset for loading DeepONet training data.

Loads split-indexed SUMO rollout files and returns (branch_input,
trunk_input, target) tuples ready for batched DeepONet training.

Branch input: ramp_control, shape (T_ctrl,) = (120,) for the constant-inflow MVP
Trunk input:  query coordinates (x, t) normalized, shape (N_query, 2)
Target:       density ρ(x, t), shape (N_query,) — z-score normalized

Query point sampling:
- During training, N_query points may be randomly sub-sampled from the full
  (N_x × T_ctrl) grid per batch to reduce memory usage.
- At evaluation time, the full grid is used.

Control augmentation:
- Each raw rollout can be exposed as the original full-control sample.
- Additional zero-padded prefix views can be generated from the same rollout:
  [u_0, ..., u_k, 0, ..., 0]. Targets for those views are restricted to
  times t <= t_k, matching the causal information available during RL.
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
        include_full_control: bool = True,
        padded_control_samples_per_rollout: int | str = 0,
        padded_control_min_prefix_steps: int = 1,
        padded_control_max_prefix_steps: int | None = None,
        padded_control_seed: int = 0,
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
            include_full_control: Include the original non-padded u(t) sample
                                  for every rollout.
            padded_control_samples_per_rollout: Number of zero-padded prefix
                                                views to synthesize per rollout,
                                                or "all" for every prefix.
            padded_control_min_prefix_steps: Smallest prefix length to use.
            padded_control_max_prefix_steps: Largest prefix length to use. If
                                             None, uses T_ctrl - 1 so there is
                                             always at least one padded future
                                             action.
            padded_control_seed: Seed for deterministic prefix selection.
        """
        self.split_file = Path(split_file)
        self.split_name = split_name
        self.n_query_points = n_query_points
        self.highway_length_m = float(highway_length_m)
        self.duration_s = float(duration_s)
        self.include_full_control = bool(include_full_control)
        self.padded_control_samples_per_rollout = padded_control_samples_per_rollout
        self.padded_control_min_prefix_steps = int(padded_control_min_prefix_steps)
        self.padded_control_max_prefix_steps = padded_control_max_prefix_steps
        self.padded_control_seed = int(padded_control_seed)

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
        self.views = self._build_views()
        if not self.views:
            raise ValueError(
                "TrafficDataset has no sample views. Enable include_full_control "
                "or set padded_control_samples_per_rollout > 0."
            )
        self.n_full_control_views = sum(
            1 for _, prefix_len in self.views if prefix_len is None
        )
        self.n_padded_control_views = len(self.views) - self.n_full_control_views

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (branch_input, trunk_input, target) for sample idx.

        Returns:
            branch_input: shape (T_ctrl,)
            trunk_input:  shape (N_query, 2)
            target:       shape (N_query,)
        """
        sample_idx, prefix_len = self.views[idx]
        sample = self.samples[sample_idx]
        density = sample["density"]
        x_grid = sample["x_grid"] / self.highway_length_m
        if prefix_len is None:
            branch = sample["ramp_control"]
            density_view = density
            t_grid = sample["t_grid"] / self.duration_s
        else:
            branch = sample["ramp_control"].copy()
            branch[prefix_len:] = 0.0
            density_view = density[:, :prefix_len]
            t_grid = sample["t_grid"][:prefix_len] / self.duration_s

        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        coords = np.stack([xx.ravel(), tt.ravel()], axis=-1).astype(np.float32)
        target = density_view.ravel().astype(np.float32)

        if self.n_query_points is not None:
            query_idx = np.random.choice(
                len(target),
                size=self.n_query_points,
                replace=self.n_query_points > len(target),
            )
            coords = coords[query_idx]
            target = target[query_idx]

        target = (target - self.density_mean) / self.density_std

        branch_input = torch.from_numpy(branch.astype(np.float32, copy=False))
        trunk_input = torch.from_numpy(coords)
        target_tensor = torch.from_numpy(target.astype(np.float32))
        return branch_input, trunk_input, target_tensor

    def _build_views(self) -> list[tuple[int, int | None]]:
        """Build full-control and padded-prefix views over loaded rollouts.

        A view is (sample_idx, prefix_len). prefix_len=None means the original
        full-control sample. Otherwise prefix_len is the number of known actions
        retained before zero-padding the future.
        """
        views: list[tuple[int, int | None]] = []
        if self.include_full_control:
            views.extend((idx, None) for idx in range(len(self.samples)))

        n_padded = self._resolve_padded_samples_per_rollout()
        if n_padded == 0:
            return views

        rng = np.random.default_rng(self.padded_control_seed)
        for sample_idx, sample in enumerate(self.samples):
            t_ctrl = int(sample["ramp_control"].shape[0])
            min_prefix = max(1, self.padded_control_min_prefix_steps)
            max_prefix_cfg = self.padded_control_max_prefix_steps
            max_prefix = (
                t_ctrl - 1 if max_prefix_cfg is None else int(max_prefix_cfg)
            )
            max_prefix = min(max_prefix, t_ctrl - 1)
            if max_prefix < min_prefix:
                raise ValueError(
                    "Invalid padded control prefix range: "
                    f"min={min_prefix}, max={max_prefix}, T_ctrl={t_ctrl}"
                )

            prefix_lengths = np.arange(min_prefix, max_prefix + 1, dtype=np.int64)
            if n_padded is not None and n_padded < len(prefix_lengths):
                prefix_lengths = np.sort(
                    rng.choice(prefix_lengths, size=n_padded, replace=False)
                )

            views.extend(
                (sample_idx, int(prefix_len)) for prefix_len in prefix_lengths
            )

        return views

    def _resolve_padded_samples_per_rollout(self) -> int | None:
        value = self.padded_control_samples_per_rollout
        if isinstance(value, str):
            if value.lower() == "all":
                return None
            value = int(value)
        return max(int(value), 0)


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


# ═══════════════════════════════════════════════════════════════════════════
# Plant-model rollout dataset (M9, draft_pipeline.md §5.2 / §5.4)
# ═══════════════════════════════════════════════════════════════════════════

from surrogate.deeponet import PlantNormalisation  # noqa: E402


class PlantRolloutDataset(Dataset):
    """Rollout-store files -> DeepONet v2 training views.

    Each item is a dict of tensors:
        branch      (2, K)    [d / 2500, q_r / 1600]
        query_k     (Nq,)     control-step index of each density query
        query_xt    (Nq, 2)   (x / L, t_k / T)
        target_rho  (Nq,)     z-scored density
        weight_rho  (Nq,)     1 + band_weight inside the shockwave band
        exit_k      (K,)      0..K-1
        exit_xt     (K, 2)    (1, t_k / T)
        target_q    (K,)      exit flow / flow_scale
        mask_q      (K,)      1 where the exit label is supervised
    mode "causal": one view per file (branch = full sequence, queries anywhere).
    mode "padded": `padded_views_per_rollout` prefix views per file (branch
    zeroed at k >= m, queries and exit labels restricted to k < m) — the
    MLP-branch ablation.
    Files may repeat (bootstrap resample); each unique file is loaded once.
    """

    def __init__(self, store_dir: str | Path, files: list[str], norm: PlantNormalisation,
                 n_query_points: int | None = 512, mode: str = "causal", padded_views_per_rollout: int = 8,
                 band_rho_min: float = 40.0, band_x_max_m: float = 1400.0, band_weight: float = 2.0,
                 seed: int = 0, highway_length_m: float = 2000.0, duration_s: float = 3600.0) -> None:
        from sumo_env.rollout import load_rollout_npz

        self.store_dir = Path(store_dir)
        self.norm = norm
        self.n_query = None if n_query_points is None else int(n_query_points)
        self.mode = str(mode)
        self.L = float(highway_length_m)
        self.T = float(duration_s)
        self.band_rho_min, self.band_x_max_m, self.band_weight = float(band_rho_min), float(band_x_max_m), float(band_weight)
        self.rng = np.random.default_rng(seed)
        self.files = list(files)
        self.unique_files = sorted(set(self.files))
        self.samples: dict[str, dict] = {}
        for fn in self.unique_files:
            arrays, meta = load_rollout_npz(self.store_dir / fn)
            density = arrays["density"].astype(np.float32)
            x_grid = arrays["x_grid"].astype(np.float32) if "x_grid" in arrays else (np.arange(density.shape[0]) + 1) * 100.0
            t_grid = arrays["t_grid"].astype(np.float32) if "t_grid" in arrays else np.arange(density.shape[1]) * 30.0
            self.samples[fn] = {
                "branch": norm.branch_input(arrays["mainline_demand"], arrays["ramp_inflow_vph"]).astype(np.float32),
                "density": density,
                "target_rho": norm.density_to_z(density).astype(np.float32),
                "outflow": arrays["outflow_vph"].astype(np.float32),
                "target_q": (arrays["outflow_vph"] / norm.flow_scale).astype(np.float32),
                "x_norm": (x_grid / self.L).astype(np.float32),
                "t_norm": (t_grid / self.T).astype(np.float32),
                "weight": self._band_weights(density, x_grid),
                "meta": meta,
                "ramp_queue": arrays["ramp_queue"].astype(np.float32),
                "mainline_demand": arrays["mainline_demand"].astype(np.float32),
                "ramp_arrival": arrays["ramp_arrival"].astype(np.float32),
                "q_ref": arrays["q_ref"].astype(np.float32) if "q_ref" in arrays else None,
            }
        first = self.samples[self.unique_files[0]]
        self.Nx, self.K = first["density"].shape
        self.views: list[tuple[str, int | None]] = []
        if self.mode == "causal":
            self.views = [(fn, None) for fn in self.files]
        elif self.mode == "padded":
            for fn in self.files:
                ms = self.rng.choice(np.arange(1, self.K), size=min(padded_views_per_rollout, self.K - 1), replace=False)
                self.views.extend((fn, int(m)) for m in np.sort(ms))
                self.views.append((fn, self.K))
        else:
            raise ValueError("mode must be 'causal' or 'padded'")

    def _band_weights(self, density: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        w = np.ones_like(density, dtype=np.float32)
        upstream = (x_grid <= self.band_x_max_m)[:, None]
        w[(density > self.band_rho_min) & upstream] += self.band_weight
        return w

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict:
        fn, m = self.views[idx]
        s = self.samples[fn]
        K, Nx = self.K, self.Nx
        branch = s["branch"].copy()
        k_max = K if m is None else int(m)
        if m is not None:
            branch[:, k_max:] = 0.0
        # density queries over the (i, k<k_max) grid
        n_cells = Nx * k_max
        if self.n_query is None:
            flat = np.arange(n_cells)
        elif self.n_query < n_cells:
            flat = self.rng.choice(n_cells, size=self.n_query, replace=False)
        else:   # short padded prefix: sample with replacement so every view has n_query cells (collate)
            flat = self.rng.choice(n_cells, size=self.n_query, replace=True)
        ii = flat // k_max
        kk = flat % k_max
        query_xt = np.stack([s["x_norm"][ii], s["t_norm"][kk]], axis=-1).astype(np.float32)
        item = {
            "branch": torch.from_numpy(branch),
            "query_k": torch.from_numpy(kk.astype(np.int64)),
            "query_xt": torch.from_numpy(query_xt),
            "target_rho": torch.from_numpy(s["target_rho"][ii, kk]),
            "weight_rho": torch.from_numpy(s["weight"][ii, kk]),
            "exit_k": torch.arange(K, dtype=torch.int64),
            "exit_xt": torch.from_numpy(np.stack([np.ones(K, np.float32), s["t_norm"]], axis=-1)),
            "target_q": torch.from_numpy(s["target_q"]),
            "mask_q": torch.from_numpy((np.arange(K) < k_max).astype(np.float32)),
            "file_idx": torch.tensor(self.unique_files.index(fn)),
        }
        return item

    # -- full-grid access for evaluation ----------------------------------
    def full_grid(self, fn: str) -> dict:
        s = self.samples[fn]
        K, Nx = self.K, self.Nx
        ii, kk = np.meshgrid(np.arange(Nx), np.arange(K), indexing="ij")
        ii, kk = ii.ravel(), kk.ravel()
        return {
            "branch": torch.from_numpy(s["branch"]).unsqueeze(0),
            "query_k": torch.from_numpy(kk.astype(np.int64)).unsqueeze(0),
            "query_xt": torch.from_numpy(np.stack([s["x_norm"][ii], s["t_norm"][kk]], -1).astype(np.float32)).unsqueeze(0),
            "exit_k": torch.arange(K, dtype=torch.int64).unsqueeze(0),
            "exit_xt": torch.from_numpy(np.stack([np.ones(K, np.float32), s["t_norm"]], -1)).unsqueeze(0),
        }


def plant_collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
