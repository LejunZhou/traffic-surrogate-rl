"""Causal space-time query dataset for the M14 GRU DeepONet ensemble."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from surrogate.deeponet import PlantNormalisation


class PlantRolloutDataset(Dataset):
    """Rollout-store files -> GRU DeepONet training views.

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
    Files may repeat (bootstrap resample); each unique file is loaded once.
    """

    def __init__(self, store_dir: str | Path, files: list[str], norm: PlantNormalisation,
                 n_query_points: int | None = 512, mode: str = "causal",
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
        if self.mode != "causal":
            raise ValueError("M14 uses causal GRU training views")
        self.views = list(self.files)

    def _band_weights(self, density: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        w = np.ones_like(density, dtype=np.float32)
        upstream = (x_grid <= self.band_x_max_m)[:, None]
        w[(density > self.band_rho_min) & upstream] += self.band_weight
        return w

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict:
        fn = self.views[idx]
        s = self.samples[fn]
        K, Nx = self.K, self.Nx
        branch = s["branch"].copy()
        k_max = K
        # density queries over the (i, k<k_max) grid
        n_cells = Nx * k_max
        if self.n_query is None:
            flat = np.arange(n_cells)
        elif self.n_query < n_cells:
            flat = self.rng.choice(n_cells, size=self.n_query, replace=False)
        else:   # sample with replacement so every item has n_query cells
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
