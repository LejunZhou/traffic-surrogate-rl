"""
PyTorch Dataset for loading DeepONet training data.

Loads processed split files (train.npz, val.npz, test.npz) and returns
(branch_input, trunk_input, target) tuples ready for batched training.

Branch input: [ramp_control; mainline_demand], shape (2 * T_ctrl,) = (240,)
Trunk input:  query coordinates (x, t) normalized, shape (N_query, 2)
Target:       density ρ(x, t), shape (N_query,) — z-score normalized

Query point sampling:
- During training, N_query points may be randomly sub-sampled from the full
  (N_x × T_ctrl) grid per batch to reduce memory usage.
- At evaluation time, the full grid is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: implement TrafficDataset


class TrafficDataset(Dataset):
    """Dataset of (branch_input, trunk_input, target) tuples for DeepONet training."""

    def __init__(
        self,
        split_file: str,
        split_info_file: str,
        n_query_points: int | None = None,
    ) -> None:
        """
        Args:
            split_file: Path to train.npz / val.npz / test.npz.
            split_info_file: Path to splits/split_info.json (normalization stats).
            n_query_points: If set, randomly sub-sample this many query points per sample.
                            If None, use the full (N_x × T_ctrl) grid.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (branch_input, trunk_input, target) for sample idx.

        Returns:
            branch_input: shape (2 * T_ctrl,)
            trunk_input:  shape (N_query, 2)
            target:       shape (N_query,)
        """
        raise NotImplementedError
