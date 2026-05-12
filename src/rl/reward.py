"""
Shared reward function for both SurrogateEnv and SumoEnv.

IMPORTANT: This function must produce identical results in both environments.
Surrogate env must denormalize density predictions before calling this function,
so that reward is always computed in physical units.

Phase 1 baseline reward:
    r(t) = -(mean density across all N_x detectors at time step t)

Future extensions (Phase 2+):
- Throughput bonus
- On-ramp queue length penalty
- Total travel time penalty
Do NOT add multi-objective terms unless explicitly requested.
"""

from __future__ import annotations

import numpy as np


def compute_reward(density: np.ndarray) -> float:
    """Compute the Phase 1 baseline reward from a density snapshot.

    Args:
        density: shape (N_x,) — density at each detector at current time step,
                 in physical units (veh/km). Must be denormalized before calling.

    Returns:
        Scalar reward = -(mean density).
    """
    density_arr = np.asarray(density, dtype=np.float32)
    if density_arr.ndim != 1:
        raise ValueError(f"density must be a 1D array, got shape {density_arr.shape}")
    if density_arr.size == 0:
        raise ValueError("density must contain at least one detector value")
    if not np.all(np.isfinite(density_arr)):
        raise ValueError("density contains NaN or Inf values")
    return -float(np.mean(density_arr))
