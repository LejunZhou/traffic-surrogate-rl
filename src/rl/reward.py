"""
Shared reward function for both SurrogateEnv and SumoEnv.

IMPORTANT: This function must produce identical results in both environments
for the same density and queue inputs. Surrogate env must denormalize density
predictions before calling, so reward is always computed in physical units.

Phase 1 shaped reward (see proposal.md §"Reward (Phase 1 shaped)"):
    r(t) = -alpha * mean(density)
           -beta  * queue_length
           -gamma * std(density)

Term motivation:
- mean(density) penalizes mainline congestion.
- queue_length penalizes ramp queue buildup, counterbalancing mean(density)
  so the policy cannot just close the ramp forever.
- std(density) penalizes spatial hotspots, encouraging uniform density.

Default weights (alpha=1.0, beta=0.1, gamma=1.0) are tunable per experiment
via the reward block in the PPO config YAML; weights = None falls back to
those defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Tunable coefficients for the shaped Phase-1 reward."""

    alpha: float = 1.0  # weight on -mean(density)
    beta: float = 0.1   # weight on -queue_length
    gamma: float = 1.0  # weight on -std(density)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RewardWeights":
        if not cfg:
            return cls()
        return cls(
            alpha=float(cfg.get("alpha", cls.alpha)),
            beta=float(cfg.get("beta", cls.beta)),
            gamma=float(cfg.get("gamma", cls.gamma)),
        )


def compute_reward(
    density: np.ndarray,
    queue_length: float,
    weights: RewardWeights | dict | None = None,
) -> float:
    """Compute the Phase-1 shaped reward.

    Args:
        density: shape (N_x,) — density at each detector at the current
                 control step, in physical units (veh/km). Must be
                 denormalized before calling.
        queue_length: current on-ramp queue length (vehicles), >= 0.
                      Both SumoEnv and SurrogateEnv use the same analytical
                      queue model so the reward is identical for the same
                      action sequence.
        weights: RewardWeights or dict with keys alpha/beta/gamma.
                 None falls back to the dataclass defaults.

    Returns:
        Scalar reward = -alpha*mean(density) - beta*queue_length - gamma*std(density).
    """
    density_arr = np.asarray(density, dtype=np.float32)
    if density_arr.ndim != 1:
        raise ValueError(f"density must be a 1D array, got shape {density_arr.shape}")
    if density_arr.size == 0:
        raise ValueError("density must contain at least one detector value")
    if not np.all(np.isfinite(density_arr)):
        raise ValueError("density contains NaN or Inf values")

    queue = float(queue_length)
    if not np.isfinite(queue):
        raise ValueError("queue_length must be finite")
    if queue < 0.0:
        raise ValueError(f"queue_length must be non-negative, got {queue}")

    if isinstance(weights, dict) or weights is None:
        w = RewardWeights.from_config(weights)
    else:
        w = weights

    mean_density = float(np.mean(density_arr))
    std_density = float(np.std(density_arr))
    return -(w.alpha * mean_density + w.beta * queue + w.gamma * std_density)
