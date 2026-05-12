"""
Shared reward function for both SurrogateEnv and SumoEnv.

IMPORTANT: This function must produce identical results in both environments
for the same density and queue inputs. Surrogate env must denormalize density
predictions before calling, so reward is always computed in physical units.

Phase 1 nonlinear shaped reward (Milestone 5c — see proposal.md
§"Reward (Phase 1 shaped)"):

    r(t) = -alpha * max(0, mean(density) - rho_freeflow)   # ReLU on density excess
           -beta  * (queue_length / queue_norm)^2           # quadratic in queue
           -gamma * std(density)                            # linear (unchanged)

Term motivation:
- The alpha-term penalizes mean density ONLY when it exceeds free-flow
  density. Operation below free-flow costs nothing, matching the physics
  intuition that an uncongested mainline doesn't need metering.
- The beta-term is quadratic so cost grows fast as the queue builds.
  Mild for short queues, expensive for long ones; "unacceptable above
  some point" behavior without a hard barrier.
- The gamma-term keeps the linear std-of-density penalty unchanged
  from M5/M5b — it's a good spatial-uniformity proxy and is the only
  term that fires symmetrically on hotspots from either side of the
  ramp.

Empirical motivation (M5b → M5c): the linear-only reward of M5b had
u=1.0 as a structural corner optimum because `(1 - u_k) * ramp_demand`
is identically 0 at that corner, making the linear queue penalty 0
regardless of beta. The ReLU-on-density + quadratic-on-queue shape
breaks this trap by:
  - penalizing the u=1.0 corner specifically when its mean density
    crosses the rho_freeflow threshold;
  - penalizing closer-to-closure policies non-linearly so their queue
    growth is meaningfully costed.

Default weights (alpha=1.0, beta=1.0, gamma=1.0, rho_freeflow=20.0,
queue_norm=100.0) are tunable per experiment via the reward block in
the PPO config YAML; weights = None falls back to those defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Tunable coefficients for the Phase-1 nonlinear shaped reward."""

    alpha: float = 1.0          # weight on -max(0, mean(density) - rho_freeflow)
    beta: float = 1.0           # weight on -(queue_length / queue_norm)^2
    gamma: float = 1.0          # weight on -std(density)
    rho_freeflow: float = 20.0  # mean-density threshold for alpha-term (veh/km)
    queue_norm: float = 100.0   # quadratic queue normalizer (vehicles)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RewardWeights":
        if not cfg:
            return cls()
        return cls(
            alpha=float(cfg.get("alpha", cls.alpha)),
            beta=float(cfg.get("beta", cls.beta)),
            gamma=float(cfg.get("gamma", cls.gamma)),
            rho_freeflow=float(cfg.get("rho_freeflow", cls.rho_freeflow)),
            queue_norm=float(cfg.get("queue_norm", cls.queue_norm)),
        )


def compute_reward(
    density: np.ndarray,
    queue_length: float,
    weights: RewardWeights | dict | None = None,
) -> float:
    """Compute the Phase-1 nonlinear shaped reward.

    Args:
        density: shape (N_x,) — density at each detector at the current
                 control step, in physical units (veh/km). Must be
                 denormalized before calling.
        queue_length: current on-ramp queue length (vehicles), >= 0.
                      Both SumoEnv and SurrogateEnv use the same analytical
                      queue model so the reward is identical for the same
                      action sequence.
        weights: RewardWeights or dict with keys alpha/beta/gamma/
                 rho_freeflow/queue_norm. None falls back to the dataclass
                 defaults.

    Returns:
        Scalar reward
            = -alpha * max(0, mean(density) - rho_freeflow)
              -beta  * (queue_length / queue_norm)^2
              -gamma * std(density).
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
    density_excess = max(0.0, mean_density - w.rho_freeflow)
    q_scaled = queue / max(w.queue_norm, 1e-6)

    return -(
        w.alpha * density_excess
        + w.beta * q_scaled * q_scaled
        + w.gamma * std_density
    )
