"""
Shared reward function for both SurrogateEnv and SumoEnv.

IMPORTANT: This function must produce identical results in both environments
for the same (density, queue, outflow) inputs. Surrogate env must denormalize
density predictions before calling, so reward is always computed in physical
units.

Milestone 7 outflow-based reward (replaces the M5c ReLU-on-density term;
see _plans/milestone_7_plan.md):

    r(t) = -delta * max(0, q_ref - q_out(t)) / q_ref     # lost mainline outflow
           -beta  * (queue_length / queue_norm)^2         # quadratic ramp queue
           -gamma * std(density) / sigma_ref              # spatial uniformity

Term motivation:
- The delta-term is the direct throughput objective. q_out is the mainline
  outflow downstream of the merge — in SumoEnv the number of vehicles that
  leave the network per control interval (exact arrival count; the det_18
  loop flow over-counts by ~1.2x, see SumoEnv.step) — so it counts
  mainline + admitted ramp vehicles. It is written as *lost*
  outflow relative to a capacity reference q_ref so that, like the other
  two terms, it is a non-positive penalty and the episode return stays <= 0.
  There is no threshold / dead zone: every vehicle-per-hour of outflow
  counts, unlike the M5c ReLU-on-mean-density proxy, which only fired above
  an ad-hoc rho_freeflow and penalised the *highest*-outflow state in the
  Phase 1 scenario (see _progress/milestone_7_progress.md).
- The beta-term is quadratic so cost grows fast as the queue builds. Mild for
  short queues, expensive for long ones.
- The gamma-term is the spatial-uniformity proxy, unchanged in form from
  M5/M5c. sigma_ref makes the scale explicit (the M5c code carried a
  hard-coded /67.7167 divisor); a value around the training-set density std
  (~6 veh/km) puts it on the same O(1)-per-step footing as the other terms.

Every term is normalised to O(1) per control step so the weights (delta,
beta, gamma) are directly comparable. scripts/balance_reward_terms.py
measures each term's episode-sum range across a constant-u sweep and
proposes weights that keep the three terms comparable ("no term dominates").

Reward inputs:
- density:      (N_x,) veh/km, physical units
- queue_length: virtual ramp queue, vehicles
- outflow_vph:  mainline outflow (network arrivals per interval), veh/h. SurrogateEnv
                has no flow prediction and passes None; that is only allowed
                when delta == 0 (surrogate path runs the two-term reward).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_LEGACY_KEYS = ("alpha", "rho_freeflow")
_ALLOWED_KEYS = {
    "delta",
    "beta",
    "gamma",
    "q_ref",
    "queue_norm",
    "sigma_ref",
    # Consumed by the envs, not by RewardWeights, but they live in the same
    # `reward:` YAML block.
    "warmup_s",
    "reward_warmup_s",
    "queue_scale",
}


@dataclass(frozen=True)
class RewardWeights:
    """Tunable coefficients for the outflow-based shaped reward."""

    delta: float = 1.0          # weight on -max(0, q_ref - outflow) / q_ref
    beta: float = 1.0           # weight on -(queue_length / queue_norm)^2
    gamma: float = 1.0          # weight on -std(density) / sigma_ref
    q_ref: float = 2970.0       # outflow reference (veh/h). IDM 1-lane capacity
                                # at tau=1.0 s; measure it with the u-sweep.
    queue_norm: float = 400.0   # quadratic queue normaliser (vehicles)
    sigma_ref: float = 6.0      # density-std normaliser (veh/km)

    def __post_init__(self) -> None:
        if self.q_ref <= 0.0:
            raise ValueError(f"q_ref must be positive, got {self.q_ref}")
        if self.queue_norm <= 0.0:
            raise ValueError(f"queue_norm must be positive, got {self.queue_norm}")
        if self.sigma_ref <= 0.0:
            raise ValueError(f"sigma_ref must be positive, got {self.sigma_ref}")
        for name in ("delta", "beta", "gamma"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RewardWeights":
        if not cfg:
            return cls()
        legacy = [k for k in _LEGACY_KEYS if k in cfg]
        if legacy:
            raise ValueError(
                f"reward config uses legacy M5c keys {legacy}. The reward is now "
                "outflow-based: replace `alpha`/`rho_freeflow` with "
                "`delta`/`q_ref` (and optionally `sigma_ref`). "
                "See _plans/milestone_7_plan.md."
            )
        unknown = sorted(set(cfg) - _ALLOWED_KEYS)
        if unknown:
            warnings.warn(
                f"reward config has unrecognised keys {unknown}; they are ignored.",
                stacklevel=2,
            )
        return cls(
            delta=float(cfg.get("delta", cls.delta)),
            beta=float(cfg.get("beta", cls.beta)),
            gamma=float(cfg.get("gamma", cls.gamma)),
            q_ref=float(cfg.get("q_ref", cls.q_ref)),
            queue_norm=float(cfg.get("queue_norm", cls.queue_norm)),
            sigma_ref=float(cfg.get("sigma_ref", cls.sigma_ref)),
        )


def _resolve_weights(weights: RewardWeights | dict | None) -> RewardWeights:
    if isinstance(weights, RewardWeights):
        return weights
    return RewardWeights.from_config(weights)


def reward_terms(
    density: np.ndarray,
    queue_length: float,
    outflow_vph: float | None,
    weights: RewardWeights | dict | None = None,
) -> dict[str, float]:
    """Compute every reward component plus the total.

    This is the single source of truth for the reward; both envs log these
    values to `info` and use `["reward"]` as the step reward, so the logged
    decomposition always sums to the reward PPO sees.

    Args:
        density: shape (N_x,) — density at each detector, veh/km.
        queue_length: virtual on-ramp queue (vehicles), >= 0.
        outflow_vph: mainline outflow at the last detector (veh/h), >= 0.
                     None is accepted only when weights.delta == 0.
        weights: RewardWeights, dict, or None (dataclass defaults).

    Returns:
        dict with keys mean_density, std_density, outflow_vph,
        lost_outflow_frac, outflow_penalty, queue_penalty, std_penalty,
        reward. Penalties are reported as non-negative magnitudes;
        reward = -(outflow_penalty + queue_penalty + std_penalty).
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

    w = _resolve_weights(weights)

    if outflow_vph is None:
        if w.delta != 0.0:
            raise ValueError(
                "outflow_vph is None but reward delta != 0. The environment "
                "must supply a mainline outflow measurement, or set delta=0 "
                "to run the two-term (queue + std) reward."
            )
        outflow = 0.0
        lost_frac = 0.0
    else:
        outflow = float(outflow_vph)
        if not np.isfinite(outflow):
            raise ValueError("outflow_vph must be finite")
        if outflow < 0.0:
            raise ValueError(f"outflow_vph must be non-negative, got {outflow}")
        lost_frac = max(0.0, w.q_ref - outflow) / w.q_ref

    mean_density = float(np.mean(density_arr))
    std_density = float(np.std(density_arr))
    q_scaled = queue / w.queue_norm

    outflow_penalty = w.delta * lost_frac
    queue_penalty = w.beta * q_scaled * q_scaled
    std_penalty = w.gamma * std_density / w.sigma_ref

    return {
        "mean_density": mean_density,
        "std_density": std_density,
        "outflow_vph": outflow,
        "lost_outflow_frac": float(lost_frac),
        "outflow_penalty": float(outflow_penalty),
        "queue_penalty": float(queue_penalty),
        "std_penalty": float(std_penalty),
        "reward": float(-(outflow_penalty + queue_penalty + std_penalty)),
    }


def compute_reward(
    density: np.ndarray,
    queue_length: float,
    outflow_vph: float | None,
    weights: RewardWeights | dict | None = None,
) -> float:
    """Scalar reward; see `reward_terms` for the decomposition and argument docs."""
    return reward_terms(density, queue_length, outflow_vph, weights)["reward"]
