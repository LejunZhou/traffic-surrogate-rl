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

M8 additions (draft_pipeline.md D10 / D11):
- q_ref_mode "offered": the outflow reference is the *offered* demand of the
  step, q_ref_k = min(d_k + r_k + drain_k, q_cap) with
  drain_k = min(Q_{k-1} * 3600 / dt, D - r_k) (the backlog the meter could
  release on top of the arrivals), so the term measures unserved demand and
  does not penalise quiet minutes. `offered_q_ref` computes it; the envs
  pass the result as `q_ref` to `reward_terms`.
- terminal_queue_weight: optional terminal cost beta_T (Q_K / queue_norm)^2
  added on the last control step (ablation only; default 0).
- form "tts" (M8 E0 finding, see _progress/m8_scenario_v2_progress.md): the
  three-term form is structurally blind to the delay a merge breakdown
  causes (the mainline backlog is served later, so the instantaneous
  lost-outflow term under-counts it), and on the profile family no dynamic
  schedule could beat a constant under it although TTS drops by 20-65 veh h.
  The TTS form is the standard ramp-metering objective:

      r_k = -(N_k + Q_k + P_k) * dt / 3600 / tts_scale        [veh h]

  with N_k = sum_i rho_{i,k} dx the vehicles on the mainline, Q_k the ramp
  queue and P_k the mainline backlog estimated by conservation,
  P_k = max(0, offered_k - served_k - N_k - Q_k) from the cumulative offered
  demand and cumulative outflow. Every quantity is available in both
  environments from (rho, q_out, Q, d, r), so the reward is identical by
  construction; the episode return is -TTS (veh h) for tts_scale = 1.

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
    "q_ref_mode",
    "q_cap",
    "terminal_queue_weight",
    "drain_allowance",
    "form",
    "tts_scale",
    "dx_km",
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
    q_ref_mode: str = "fixed"   # "fixed" | "offered" (per-step reference, see offered_q_ref)
    q_cap: float | None = None  # cap for the offered reference (default: q_ref)
    terminal_queue_weight: float = 0.0   # beta_T for the terminal (Q_K / queue_norm)^2 cost
    drain_allowance: bool = True         # offered mode: add the releasable backlog to the reference
    form: str = "three_term"             # "three_term" | "tts"
    tts_scale: float = 1.0               # tts form: reward = -(N + Q + P) dt/3600 / tts_scale
    dx_km: float = 0.1                   # detector spacing used for N = sum(rho) dx

    def __post_init__(self) -> None:
        if self.form not in ("three_term", "tts"):
            raise ValueError(f"reward form must be 'three_term' or 'tts', got {self.form!r}")
        if self.tts_scale <= 0.0 or self.dx_km <= 0.0:
            raise ValueError("tts_scale and dx_km must be positive")
        if self.q_ref_mode not in ("fixed", "offered"):
            raise ValueError(f"q_ref_mode must be 'fixed' or 'offered', got {self.q_ref_mode!r}")
        if self.q_cap is not None and self.q_cap <= 0.0:
            raise ValueError(f"q_cap must be positive, got {self.q_cap}")
        if not np.isfinite(self.terminal_queue_weight) or self.terminal_queue_weight < 0.0:
            raise ValueError("terminal_queue_weight must be finite and >= 0")
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
            q_ref_mode=str(cfg.get("q_ref_mode", cls.q_ref_mode)),
            q_cap=None if cfg.get("q_cap") is None else float(cfg["q_cap"]),
            terminal_queue_weight=float(cfg.get("terminal_queue_weight", cls.terminal_queue_weight)),
            drain_allowance=bool(cfg.get("drain_allowance", cls.drain_allowance)),
            form=str(cfg.get("form", cls.form)),
            tts_scale=float(cfg.get("tts_scale", cls.tts_scale)),
            dx_km=float(cfg.get("dx_km", cls.dx_km)),
        )

    @property
    def q_cap_value(self) -> float:
        return float(self.q_cap) if self.q_cap is not None else float(self.q_ref)


def offered_q_ref(
    weights: RewardWeights | dict | None,
    mainline_vph: float,
    ramp_arrival_vph: float,
    queue_before: float = 0.0,
    discharge_vph: float = 1600.0,
    dt_ctrl_s: float = 30.0,
) -> float:
    """Per-step outflow reference (D10).

    fixed:   q_ref
    offered: min(d_k + r_k + drain_k, q_cap), drain_k = min(Q_{k-1} * 3600/dt, D - r_k)^+
    """
    w = _resolve_weights(weights)
    if w.q_ref_mode == "fixed":
        return float(w.q_ref)
    d = max(float(mainline_vph), 0.0)
    r = max(float(ramp_arrival_vph), 0.0)
    drain = 0.0
    if w.drain_allowance:
        drain = min(max(float(queue_before), 0.0) * 3600.0 / float(dt_ctrl_s), max(float(discharge_vph) - r, 0.0))
    return float(max(min(d + r + drain, w.q_cap_value), 1.0))


def _resolve_weights(weights: RewardWeights | dict | None) -> RewardWeights:
    if isinstance(weights, RewardWeights):
        return weights
    return RewardWeights.from_config(weights)


def reward_terms(
    density: np.ndarray,
    queue_length: float,
    outflow_vph: float | None,
    weights: RewardWeights | dict | None = None,
    q_ref: float | None = None,
    terminal: bool = False,
    backlog_veh: float = 0.0,
    dt_ctrl_s: float = 30.0,
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
        q_ref: per-step outflow reference (veh/h). None uses weights.q_ref;
               the envs pass `offered_q_ref(...)` in offered mode.
        terminal: True on the last control step; adds the terminal queue cost
                  weights.terminal_queue_weight * (queue / queue_norm)^2.
        backlog_veh: mainline backlog estimate P_k (tts form only; see
                  `backlog_estimate`).
        dt_ctrl_s: control interval (tts form).

    Returns:
        dict with keys mean_density, std_density, outflow_vph, q_ref,
        lost_outflow_frac, outflow_penalty, queue_penalty, std_penalty,
        terminal_penalty, reward. Penalties are reported as non-negative
        magnitudes; reward = -(outflow + queue + std + terminal penalties).
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
    q_ref_k = float(w.q_ref if q_ref is None else q_ref)
    if not np.isfinite(q_ref_k) or q_ref_k <= 0.0:
        raise ValueError(f"q_ref must be positive and finite, got {q_ref_k}")

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
        lost_frac = max(0.0, q_ref_k - outflow) / q_ref_k

    mean_density = float(np.mean(density_arr))
    std_density = float(np.std(density_arr))
    q_scaled = queue / w.queue_norm

    outflow_penalty = w.delta * lost_frac
    queue_penalty = w.beta * q_scaled * q_scaled
    std_penalty = w.gamma * std_density / w.sigma_ref
    terminal_penalty = w.terminal_queue_weight * q_scaled * q_scaled if terminal else 0.0
    # tts form components (veh h per step)
    on_road = float(np.sum(density_arr)) * w.dx_km
    backlog = max(float(backlog_veh), 0.0)
    dt_h = float(dt_ctrl_s) / 3600.0
    tts_step = (on_road + queue + backlog) * dt_h
    if w.form == "tts":
        terminal_penalty = w.terminal_queue_weight * (queue + backlog) * dt_h / w.tts_scale if terminal else 0.0
        reward = -(tts_step / w.tts_scale + terminal_penalty)
    else:
        reward = -(outflow_penalty + queue_penalty + std_penalty + terminal_penalty)

    return {
        "mean_density": mean_density,
        "std_density": std_density,
        "outflow_vph": outflow,
        "q_ref": q_ref_k,
        "lost_outflow_frac": float(lost_frac),
        "outflow_penalty": float(outflow_penalty),
        "queue_penalty": float(queue_penalty),
        "std_penalty": float(std_penalty),
        "terminal_penalty": float(terminal_penalty),
        "on_road_veh": float(on_road),
        "backlog_veh": float(backlog),
        "tts_step_veh_h": float(tts_step),
        "tts_penalty": float(tts_step / w.tts_scale),
        "reward": float(reward),
    }


def backlog_estimate(cum_offered_veh: float, cum_served_veh: float, on_road_veh: float, queue_veh: float) -> float:
    """Mainline backlog by conservation: offered - served - on road - ramp queue, >= 0.
    Both environments use this estimate (the surrogate has no pending count);
    SumoEnv logs the true pending count next to it for diagnostics."""
    return max(float(cum_offered_veh) - float(cum_served_veh) - float(on_road_veh) - float(queue_veh), 0.0)


def compute_reward(
    density: np.ndarray,
    queue_length: float,
    outflow_vph: float | None,
    weights: RewardWeights | dict | None = None,
    q_ref: float | None = None,
    terminal: bool = False,
    backlog_veh: float = 0.0,
    dt_ctrl_s: float = 30.0,
) -> float:
    """Scalar reward; see `reward_terms` for the decomposition and argument docs."""
    return reward_terms(density, queue_length, outflow_vph, weights, q_ref=q_ref, terminal=terminal,
                        backlog_veh=backlog_veh, dt_ctrl_s=dt_ctrl_s)["reward"]
