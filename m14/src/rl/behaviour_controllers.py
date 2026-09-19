"""Behavior controllers for the M14 round-zero SUMO dataset.

The configured mixture includes wide PI-ALINEA, dithered PI-ALINEA,
store-and-flush schedules, feedforward schedules, random signals and constants.
Each controller has reset(env), a callable (observation, info) interface, and
serializable metadata. Seeded controller IDs preserve the original M14 draws.
"""

from __future__ import annotations

import copy

import numpy as np

from rl.baseline_controllers import PIALINEAController
from sumo_env.dataset_generation import sample_ramp_control
from sumo_env.demand_profiles import DemandProfile

CONSTANT_GRID = [round(0.05 * i, 2) for i in range(21)]


class ConstantController:
    def __init__(self, u: float) -> None:
        self.u = float(np.clip(u, 0.0, 1.0))
        self.spec = {"type": "constant", "u": self.u}

    def reset(self, env=None) -> None:
        pass

    def __call__(self, obs, info=None):
        return np.array([self.u], dtype=np.float32)


class ScheduleController:
    """Open-loop schedule u_k, k = 0..K-1."""

    def __init__(self, schedule: np.ndarray, spec: dict | None = None) -> None:
        self.schedule = np.clip(np.asarray(schedule, dtype=np.float32).reshape(-1), 0.0, 1.0)
        self.spec = spec or {"type": "schedule", "schedule": self.schedule.tolist()}
        self._k = 0

    def reset(self, env=None) -> None:
        self._k = 0

    def __call__(self, obs, info=None):
        u = self.schedule[min(self._k, len(self.schedule) - 1)]
        self._k += 1
        return np.array([u], dtype=np.float32)


def store_and_flush_schedule(
    profile: DemandProfile, u_pre: float, u_low: float, u_high: float,
    lead_min: float, lag_min: float, K: int = 120, dt_ctrl_s: float = 30.0,
) -> np.ndarray:
    """u = u_pre before the storage window, u_low inside, u_high after.
    The window brackets the mainline peak (or the ramp surge for a constant
    mainline): [t_c - w - lead, t_c + w + lag]."""
    mp = profile.params.get("mainline", {})
    rp = profile.params.get("ramp", {})
    fam = mp.get("family")
    if fam in ("peak", "double"):
        tc, w = float(mp["t_center"]), float(mp["half_width"])
        t0, t1 = tc - w, tc + w
    elif fam in ("step", "plateau"):
        t0 = float(mp["t1"]); t1 = float(mp.get("t2", t0 + float(mp.get("length", 30.0))))
    elif rp.get("family") in ("surge", "early_surge"):
        t0 = float(rp["t_start"]); t1 = t0 + float(rp["length"])
    else:
        t0, t1 = 15.0, 35.0
    t0 = max(0.0, t0 - lead_min); t1 = min(60.0, t1 + lag_min)
    t_min = (np.arange(K) * dt_ctrl_s + 0.5 * dt_ctrl_s) / 60.0
    u = np.where(t_min < t0, u_pre, np.where(t_min < t1, u_low, u_high))
    return u.astype(np.float32)


def feedforward_schedule(profile: DemandProfile, capacity_vph: float, discharge_vph: float, lag_steps: int = 0,
                         u_min: float = 0.05, K: int = 120, n_lanes: int = 1) -> np.ndarray:
    """Capacity-tracking feedforward: u_k = clip((C - d_{k+lag} / n_lanes) / D, u_min, 1).
    Releases whatever merge-lane margin the mainline leaves (C = per-lane merge
    capacity; the mainline is split evenly over n_lanes, M15); the natural
    'anticipative' schedule against which constants are measured in E0."""
    d = profile.mainline_vph[:K] / max(int(n_lanes), 1)
    idx = np.clip(np.arange(K) + int(lag_steps), 0, K - 1)
    u = (float(capacity_vph) - d[idx]) / float(discharge_vph)
    return np.clip(u, u_min, 1.0).astype(np.float32)




class ObsController:
    """Adapter: obs-only callable (e.g. PIALINEAController) -> (obs, info) contract."""

    def __init__(self, inner, spec: dict) -> None:
        self.inner = inner
        self.spec = spec

    def reset(self, env=None) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def __call__(self, obs, info=None):
        return self.inner(obs)


class DitherController:
    """PI-ALINEA plus held Gaussian action dither (M13).

    u_applied = clip(u_alinea + eps, 0, 1) with eps ~ N(0, sigma) redrawn every
    `hold_steps` control steps. The inner integrator continues from the applied
    rate (actuator-saturation semantics), so the loop stays closed on what the
    traffic actually saw.
    """

    def __init__(self, inner: PIALINEAController, sigma: float, hold_steps: int, seed: int, discharge_vph: float,
                 spec: dict) -> None:
        self.inner = inner
        self.sigma = float(sigma)
        self.hold_steps = max(1, int(hold_steps))
        self.seed = int(seed)
        self.discharge_vph = float(discharge_vph)
        self.spec = spec
        self.reset()

    def reset(self, env=None) -> None:
        self.inner.reset()
        self.rng = np.random.default_rng(self.seed)
        self._k = 0
        self._eps = 0.0

    def __call__(self, obs, info=None):
        u_inner = float(np.asarray(self.inner(obs)).reshape(-1)[0])
        if self._k % self.hold_steps == 0:
            self._eps = float(self.rng.normal(0.0, self.sigma))
        u = float(np.clip(u_inner + self._eps, 0.0, 1.0))
        self.inner._r_prev = u * self.discharge_vph
        self._k += 1
        return np.array([u], dtype=np.float32)




# ── plan --------------------------------------------------------------------

TYPE_IDS = {"constant": 1, "alinea": 2, "store_flush": 3, "feedforward": 4, "random_signal": 6,
            "alinea_wide": 7, "alinea_dither": 8}

DEFAULT_SHARES = {"alinea_wide": 0.25, "alinea_dither": 0.15, "store_flush": 0.25, "feedforward": 0.15, "random_signal": 0.10, "constant": 0.10}


FEEDFORWARD_CAPACITY_VPH = (2250.0, 2550.0)   # round-0 feedforward capacity draw range, single-lane merge (v2/v3/v3b)


def build_mixture_plan(n: int, seed: int, shares: dict | None = None, round_index: int = 0,
                       sumo_seed_base: int = 72000,
                       per_entry_seeds: bool = True, feedforward_capacity_vph=None) -> list[dict]:
    """Deterministic list of rollout specs for round 0.

    Each spec: {index, round, profile_set: "train", profile_draw: int, sumo_seed,
    controller: {...}}. Controller parameters are drawn from a seeded rng so
    the plan is reproducible; the profile itself is sampled inside the worker
    from the family with `profile_draw` as the (set, index) key.

    per_entry_seeds=False uses one
    rng shared by all draws in type order. per_entry_seeds=True (M14 default) seeds
    every draw with SeedSequence([seed, TYPE_IDS[type], j]) and the shuffle
    with SeedSequence([seed, 0, n]), so adding or resizing a type leaves the
    other types' parameters unchanged.

    An explicit "feedforward" share disables the legacy fold (one third of the
    store_flush draws becoming feedforward). feedforward_capacity_vph=(lo, hi)
    sets the capacity draw range of the feedforward schedules (default: the
    single-lane 2250-2550).
    """
    ff_range = tuple(float(v) for v in (feedforward_capacity_vph or FEEDFORWARD_CAPACITY_VPH))
    shares = dict(shares or DEFAULT_SHARES)
    fold_feedforward = "feedforward" not in shares
    total = sum(shares.values())
    counts = {k: int(round(n * v / total)) for k, v in shares.items()}
    diff = n - sum(counts.values())
    first = next(iter(counts))
    counts[first] += diff
    rng = np.random.default_rng(seed)
    plan: list[dict] = []
    idx = 0
    for ctype, count in counts.items():
        if ctype not in TYPE_IDS:
            raise ValueError(f"unknown controller type {ctype!r}")
        for j in range(count):
            draw_rng = np.random.default_rng(np.random.SeedSequence([int(seed), TYPE_IDS[ctype], j])) if per_entry_seeds else rng
            spec = _draw_controller_spec(ctype, j, draw_rng, fold_feedforward=fold_feedforward, ff_range=ff_range)
            plan.append({
                "index": idx, "round": int(round_index), "profile_set": "train",
                "profile_draw": int(1_000_000 * (round_index + 1) + idx),
                "sumo_seed": int(sumo_seed_base + idx),
                "controller": spec,
            })
            idx += 1
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0, int(n)])) if per_entry_seeds else rng
    order = shuffle_rng.permutation(len(plan))
    plan = [plan[i] for i in order]
    for i, p in enumerate(plan):
        p["index"] = i
    return plan


def enforce_storage_mandatory(plan: list[dict], family, frac: float, vph: float = 2500.0, stride: int = 10_000,
                              max_attempts: int = 100, set_name: str = "train", n_lanes: int = 1) -> dict:
    """Re-draw the profiles of the first ceil(frac * n) plan entries until their
    peak merge load max_k(d_k / n_lanes + r_k) exceeds `vph` (storage-mandatory
    profiles, where ramp metering matters; n_lanes = 1 is the peak total demand). Draws advance by `stride` per attempt so they never
    collide with other entries or with later rounds. Records profile_draw,
    profile_base_draw, profile_attempts, storage_mandatory and peak_total_vph
    on every entry (in place) and returns a summary."""
    n = len(plan)
    n_mand = int(np.ceil(float(frac) * n)) if frac > 0 else 0
    attempts_max = 0
    for i, p in enumerate(plan):
        base = int(p.get("profile_base_draw", p["profile_draw"]))
        p["profile_base_draw"] = base
        p["storage_mandatory"] = bool(i < n_mand)
        draw, attempt = base, 0
        prof = family.sample_by_key(set_name, draw)
        if p["storage_mandatory"]:
            while prof.peak_merge_load_vph(n_lanes) <= vph:
                attempt += 1
                if attempt > max_attempts:
                    raise RuntimeError(f"no storage-mandatory profile within {max_attempts} draws from {base}")
                draw = base + stride * attempt
                prof = family.sample_by_key(set_name, draw)
        p["profile_draw"] = int(draw)
        p["profile_attempts"] = int(attempt)
        p["peak_total_vph"] = float(prof.peak_total_vph)
        p["peak_merge_load_vph"] = float(prof.peak_merge_load_vph(n_lanes))
        attempts_max = max(attempts_max, attempt)
    above = [p for p in plan if p["peak_merge_load_vph"] > vph]
    return {"n": n, "n_mandatory": n_mand, "threshold_vph": float(vph), "n_lanes": int(n_lanes),
            "frac_above_threshold": len(above) / max(n, 1), "max_attempts": attempts_max}


def _draw_alinea_wide(rng: np.random.Generator) -> dict:
    """Wide ALINEA / PI-ALINEA draw (M13): set-points on both sides of the
    critical density, gains from sluggish to oscillating, an optional ALINEA/Q
    queue override, so the data contains the over-restrictive and the
    over-permissive behaviours an early PPO policy produces."""
    pi = bool(rng.uniform() < 0.5)
    qmax = rng.choice([0, 100, 200])
    return {"ki": float(rng.uniform(5, 60)), "kp": float(rng.uniform(0.5, 12)) if pi else 0.0,
            "rho_set": float(rng.uniform(18, 50)), "det": int(rng.choice([10, 11, 12, 13])),
            "u_init": float(rng.uniform(0.1, 0.8)), "queue_max": None if qmax == 0 else float(qmax)}


def _draw_controller_spec(ctype: str, j: int, rng: np.random.Generator, fold_feedforward: bool = True,
                          ff_range=FEEDFORWARD_CAPACITY_VPH) -> dict:
    if ctype == "constant":
        return {"type": "constant", "u": float(CONSTANT_GRID[j % len(CONSTANT_GRID)])}
    if ctype == "alinea":
        pi = bool(rng.uniform() < 0.5)
        return {"type": "alinea", "ki": float(rng.uniform(10, 40)), "kp": float(rng.uniform(1, 10)) if pi else 0.0,
                "rho_set": float(rng.uniform(26, 40)), "det": int(rng.choice([11, 12, 13])),
                "u_init": float(rng.uniform(0.2, 0.6))}
    if ctype == "feedforward" or (ctype == "store_flush" and fold_feedforward and j % 3 == 2):
        # E0 finding: flushing at u = 1 jams, capacity-tracking schedules are the anticipative reference
        return {"type": "feedforward", "capacity_vph": float(rng.uniform(ff_range[0], ff_range[1])),
                "lag_steps": int(rng.integers(0, 3)), "u_min": float(rng.uniform(0.0, 0.15))}
    if ctype == "store_flush":
        return {"type": "store_flush", "u_pre": float(rng.uniform(0.25, 0.6)), "u_low": float(rng.uniform(0.0, 0.3)),
                "u_high": float(rng.uniform(0.45, 0.8)), "lead_min": float(rng.uniform(0, 5)),
                "lag_min": float(rng.uniform(0, 10))}
    if ctype == "random_signal":
        family = str(rng.choice(["piecewise_constant", "smooth", "ramp_step"]))
        return {"type": "random_signal", "family": family, "seed": int(rng.integers(0, 2**31 - 1))}
    if ctype == "alinea_wide":
        return {"type": "alinea_wide", **_draw_alinea_wide(rng)}
    if ctype == "alinea_dither":
        return {"type": "alinea_dither", **_draw_alinea_wide(rng), "sigma": float(rng.uniform(0.05, 0.2)),
                "hold_steps": int(rng.choice([1, 2, 4])), "seed": int(rng.integers(0, 2**31 - 1))}
    raise ValueError(f"unknown controller type {ctype!r}")


def _build_pialinea(spec: dict, env) -> PIALINEAController:
    """PI-ALINEA from a spec on the v2 observation. The ramp feature of the v2
    observation is r / ramp_norm_vph (not the 400-800 min-max of run 7), so the
    ALINEA/Q override de-normalises with [0, ramp_norm_vph]."""
    return PIALINEAController(
        ki=spec["ki"], kp=spec.get("kp", 0.0), rho_set=spec["rho_set"], detector_index=spec["det"],
        discharge_vph=float(env.ramp_discharge_vph), u_init=spec.get("u_init", 0.5),
        dt_ctrl_s=float(env.dt_ctrl), n_detectors=int(env.N_x),
        density_mean=float(env.density_mean), density_std=float(env.density_std),
        queue_scale=float(env.queue_scale), observe_ramp_demand=bool(env.observe_ramp_demand),
        min_ramp_demand=0.0, max_ramp_demand=float(getattr(env, "ramp_norm_vph", 1000.0)),
        queue_max=spec.get("queue_max"),
    )


def make_controller_from_spec(spec: dict, env, profile: DemandProfile | None = None):
    """Rebuild a behaviour controller inside a worker from its spec."""
    spec = copy.deepcopy(spec)
    ctype = spec["type"]
    K = int(env.T_ctrl)
    if ctype == "constant":
        return ConstantController(spec["u"])
    if ctype in ("alinea", "alinea_wide"):
        return ObsController(_build_pialinea(spec, env), spec)
    if ctype == "alinea_dither":
        return DitherController(_build_pialinea(spec, env), spec["sigma"], spec["hold_steps"], spec["seed"],
                                float(env.ramp_discharge_vph), spec)
    if ctype == "store_flush":
        if profile is None:
            raise ValueError("store_flush needs the episode profile")
        sched = store_and_flush_schedule(profile, spec["u_pre"], spec["u_low"], spec["u_high"],
                                         spec["lead_min"], spec["lag_min"], K=K, dt_ctrl_s=float(env.dt_ctrl))
        return ScheduleController(sched, spec)
    if ctype == "feedforward":
        if profile is None:
            raise ValueError("feedforward needs the episode profile")
        n_lanes = int((getattr(env, "sumo_config", {}) or {}).get("network", {}).get("num_lanes", 1))
        sched = feedforward_schedule(profile, spec["capacity_vph"], float(env.ramp_discharge_vph),
                                     int(spec.get("lag_steps", 0)), float(spec.get("u_min", 0.05)), K=K, n_lanes=n_lanes)
        return ScheduleController(sched, spec)
    if ctype == "random_signal":
        rng = np.random.default_rng(int(spec["seed"]))
        return ScheduleController(sample_ramp_control(spec["family"], K, rng), spec)
    if ctype == "schedule":
        return ScheduleController(np.asarray(spec["schedule"], dtype=np.float32), spec)
    raise ValueError(f"unknown controller type {ctype!r}")
