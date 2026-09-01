"""Classical feedback ramp-metering baselines: ALINEA and PI-ALINEA.

ALINEA (Papageorgiou, Hadj-Salem & Blosseville, TRR 1320, 1991) is an
integral regulator that holds the density just downstream of the merge at
a set-point near the critical density:

    r(k) = r(k-1) + K_I * (rho_set - rho(k))

PI-ALINEA (Wang, Kosmatopoulos, Papageorgiou & Papamichail, IEEE T-ITS
2014) adds a proportional term acting on the density *change* (damping,
important with measurement/transport lag):

    r(k) = r(k-1) - K_P * (rho(k) - rho(k-1)) + K_I * (rho_set - rho(k))

Both output a metering rate r in veh/h, mapped to the project's action
u = r / ramp_discharge_vph (green fraction of the 1600 vph saturation
flow) and clamped to [u_min, u_max]. Anti-windup: the *clamped* rate is
stored as r(k-1), so the integrator does not wind up at the bounds.

Optional queue override (ALINEA/Q lower bound, Smaragdis & Papageorgiou
2003): when the ramp queue w exceeds `queue_max`, release at least
r_w = d_ramp + (w - queue_max) * 3600 / dt_ctrl.

The controller consumes the SumoEnv/SurrogateEnv observation vector
[N_x z-scored densities, mainline-demand norm, (ramp-demand norm),
time norm, queue / queue_scale] and de-normalizes what it needs; it has
no SUMO dependency and is unit-testable with numpy alone.

Spec strings (parsed by `make_controller`, accepted by the eval scripts
anywhere a policy path or `u=0.5` is accepted):

    alinea:ki=35,rho=30,det=14
    pialinea:kp=4,ki=35,rho=30,det=14
    optional extras: u0=0.5, umin=0, umax=1, cap=1600, qmax=100
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CONTROLLER_NAMES = ("alinea", "pialinea")


class PIALINEAController:
    """PI-ALINEA; ALINEA is the special case kp=0."""

    def __init__(
        self,
        ki: float,
        kp: float = 0.0,
        rho_set: float = 30.0,
        detector_index: int = 14,
        discharge_vph: float = 1600.0,
        u_init: float = 0.5,
        u_min: float = 0.0,
        u_max: float = 1.0,
        dt_ctrl_s: float = 30.0,
        n_detectors: int = 19,
        density_mean: float = 0.0,
        density_std: float = 1.0,
        queue_scale: float = 100.0,
        observe_ramp_demand: bool = False,
        min_ramp_demand: float = 400.0,
        max_ramp_demand: float = 800.0,
        queue_max: float | None = None,
        name: str | None = None,
    ) -> None:
        if not 0 <= int(detector_index) < int(n_detectors):
            raise ValueError(f"detector_index {detector_index} outside [0, {n_detectors})")
        if discharge_vph <= 0:
            raise ValueError("discharge_vph must be positive")
        self.ki = float(ki)
        self.kp = float(kp)
        self.rho_set = float(rho_set)
        self.detector_index = int(detector_index)
        self.discharge_vph = float(discharge_vph)
        self.u_init = float(u_init)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.dt_ctrl_s = float(dt_ctrl_s)
        self.n_detectors = int(n_detectors)
        self.density_mean = float(density_mean)
        self.density_std = max(float(density_std), 1e-6)
        self.queue_scale = float(queue_scale)
        self.observe_ramp_demand = bool(observe_ramp_demand)
        self.min_ramp_demand = float(min_ramp_demand)
        self.max_ramp_demand = float(max_ramp_demand)
        self.queue_max = None if queue_max is None else float(queue_max)
        self._name = name or ("ALINEA" if self.kp == 0.0 else "PI-ALINEA")
        self.reset()

    # -- interface used by scripts/eval_sumo_baselines.py -----------------
    @property
    def label(self) -> str:
        parts = [self._name]
        if self.kp != 0.0:
            parts.append(f"kp{self.kp:g}")
        parts.append(f"ki{self.ki:g}")
        parts.append(f"rho{self.rho_set:g}")
        parts.append(f"d{self.detector_index}")
        if self.queue_max is not None:
            parts.append(f"q{self.queue_max:g}")
        return " ".join(parts)

    def reset(self) -> None:
        """Start of a new episode: rate at u_init, no previous density."""
        self._r_prev = self.u_init * self.discharge_vph
        self._rho_prev: float | None = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        rho = float(obs[self.detector_index]) * self.density_std + self.density_mean
        rho_prev = rho if self._rho_prev is None else self._rho_prev
        r = (
            self._r_prev
            - self.kp * (rho - rho_prev)
            + self.ki * (self.rho_set - rho)
        )
        if self.queue_max is not None:
            queue = float(obs[-1]) * self.queue_scale
            r_w = self._ramp_demand_vph(obs) + (queue - self.queue_max) * 3600.0 / self.dt_ctrl_s
            r = max(r, r_w)
        u = float(np.clip(r / self.discharge_vph, self.u_min, self.u_max))
        self._r_prev = u * self.discharge_vph  # anti-windup: store clamped rate
        self._rho_prev = rho
        return np.array([u], dtype=np.float32)

    # -- helpers -----------------------------------------------------------
    def _ramp_demand_vph(self, obs: np.ndarray) -> float:
        """Ramp arrival rate for the queue override: from the observation if
        present, else the (conservative) maximum configured level."""
        span = self.max_ramp_demand - self.min_ramp_demand
        if self.observe_ramp_demand and span > 0 and len(obs) >= self.n_detectors + 4:
            return float(obs[self.n_detectors + 1]) * span + self.min_ramp_demand
        return self.max_ramp_demand


# -- spec parsing ----------------------------------------------------------

_PARAM_KEYS = {
    "ki": "ki",
    "kp": "kp",
    "rho": "rho_set",
    "det": "detector_index",
    "u0": "u_init",
    "umin": "u_min",
    "umax": "u_max",
    "cap": "discharge_vph",
    "qmax": "queue_max",
}


def is_controller_spec(spec: str) -> bool:
    return str(spec).partition(":")[0].lower() in CONTROLLER_NAMES


def _load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_controller(spec: str, env_cfg: dict | None = None) -> PIALINEAController:
    """Build a controller from a spec string plus the eval env config.

    `env_cfg` is the PPO config's `env` block (as used by SumoEnv): supplies
    density_mean/std, queue_norm_scale, observe_ramp_demand,
    ramp_demand_levels, and (via its `sumo_config` yaml) n_detectors,
    dt_ctrl_s and ramp_discharge_vph. Spec parameters win over config.
    """
    env_cfg = env_cfg or {}
    name, _, params_str = str(spec).partition(":")
    name = name.lower()
    if name not in CONTROLLER_NAMES:
        raise ValueError(f"unknown controller {name!r} (expected one of {CONTROLLER_NAMES})")

    params: dict[str, float] = {}
    for token in filter(None, (t.strip() for t in params_str.split(","))):
        key, sep, value = token.partition("=")
        key = key.strip().lower()
        if not sep or key not in _PARAM_KEYS:
            raise ValueError(
                f"bad controller parameter {token!r} (known: {sorted(_PARAM_KEYS)})"
            )
        params[_PARAM_KEYS[key]] = float(value)
    if name == "alinea" and params.get("kp"):
        raise ValueError("kp is only valid for pialinea: (ALINEA is the kp=0 case)")
    if "ki" not in params:
        raise ValueError(f"{name}: spec must set ki=<gain veh/h per veh/km>, got {spec!r}")

    # Scenario constants from the sumo yaml the env config points at.
    n_detectors, dt_ctrl_s, discharge = 19, 30.0, 1600.0
    sumo_cfg_path = env_cfg.get("sumo_config")
    if sumo_cfg_path and Path(sumo_cfg_path).exists():
        sumo_cfg = _load_yaml(sumo_cfg_path)
        n_detectors = int(sumo_cfg.get("detectors", {}).get("n_detectors", n_detectors))
        dt_ctrl_s = float(sumo_cfg.get("simulation", {}).get("dt_ctrl_s", dt_ctrl_s))
        discharge = float(sumo_cfg.get("demand", {}).get("ramp_discharge_vph", discharge))
    # env-level overrides mirror SumoEnv's own resolution order
    discharge = float(
        env_cfg.get(
            "ramp_discharge_vph",
            (env_cfg.get("sumo_overrides") or {}).get("demand", {}).get("ramp_discharge_vph", discharge),
        )
    )
    ramp_levels = env_cfg.get("ramp_demand_levels") or [env_cfg.get("ramp_demand_vph", 800.0)]

    kwargs = dict(
        kp=0.0,
        detector_index=14,
        discharge_vph=discharge,
        dt_ctrl_s=dt_ctrl_s,
        n_detectors=n_detectors,
        density_mean=float(env_cfg.get("density_mean", 0.0)),
        density_std=float(env_cfg.get("density_std", 1.0)),
        queue_scale=float(env_cfg.get("queue_norm_scale", 100.0)),
        observe_ramp_demand=bool(env_cfg.get("observe_ramp_demand", False)),
        min_ramp_demand=float(min(ramp_levels)),
        max_ramp_demand=float(max(ramp_levels)),
        name="ALINEA" if name == "alinea" else "PI-ALINEA",
    )
    kwargs.update(params)
    kwargs["detector_index"] = int(kwargs["detector_index"])
    return PIALINEAController(**kwargs)
