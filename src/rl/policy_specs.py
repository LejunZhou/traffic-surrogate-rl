"""
Policy specs accepted everywhere a policy is evaluated (M8).

    u=0.3                       constant metering rate
    alinea:ki=15,rho=37,det=12  ALINEA / PI-ALINEA (rl.baseline_controllers)
    mpc:<ensemble_dir>[,H=20,iters=30,...]   Surrogate-MPC (rl.surrogate_mpc)
    <path>.zip                  SB3 PPO policy (symmetric action box detected)

`make_policy_callable(spec, env, env_cfg)` returns a controller with the
(obs, info) calling convention of sumo_env.rollout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ConstantPolicy:
    def __init__(self, u: float) -> None:
        self.u = float(u)
        self.spec = {"type": "constant", "u": self.u}
        self.label = f"constant u={self.u:.2f}"

    def reset(self, env=None) -> None:
        pass

    def __call__(self, obs, info=None):
        return np.array([self.u], dtype=np.float32)


class SB3Policy:
    def __init__(self, path: str, deterministic: bool = True) -> None:
        from stable_baselines3 import PPO

        self.path = str(path)
        self.model = PPO.load(self.path, device="cpu")
        self.symmetric = bool(np.asarray(self.model.action_space.low).reshape(-1)[0] < 0.0)
        self.deterministic = bool(deterministic)
        self.spec = {"type": "policy", "path": self.path, "symmetric": self.symmetric}
        self.label = f"ppo ({Path(self.path).parent.name}/{Path(self.path).stem})"

    def reset(self, env=None) -> None:
        pass

    def __call__(self, obs, info=None):
        obs = np.asarray(obs, dtype=np.float32)
        expected = int(self.model.observation_space.shape[0])
        if obs.shape[0] != expected:
            raise ValueError(f"policy expects {expected}-dim observations, env gives {obs.shape[0]}")
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if self.symmetric:
            a = (a + 1.0) / 2.0
        return np.clip(a, 0.0, 1.0)


class ObsOnly:
    def __init__(self, inner, label: str, spec: dict) -> None:
        self.inner, self.label, self.spec = inner, label, spec

    def reset(self, env=None) -> None:
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def __call__(self, obs, info=None):
        return self.inner(obs)


def make_policy_callable(spec: str, env=None, env_cfg: dict | None = None):
    spec = str(spec)
    if spec.startswith("u="):
        return ConstantPolicy(float(spec[2:]))
    head = spec.partition(":")[0].lower()
    if head in ("alinea", "pialinea"):
        from rl.baseline_controllers import make_controller

        cfg = dict(env_cfg or {})
        if env is not None:
            cfg.setdefault("density_mean", env.density_mean)
            cfg.setdefault("density_std", env.density_std)
            cfg.setdefault("queue_norm_scale", env.queue_scale)
            cfg.setdefault("observe_ramp_demand", env.observe_ramp_demand)
            cfg.setdefault("ramp_discharge_vph", env.ramp_discharge_vph)
        ctrl = make_controller(spec, cfg)
        return ObsOnly(ctrl, ctrl.label, {"type": head, "spec": spec})
    if head == "mpc":
        from rl.surrogate_mpc import SurrogateMPC

        return SurrogateMPC.from_spec(spec, env=env, env_cfg=env_cfg)
    if head == "onestep_mpc":
        from rl.surrogate_mpc import SurrogateMPC

        return SurrogateMPC.from_spec(spec, env=env, env_cfg=env_cfg)
    return SB3Policy(spec)


def policy_label(spec: str) -> str:
    spec = str(spec)
    if spec.startswith("u="):
        return f"constant u={float(spec[2:]):.2f}"
    if spec.partition(":")[0].lower() in ("alinea", "pialinea", "mpc", "onestep_mpc"):
        return spec
    p = Path(spec)
    return f"{p.parent.name}/{p.stem}"
