"""Behaviour-controller mixture (round 0) and the M13 additions:
per-entry seeding, wide / dithered ALINEA, random-policy actors, and the
storage-mandatory profile re-draw."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl.behaviour_controllers import (  # noqa: E402
    DitherController,
    build_mixture_plan,
    enforce_storage_mandatory,
    make_controller_from_spec,
)
from sumo_env.demand_profiles import ProfileFamily  # noqa: E402

M14_MIXTURE = {"alinea_wide": 0.25, "alinea_dither": 0.15, "store_flush": 0.25,
               "feedforward": 0.15, "random_signal": 0.10, "constant": 0.10}


def _fake_env(obs_dim: int = 23):
    from gymnasium import spaces

    return SimpleNamespace(T_ctrl=120, N_x=19, ramp_discharge_vph=1200.0, dt_ctrl=30.0, density_mean=20.0,
                           density_std=12.0, queue_scale=100.0, observe_ramp_demand=True, ramp_norm_vph=1000.0,
                           observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32))


def _counts(plan):
    c = {}
    for p in plan:
        c[p["controller"]["type"]] = c.get(p["controller"]["type"], 0) + 1
    return c


def test_m14_plan_counts_and_types():
    plan = build_mixture_plan(240, 13, M14_MIXTURE, per_entry_seeds=True)
    assert len(plan) == 240
    assert _counts(plan) == {"alinea_wide": 60, "alinea_dither": 36, "store_flush": 60,
                             "feedforward": 36, "random_signal": 24, "constant": 24}
    assert sorted(p["index"] for p in plan) == list(range(240))
    assert len({p["profile_draw"] for p in plan}) == 240 and len({p["sumo_seed"] for p in plan}) == 240



def test_per_entry_seeds_isolate_types():
    base = build_mixture_plan(120, 7, {"alinea_wide": 0.5, "constant": 0.5}, per_entry_seeds=True)
    more = build_mixture_plan(120, 7, {"alinea_wide": 0.5, "random_signal": 0.25, "constant": 0.25}, per_entry_seeds=True)
    specs = lambda plan, t: sorted(json.dumps(p["controller"], sort_keys=True) for p in plan if p["controller"]["type"] == t)
    assert specs(base, "alinea_wide") == specs(more, "alinea_wide")
    assert build_mixture_plan(50, 3, M14_MIXTURE, per_entry_seeds=True) == build_mixture_plan(50, 3, M14_MIXTURE, per_entry_seeds=True)



def test_dither_controller_bounds_and_writeback():
    env = _fake_env()
    plan = build_mixture_plan(4, 1, {"alinea_dither": 1.0}, per_entry_seeds=True)
    spec = plan[0]["controller"]
    spec["hold_steps"] = 2
    ctrl = make_controller_from_spec(spec, env)
    assert isinstance(ctrl, DitherController)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, (12, 23)).astype(np.float32)
    ctrl.reset(env)
    us = []
    for k in range(12):
        u = ctrl(obs[k])
        assert u.shape == (1,) and 0.0 <= float(u[0]) <= 1.0
        assert ctrl.inner._r_prev == pytest.approx(float(u[0]) * 1200.0)
        us.append(float(u[0]))
    eps = ctrl._eps
    ctrl2 = make_controller_from_spec(spec, env); ctrl2.reset(env)
    assert [float(ctrl2(obs[k])[0]) for k in range(12)] == us
    assert ctrl2._eps == eps



def test_alinea_wide_queue_override_denormalises_ramp_feature():
    env = _fake_env()
    spec = {"type": "alinea_wide", "ki": 20.0, "kp": 0.0, "rho_set": 30.0, "det": 12, "u_init": 0.5, "queue_max": 200.0}
    ctrl = make_controller_from_spec(spec, env)
    inner = ctrl.inner
    assert inner.queue_max == 200.0
    obs = np.zeros(23, dtype=np.float32); obs[19] = 0.7; obs[20] = 0.6
    assert inner._ramp_demand_vph(obs) == pytest.approx(600.0)


def test_enforce_storage_mandatory():
    family = ProfileFamily.load(PROJECT_ROOT / "configs/demand.yaml")
    plan = build_mixture_plan(40, 13, M14_MIXTURE, per_entry_seeds=True)
    summary = enforce_storage_mandatory(plan, family, 0.5, 2500.0)
    assert summary["n_mandatory"] == 20 and summary["frac_above_threshold"] >= 0.5
    mandatory = [p for p in plan if p["storage_mandatory"]]
    assert len(mandatory) == 20 and all(p["peak_total_vph"] > 2500.0 for p in mandatory)
    assert len({p["profile_draw"] for p in plan}) == 40
    for p in plan[:8]:
        assert family.sample_by_key("train", p["profile_draw"]).peak_total_vph == pytest.approx(p["peak_total_vph"])
    again = build_mixture_plan(40, 13, M14_MIXTURE, per_entry_seeds=True)
    enforce_storage_mandatory(again, family, 0.5, 2500.0)
    assert [p["profile_draw"] for p in again] == [p["profile_draw"] for p in plan]
