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
    DEFAULT_SHARES,
    DitherController,
    build_mixture_plan,
    enforce_storage_mandatory,
    make_controller_from_spec,
)
from sumo_env.demand_profiles import ProfileFamily  # noqa: E402

MIX2 = {"alinea_wide": 0.35, "alinea_dither": 0.15, "random_policy": 0.10, "store_flush": 0.10,
        "feedforward": 0.05, "random_signal": 0.15, "constant": 0.10}


def _fake_env(obs_dim: int = 23):
    from gymnasium import spaces

    return SimpleNamespace(T_ctrl=120, N_x=19, ramp_discharge_vph=1600.0, dt_ctrl=30.0, density_mean=20.0,
                           density_std=12.0, queue_scale=100.0, observe_ramp_demand=True, ramp_norm_vph=1000.0,
                           observation_space=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32))


def _counts(plan):
    c = {}
    for p in plan:
        c[p["controller"]["type"]] = c.get(p["controller"]["type"], 0) + 1
    return c


def test_mix2_plan_counts_and_types():
    plan = build_mixture_plan(240, 13, MIX2, per_entry_seeds=True)
    assert len(plan) == 240
    assert _counts(plan) == {"alinea_wide": 84, "alinea_dither": 36, "random_policy": 24, "store_flush": 24,
                             "feedforward": 12, "random_signal": 36, "constant": 24}
    assert sorted(p["index"] for p in plan) == list(range(240))
    assert len({p["profile_draw"] for p in plan}) == 240 and len({p["sumo_seed"] for p in plan}) == 240


def test_legacy_fold_and_missing_legacy_policy():
    plan = build_mixture_plan(120, 0, DEFAULT_SHARES, legacy_policy_path=None)
    c = _counts(plan)
    assert "legacy_policy" not in c
    assert c["alinea"] == 48                      # 0.25 + 0.15 of 120
    assert c["feedforward"] == 8 and c["store_flush"] == 16   # one third of the 24 store_flush draws


def test_per_entry_seeds_isolate_types():
    base = build_mixture_plan(120, 7, {"alinea_wide": 0.5, "constant": 0.5}, per_entry_seeds=True)
    more = build_mixture_plan(120, 7, {"alinea_wide": 0.5, "random_signal": 0.25, "constant": 0.25}, per_entry_seeds=True)
    specs = lambda plan, t: sorted(json.dumps(p["controller"], sort_keys=True) for p in plan if p["controller"]["type"] == t)
    assert specs(base, "alinea_wide") == specs(more, "alinea_wide")
    assert build_mixture_plan(50, 3, MIX2, per_entry_seeds=True) == build_mixture_plan(50, 3, MIX2, per_entry_seeds=True)


def test_default_path_reproduces_round0_plan():
    """The shared-rng path must keep reproducing the stored round-0 plan (generated
    with --target-total, i.e. need-counts as shares and an absolute legacy path)."""
    plan_file = PROJECT_ROOT / "data/plant_v2/round0/generation_plan.json"
    legacy = PROJECT_ROOT / "runs/rl/ppo_sumo_m7_run7_range_m7_seed1_20260830_114758/best_model_multiseed.zip"
    if not plan_file.exists() or not legacy.exists():
        pytest.skip("round-0 store or run-7 checkpoint not present")
    stored = json.loads(plan_file.read_text())
    c = {}
    for q in stored:
        c[q["controller"]["type"]] = c.get(q["controller"]["type"], 0) + 1
    shares = {k: float(v) for k, v in {"constant": c.get("constant", 0), "alinea": c.get("alinea", 0),
                                         "store_flush": c.get("store_flush", 0) + c.get("feedforward", 0),
                                         "legacy_policy": c.get("legacy_policy", 0), "random_signal": c.get("random_signal", 0)}.items() if v > 0}
    plan = build_mixture_plan(len(stored), 0, shares, legacy_policy_path=str(legacy.resolve()), sumo_seed_base=50000)
    assert all(p["controller"] == q["controller"] and p["profile_draw"] == q["profile_draw"] and p["sumo_seed"] == q["sumo_seed"]
               for p, q in zip(plan, stored))


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
        assert ctrl.inner._r_prev == pytest.approx(float(u[0]) * 1600.0)
        us.append(float(u[0]))
    eps = ctrl._eps
    ctrl2 = make_controller_from_spec(spec, env); ctrl2.reset(env)
    assert [float(ctrl2(obs[k])[0]) for k in range(12)] == us
    assert ctrl2._eps == eps


def test_random_policy_controller():
    env = _fake_env()
    spec = {"type": "random_policy", "seed": 5, "log_std_init": -2.0, "init_u": 0.3, "action_gain": 0.01}
    ctrl = make_controller_from_spec(spec, env); ctrl.reset(env)
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 1, (200, 23)).astype(np.float32)
    us = np.array([float(ctrl(o)[0]) for o in obs])
    assert us.shape == (200,) and us.min() >= 0.0 and us.max() <= 1.0
    assert abs(us.mean() - 0.3) < 0.03            # near-constant + exploration noise
    ctrl2 = make_controller_from_spec(spec, env); ctrl2.reset(env)
    assert np.allclose([float(ctrl2(o)[0]) for o in obs], us)
    wide = make_controller_from_spec({**spec, "action_gain": 1.0, "log_std_init": -6.0}, env); wide.reset(env)
    uw = np.array([float(wide(o)[0]) for o in obs])
    assert uw.std() > 0.05                        # state-dependent map
    with pytest.raises(ValueError):
        ctrl.reset(_fake_env(obs_dim=25))


def test_alinea_wide_queue_override_denormalises_v2_ramp_feature():
    env = _fake_env()
    spec = {"type": "alinea_wide", "ki": 20.0, "kp": 0.0, "rho_set": 30.0, "det": 12, "u_init": 0.5, "queue_max": 200.0}
    ctrl = make_controller_from_spec(spec, env)
    inner = ctrl.inner
    assert inner.queue_max == 200.0
    obs = np.zeros(23, dtype=np.float32); obs[19] = 0.7; obs[20] = 0.6
    assert inner._ramp_demand_vph(obs) == pytest.approx(600.0)


def test_enforce_storage_mandatory():
    family = ProfileFamily.load(PROJECT_ROOT / "configs/profiles/family_v1.yaml")
    plan = build_mixture_plan(40, 13, MIX2, per_entry_seeds=True)
    summary = enforce_storage_mandatory(plan, family, 0.5, 2500.0)
    assert summary["n_mandatory"] == 20 and summary["frac_above_threshold"] >= 0.5
    mandatory = [p for p in plan if p["storage_mandatory"]]
    assert len(mandatory) == 20 and all(p["peak_total_vph"] > 2500.0 for p in mandatory)
    assert len({p["profile_draw"] for p in plan}) == 40
    for p in plan[:8]:
        assert family.sample_by_key("train", p["profile_draw"]).peak_total_vph == pytest.approx(p["peak_total_vph"])
    again = build_mixture_plan(40, 13, MIX2, per_entry_seeds=True)
    enforce_storage_mandatory(again, family, 0.5, 2500.0)
    assert [p["profile_draw"] for p in again] == [p["profile_draw"] for p in plan]
