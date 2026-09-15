"""
Unit tests for the outflow-based shaped reward (Milestone 7).

Pure numpy — no SUMO, torch, or checkpoints required.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl.reward import RewardWeights, compute_reward, reward_terms

UNIT = RewardWeights(delta=1.0, beta=1.0, gamma=1.0, q_ref=3000.0, queue_norm=400.0, sigma_ref=6.0)


def test_terms_sum_to_reward():
    rho = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    t = reward_terms(rho, queue_length=200.0, outflow_vph=1500.0, weights=UNIT)
    assert t["reward"] == pytest.approx(-(t["outflow_penalty"] + t["queue_penalty"] + t["std_penalty"]))
    assert compute_reward(rho, 200.0, 1500.0, UNIT) == pytest.approx(t["reward"])
    assert t["reward"] <= 0.0


def test_component_values():
    rho = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    t = reward_terms(rho, queue_length=200.0, outflow_vph=1500.0, weights=UNIT)
    assert t["lost_outflow_frac"] == pytest.approx(0.5)
    assert t["outflow_penalty"] == pytest.approx(0.5)
    assert t["queue_penalty"] == pytest.approx((200.0 / 400.0) ** 2)
    assert t["std_penalty"] == pytest.approx(float(np.std(rho)) / 6.0)
    assert t["mean_density"] == pytest.approx(20.0)


def test_outflow_monotone_and_saturates_at_q_ref():
    rho = np.full(19, 18.0, dtype=np.float32)
    r_low = compute_reward(rho, 0.0, 1000.0, UNIT)
    r_mid = compute_reward(rho, 0.0, 2000.0, UNIT)
    r_cap = compute_reward(rho, 0.0, 3000.0, UNIT)
    r_over = compute_reward(rho, 0.0, 3500.0, UNIT)
    assert r_low < r_mid < r_cap
    # No reward for exceeding the reference: the term is a *lost*-outflow penalty.
    assert r_cap == pytest.approx(r_over)
    assert r_cap == pytest.approx(0.0)  # uniform density + no queue + full outflow


def test_weights_scale_terms_independently():
    rho = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    base = reward_terms(rho, 100.0, 1500.0, UNIT)
    heavy = reward_terms(
        rho, 100.0, 1500.0,
        RewardWeights(delta=3.0, beta=1.0, gamma=1.0, q_ref=3000.0, queue_norm=400.0, sigma_ref=6.0),
    )
    assert heavy["outflow_penalty"] == pytest.approx(3.0 * base["outflow_penalty"])
    assert heavy["queue_penalty"] == pytest.approx(base["queue_penalty"])
    assert heavy["std_penalty"] == pytest.approx(base["std_penalty"])


def test_none_outflow_requires_delta_zero():
    rho = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    with pytest.raises(ValueError, match="delta"):
        compute_reward(rho, 0.0, None, UNIT)
    two_term = RewardWeights(delta=0.0, beta=1.0, gamma=1.0, q_ref=3000.0, queue_norm=400.0, sigma_ref=6.0)
    t = reward_terms(rho, 100.0, None, two_term)
    assert t["outflow_penalty"] == 0.0
    assert t["reward"] == pytest.approx(-(t["queue_penalty"] + t["std_penalty"]))


def test_from_config_rejects_legacy_keys():
    with pytest.raises(ValueError, match="alpha"):
        RewardWeights.from_config({"alpha": 1.0, "beta": 1.0})
    with pytest.raises(ValueError, match="rho_freeflow"):
        RewardWeights.from_config({"rho_freeflow": 20.0})


def test_from_config_reads_new_keys_and_ignores_env_keys():
    w = RewardWeights.from_config(
        {"delta": 2.0, "beta": 0.5, "gamma": 0.8, "q_ref": 2500.0, "queue_norm": 300.0,
         "sigma_ref": 5.0, "warmup_s": 90}
    )
    assert (w.delta, w.beta, w.gamma) == (2.0, 0.5, 0.8)
    assert (w.q_ref, w.queue_norm, w.sigma_ref) == (2500.0, 300.0, 5.0)
    assert RewardWeights.from_config(None) == RewardWeights()


def test_input_validation():
    rho = np.array([10.0, 20.0], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_reward(np.array([[1.0, 2.0]]), 0.0, 1000.0, UNIT)
    with pytest.raises(ValueError):
        compute_reward(np.array([np.nan, 1.0]), 0.0, 1000.0, UNIT)
    with pytest.raises(ValueError):
        compute_reward(rho, -1.0, 1000.0, UNIT)
    with pytest.raises(ValueError):
        compute_reward(rho, 0.0, -5.0, UNIT)
    with pytest.raises(ValueError):
        RewardWeights(q_ref=0.0)
    with pytest.raises(ValueError):
        RewardWeights(delta=-1.0)


# ── M8 additions: offered reference, terminal cost, TTS form ──────────────

def test_offered_q_ref_modes():
    from rl.reward import offered_q_ref

    w = RewardWeights(q_ref=2476.0, q_ref_mode="offered", q_cap=2476.0)
    assert offered_q_ref(w, 1500.0, 400.0, 0.0, 1600.0, 30.0) == pytest.approx(1900.0)
    # backlog of 10 vehicles adds min(10 * 120, D - r) = 1200 -> capped at q_cap
    assert offered_q_ref(w, 1500.0, 400.0, 10.0, 1600.0, 30.0) == pytest.approx(2476.0)
    assert offered_q_ref(RewardWeights(q_ref=2476.0), 1500.0, 400.0, 50.0) == pytest.approx(2476.0)
    with pytest.raises(ValueError):
        RewardWeights(q_ref_mode="bogus")


def test_terminal_queue_cost_only_on_terminal_step():
    rho = np.full(19, 20.0, dtype=np.float32)
    w = RewardWeights(delta=1.0, beta=1.0, gamma=0.0, q_ref=2000.0, queue_norm=100.0, terminal_queue_weight=2.0)
    a = reward_terms(rho, 50.0, 2000.0, w, terminal=False)
    b = reward_terms(rho, 50.0, 2000.0, w, terminal=True)
    assert a["terminal_penalty"] == 0.0 and b["terminal_penalty"] == pytest.approx(2.0 * 0.25)
    assert b["reward"] == pytest.approx(a["reward"] - 0.5)


def test_tts_form_and_backlog_estimate():
    from rl.reward import backlog_estimate

    rho = np.full(19, 20.0, dtype=np.float32)          # 20 veh/km x 19 x 0.1 km = 38 vehicles on the road
    w = RewardWeights.from_config({"form": "tts", "tts_scale": 1.0, "dx_km": 0.1})
    t = reward_terms(rho, 12.0, 1800.0, w, backlog_veh=50.0, dt_ctrl_s=30.0)
    assert t["on_road_veh"] == pytest.approx(38.0)
    assert t["tts_step_veh_h"] == pytest.approx((38.0 + 12.0 + 50.0) * 30.0 / 3600.0)
    assert t["reward"] == pytest.approx(-t["tts_step_veh_h"])
    # three-term components are still reported (for parity logs) but do not enter the reward
    assert t["queue_penalty"] > 0.0
    assert backlog_estimate(100.0, 40.0, 38.0, 12.0) == pytest.approx(10.0)
    assert backlog_estimate(100.0, 90.0, 38.0, 12.0) == 0.0
    # an hour at 38 vehicles on the road, no queue, no backlog = 38 veh h
    total = sum(reward_terms(rho, 0.0, 2000.0, w)["reward"] for _ in range(120))
    assert total == pytest.approx(-38.0)


def test_rescore_return_matches_step_sum():
    from sumo_env.rollout import rescore_return

    K, Nx = 120, 19
    rng = np.random.default_rng(0)
    arrays = {"density": rng.uniform(5, 40, (Nx, K)).astype(np.float32), "outflow_vph": np.full(K, 1800.0, np.float32),
              "ramp_queue": np.linspace(0, 30, K).astype(np.float32), "mainline_demand": np.full(K, 1500.0, np.float32),
              "ramp_arrival": np.full(K, 400.0, np.float32), "q_ref": np.full(K, 1900.0, np.float32)}
    w = RewardWeights.from_config({"form": "tts"})
    r = rescore_return(arrays, w, warmup_s=90.0)
    # manual: skip the first 3 steps, cumulative conservation backlog
    tot = 0.0; off = srv = 0.0
    for k in range(K):
        off += 1900 * 30 / 3600; srv += 1800 * 30 / 3600
        on_road = arrays["density"][:, k].sum() * 0.1
        bl = max(off - srv - on_road - arrays["ramp_queue"][k], 0.0)
        if k >= 3:
            tot -= (on_road + arrays["ramp_queue"][k] + bl) * 30 / 3600
    assert r["return"] == pytest.approx(tot, rel=1e-5)
