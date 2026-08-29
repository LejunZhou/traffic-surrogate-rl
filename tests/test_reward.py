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
