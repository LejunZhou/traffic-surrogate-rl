"""
Smoke test for SurrogateEnv (Milestone 4).

Loads the latest M3 checkpoint under runs/surrogate/, instantiates the
env on wilson's scenario constants, drives random + deterministic
rollouts, and runs gymnasium's check_env.

If no checkpoint is available (M3 hasn't finished yet on a fresh
machine), the tests skip rather than fail — keeps the repo green for
fresh clones.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_checkpoint() -> Path | None:
    from rl.surrogate_env import find_latest_checkpoint

    explicit = os.environ.get("SURROGATE_CHECKPOINT")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    return find_latest_checkpoint(PROJECT_ROOT / "runs" / "surrogate")


@pytest.fixture(scope="module")
def env_and_checkpoint():
    from rl.surrogate_env import SurrogateEnv

    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        pytest.skip(
            "No M3 checkpoint found under runs/surrogate/. "
            "Train one first: "
            "python -m surrogate.train --config configs/surrogate/baseline.yaml"
        )
    env = SurrogateEnv(
        surrogate_checkpoint=str(checkpoint),
        config={
            "sumo_config": str(PROJECT_ROOT / "configs" / "sumo" / "phase1_1.yaml"),
            "demand_profiles": [1500.0],
            "reward": {"alpha": 1.0, "beta": 0.1, "gamma": 1.0},
        },
    )
    return env, checkpoint


def test_spaces(env_and_checkpoint):
    env, _ = env_and_checkpoint
    assert env.N_x == 19
    assert env.T_ctrl == 120
    assert env.observation_space.shape == (22,)  # 19 density + demand + time + queue
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0


def test_reset_shape_and_queue(env_and_checkpoint):
    env, _ = env_and_checkpoint
    obs, info = env.reset(seed=0)
    assert obs.shape == (22,)
    assert obs.dtype == np.float32
    assert info["k"] == 0
    assert info["analytical_queue"] == 0.0
    # queue at reset (obs index 21) must be exactly 0
    assert obs[21] == 0.0
    # time at reset (obs index 20) must be exactly 0
    assert obs[20] == 0.0
    assert np.all(np.isfinite(obs))


def test_step_clipping_and_termination(env_and_checkpoint):
    env, _ = env_and_checkpoint
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(np.array([1.5]))
    assert info["u"] == 1.0
    assert not terminated and not truncated
    assert obs.shape == (22,)
    assert np.isfinite(reward)

    obs, reward, terminated, truncated, info = env.step(np.array([-0.3]))
    assert info["u"] == 0.0

    # Drive the rest of the episode with u=0.5 (queue should grow steadily).
    last_queue = info["analytical_queue"]
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
        assert obs.shape == (22,)
        assert np.isfinite(reward)
        # Queue should be monotone non-decreasing under u≤1.
        assert info["analytical_queue"] >= last_queue - 1e-9
        last_queue = info["analytical_queue"]
    assert info["k"] == env.T_ctrl


def test_queue_growth_under_closed_ramp(env_and_checkpoint):
    env, _ = env_and_checkpoint
    env.reset(seed=2)
    # Closed ramp for 10 steps: queue should grow exactly
    # (1 - 0) * 800 * 30 / 3600 = 6.6666... per step.
    expected_growth_per_step = env.ramp_demand_vph * env.dt_ctrl_s / 3600.0
    for k in range(10):
        _, _, _, _, info = env.step(np.array([0.0]))
        expected = (k + 1) * expected_growth_per_step
        assert info["analytical_queue"] == pytest.approx(expected, rel=1e-6)


def test_queue_constant_under_open_ramp(env_and_checkpoint):
    env, _ = env_and_checkpoint
    env.reset(seed=3)
    # Open ramp: queue should stay at 0 (never grows from 0).
    for _ in range(20):
        _, _, _, _, info = env.step(np.array([1.0]))
        assert info["analytical_queue"] == 0.0


def test_random_rollout(env_and_checkpoint):
    env, _ = env_and_checkpoint
    rng = np.random.default_rng(42)
    obs, _ = env.reset(seed=42)
    rewards = []
    for step in range(env.T_ctrl):
        action = rng.uniform(0.0, 1.0, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated:
            assert step == env.T_ctrl - 1
            break
    assert len(rewards) == env.T_ctrl
    # Under the M5c nonlinear reward, expected per-step magnitude is much
    # smaller than the M5/M5b linear form: alpha-term ~ 0 (mean rho rarely
    # crosses 20 under random actions), beta * (queue/100)^2 stays below
    # ~16 for queue < 400, gamma * std is ~5. Total per-step well within
    # (-50, 0). Tighter bound catches real regressions while staying safe.
    assert -50.0 < float(np.mean(rewards)) < 0.0


def test_check_env(env_and_checkpoint):
    from gymnasium.utils.env_checker import check_env

    env, _ = env_and_checkpoint
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
