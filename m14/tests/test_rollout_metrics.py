"""Episode metrics retain queue statistics at their original sampling rates."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sumo_env.rollout import rollout_episode  # noqa: E402


class QueueStatsEnv:
    T_ctrl = 2
    N_x = 2
    dt_ctrl = 30.0
    x_grid = np.array([0.0, 100.0])

    def __init__(self, queue_stats):
        self.queue_stats = queue_stats

    def reset(self, options):
        self.k = 0
        return np.zeros(2), {}

    def step(self, action):
        self.k += 1
        info = {
            "density": np.zeros(2),
            "outflow_vph": 0.0,
            "queue_after": float(self.k * 2),
        }
        if self.k == self.T_ctrl:
            info.update(self.queue_stats)
        return np.zeros(2), 0.0, self.k == self.T_ctrl, False, info


@pytest.mark.parametrize(
    "queue_stats",
    [
        {},
        {"episode_queue_mean": 1.75, "episode_queue_max": 7.0},
        {"episode_queue_mean": 1.75},
        {"episode_queue_max": 7.0},
    ],
)
def test_rollout_preserves_simulation_step_queue_statistics(queue_stats):
    result = rollout_episode(QueueStatsEnv(queue_stats), lambda obs: 0.5)
    metrics = result["metrics"]

    # The 30 s observations miss a 1 s peak of 7 vehicles; retain both values.
    assert metrics["max_queue"] == 4.0
    np.testing.assert_array_equal(result["arrays"]["ramp_queue"], [2.0, 4.0])
    for key in ("episode_queue_mean", "episode_queue_max"):
        if key in queue_stats:
            assert metrics[key] == queue_stats[key]
            assert isinstance(metrics[key], float)
        else:
            assert key not in metrics
