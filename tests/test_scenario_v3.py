"""Scenario v3 (M14): 60 km/h ramp, stop line 100 m before the merge, finite
ramp storage enforced as an actuator constraint in SumoEnv. Needs the sumo binary."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sumo_env.demand_profiles import DemandProfile  # noqa: E402
from utils.config import load_config  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("sumo") is None, reason="sumo binary not available")


def _env(tmp_path):
    from rl.sumo_env_wrapper import SumoEnv

    cfg = dict(load_config(str(PROJECT_ROOT / "configs/experiments/round0_v3.yaml"))["env"])
    cfg["project_root"] = str(PROJECT_ROOT)
    cfg["network_dir"] = str(tmp_path / "net_v3")
    return SumoEnv(cfg)


def test_v3_geometry_and_stop_line(tmp_path):
    env = _env(tmp_path)
    assert env.ramp_queue_max_veh == 28.0 and env.ramp_stopline_offset_m == 100.0
    profile = DemandProfile.constant(1500.0, 900.0)
    env.reset(options={"profile": profile})
    import traci
    traci.switch(env._traci_label)
    assert abs(traci.lane.getMaxSpeed("ramp_0") - 16.67) < 0.05                      # 60 km/h ramp
    ramp_len = traci.lane.getLength("ramp_0")
    assert 280.0 < ramp_len <= 300.0
    env.step(np.array([1.0], dtype=np.float32))                                     # forces the depart position to resolve
    assert env._ramp_depart_pos not in (None, "free")
    assert abs(float(env._ramp_depart_pos) - (ramp_len - 100.0)) < 1e-6
    env.close()


def test_v3_queue_cap_binds(tmp_path):
    env = _env(tmp_path)
    profile = DemandProfile.constant(1200.0, 900.0)      # 7.5 arrivals per 30 s step, policy closes the meter
    env.reset(options={"profile": profile})
    queues, overrides, applied = [], [], []
    for _ in range(16):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        queues.append(info["queue_after"] if "queue_after" in info else info["analytical_queue"])
        overrides.append(info["u_override"]); applied.append(info["u"])
        assert info["u_requested"] == 0.0
    env.close()
    assert max(queues) <= 28.0 + 1.0                       # cap holds (SUMO inserts with at most one step of lag)
    assert any(overrides) and max(applied) > 0.0           # the meter was forced open
    assert queues[-1] > 20.0                               # and the queue really sits near the cap, not drained
