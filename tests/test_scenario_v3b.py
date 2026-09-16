"""Scenario v3b (M14): 200 m ramp at 120 km/h = acceleration segment only, meter at its
start, discharge 1200 veh/h, no storage cap (queue priced by the TTS reward). Needs sumo."""

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

    cfg = dict(load_config(str(PROJECT_ROOT / "configs/experiments/round0_v3b.yaml"))["env"])
    cfg["project_root"] = str(PROJECT_ROOT)
    cfg["network_dir"] = str(tmp_path / "net_v3b")
    return SumoEnv(cfg)


def test_v3b_geometry_and_stop_line(tmp_path):
    env = _env(tmp_path)
    assert env.ramp_queue_max_veh is None and env.ramp_stopline_offset_m == 200.0 and env.ramp_discharge_vph == 1200.0
    assert env.merge_station_lanes == "mainline" and env.density_method == "occupancy"
    profile = DemandProfile.constant(1500.0, 900.0)
    env.reset(options={"profile": profile})
    import traci
    traci.switch(env._traci_label)
    assert abs(traci.lane.getMaxSpeed("ramp_0") - 33.33) < 0.05                      # 120 km/h ramp
    ramp_len = traci.lane.getLength("ramp_0")
    assert 175.0 < ramp_len <= 200.0                                                # 10° entry: the merge junction absorbs ~20 m of the 200 m edge
    env.step(np.array([1.0], dtype=np.float32))                                     # forces the depart position to resolve
    assert env._ramp_depart_pos not in (None, "free")
    assert float(env._ramp_depart_pos) == 0.0                                        # meter at the ramp start
    env.close()


def test_v3b_no_cap_queue_grows(tmp_path):
    env = _env(tmp_path)
    profile = DemandProfile.constant(1200.0, 900.0)      # 7.5 arrivals per 30 s step, policy closes the meter
    env.reset(options={"profile": profile})
    queues, overrides = [], []
    for _ in range(16):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        queues.append(info["queue_after"]); overrides.append(info["u_override"])
        assert info["u"] == 0.0 and info["queue_max_veh"] == -1.0
    env.close()
    assert not any(overrides)                             # no forced-open rule
    assert queues[-1] > 100.0                             # the queue simply accumulates (16 × 7.5 = 120)


def test_merge_station_density_uses_through_lane_only():
    from rl.sumo_env_wrapper import density_from_loops

    n, step = 30, 1.0
    # station 0: single loop at 20 % occupancy; station 1: ramp lane jammed (100 %), through lane 20 %
    occ = [np.array([20.0 * n]), np.array([100.0 * n, 20.0 * n])]
    counts = np.array([5.0, 8.0]); speeds = np.array([5.0 * 30.0, 8.0 * 20.0]); sc = np.array([5, 8])
    args = (counts, speeds, sc, occ, n, step, "occupancy", 5.0, 5.0, 142.857)
    d_mean, _, _ = density_from_loops(*args, merge_station_lanes="mean")
    d_main, _, _ = density_from_loops(*args, merge_station_lanes="mainline")
    assert abs(d_mean[0] - 40.0) < 1e-3 and abs(d_main[0] - 40.0) < 1e-3          # single-loop stations unchanged
    assert abs(d_mean[1] - (142.857 + 40.0) / 2) < 1e-3                             # old rule: lane average
    assert abs(d_main[1] - 40.0) < 1e-3                                             # new rule: through lane only
    with pytest.raises(ValueError):
        density_from_loops(*args, merge_station_lanes="sum")


def test_surrogate_env_reads_meter_discharge_from_scenario(tmp_path):
    """SurrogateVecEnv must use the scenario file's D when the env config does not set it (parity with SumoEnv)."""
    import inspect
    from rl import surrogate_vec_env as sve
    src = inspect.getsource(sve.SurrogateVecEnv.__init__)
    assert '_demand.get("ramp_discharge_vph", 1600.0)' in src and 'ramp_queue_max_veh", _demand.get' in src
    cfg = load_config(str(PROJECT_ROOT / "configs/rl/env_v3b.yaml"))["env"]
    sc = load_config(str(PROJECT_ROOT / cfg["sumo_config"]))["demand"]
    assert cfg["ramp_discharge_vph"] == sc["ramp_discharge_vph"] == 1200 and sc["ramp_queue_max_veh"] is None
