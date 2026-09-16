"""Scenario v4 (M15): three-lane mainline, v3b ramp, lane-averaged observation, breakdown
threshold 30 veh/km on the through-lane mean, demand family v3 scaled to the 3-lane capacity.
The SUMO tests skip without a sumo binary; the config / metric tests run everywhere."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from sumo_env.demand_profiles import DemandProfile, ProfileFamily, load_profile_set  # noqa: E402
from sumo_env.rollout import breakdown_flags, episode_metrics  # noqa: E402
from utils.config import load_config  # noqa: E402

needs_sumo = pytest.mark.skipif(shutil.which("sumo") is None, reason="sumo binary not available")
SCENARIO = PROJECT_ROOT / "configs/sumo/scenario_v4.yaml"
ROUND0 = PROJECT_ROOT / "configs/experiments/round0_v4.yaml"
ENV_OVERLAY = PROJECT_ROOT / "configs/rl/env_v4.yaml"
FAMILY = PROJECT_ROOT / "configs/profiles/family_v3.yaml"


def _env(tmp_path):
    from rl.sumo_env_wrapper import SumoEnv

    cfg = dict(load_config(str(ROUND0))["env"])
    cfg["project_root"] = str(PROJECT_ROOT)
    cfg["network_dir"] = str(tmp_path / "net_v4")
    cfg.pop("density_stats_from", None)
    return SumoEnv(cfg)


@needs_sumo
def test_v4_geometry_three_lanes(tmp_path):
    env = _env(tmp_path)
    assert env.N_x == 19 and env.merge_station_lanes == "mainline" and env.density_method == "occupancy"
    assert env.breakdown_density_veh_km == 30.0
    assert env.ramp_queue_max_veh is None and env.ramp_stopline_offset_m == 200.0 and env.ramp_discharge_vph == 1200.0
    loops = [len(l) for l in env.det_ids_per_lane]
    assert loops[12] == 4 and all(n == 3 for i, n in enumerate(loops) if i != 12)   # 1300 m = accel edge: 3 through + 1 accel lane
    env.reset(options={"profile": DemandProfile.constant(4500.0, 600.0)})
    import traci
    traci.switch(env._traci_label)
    assert traci.edge.getLaneNumber("highway_pre") == 3 and traci.edge.getLaneNumber("highway_post") == 3
    assert traci.edge.getLaneNumber("highway_accel") == 4
    assert abs(traci.lane.getMaxSpeed("ramp_0") - 33.33) < 0.05
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert float(env._ramp_depart_pos) == 0.0                                        # meter at the ramp start
    assert info["teleports"] == 0 if "teleports" in info else True
    assert len(env.last_per_lane_density) == 19 and env.last_per_lane_density[12].shape == (4,)
    env.close()


def test_merge_station_density_is_mean_over_through_lanes():
    from rl.sumo_env_wrapper import density_from_loops

    n, step = 30, 1.0
    # station 0: three through lanes 20/20/20 % -> 40 veh/km; station 1 (merge): accel lane 100 %,
    # through lanes 38.5/9.5/9.5 % (lane-0 jam 77 with free neighbours 19) -> mean 38.3
    occ = [np.array([20.0, 20.0, 20.0]) * n, np.array([100.0, 38.5, 9.5, 9.5]) * n]
    counts = np.array([15.0, 20.0]); speeds = np.array([15.0 * 30.0, 20.0 * 10.0]); sc = np.array([15, 20])
    d, _, _ = density_from_loops(counts, speeds, sc, occ, n, step, "occupancy", 5.0, 5.0, 142.857, merge_station_lanes="mainline")
    assert abs(d[0] - 40.0) < 1e-3
    assert abs(d[1] - (77.0 + 19.0 + 19.0) / 3) < 1e-3
    assert d[1] < 60.0 < 77.0                                       # why the v3b threshold (60) would never fire on v4


def test_breakdown_threshold_is_a_parameter():
    rho = np.full((19, 120), 15.0); rho[10, 40:] = 38.0             # lane-0 jam seen through the lane mean
    assert not breakdown_flags(rho)["breakdown"]                     # single-lane default 60
    bf = breakdown_flags(rho, threshold=30.0)
    assert bf["breakdown"] and bf["breakdown_onset_min"] == 20.0 and bf["breakdown_density_veh_km"] == 30.0
    arrays = {"density": rho, "ramp_queue": np.zeros(120), "outflow_vph": np.full(120, 5000.0),
              "mainline_demand": np.full(120, 4500.0), "ramp_arrival": np.full(120, 600.0),
              "reward": np.zeros(120), "action": np.ones(120)}
    m = episode_metrics(arrays, 30.0, 0.1, breakdown_density=30.0)
    assert m["breakdown"] and m["breakdown_density_veh_km"] == 30.0
    assert not episode_metrics(arrays, 30.0, 0.1)["breakdown"]


def test_scenario_file_sets_lanes_and_threshold():
    sc = load_config(str(SCENARIO))
    assert sc["network"]["num_lanes"] == 3 and sc["network"]["ramp_entry_angle_deg"] == 10.0
    assert sc["detectors"]["merge_station_lanes"] == "mainline" and sc["detectors"]["breakdown_density_veh_km"] == 30.0
    assert sc["demand"]["ramp_discharge_vph"] == 1200 and sc["demand"]["ramp_queue_max_veh"] is None
    assert not sc["vehicle"].get("lane_change")                      # LC2013 defaults (user decision 2026-09-15)


def test_e0_constants_follow_the_round0_config():
    from run_scenario_characterisation import _e0_constants

    storage, ff, ins = _e0_constants(load_config(str(PROJECT_ROOT / "configs/experiments/round0_v3b.yaml")))
    assert (storage, ff, ins) == (2500.0, [2300.0, 2400.0, 2500.0], 1040.0)   # v3b unchanged
    cfg = load_config(str(ROUND0))
    storage, ff, ins = _e0_constants(cfg)
    # lane-aware rule (M15): per-lane merge load d / 3 + r against the ~2450 veh/h lane-0 capacity
    assert storage == float(cfg["dataset"]["storage_mandatory_vph"]) and 2300.0 <= storage <= 2600.0
    assert len(ff) == 3 and max(ff) <= storage + 1e-6 and min(ff) >= storage - 200.0
    assert ins >= 800.0                                              # >= any ramp rate of the family
    lo, hi = cfg["dataset"]["feedforward_capacity_vph"]
    assert lo < storage <= hi
    p = DemandProfile.constant(5400.0, 700.0)
    assert abs(p.peak_merge_load_vph(3) - 2500.0) < 1e-3 and p.peak_merge_load_vph(1) == p.peak_total_vph


def test_family_v3_is_scaled_to_the_three_lane_capacity():
    cfg = load_config(str(ROUND0))
    C_lane = float(cfg["dataset"]["storage_mandatory_vph"])
    fam = ProfileFamily.load(FAMILY)
    assert fam.version == 3
    peaks = []
    for i in range(400):
        p = fam.sample_by_key("train", i)
        assert p.K == 120 and p.ramp_blocks.max() <= 800.0 + 1e-3     # meter cap unchanged (2/3 of D = 1200)
        assert p.mainline_blocks.max() <= 5750.0 + 1e-3               # mainline alone (~6150 served) never jams
        fam_m = p.params["mainline"]["family"]
        if fam_m == "peak":
            assert np.ptp(p.mainline_blocks[9:]) < 5.0, p.params    # over by minute 45
        elif fam_m == "step":
            assert p.params["mainline"]["t2"] <= 45.0 + 1e-6
        peaks.append(p.peak_merge_load_vph(3))
    frac_storage = float(np.mean(np.asarray(peaks) > C_lane))
    assert 0.25 <= frac_storage <= 0.65, frac_storage                # a storage-mandatory regime exists, not the whole family
    for name, n in (("val", 18), ("test", 30), ("ood", 12)):
        ps = load_profile_set(PROJECT_ROOT / "configs/profiles/v3" / f"{name}.json")
        assert len(ps) == n and all(p.ramp_blocks.max() <= 800.0 + 1e-3 for p in ps)


def test_env_v4_overlay_is_consistent_with_the_scenario():
    env = load_config(str(ENV_OVERLAY))["env"]
    sc = load_config(str(PROJECT_ROOT / env["sumo_config"]))
    assert env["sumo_config"].endswith("scenario_v4.yaml") and sc["network"]["num_lanes"] == 3
    assert env["ramp_discharge_vph"] == sc["demand"]["ramp_discharge_vph"] == 1200
    assert env["profiles"]["family"].endswith("family_v3.yaml") and env["eval_profiles"].endswith("v3/val.json")
    r0 = load_config(str(ROUND0))
    assert r0["env"]["sumo_config"] == env["sumo_config"] and r0["env"]["profiles"]["family"] == env["profiles"]["family"]
    assert env["observation"]["demand_norm"] == r0["env"]["observation"]["demand_norm"] == 6000.0   # ~ mainline capacity


def test_surrogate_env_reads_breakdown_threshold_from_scenario():
    import inspect
    from rl import surrogate_vec_env as sve
    src = inspect.getsource(sve.SurrogateVecEnv.__init__)
    assert 'breakdown_density_veh_km' in src and '_dets.get("breakdown_density_veh_km", 60.0)' in src
