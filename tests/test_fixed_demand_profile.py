"""Fixed-template demand and route timing regression tests."""
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sumo_env.demand_profiles import sample_joint_demand_profile
from sumo_env.network_builder import _write_routes

@pytest.fixture
def scenario():
    cfg = yaml.safe_load((ROOT / "configs/sumo/phase1_1.yaml").read_text())
    cfg["simulation"].update(duration_s=120, dt_ctrl_s=30)
    return cfg


def test_acc_reference_demand_profile_and_scaled_family():
    cfg = yaml.safe_load(
        (ROOT / "configs/experiments/dataset_time_varying.yaml").read_text()
    )
    spec = cfg["dataset"]["demand_profile_family"]
    nominal_main, nominal_ramp = sample_joint_demand_profile(
        spec, 120, 30, np.random.default_rng(42), sample_index=0
    )

    np.testing.assert_array_equal(nominal_main[:20], 1250)
    np.testing.assert_array_equal(nominal_main[20:40], 1600)
    assert nominal_main[40] == 1600 and nominal_main[49] == 1780
    np.testing.assert_array_equal(nominal_main[50:90], 1800)
    assert nominal_main[90] == 1800 and nominal_main[99] == 1170
    np.testing.assert_array_equal(nominal_main[100:], 1100)

    np.testing.assert_array_equal(nominal_ramp[:20], 250)
    np.testing.assert_array_equal(nominal_ramp[20:40], 400)
    assert nominal_ramp[40] == 400 and nominal_ramp[49] == 850
    np.testing.assert_array_equal(nominal_ramp[50:80], 900)
    assert nominal_ramp[80] == 900 and nominal_ramp[89] == 360
    np.testing.assert_array_equal(nominal_ramp[90:], 300)

    a = sample_joint_demand_profile(spec, 120, 30, np.random.default_rng(7), sample_index=1)
    b = sample_joint_demand_profile(spec, 120, 30, np.random.default_rng(7), sample_index=1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert 990 <= a[0].min() <= a[0].max() <= 1980
    assert 225 <= a[1].min() <= a[1].max() <= 990


def test_route_intervals_match_labels_and_zero_demand(tmp_path, scenario):
    scenario["demand"]["mainline_demand_profile"] = [1500, 1500, 0, 2100]
    path = tmp_path / "routes.xml"
    _write_routes(path, scenario)
    flows = ET.parse(path).getroot().findall("flow")
    assert [(float(f.get("begin")), float(f.get("end")), float(f.get("vehsPerHour"))) for f in flows] == [
        (0, 60, 1500), (90, 120, 2100)
    ]


def test_route_intervals_prepend_constant_warmup_and_shift_profile(tmp_path, scenario):
    scenario["simulation"]["warmup_s"] = 60
    scenario["demand"]["mainline_demand_profile"] = [1500, 1500, 0, 2100]
    path = tmp_path / "routes_warmup.xml"
    _write_routes(path, scenario)
    flows = ET.parse(path).getroot().findall("flow")
    assert [
        (float(f.get("begin")), float(f.get("end")), float(f.get("vehsPerHour")))
        for f in flows
    ] == [(0, 120, 1500), (150, 180, 2100)]


