"""Ramp measurements must follow SUMO entry events, including delayed requests."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("traci")

runner = importlib.import_module("sumo_env.run_simulation")


class DelayedTraCI:
    """Accept requests immediately, but confirm road entries on a later step."""

    def __init__(self, departures, discards=None, reject_once=None):
        self.departures = departures
        self.discards = discards or {}
        self.reject_once = set(reject_once or ())
        self.time = 0
        self.pending = set()
        self.accepted = []
        self.departed = []
        self.closed = False
        self.simulation = self
        self.exceptions = runner.traci.exceptions
        self.vehicle = SimpleNamespace(add=self.add)
        self.edge = SimpleNamespace(getLastStepVehicleNumber=lambda _: 0)
        self.inductionloop = SimpleNamespace(
            getLastStepVehicleNumber=lambda _: 0,
            getLastStepMeanSpeed=lambda _: -1.0,
            getLastStepOccupancy=lambda _: 0.0,
        )

    def start(self, command):
        pass

    def close(self):
        self.closed = True

    def add(self, vehID, **kwargs):
        if vehID in self.reject_once:
            self.reject_once.remove(vehID)
            raise self.exceptions.TraCIException("rejected request")
        assert vehID not in self.accepted
        self.accepted.append(vehID)
        self.pending.add(vehID)

    def simulationStep(self):
        self.time += 1
        entering = {v for v in self.pending if self.departures.get(v) == self.time}
        discarded = {v for v in self.pending if self.discards.get(v) == self.time}
        self.pending -= entering | discarded
        # Mainline departures must never reduce the ramp queue.
        self.departed = [f"mainline_{self.time}", *sorted(entering)]

    def getTime(self):
        return self.time

    def getDepartedIDList(self):
        return self.departed

    def getPendingVehicles(self):
        return sorted(self.pending)

    def getStartingTeleportNumber(self):
        return 0


@pytest.fixture
def config():
    cfg = yaml.safe_load((ROOT / "configs/sumo/phase1_1.yaml").read_text())
    cfg["simulation"].update(duration_s=16, dt_ctrl_s=4, step_length_s=1)
    cfg["demand"].update(ramp_demand_vph=900, ramp_discharge_vph=3600)
    return cfg


def install_delayed_sumo(monkeypatch, **kwargs):
    fake = DelayedTraCI({"ramp_0": 6, "ramp_1": 14, "ramp_3": 16}, **kwargs)
    monkeypatch.setattr(runner, "traci", fake)
    return fake


def assert_delayed_measurements(data, first_pending=1):
    # Arrivals at seconds 4, 8, 12, 16; actual entries at 6, 14, 16.
    np.testing.assert_array_equal(data["ramp_departed_count"], [0, 1, 0, 2])
    np.testing.assert_array_equal(data["ramp_pending_count"], [first_pending, 1, 2, 1])
    np.testing.assert_array_equal(data["ramp_queue"], [1, 1, 2, 1])
    np.testing.assert_array_equal(data["ramp_inflow_vph"], [0, 900, 0, 1800])
    np.testing.assert_array_equal(data["ramp_control"], [0, 0.25, 0, 0.5])
    np.testing.assert_array_equal(data["ramp_control_cmd"], np.ones(4))
    np.testing.assert_array_equal(
        data["ramp_queue"] + np.cumsum(data["ramp_departed_count"]), [1, 2, 3, 4]
    )
    assert data["ramp_departed_count"].dtype == np.int32
    assert data["ramp_inflow_vph"].dtype == np.float32


@pytest.mark.parametrize("reject_once", [None, {"ramp_0"}])
def test_delayed_and_rejected_requests_keep_demand_queued(monkeypatch, config, reject_once):
    fake = install_delayed_sumo(monkeypatch, reject_once=reject_once)
    data = runner.run_simulation("net", "routes", "detectors", np.ones(4), config)
    # A rejected first request is still demand, but is not yet pending in SUMO.
    assert_delayed_measurements(data, first_pending=0 if reject_once else 1)
    # One submitted ID per arrival despite multiple seconds of insertion delay.
    assert fake.accepted == ["ramp_0", "ramp_1", "ramp_2", "ramp_3"]
    assert data["metadata"]["insert_success"] == 4
    assert data["metadata"]["insert_rejected"] == bool(reject_once)
    assert data["metadata"]["ramp_departed_total"] == 3
    assert data["metadata"]["ramp_pending_final"] == 1
    assert data["metadata"]["ramp_flow_measurement"] == "confirmed_departures"
    assert fake.closed


def test_discarded_requests_can_be_retried_without_losing_arrivals(monkeypatch, config):
    fake = DelayedTraCI(
        {"ramp_1": 7, "ramp_2": 9, "ramp_4": 15, "ramp_5": 16},
        discards={"ramp_0": 5, "ramp_3": 13},
    )
    monkeypatch.setattr(runner, "traci", fake)
    data = runner.run_simulation("net", "routes", "detectors", np.ones(4), config)
    np.testing.assert_array_equal(data["ramp_departed_count"], [0, 1, 1, 2])
    np.testing.assert_array_equal(data["ramp_queue"], [1, 1, 1, 0])
    np.testing.assert_array_equal(data["ramp_pending_count"], [1, 1, 1, 0])
    assert data["metadata"]["insert_success"] == 6
    assert data["metadata"]["ramp_departed_total"] == 4
    assert data["metadata"]["ramp_discarded_total"] == 2


def test_open_loop_measures_entries_but_preserves_command_branch_input(monkeypatch, config):
    install_delayed_sumo(monkeypatch)
    config["demand"]["ramp_model"] = "open_loop"
    data = runner.run_simulation("net", "routes", "detectors", np.ones(4), config)
    np.testing.assert_array_equal(data["ramp_departed_count"], [0, 1, 0, 2])
    np.testing.assert_array_equal(data["ramp_inflow_vph"], [0, 900, 0, 1800])
    np.testing.assert_array_equal(data["ramp_queue"], np.zeros(4))
    np.testing.assert_array_equal(data["ramp_control"], np.ones(4))


@pytest.mark.parametrize("writer", ["dataset", "single"])
def test_npz_writers_save_confirmed_entries(monkeypatch, config, tmp_path, writer):
    install_delayed_sumo(monkeypatch)
    config["output"].update(raw_dir=str(tmp_path), network_dir=str(tmp_path / "network"))
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(yaml.safe_dump(config))

    def build_network(output_dir, cfg):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"net": "net", "route": "routes"}

    module = importlib.import_module(
        "sumo_env.dataset_generation" if writer == "dataset" else "run_rollout"
    )
    monkeypatch.setattr(module, "build_network", build_network)
    monkeypatch.setattr(module, "build_detector_file", lambda *args: "detectors")
    monkeypatch.setattr(module, "plot_trajectory", lambda **kwargs: None)
    if writer == "dataset":
        ds = yaml.safe_load((ROOT / "configs/experiments/dataset_constant_inflow.yaml").read_text())
        ds["base_sumo_config"] = str(scenario)
        ds["dataset"]["n_samples"] = 1
        ds["output"].update(config["output"], save_heatmaps=False)
        ds_path = tmp_path / "dataset.yaml"
        ds_path.write_text(yaml.safe_dump(ds))
        monkeypatch.setattr(module, "sample_ramp_control", lambda *args: np.ones(4))
        paths = module.generate_dataset(str(ds_path), project_root=ROOT)
        saved = paths[0]
    else:
        monkeypatch.setattr(sys, "argv", ["run_rollout.py", "--config", str(scenario), "--ramp-rate", "1"])
        module.main()
        saved = tmp_path / "sim_0000.npz"
    with np.load(saved, allow_pickle=False) as data:
        assert_delayed_measurements(data)
        assert data["ramp_flow_measurement"].item() == "confirmed_departures"
        assert data["ramp_model"].item() == "metered_queue"
        assert data["ramp_ref_vph"].item() == 3600
