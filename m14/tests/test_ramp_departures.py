"""The M14 environment prices waiting arrivals until SUMO confirms entry."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("traci")
from rl import sumo_env_wrapper as wrapper


class DelayedTraCI:
    def __init__(self, departures, discards=None, reject_once=None):
        self.departures = departures
        self.discards = discards or {}
        self.reject_once = set(reject_once or ())
        self.time = 0
        self.pending = set()
        self.accepted = []
        self.departed = []
        self.simulation = self
        self.exceptions = wrapper.traci.exceptions
        self.vehicle = SimpleNamespace(add=self.add)
        self.edge = SimpleNamespace(getLastStepVehicleNumber=lambda _: 0)
        self.inductionloop = SimpleNamespace(
            getLastStepVehicleNumber=lambda _: 0,
            getLastStepMeanSpeed=lambda _: -1.0,
            getLastStepOccupancy=lambda _: 0.0,
        )

    def switch(self, label):
        pass

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
        self.departed = [f"mainline_{self.time}", *sorted(entering)]

    def getTime(self):
        return self.time

    def getDepartedIDList(self):
        return self.departed

    def getPendingVehicles(self):
        return sorted(self.pending)

    def getStartingTeleportNumber(self):
        return 0

    def getArrivedNumber(self):
        return 0


def interval_env():
    """Initialize the interval bookkeeping without launching a SUMO process."""
    env = wrapper.SumoEnv.__new__(wrapper.SumoEnv)
    for name, value in {
        "_traci_label": "test", "N_x": 1, "det_ids_per_lane": [["loop"]],
        "dt_ctrl_steps": 4, "dt_ctrl": 4, "step_len": 1, "warmup_s": 0,
        "ramp_discharge_vph": 3600, "_ramp_arrival_accumulator": 0.0,
        "_ramp_release_accumulator": 0.0, "_virtual_queue_length": 0.0,
        "_ramp_depart_pos": "0", "_veh_counter": 0, "_insert_attempts": 0,
        "_insert_success": 0, "_insert_rejected": 0, "_teleports": 0,
        "_arrived_vehicles": 0, "_queue_samples": [], "_physical_ramp_samples": [],
        "density_method": "occupancy", "vehicle_length_m": 5.0,
        "occupancy_effective_length_m": 5.0, "jam_density_veh_km": 142.857,
        "merge_station_lanes": "mainline", "max_depart_delay_s": -1,
    }.items():
        setattr(env, name, value)
    env._reset_insertion_bookkeeping()
    return env


@pytest.mark.parametrize("reject_once", [None, {"ramp_0"}])
def test_delayed_and_rejected_requests_keep_demand_queued(monkeypatch, reject_once):
    fake = DelayedTraCI({"ramp_0": 6, "ramp_1": 14, "ramp_3": 16}, reject_once=reject_once)
    monkeypatch.setattr(wrapper, "traci", fake)
    env = interval_env()
    infos = [env._advance_control_interval(1.0, 900.0)[3] for _ in range(4)]
    np.testing.assert_array_equal([i["ramp_released"] for i in infos], [0, 1, 0, 2])
    np.testing.assert_array_equal([i["queue_after"] for i in infos], [1, 1, 2, 1])
    np.testing.assert_array_equal([i["ramp_inflow_vph"] for i in infos], [0, 900, 0, 1800])
    np.testing.assert_array_equal(
        np.array([i["queue_after"] for i in infos]) + np.cumsum([i["ramp_released"] for i in infos]),
        [1, 2, 3, 4],
    )
    assert infos[0]["pending_ramp"] == (0 if reject_once else 1)
    assert fake.accepted == ["ramp_0", "ramp_1", "ramp_2", "ramp_3"]
    assert env._insert_success == 4 and env._insert_rejected == bool(reject_once)


def test_discarded_requests_are_retried_without_losing_arrivals(monkeypatch):
    fake = DelayedTraCI(
        {"ramp_1": 7, "ramp_2": 9, "ramp_4": 15, "ramp_5": 16},
        discards={"ramp_0": 5, "ramp_3": 13},
    )
    monkeypatch.setattr(wrapper, "traci", fake)
    env = interval_env()
    infos = [env._advance_control_interval(1.0, 900.0)[3] for _ in range(4)]
    np.testing.assert_array_equal([i["ramp_released"] for i in infos], [0, 1, 1, 2])
    np.testing.assert_array_equal([i["queue_after"] for i in infos], [1, 1, 1, 0])
    assert env._insert_success == 6 and env._ramp_departed == 4
    assert env._discarded_ramp == 2
