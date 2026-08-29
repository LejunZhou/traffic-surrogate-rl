"""
Metered on-ramp queue shared by the dataset generator (run_simulation.py)
and, in spirit, SumoEnv (which carries the same accumulators inline plus
SUMO pending-vehicle bookkeeping).

Vehicles arrive at `arrival_vph` and wait in a virtual queue upstream of the
meter; the meter releases at most u * discharge_vph, and never more than the
queue holds. With discharge_vph > arrival_vph the released flow can exceed
the arrival rate while a queue exists (e.g. u = 1 after a closure), which is
what makes a backlog drainable (M7 progress §7.10).
"""

from __future__ import annotations


class MeteredRampQueue:
    def __init__(self, arrival_vph: float, discharge_vph: float, step_len_s: float) -> None:
        if arrival_vph < 0.0 or discharge_vph <= 0.0 or step_len_s <= 0.0:
            raise ValueError("arrival_vph >= 0, discharge_vph > 0 and step_len_s > 0 required")
        self.arrival_rate = float(arrival_vph) / 3600.0
        self.discharge_rate = float(discharge_vph) / 3600.0
        self.step_len = float(step_len_s)
        self.queue = 0.0
        self._arrival_acc = 0.0
        self._release_acc = 0.0
        self.total_arrivals = 0
        self.total_released = 0

    def step(self, u: float) -> int:
        """Advance one sub-step: add arrivals, return how many vehicles the meter
        may release now (green fraction u in [0, 1]). Unused release capacity
        is not banked (same as SumoEnv)."""
        u = min(max(float(u), 0.0), 1.0)
        self._arrival_acc += self.arrival_rate * self.step_len
        n_arr = int(self._arrival_acc)
        self._arrival_acc -= n_arr
        self.queue += n_arr
        self.total_arrivals += n_arr
        self._release_acc += u * self.discharge_rate * self.step_len
        n_cap = int(self._release_acc)
        self._release_acc -= n_cap
        return min(n_cap, int(self.queue))

    def on_released(self, n: int = 1) -> None:
        """Remove n vehicles that were actually inserted into SUMO from the queue."""
        self.queue = max(self.queue - n, 0.0)
        self.total_released += n
