"""
Metered on-ramp queue shared by the dataset generator (run_simulation.py)
and, in spirit, SumoEnv (which carries the same accumulators inline plus
SUMO pending-vehicle bookkeeping).

Vehicles arrive at `arrival_vph` and wait in a virtual queue upstream of the
meter; the meter releases at most u * discharge_vph, and never more than the
queue holds. With discharge_vph > arrival_vph the released flow can exceed
the arrival rate while a queue exists (e.g. u = 1 after a closure), which is
what makes a backlog drainable (M7 progress §7.10).

M8: the arrival rate may vary per sub-step (`step(u, arrival_vph=r_k)`), so
time-varying ramp-arrival profiles use the same recursion as the constant
case. `analytic_queue_step` gives the analytic per-control-step version used by
SurrogateVecEnv (release = min(Q + a_k, u_k * D * dt / 3600)).

M14 (scenario v3): the ramp has finite storage. `queue_override_rate` is the
smallest green fraction that keeps the queue at or below `queue_max` over the
next control step (the ALINEA/Q rule applied as an actuator constraint); both
SumoEnv and SurrogateVecEnv apply u = max(u_policy, u_min), so the constraint
is identical by construction in the two environments.
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

    def set_arrival_vph(self, arrival_vph: float) -> None:
        if arrival_vph < 0.0:
            raise ValueError("arrival_vph must be >= 0")
        self.arrival_rate = float(arrival_vph) / 3600.0

    def step(self, u: float, arrival_vph: float | None = None) -> int:
        """Advance one sub-step: add arrivals (at `arrival_vph` if given, else
        the current rate), return how many vehicles the meter may release now
        (green fraction u in [0, 1]). Unused release capacity is not banked
        (same as SumoEnv)."""
        if arrival_vph is not None:
            self.set_arrival_vph(arrival_vph)
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


def analytic_queue_step(
    queue_before: float, u: float, arrival_vph: float, discharge_vph: float, dt_ctrl_s: float
) -> tuple[float, float, float]:
    """One control-step of the continuous queue recursion used by the surrogate env
    (and by the reward's offered-demand reference):

        a        = r_k * dt / 3600
        release  = min(Q_{k-1} + a, u_k * D * dt / 3600)
        Q_k      = Q_{k-1} + a - release

    Returns (queue_after, released_vehicles, arrivals_vehicles).
    """
    u = min(max(float(u), 0.0), 1.0)
    arrivals = float(arrival_vph) * dt_ctrl_s / 3600.0
    capacity = u * float(discharge_vph) * dt_ctrl_s / 3600.0
    available = float(queue_before) + arrivals
    released = min(available, capacity)
    return max(available - released, 0.0), released, arrivals


def queue_override_rate(
    queue_before: float, arrival_vph: float, queue_max: float | None, discharge_vph: float, dt_ctrl_s: float
) -> float:
    """Minimum green fraction u_min in [0, 1] such that the queue after one
    control step does not exceed `queue_max`:

        Q_{k-1} + a - u_min * D * dt / 3600 <= queue_max

    Returns 0.0 when no cap is set or the cap cannot bind. When even u = 1
    cannot hold the cap (arrivals exceed the discharge capacity), returns 1.0
    and the queue exceeds the cap by the shortfall."""
    if queue_max is None or queue_max <= 0.0:
        return 0.0
    arrivals = float(arrival_vph) * dt_ctrl_s / 3600.0
    excess = float(queue_before) + arrivals - float(queue_max)
    if excess <= 0.0:
        return 0.0
    capacity = float(discharge_vph) * dt_ctrl_s / 3600.0
    return float(min(max(excess / max(capacity, 1e-9), 0.0), 1.0))
