"""Metered ramp queue (numpy/SUMO-free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sumo_env.ramp_queue import MeteredRampQueue  # noqa: E402


def _run(q, u, seconds):
    released = 0
    for _ in range(int(seconds)):
        n = q.step(u)
        q.on_released(n)
        released += n
    return released


def test_closed_meter_accumulates_arrivals():
    q = MeteredRampQueue(arrival_vph=800, discharge_vph=1600, step_len_s=1.0)
    assert _run(q, 0.0, 3600) == 0
    assert 795 <= q.queue <= 800


def test_half_green_passes_full_demand_without_queue():
    q = MeteredRampQueue(arrival_vph=800, discharge_vph=1600, step_len_s=1.0)
    released = _run(q, 0.5, 3600)
    assert 795 <= released <= 800
    assert q.queue <= 1


def test_full_green_drains_backlog_above_arrival_rate():
    q = MeteredRampQueue(arrival_vph=800, discharge_vph=1600, step_len_s=1.0)
    _run(q, 0.0, 600)                     # 10-min closure -> ~133 queued
    backlog = q.queue
    released_5min = _run(q, 1.0, 300)     # flush at 1600 vph capacity
    assert backlog > 120
    assert released_5min > 800 * 300 / 3600 * 1.2   # clearly above the arrival rate
    assert q.queue < backlog


def test_release_never_exceeds_queue_or_capacity():
    q = MeteredRampQueue(arrival_vph=400, discharge_vph=1600, step_len_s=1.0)
    released = _run(q, 1.0, 3600)
    assert released <= q.total_arrivals
    assert 395 <= released <= 400
