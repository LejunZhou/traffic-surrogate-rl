# Confirmed ramp entries progress

## 2026-09-11

- Changed the generator to subtract only ramp IDs confirmed by
  `getDepartedIDList()` after a simulation step. Pending requests remain queued
  and cannot be resubmitted; discarded requests retain demand for later retry.
- Saved `ramp_departed_count`, `ramp_pending_count`, and
  `ramp_flow_measurement="confirmed_departures"` in both `.npz` writers.
- The single-rollout writer now also saves the command, queue, ramp inflow,
  model, reference flow, and discharge capacity already exposed by the runner.
- Added behavioral regression checks for insertion delays, rejections, discards,
  open-loop compatibility, conservation, and persistence through both writers.
- Validation passed: `tests/test_ramp_departures.py` and `tests/test_ramp_queue.py`
  report **10 passed**, using Python 3.11, pytest 7.4.0, and the installed SUMO
  TraCI tools. Checks cover both `.npz` writers and queue conservation.
- Live SUMO validation passed for both the dataset generator and single-rollout
  CLI. The temporary dataset scenario kept the 200 m ramp, set a temporary low
  ramp speed to create insertion delay, then restored speed. At 900 vph arrivals,
  the eight 30-second intervals recorded confirmed entries
  `[0, 0, 13, 13, 1, 0, 0, 10]`, pending counts
  `[0, 0, 0, 0, 10, 10, 10, 0]`, and queue
  `[7, 15, 9, 4, 10, 18, 25, 23]`. All fields matched independently captured
  TraCI events, and queue plus cumulative entries equalled cumulative arrivals.
- The 10 delayed requests entered after the control command changed to zero:
  measurement follows SUMO's confirmed entry time. This change corrects counting;
  it does not cancel previously submitted requests when an action changes.
- Live checks used installed SUMO 1.20.0; the project's SUMO 1.27.1 pin remains
  unchanged. The tests validate accounting and persistence, not equivalence of
  traffic physics across SUMO versions. Temporary verification script:
  `/tmp/verify_sumo_confirmed_ramp_entries.py`; outputs:
  `/var/folders/4n/q95jyw_94kq_x0zf37fb2wch0000gn/T/sumo-confirmed-ramp-4tdzta0o/`.
- The exit-boundary average was checked in both live output paths and is intact.
- No existing datasets or production network geometry were modified. Historical
  aggregate files cannot recover delayed entry timing and need regeneration for
  these corrected measurements.
