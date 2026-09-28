# Fixed demand generator restoration — 2026-09-25

Restored the exact seven-segment one-hour schedule from stash@{0}
(pre-update-2026-09-20-local-work), with independent 0.9–1.1 channel scaling,
nominal samples every 11 indices, and a discarded 180-second pre-roll.

Integrated profile helpers into the existing demand_profiles module, preserving
ProfileFamily and frozen random profiles. Updated root dataset generation,
route construction, and simulation collection for both demand channels and
warmup metadata. Preserved the older ramp_warmup_s insertion-delay option.
Explicit mainline_blocks remain absolute simulator intervals. Added --no-splits
support to the regular dataset CLI for single-sample generation.

The config uses the current phase1_1 scenario. No stashed scenario changes,
DeepONet/PPO changes, paper files, or m14 files were restored. The stash is intact.
README documents commands and data semantics. Changes remain uncommitted.

Verification:
- PYTHONPATH=src:/opt/homebrew/share/sumo/tools python -m pytest
  tests/test_fixed_demand_profile.py tests/test_demand_profiles.py
  tests/test_ramp_departures.py tests/test_ramp_queue.py -q: 24 passed.
- Two full 3600-second episodes plus 180-second pre-roll on installed SUMO 1.20.0:
  both saved 19x120 density arrays, zero teleports, exact nominal values for
  sample 0, scaled demand for sample 1. Verified relative time starts at zero,
  integer arrival/confirmed-departure queue conservation, and split output.
- The first smoke assertion incorrectly assumed continuous arrivals; the existing
  meter accumulates fractional demand and admits integer arrivals. Rechecked the
  saved outputs with that exact discrete accounting; both pass.
- Smoke output: /var/folders/4n/q95jyw_94kq_x0zf37fb2wch0000gn/T/fixed-demand-smoke-7pneyply
- git diff --check: clean.

Limit: the smoke run validates integration on SUMO 1.20.0, not reproduction of
paper results using SUMO 1.27.1. The full 1000-episode dataset was not launched.

## Recording-boundary queue reset

User requested Q(0)=0 for the saved horizon. The runner now cancels pending
warmup ramp requests and initializes a new meter (zero queue and fractional
arrival/release accumulators) before recording. Vehicles already admitted to
the road are preserved. Warmup diagnostics retain the pre-reset values, and
NPZ output adds recording_initial_ramp_queue=0. End-of-interval ramp_queue
values retain their original meaning. Existing datasets are not modified.

Regression coverage includes accumulated warmup queue with both closed and
open meters, a pending request scheduled to enter after the boundary, and
fractional arrivals that would otherwise change the first recorded queue.

Validation: 26 targeted tests passed; git diff --check clean. This follow-up
has not been committed or pushed.

## M14 60 km/h road adaptation

The fixed dataset now references m14/configs/scenario.yaml, including 16.67 m/s
ramp speed, 10-degree entry, 1200 veh/h discharge, speedDev 0.03, and occupancy
density. Moved the existing density_from_loops helper to sumo_env.detectors
and re-export it through the RL module's import. The dataset runner now uses
that same helper, retaining q/v behavior for old configs and respecting
through-lane-only merge density and 142.857 veh/km clipping for M14.

Added a regression test that loads the actual configured scenario and checks
meter capacity, ramp speed, occupancy density, and exclusion of the ramp lane.
A broader test attempt yielded 32 passes and 7 failures because gymnasium is
not installed in the available Python environment; those failures are in
RL-dependent scenario tests. No dependencies were changed.

M14 road adaptation checks completed: 27 targeted root tests passed. Two full
root-generator episodes used the generated ramp speed 16.67 m/s and recorded
1200 veh/h meter capacity, bounded occupancy density, and zero initial queue,
with zero teleports on SUMO 1.20.0. Subsequent user steering moved the schedule
into scenario.yaml for the whole study; see m14_fixed_schedule_progress.md.
