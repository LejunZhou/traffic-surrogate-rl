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
