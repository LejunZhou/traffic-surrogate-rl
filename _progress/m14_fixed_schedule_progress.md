# M14 fixed schedule integration — 2026-09-27

User explicitly selected the entire M14 study, including training and evaluation.
The schedule now lives under demand.profile_segments in m14/configs/scenario.yaml.
Scalars reflect the initial 1250/250 veh/h rates. Both M14 training configs point
the profile-family loader at scenario.yaml. ProfileFamily accepts a scenario
and yields the exact 120-step profile, preserving the old parametric-family API.
Fixed profiles carry a derived 20–45 min storage window for behavior controllers.

Evaluation descriptors point to the scenario rather than embedding copied
arrays. Counts remain 18 validation, 30 test × 3 seeds, 12 repeat × 3 seeds;
SUMO seeds are disjoint (10000, 20000, 30000 ranges). The ood key is retained
for compatibility but tables and the repeat-set plot explicitly identify fixed
repeats rather than out-of-distribution demand. The profile-set generator and
smoke driver support these descriptors. Legacy random family remains available
only when explicitly selected through configs/demand.yaml.

The root restored generator reads the same scenario segments and now uses
unit scaling (exact schedule). Its pre-recording queue reset remains in place;
it now honors M14's stop-line placement as well. Native M14 starts empty as
before. Existing datasets and checkpoints are not modified; use fresh output
for the changed experiment.

Initial verification: 27 root generator tests and 9 M14 demand-profile tests
passed. A temporary environment under /private/tmp/m14-fixed-demand-tests
adds pytest and stable-baselines3 atop the existing cs285 Python, without
modifying that environment. Full M14 validation completed below.

Verification:
- Full M14 suite: 90 passed initially; 3 report tests exposed a pre-existing
  Python 3.11 f-string backslash syntax error. Moved the LaTeX separator out
  of the f-string expression; the complete 10-test report/resume module then
  passed. All 93 M14 tests have passed across the full run and targeted rerun.
- Root generator/profile/queue suite: 27 passed.
- Native `m14/run.py simulate --policy u=0.5 --seed 10000` completed all 120
  intervals on local SUMO 1.20.0 with zero teleports. Compared saved mainline_demand
  and ramp_arrival to scenario sampling: exact equality at every interval.
  Output: /private/tmp/m14-fixed-schedule-check/rollout.npz.
- git diff --check: clean.

This is an integration check, not reproduction on the paper's pinned SUMO
1.27.1. Full data generation, PPO training, and study evaluation were not run.
Changes remain uncommitted, including the earlier queue reset and M14 road
adaptation requested in this task. No existing experiment output was overwritten.

