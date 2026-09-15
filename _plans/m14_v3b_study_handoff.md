# Handoff: run the scenario-v3b study (M14) on the Windows machine

State on 2026-09-15 (Mac): scenario v3b, demand family v2 and the study driver are in git
(commit `2a4d5d2` and later); the E0 characterisation on v3b is done. Nothing of the study
itself has been run. Context: `_progress/m14_scenario_v3_progress.md` (§8–§12 for v3b),
`_plans/m14_scenario_v3_plan.md`, `_progress/m13_round0_budget_progress.md` (what to expect
from the aggregation loop), README "scenario v3b" lines.

## Prerequisites
- Clone at the current `main`. Python env with the project installed (`pip install -e ".[dev]"`)
  and SUMO on PATH: either the `eclipse-sumo` wheel in the env (gives `sumo`, `netconvert`)
  or a SUMO install with `SUMO_HOME` set. The driver aborts if `sumo` is not on PATH.
- Run the driver from Git Bash (or WSL): `sh scripts/run_v3b_study.sh`. It uses `sh`
  features only (background jobs, `wait`, `$(( ))`, `printf`, `tr`).
- Optional: copy `data/plant_v3b/round0/` (404 E0 rollouts, 15 MB, git-ignored) from the Mac.
  Without it the driver regenerates E0 first (`_progress/m14_e0_v3b_characterisation.json`
  already exists in git, so delete or rename it if E0 must be regenerated: the driver runs
  E0 only when the report is missing, and the round-0 stage then starts from an empty store —
  the E0 rollouts are simply extra round-0 data, so both paths are valid).
- Quick check before the long run: `PYTHONPATH=src python -m pytest tests -q` (66 pass, 7 skip
  without SUMO) and `SMOKE=1 sh scripts/run_v3b_study.sh` (≈ 15 min, isolated paths, then
  delete `data/plant_v3b/smoke_round0`, `runs/surrogate/plant_v3b_smoke`,
  `runs/aggregation/v3b_smoke_s0`, `runs/study/v3b_smoke`, `runs/ledger/v3b_smoke*`).

## The run
```
sh scripts/run_v3b_study.sh                      # demo scale: seed 0, 3 rounds x 300k steps, direct 200/700 EE, 8 workers
WORKERS=12 sh scripts/run_v3b_study.sh           # more SUMO workers if the machine has the cores
```
Stages (idempotent, each skipped when its output exists): E0 → round-0 dataset
(`configs/experiments/round0_v3b.yaml`, 288 mixture rollouts on top of E0) → 5-member plant
DeepONet ensemble `runs/surrogate/plant_v3b_r0` → gate on val/test (`eval_val.json`,
`eval_test.json`) → aggregation loop `runs/aggregation/v3b_s0` (stop rule δ 2, patience 2,
fine-tune 20 epochs) in parallel with direct SUMO PPO `runs/study/v3b/direct_ppo_{200,700}ee_s0`
→ ALINEA tuning + constants (`_progress/m14_alinea_tuning_v3b.json`) → checkpoint selection
→ manifest `runs/study/v3b/arms.json` → final evaluation on the family-v2 T and O sets
(`runs/study/v3b/eval/*.jsonl`) → figures `_progress/figures/m14_v3b/`.
Logs: `runs/logs/v3b_*.log`. Expected wall time ≈ 5 h on 10 cores.
Scenario selection is automatic: the driver exports `SCENARIO_OVERLAY=configs/rl/env_v3b.yaml:
runs/study/v3b/env_study.yaml` and `PROFILE_SETS_DIR=configs/profiles/v2`; do not run the
individual scripts for v3b without those two variables.

## What to record when it finishes (progress file §13, "v3b study")
1. Round-0 gate (val and test): return error, false-breakdown rate, calibration slope, pass/fail
   (v2 at 692 rollouts: 0.06–0.11 / 0 / in range; at 240: 0.078–0.081 false breakdown).
2. Per round: SUMO breakdown rate of the top-3, gap surrogate − SUMO, best V return,
   stop-rule status (`runs/aggregation/v3b_s0/rounds.json`, `study.json`).
3. Final table on T (30 profiles × 3 seeds) and O (12 × 3): mean return, p10, breakdown rate,
   end queue, for A0, A1 per round, B at 200/700, ALINEA, constant — from the final-eval log
   or `runs/study/v3b/eval/*.jsonl`; compare with the v2 demo study
   (`_progress/m12_study_progress.md`: A1 −55.0 at 800 EE, B −64.1 at 844 EE, ALINEA −60.6 on T)
   keeping in mind returns are not comparable across scenarios, only rankings and margins are.
4. Ramp-blocking check on the round-0 store and the aggregation rollouts:
   `pending_ramp_max` in each rollout's metrics (expected > 0 only in flush-into-jam episodes).
5. Figures 1, 2, 4 in `_progress/figures/m14_v3b/` (git-ignored: attach or describe).
Then update `_plans/milestones_overview_plan.md` (M14 done) and the README line, and commit
without the generated data (`data/plant_*/`, `runs/` are git-ignored).

## Known behaviours
- The E0 gate on v3b is 2/8 (per-profile constant) and 3/8 (single constant u = 0.4): the
  anticipation lever exists on ≈ a third of the loaded profiles; expect a smaller RL margin over
  ALINEA than on v2 (progress §11–§12).
- Ramp blocking (released vehicles not inserted) happens only when ≥ ≈ 860 veh/h is released
  into a jammed merge; SumoEnv keeps them in the virtual queue, the surrogate cannot reproduce it.
- The M13 loops regressed after round 4 because 20 fine-tune epochs on 54 new rollouts did not
  track; at 3 rounds this is not expected to bite, but watch the gap column.
