# M14 rerun with a 60 km/h ramp on Colab: progress

Plan: `_plans/m14_ramp60_colab_plan.md`. Earlier steps of the same rerun: M14 progress §17 (reward charges
Q_{k+1} in both envs) and §18 (60 km/h ramp, probe-vehicle speeds).

## 1. Implementation (2026-09-22/23, all in `m14/` unless noted)
- **Direct SUMO-PPO resume** (`src/rl/train_ppo.py`, `training.resume`): latest `ppo_sumo_<N>_steps.zip` loaded with
  optimizer state and step count; `prepare_resume` keeps `progress.csv` / `monitor.csv` as `*_part<n>.csv`, cuts
  `eval/evaluations.npz` to timesteps ≤ N (full copy kept as `evaluations_part<n>.npz`), cuts the ledger to N/120
  training episodes + 18 per kept evaluation (the rest to `ledger_lost.jsonl` in the run dir), logs `resume_log.jsonl`;
  the EvalCallback is pre-loaded with the kept history and best mean; the SUMO-seed index continues at N/120 and the
  RNG is reseeded with seed + N (no replay of the first profiles). `learn(total − N, reset_num_timesteps=False)`.
- **`run.py`**: `sumo-ppo` resumes instead of refusing, default budget 1000, validation every 9600 steps above 200
  episodes (was 24 000 above 700); new `e0` stage (E0 + capacity comparison) and `tables` stage (final-ensemble
  evaluation on the aggregation store's val/test splits + table builder); `evaluate` adds separate ALINEA and
  PI-ALINEA arms and Surrogate-MPC on the last aggregation ensemble (`--mpc/--no-mpc`, `--mpc-args iters=30`);
  `baselines` grid options (`--dets --rhos --kis --kps`); `pipeline --parallel` (data, then [deeponet → surrogate-ppo]
  ‖ sumo-ppo ‖ baselines as child processes with `runs/logs/pipeline_<branch>.log`, then evaluate, tables, report);
  `pipeline` refuses to generate data when the capacity check recommends a rescale (`--ignore-capacity-check`);
  `pipeline --recover-interrupted` sets `M14_RECOVER_INTERRUPTED=1`.
- **Recovery after a lost session** (opt-in): `run_aggregation_loop.py --recover-interrupted` moves an interrupted
  round's artifacts, store entries and ledger rows to `interrupted_r<j>_<t>/` and redoes the round (the m14 extraction
  refused and asked for a new study name); `profile_eval.evaluate_on_profiles` moves a request whose marker is
  still "running" and whose JSONL was never written (so no ledger row exists: results are published JSONL →
  summary → ledger → complete marker) to `<name>.interrupted_<t>/` and runs it again. Lost episodes are not charged,
  in both mechanisms, so costs are those of an uninterrupted run.
- `tune_alinea_profiles.py`: `--kps` (default [4] = old behaviour); report records `best_pure_alinea`,
  `best_pi_alinea` and the grid. `build_arms_manifest.py`: `--pure-alinea/--pi-alinea` (ALINEA charged its own
  candidates, PI-ALINEA the joint search, as in the paper: 288 / 324 on v3b); MPC cost = data + model training
  behind its ensemble (cumulative SUMO episodes of that round, no PPO time).
- New `scripts/build_paper_tables.py`: Table I (final ensemble, mean over trajectories of density / exit-flow
  rel-L2 and MAE, return relative error and MAE); Table II (TTS, 1-s mean queue, mean of per-episode max queue,
  completed trips, SUMO episodes, summed compute hours; Surrogate-PPO charged for every round run); headline
  1 − mean(TTS_a)/mean(TTS_b), paired bootstrap (5000) vs PI-ALINEA, ALINEA, SUMO-PPO. Outputs
  `runs/study/m14/tables/{tables.json, tables.md, table2_rows.tex}`.
- New `scripts/compare_e0_capacity.py` + reference `configs/reference/e0_ramp120kmh.json` (copy of
  `_progress/m14_e0_v3b_characterisation.json`): per-profile first-breakdown constant rate, mean shift; verdict
  < 0.5 step keep, 0.5–1 step keep (the 30° → 10° change was −67 veh/h, 5/9 profiles one step lower, and the
  published study did not rescale), ≥ 1 step rescale. Check on the 30° vs 10° reports reproduces §14 (5 of 9 lower).
- `colab/m14_ramp60.ipynb` (repo root, generated): parameters, Drive working copy `MyDrive/m14_ramp60` (code copied
  from a GitHub clone once; `UPDATE_CODE` refreshes code only), SUMO wheel 1.27.1, one-episode check that prints the
  ramp speed from the built net, E0 + capacity verdict (decision point), background pipeline with double-launch
  guard, status cell, tables, results archive.
- Docs: `m14/README.md`, `m14/docs/workflow.md`.

## 2. Verification
- Unit tests: `tests/test_resume_and_tables.py` (7: checkpoint ordering, resume cut-back of npz / ledger / logs,
  eval-history restore, paired reduction, capacity verdicts, both recovery paths). Full suite 79 passed, 1 failed
  (pre-existing `test_standalone.py` `dateutil` failure, §17 of the M14 progress; 79 passed after the recovery tests).
- `run.py pipeline --dry-run --parallel`: command sequence as intended.
- `run.py smoke` extended (direct PPO 240 steps, final model removed, resumed to 480 with assertions on checkpoints,
  evaluations and ledger; evaluate with MPC `H=4,iters=2`; tables): see §3.
- Not verifiable locally: the notebook on Colab itself (cells syntax-checked only).

## 3. Smoke result (2026-09-23, Windows, 4 workers, exit 0)
Every stage ran: 18-rollout store, 2-member ensemble, one aggregation round, direct PPO 240 steps → final model
removed → resumed to 480 (**resume check passed**: checkpoints and evaluations at 120/240/360/480, ledger
4 direct_ppo + 4 eval_val, `resume_log.jsonl` part 1 from 240 steps), ALINEA tuning with the new report keys,
manifest with separate ALINEA / PI-ALINEA arms and Surrogate-MPC on `ensemble_r1` (`H=4,iters=2`), final
evaluation of 10 arms on ID and OOD, Table I (final ensemble on the aggregation store's val/test), Table II
(all 8 paper rows), headline reductions, figures. Smoke numbers are integration diagnostics only (1-epoch models,
1 profile). SUMO printed "peer shutdown" messages after the last evaluation had been summarised (worker
processes exiting at pool shutdown); no episode was lost. Full suite after all edits: 79 passed, 1 pre-existing
failure. Smoke artifacts deleted.

## 4. ALINEA grid widened (user, 2026-09-23)
Reason (fair baseline): on the 120 km/h study the selected PI-ALINEA (kp 4, ki 20, ρ̂ 26, det 13) had its set-point
and detector at the edge of the grid, no detector downstream of the bottleneck (the lane drop at 1400 m) was tried,
and kp had one value. New tuner defaults: det {12, 13, 14, 15} (1300–1600 m), ρ̂ {20, 23, 26, 30, 34}, ki {10, 20, 35},
kp {2, 4, 8}: stage 1 60 × 6 = 360 episodes, stage 2 (4 + 2 × 3) × 18 = 180, total 540 (was 324). The report now
records the grid and an `edge_check` per selected controller (warning printed; notebook cell 7 shows it); rule: if a
selected value is on the edge, widen in that direction and retune. Test `test_alinea_edge_check`; docs updated.

## Open
- Commit + push so the notebook's clone contains these changes.
- Explanation of the Surrogate-MPC negative result; seeds 1–2; paper edits.
