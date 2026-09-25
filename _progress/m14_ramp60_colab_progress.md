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

## 5. Colab run, seed 0 (L4 runtime, 12 workers, code 37997fe)
- 2026-09-23 05:34 UTC: working copy made. One-episode check normal (TTS 23.2 veh·h, no breakdown, 10.8 s).
- E0 (404 rollouts) finished about 05:44. **Capacity verdict: keep the constants.** 9 profiles break down in both
  studies: 3 lower, 5 same, 1 higher; mean shift −27 veh/h (grid step 120). E0 gate: 3/8 storage-mandatory profiles
  pass (120 km/h: 2/8), with best single constant u = 0.4 (was 0.5). The E0 gate prints FAILED, as it did at 120 km/h
  (2/11 peaked, 2/8 storage; now 3/11, 3/8; against the single constant 6/11, 5/8, was 4/11, 4/8). The published study
  went ahead on the same failure. Hand schedules lose to the best constant on the step-plateau profiles 5 and 8, the
  known E0 grid weakness. The pipeline does not stop on this gate. The insertion check (u = 0.87) matches 120 km/h
  within 0.01 in served/offered on all 20 profiles; e0[112] no longer breaks down (0.945, was 0.940 with breakdown).
- 05:44: pipeline launched (`--parallel --recover-interrupted --workers 12`). Round-0 data at ≈ 1.5 s/episode, 158/288
  by 05:48. Output: `runs/logs/pipeline_driver.log` on Drive.
- 05:51: round-0 data done (414 s; store 692 = 486 train / 105 val / 101 test, breakdown rate 0.52); the three branches
  started.
- **DeepONet ensemble** (5 members in parallel, 598 s). Gate PASSED on both splits. Val: return error 0.089, false
  breakdown 0.000, calibration slope 1.37. Test: 0.098 / 0.019 / 1.44. The return error is close to the 0.10 limit
  (120 km/h round 0: ≈ 0.05). Aggregation round 1 started 06:09:46.
- **ALINEA tuning** done 06:16 (540 episodes). Best PI-ALINEA `kp=2,ki=20,rho=26,det=13` (V −48.5, TTS 44.3). Best pure
  ALINEA `ki=10,rho=26,det=13` (V −55.0): the only pure candidate without a catastrophic episode; `ki=20,rho=26` has
  mean −51.3 but one catastrophic episode. Best constant u = 0.4 (V −78.5). Detector 13 (end of the acceleration lane)
  wins; stations 14–15 downstream are much worse (−80 to −98). **Edge check**: pure ALINEA ki = 10 is the lowest value
  tried (a real edge). PI-ALINEA kp = 2 is the lowest value tried, but kp = 0 with the same ki/ρ̂/det is the pure ALINEA
  `ki=20,rho=26,det=13` (−51.3, worse than kp 2), so the kp optimum is bracketed and that flag is a false alarm.
- **ALINEA retune (user, 2026-09-23 ~06:50)** because pure ALINEA ki = 10 was on the grid edge. Local grid det 12–14,
  ρ̂ 23/26/30, ki 5/10/20, kp 2/4/8, 4 workers, same ledger study `m14_alinea` (both searches are charged). The first
  search is kept as `runs/study/m14/alinea_tuning_r1.json` + `runs/eval/m14_alinea_r1/`. Log: `runs/logs/alinea_retune.log`.
  The new `alinea_tuning.json` is what the final evaluation reads.
  Retune stages 1–2 done 07:10 (constants still running). ki = 5 is worse than ki = 10/20 at every det/ρ̂ in stage 1
  (det 13: −62.5…−70.6 vs −43.6…−54.5), so no ki = 5 candidate reached stage 2. Stage 2 is the same candidates as
  before plus `alinea:ki=10,rho=23,det=13` (−58.3). Pure ALINEA is now inside the grid (no warning). PI-ALINEA again
  `kp=2,ki=20,rho=26,det=13` (−48.5), flagged for ki = 20 (highest in the new grid) and kp = 2 (lowest). Both are
  bracketed by the first search: ki = 35 at ρ̂ 26 / det 13 scored −53.9 in stage 1 vs −43.6 for ki = 20, and kp = 0
  (pure ALINEA `ki=20,rho=26,det=13`, −51.3) is worse than kp = 2. So both flags are grid artifacts and the ALINEA
  baselines are settled.
  Retune finished 07:22: `alinea_tuning.json` has best pure ALINEA `ki=10,rho=26,det=13` (V −55.0, no edge), best
  PI-ALINEA `kp=2,ki=20,rho=26,det=13` (V −48.5, TTS 44.3; flags explained above), best constant u = 0.4 (V −78.5).
  Same controllers as the first search.
- Aggregation round 2 (2291 s): selected 216 000 steps, surrogate −49.4 / **SUMO V −45.6** (TTS 41.4, 0 breakdowns),
  gap +3.8. Already ahead of tuned PI-ALINEA on V (−48.5 / TTS 44.3). Cumulative 800 EE. Ensemble spread on the new
  rollouts 0.163 → 0.216 (return error; this went up after fine-tuning, to watch). Round 3 PPO 1549 s, top-3 by
  surrogate V ≈ −52.0.
- Aggregation round 3 (1721 s): selected 168 000 steps, surrogate −52.0 / **SUMO V −43.4** (TTS 39.4, 0 breakdowns),
  gap +8.6; spread on new rollouts 0.225 → 0.122. Cumulative 854 EE (the 120 km/h study's round 3 was also 854 EE).
  Improvement over round 2 is +2.2 (≥ δ = 2), so the stop rule has not fired. Round 4 PPO 1608 s (top-3 surrogate V
  ≈ −47.5). During its SUMO validation one SUMO instance hit "Address already in use" on its TraCI port. traci
  retried on a different port automatically, so this is harmless (three branches start SUMO at the same time).
- Aggregation round 4 (1782 s): selected 72 000 steps, surrogate −47.6 / SUMO V −44.0 (TTS 40.1), gap +3.6; spread
  0.118 → 0.044; cumulative 908 EE. Stop rule: improvement −0.6 < 2 (1/2). Round 5 PPO 1580 s (top-3 surrogate V
  ≈ −43.2); SUMO validation at 08:50. Best so far is round 3 (−43.4).
- **Aggregation done 09:10** (9670 s). Round 5 selected 48 000 steps, surrogate −43.3 / SUMO V −43.3 (gap 0.0), spread
  0.060 → 0.046, cumulative 962 EE. Stop rule fired (+0.1 < 2, 2/2). **A1 = round 5, V −43.3 at 962 EE** (round 3
  −43.4 is statistically the same). The plateau and 962 EE are identical to the 120 km/h study.
- **Direct PPO crashed at 09:10:30**, at 76 320 steps (636 episodes): `FileNotFoundError` on
  `runs/ledger/m14_direct_1000_s0.jsonl`, and run.py then reported `/content/drive/MyDrive/m14_ramp60/runs` missing.
  A transient Google Drive (FUSE) outage: the ledger file exists on Drive with 638 direct_ppo + 126 eval_val rows, all
  valid JSON. The pipeline driver stopped ("parallel branches failed: direct: sumo-ppo exited 1") before
  evaluate/tables. Latest checkpoint 67 200 steps (560 episodes, 7 validations kept). Recovery: relaunch cell 6.
  sumo-ppo resumes from 67 200 (≈ 76 episodes redone, moved to ledger_lost), the other stages are reused. Side
  effect: the resumed aggregation loop rewrites `study.json` with a near-zero `wall_s`. That field is cosmetic; costs
  come from the ledger and rounds.json.
- The Colab kernel/runtime had been reset (cell 7 showed `driver_alive` undefined and the wrong working directory).
  The user reran cells 1–3; cell 7 confirmed the state on Drive. Cell 6 relaunched at 09:17 (pid 4598).
  `resume_log.jsonl`: part 1, resumed from 67 200 steps, 560 training episodes, 7 validations kept, 78 ledger rows
  moved to `ledger_lost.jsonl` (not charged). Old logs kept as `progress_part1.csv` / `monitor_part1.csv`.
- Likely cause of the reset: Colab's idle timeout. No cell had run since about 06:50 and the runtime went at about
  09:10; background processes don't count as activity. Fix (user request): new notebook cell **6b watchdog**
  (`colab/m14_ramp60.ipynb`, generator in the session scratchpad). It keeps a foreground cell busy, prints a status
  line every 5 min, and relaunches the pipeline (at most 3 times) if the driver stops before `tables.md` exists. It
  cannot survive a recycled runtime. Given to the user at about 09:25 to paste into the running session; not
  committed yet.
- After the relaunch: the surrogate branch reran its quick checks and finished at about 09:23 (nothing retrained).
  Direct PPO checkpoint 76 800 at 09:32; about 714 episodes by 09:51 (≈ 13 s/episode now that only direct PPO runs), about 833 by 10:21, about 955 by 10:48.
- **Direct PPO finished 10:58** (1000 training episodes; 1216 EE with 12 validations). The final evaluation started
  10:59. Manifest `arms.json`, 10 arms with charged SUMO episodes: A0 692, A1 746/800/854/908/962, B 1216, ALINEA 666
  (pure-ALINEA candidates of both searches), PI-ALINEA 882 (all feedback candidates of both searches), tuned constant
  396 (both constant sweeps), fixed u = 0 / 0.5 / 1: 0, Surrogate-MPC 962 (`ensemble_r5`, iters = 30).
- Test set T (90 episodes) as they come in: A0 −58.9 (TTS 54.8); A1 by round: r1 −58.9 (same checkpoint as A0), r2
  −49.5, r3 −45.3 (TTS 41.3), r4 −48.8, **r5 (final) −45.4 (TTS 41.3, 0 breakdowns)**; **direct PPO −50.6 (TTS 46.6)**;
  ALINEA −56.3 (TTS 51.4, 18 % breakdowns). **PI-ALINEA −51.6 (TTS 47.3, 0 breakdowns, 3 catastrophic)**; tuned
  constant u = 0.4 −85.6 (TTS 74.3). Preliminary TTS ratio on T: A1 is 12.7 % below PI-ALINEA and 11.4 % below direct
  PPO. Surrogate-MPC on T started 11:31: ≈ 9 min per batch of 12 episodes, 24/90 by 11:49 (≈ 70 min for T). The OOD
  set comes after that, so tables are expected around 14:15–14:30.
- **Surrogate-MPC on T (finished 12:39, 4088 s): −102.0, TTS 85.9, 56 % breakdowns, 21 catastrophic.** The negative
  result repeats (120 km/h: TTS 89.9, 54 % breakdowns).
- OOD set O (36 episodes, about 1 min per arm): A1 r5 −54.1 (TTS 50.2, 0 breakdowns); A1 r3 −54.0 (TTS 50.2); direct
  PPO −55.1 (TTS 51.6, 3 % breakdowns); PI-ALINEA −57.2 (TTS 53.2); ALINEA −61.1 (TTS 56.5, 17 % breakdowns).
  Preliminary O TTS ratio: A1 is 5.7 % below PI-ALINEA and 2.7 % below direct PPO (smaller than on T; 120 km/h had
  −12 % on O). Surrogate-MPC on O was running at 12:51 (≈ 27 min), so tables are now expected around 13:30.
- Final evaluation done 13:21: Surrogate-MPC on O −107.4, TTS 91.5, 58 % breakdowns, 12 catastrophic (1650 s). Tables
  and report stages next.
- Direct PPO: about 447 episodes by 08:20; best_model updated again at 08:08. About 553 episodes by 08:50.
- Direct PPO validation (V, 18 profiles): 9600 −83.3, 19 200 −84.3, 28 800 −79.2, 38 400 −69.0. 344 episodes by 07:50,
  ≈ 15 s/episode.
- Aggregation round 1 (2119 s): selected 240 000 steps, surrogate −61.0 / SUMO V −56.0 (gap +5.0), ensemble spread on
  the new rollouts 0.092 → 0.042, cumulative 746 EE. Round 2 started about 06:45.
- Direct PPO: first validation at 9600 steps V −83.3 (06:28); 152 episodes by 06:47 (train ep_rew_mean −76), ≈ 15 s/episode.
- Direct PPO: 64 training episodes by 06:20. About 27 s/episode while ALINEA tuning shared the cores, then about 14 s.

- **Pipeline finished 13:21** (tables, figures 1, 2, 4, 5, 7 in `reports/figures`). Total ≈ 7 h 45 min wall from E0
  start, including the 09:10 reset. Local copy of the tables: session scratchpad `tables_ramp60.md`; on Drive:
  `runs/study/m14/tables/{tables.md, tables.json, table2_rows.tex}`.

## 6. Results, seed 0 (60 km/h ramp) vs the 120 km/h study (M14 progress §14 addendum and §16)

**Table I** (final ensemble `ensemble_r5`, aggregation store val 105 / test 101 trajectories):

| metric | 60 km/h val / test | 120 km/h val / test |
|---|---|---|
| density rel-L2 | 18.6 % / 18.2 % | 18.2 % / 17.9 % |
| density MAE (veh/km) | 2.47 / 2.55 | 2.36 / 2.53 |
| exit-flow rel-L2 | 5.0 % / 5.0 % | 5.0 % / 5.2 % |
| exit-flow MAE (veh/h) | 70.0 / 69.9 | 70.9 / 71.8 |
| return relative error | 5.5 % / 4.6 % | 5.8 % / 6.4 % |
| return MAE (veh h) | 4.30 / 3.59 | 4.21 / 4.68 |

**Table II** (TTS veh h, ID = test T 90 episodes / OOD = O 36; mean q and max q in vehicles; SUMO episodes charged;
compute = summed compute hours on the Colab L4 runtime):

| method | ID TTS | ID mean q | ID max q | OOD TTS | OOD mean q | EE | compute h | 120 km/h ID / OOD TTS |
|---|---|---|---|---|---|---|---|---|
| u = 0 | 259.2 | 232.1 | 456.5 | 262.4 | 229.7 | 0 | 0 | — |
| u = 0.5 | 77.1 | 10.5 | 25.7 | 92.1 | 5.9 | 0 | 0 | 77.1 / 92.6 |
| u = 1 | 82.1 | 0.1 | 1.0 | 93.4 | 0.1 | 0 | 0 | — |
| ALINEA | 51.4 | 20.7 | 54.7 | 56.5 | 18.4 | 666 | 3.8 | 42.6 / 51.1 |
| PI-ALINEA | 47.3 | 18.0 | 47.2 | 53.2 | 18.2 | 882 | 4.9 | 42.6 / 50.6 |
| SUMO-PPO (1000 EE nominal) | 46.6 | 17.6 | 41.3 | 51.6 | 17.2 | 1216 | 5.5 | 41.1 / 47.9 (1209 EE) |
| **Surrogate-PPO** | **41.3** | **12.3** | **31.4** | **50.2** | 15.2 | 962 | 8.1 | 37.9 / 44.3 (962 EE) |
| Surrogate-MPC | 85.9 | 1.5 | 13.4 | 91.5 | 2.3 | 962 | 5.7 | 89.9 / 91.1 |

**Headline TTS reductions of Surrogate-PPO** (1 − mean ratio, paired bootstrap 95 % CI):

| vs | test (60 km/h) | OOD (60 km/h) | 120 km/h test / OOD |
|---|---|---|---|
| PI-ALINEA | **12.7 % [8.7, 16.1]** | 5.7 % [−1.8, 11.6] | 11.1 % [7.8, 14.2] / 12.5 % [7.2, 17.0] |
| ALINEA | 19.6 % [15.5, 23.2] | 11.2 % [2.4, 18.4] | — |
| SUMO-PPO | **11.2 % [7.6, 14.3]** | 2.7 % [−6.0, 9.4] | ≈ 7.8 % / 7.5 % (TTS ratio; return diff +3.3 [1.5, 5.3] / +3.8 [1.9, 5.8]) |

Reading:
- The ID result reproduces and is slightly stronger at 60 km/h: −12.7 % vs PI-ALINEA and −11.2 % vs direct SUMO-PPO
  on the test set. Surrogate-PPO uses 962 SUMO episodes against 1216 for direct PPO, with no breakdowns and the
  lowest mean queue among the working controllers (12.3 vs 18.0 veh for PI-ALINEA).
- **The OOD advantage does not reproduce with confidence:** −5.7 % vs PI-ALINEA and −2.7 % vs direct PPO, both CIs
  spanning 0 (120 km/h: −12.5 % vs PI-ALINEA, significant). Surrogate-PPO is still the best arm on OOD by the mean.
  Possible reasons: the smaller OOD set (36 episodes); the 60 km/h ramp changes merge behaviour on the OOD demand
  shapes (to check per profile); the round-0 return error was higher at 60 km/h (8.9–9.8 % vs ≈ 5 %) before
  aggregation.
- The aggregation trajectory matches 120 km/h: gains until round 3, plateau, stop rule at round 5, 962 EE. A1 round 3
  and round 5 are equal on T and O.
- Pure ALINEA is weaker than PI-ALINEA here (51.4 vs 47.3 ID; equal at 120 km/h). Its tuned integral gain is lower
  (ki 10 vs 20) because the no-catastrophic selection rule excluded ki = 20.
- Surrogate-MPC fails the same way as at 120 km/h (56–58 % breakdowns, TTS ≈ 86–91): the negative result is robust.
- The compute column is on a different machine from the 120 km/h study, so hours are not comparable across studies.
  Within this study Surrogate-PPO costs more compute (8.1 h: SUMO data + 5 × ≈ 30 min PPO + ensemble training) than
  direct PPO (5.5 h ledger; direct PPO's optimisation overhead is not in its ledger). The sample-efficiency claim is
  about SUMO episodes, not compute hours; say so in the paper.

### Compute breakdown (arms.json, rounds.json, member metrics.csv, runs/commands.jsonl)
| | Surrogate-PPO | SUMO-PPO |
|---|---|---|
| SUMO episodes (summed wall) | 962 → ≈ 4.6 h (692 initial ≈ 3.2 h + 270 in rounds ≈ 1.35 h) | 1216 → 5.47 h (16.2 s/episode) |
| DeepONet training (summed over members) | ≈ 1.05 h (round 0: 5 × 570 s; rounds 1–5: 5 × 5 × ≈ 37 s fine-tune) | – |
| PPO process | 2.43 h (5 rounds × 1550–2080 s; 300k surrogate steps/round = 1.5 M steps) | not charged (gradient updates on 120k steps, minutes) |
| total charged | 8.1 h | 5.5 h |
| elapsed on Colab | ≈ 3.1 h (E0 581 s + data 415 s + ensemble 599 s + loop 9676 s; SUMO 12 in parallel) | ≈ 5 h (05:51–09:10 + 6119 s after resume; one training simulation at a time) |

- Why more compute: 254 fewer SUMO episodes save only ≈ 1 h of simulation (a 60-min episode of this corridor takes
  ≈ 16 s), while the surrogate side adds ≈ 3.5 h (PPO 2.4 h + DeepONet 1.05 h). A surrogate step is ≈ 23× faster than a
  SUMO step (≈ 170 vs ≈ 7.4 steps/s) but the loop takes 12.5× more steps (1.5 M vs 120k).
- Break-even: with ≈ 3.5 h surrogate overhead and 254 episodes saved, Surrogate-PPO is cheaper in summed compute once a
  SUMO episode costs more than ≈ 50 s (3× this corridor).
- The "PPO process" time is mostly surrogate simulation, not learning: ppo_r5.log shows 236 steps/s for a rollout
  alone and 196 steps/s overall, so ≈ 80 % of it is stepping the DeepONet ensemble (≈ 0.56 s per surrogate episode vs
  ≈ 16 s per SUMO episode); gradient updates + surrogate evaluation are ≈ 20 % (KL early stop in 30 of 40 iterations).
  Direct PPO's stepping is its SUMO time; its updates (250 × 480-sample rollouts) take minutes.
- Bottleneck found (local benchmark, random-weight members, 16 envs, 2026-09-23): `DeepONetEnsemble.predict_step`
  reruns the causal GRU over the whole 120-step history at every control step to read one output; `torch.gru` is
  87 % of `step_wait` (277 env-steps/s). Carrying the GRU hidden state per env (one GRU step per control step) gives
  the same outputs (max difference 2e-6 veh/km) with 19–22× less plant time; projected ≈ 7× env throughput and
  ≈ 3× shorter surrogate PPO rounds.
- Implemented 2026-09-23 (before seeds 1–2): `BranchCache` + `predict_step_cached` /
  `predict_all_members_step_cached` in `surrogate/deeponet.py`, used by `SurrogateVecEnv` by default
  (`env.incremental_branch: false` = old path); stale rows (reset slot, member switch) are rebuilt from the history
  prefix. Thread cap: `training.torch_threads` or `M14_TORCH_THREADS`, set for all children by
  `run.py --torch-threads N`. Tests: cached vs full step (member switch, row reset, all-member path) and env
  trajectories cached vs uncached over an episode boundary in sample / mean / pessimistic mode; m14 suite 85 passed;
  `run.py smoke --workers 4 --torch-threads 2` passed (thread cap reached the PPO child processes).
  Local benchmark, production-size random members, 16 envs, PPO with the ppo.yaml settings:

  | torch threads | env alone, old → cached (steps/s) | PPO learn, old → cached (steps/s) |
  |---|---|---|
  | 14 (default, all cores) | 270 → 3375 | 262 → 1349 |
  | 4 | 497 → 6024 | 459 → 2125 |
  | 2 | – → 6264 | – → 1993 |
  | 1 | – → 5678 | – → 1546 |

  So PPO is ≈ 5× faster from the cache alone and ≈ 8× with a 2–4 thread cap; gradient updates are now the larger
  share. Seed 0's compute column was measured with the old code (report compute from seeds run with the new code,
  or rerun seed 0's loop).
- Waste in the loop: the selected checkpoints in rounds 1–5 were at 240k, 216k, 168k, 72k, 48k of 300k steps, so
  later rounds could run far fewer steps. Surrogate inference and DeepONet training run on CPU (`env_surrogate.yaml`
  device cpu); the L4 GPU is idle.

## 7. Seeds 1–2 setup (2026-09-23)
- `run.py seeds --new-seeds 1 2`: surrogate-PPO and direct PPO of each new seed as four concurrent branches (shared
  round-0 data, round-0 ensemble, tuned baselines), then `evaluate --seeds 0 1 2` (seed 0's evaluations reused by
  fingerprint; MPC stays on seed 0's ensemble), `tables --seeds 0 1 2`, `report`.
- `build_paper_tables.py --seeds`: per-seed tables in `tables/seed_<s>/`, summary in `tables/tables.md` (Tables I/II
  mean ± sd over seeds, per-seed TTS, headline reductions with a hierarchical bootstrap over seeds then episodes).
- Notebook `colab/m14_seeds.ipynb` works in the same Drive folder as seed 0, pins the code commit, keeps a copy of the
  seed-0-only tables, and has the watchdog.
- Tests: CLI dry run of `seeds`, per-branch seed propagation, seed summary maths; m14 suite 88 passed.
- Launched on Colab 2026-09-24 06:57 UTC (commit c3d07c0; 6 SUMO workers per branch, 4 torch threads). Seed 1
  round 1 surrogate PPO took 458 s (seed 0: 1946 s, ≈ 4.2× faster on Colab). Direct PPO runs at 7–9 steps/s
  (≈ 4–5 h for 120k steps), the long pole as expected.
- Aggregation loops finished (stop rule in both, as for seed 0), PPO 6–8 min per round:

  | seed | V per round (r1 → last) | selected | SUMO V | EE (selected / charged) | loop wall |
  |---|---|---|---|---|---|
  | 0 | −56.0, −45.6, −43.4, −44.0, −43.3 | r5 | −43.3 | 962 / 962 | 2.7 h |
  | 1 | −56.6, −46.2, −42.6, …(r4–r5 no gain ≥ 2) | r3 | −42.6 | 854 / 962 | 58 min |
  | 2 | −52.1, −43.8, −44.7, −43.2 | r4 | −43.2 | 908 / 908 | 48 min |

  Direct PPO seeds 1–2 at 28.8k / 120k steps at 08:14 UTC (≈ 24 % in 77 min → done ≈ 12:15 UTC).
- Direct PPO seeds 1–2 finished 11:31 / 11:33 UTC (≈ 4.6 h each; 1216 / 1212 SUMO episodes). Manifest rebuilt for
  seeds 0–2 (seed-0 evaluations reused as intended); final evaluation of the 13 new points started 11:34 UTC
  (≈ 285 s per point on test with 6 workers → ≈ 1.5 h incl. OOD).
- Early signal (12:17 UTC): seed 1's selected Surrogate-PPO (round 3, best V −42.6) has test TTS **46.5** veh h,
  1 breakdown and 5 catastrophic episodes in 90 (seed 0: 41.3, none); PI-ALINEA 47.3, seed-0 SUMO-PPO 46.6. The
  validation score did not predict test TTS for this seed; the three-seed margin over PI-ALINEA will be smaller than
  seed 0's 12.7 %.
- Test set complete (12:43 UTC), test TTS (veh h) of the selected policies:

  | arm | seed 0 | seed 1 | seed 2 | mean |
  |---|---|---|---|---|
  | Surrogate-PPO | 41.3 | 46.5 (r3; 1 bd, 5 cat.) | 44.4 (r4; 6 bd, 3 cat.) | ≈ 44.1 |
  | SUMO-PPO (1000 EE) | 46.6 | 62.2 (10 cat.) | 65.8 (12 cat.) | ≈ 58.2 |
  | PI-ALINEA / ALINEA | 47.3 / 51.4 | same | same | |

  Direct PPO is far more seed-sensitive than Surrogate-PPO: seeds 1–2 are worse than pure ALINEA. Rough three-seed
  margins: Surrogate-PPO ≈ 7 % below PI-ALINEA, ≈ 24 % below SUMO-PPO (tables pending, with seed-level CIs).

## 8. Results over policy seeds 0–2 (60 km/h ramp, 2026-09-24)
Sweep finished 13:19 UTC (`runs/study/m14/tables/tables.md`, per-seed tables in `tables/seed_<s>/`, seed-0-only copy
in `tables_seed0_only/`). Mean ± sd over seeds; baselines are seed-independent.

| method | ID TTS | ID max q | OOD TTS | OOD max q | SUMO episodes | compute h |
|---|---|---|---|---|---|---|
| ALINEA | 51.39 | 54.7 | 56.49 | 54.2 | 666 | 3.8 |
| PI-ALINEA | 47.34 | 47.2 | 53.19 | 52.6 | 882 | 4.9 |
| SUMO-PPO | 58.19 ± 10.22 | 61.0 ± 17.1 | 69.36 ± 15.42 | 62.2 ± 13.0 | 1215 ± 2 | 5.4 ± 0.1 |
| **Surrogate-PPO** | **44.10 ± 2.61** | **38.2 ± 6.3** | **49.82 ± 2.23** | **41.1 ± 1.9** | 944 ± 31 | 6.8 ± 1.2 |
| Surrogate-MPC | 85.91 | 13.4 | 91.48 | 20.2 | 962 | 5.7 |

Per seed (ID / OOD TTS): Surrogate-PPO 41.34 / 50.17, 46.55 / 51.86, 44.40 / 47.44; SUMO-PPO 46.57 / 51.55,
62.19 / 78.20, 65.79 / 78.33. Table I is unchanged across seeds (test density rel-L2 18.18 ± 0.15 %, outflow
4.99 ± 0.01 %, return 4.71 ± 0.23 %).

Headline reductions of Surrogate-PPO over all seeds (95 % CI, hierarchical bootstrap over seeds then episodes):

| vs | test (seed 0 alone) | test, 3 seeds | OOD (seed 0 alone) | OOD, 3 seeds |
|---|---|---|---|---|
| PI-ALINEA | 12.7 [8.7, 16.1] | **6.8 [0.6, 12.8]** (12.7 / 1.7 / 6.2) | 5.7 [−1.8, 11.6] | **6.3 [0.5, 11.2]** (5.7 / 2.5 / 10.8) |
| ALINEA | 19.6 [15.5, 23.2] | 14.2 [8.4, 19.6] | 11.2 [2.4, 18.4] | 11.8 [5.5, 17.1] |
| SUMO-PPO | 11.2 [7.6, 14.3] | **24.2 [11.8, 32.7]** (11.2 / 25.2 / 32.5) | 2.7 [−6.0, 9.4] | **28.2 [4.0, 39.9]** (2.7 / 33.7 / 39.4) |

Reading:
- Seed 0 was Surrogate-PPO's best seed and SUMO-PPO's best seed. Over three seeds Surrogate-PPO still beats
  PI-ALINEA in every seed on both sets, but by 6.8 % (ID) / 6.3 % (OOD), both CIs just above 0; the OOD margin is now
  significant (seed 0 alone was not).
- Direct SUMO-PPO at ~1200 episodes is strongly seed-dependent (ID 46.6 / 62.2 / 65.8; seeds 1–2 worse than pure
  ALINEA, 10–12 catastrophic test episodes). Surrogate-PPO's seed spread is 4× smaller (sd 2.6 vs 10.2 ID). The
  robust claim for the paper: at a comparable (slightly smaller) SUMO budget, surrogate training is much more
  reliable than direct training (−24 % / −28 % TTS), and it beats a tuned PI-ALINEA modestly but consistently.
- Surrogate-PPO keeps the smallest ramp queues of the working controllers (max queue 38 vs 47 for PI-ALINEA ID).
- Validation selection is imperfect: seed 1 had the best validation score (−42.6) but the worst test TTS (46.5,
  1 breakdown, 5 catastrophic episodes); seed 2 has 6/90 breakdowns. Worth a per-profile look.
- Compute: seeds 1–2 ran with the incremental GRU (loops 48–58 min); the 6.8 ± 1.2 h mean mixes code versions, so
  report per-seed or re-measure seed 0 before quoting compute. Timing check (per-seed tables, `study.json`,
  `rounds.json` fields `ppo_wall_s` / `wall_s`):

  | | seed 0 (old code) | seed 1 | seed 2 |
  |---|---|---|---|
  | rounds run / selected | 5 / r5 | 5 / r3 | 4 / r4 |
  | PPO per round (s) | 1550–2080 | 340–458 | 367–433 |
  | PPO total | 2.43 h | 32.6 min | 26.4 min |
  | loop wall (PPO + SUMO val + fine-tune) | 2.7 h | 58.1 min | 48.2 min |
  | non-PPO per round (s) | ≈ 185 | 262–370 | 288–394 |
  | Surrogate-PPO charged compute | 8.1 h | 6.3 h | 5.9 h |
  | SUMO-PPO charged compute / wall | 5.5 h / ≈ 5 h | 5.4 h / 4.6 h | 5.4 h / 4.6 h |

  With the fast code the surrogate charge is ≈ 4.0 h fixed (692 initial episodes ≈ 3.2 h + round-0 ensemble
  0.8 h) + ≈ 54 SUMO episodes and ≈ 11–13 min per round. The remaining 0.5–0.9 h gap to direct PPO is about the
  round-0 ensemble training; SUMO time alone is lower for Surrogate-PPO (908–962 vs ≈ 1215 episodes). Non-PPO time
  per round rose vs seed 0 because four processes shared the runtime (6 SUMO workers per branch vs 12). Elapsed
  time favours Surrogate-PPO (≈ 27 min E0/data/ensemble + ≈ 1 h loop vs 4.6 h direct PPO), but only because its
  SUMO data is collected in parallel while direct PPO runs one simulation at a time. Sweep wall: 06:57 → 13:19 UTC
  (6.4 h; direct PPO 4.6 h was the long pole, evaluation 1.75 h).
- Table fix after the run: identical-seed rows printed "± 0.00" from float noise; `_pm` now treats sd below 1e-9 of
  the mean as zero (rebuild the tables to clean the file; values unchanged).

## 9. Direct SUMO-PPO trained longer: 1000 → 1200 episodes (setup, 2026-09-24)
Goal: give direct PPO at least Surrogate-PPO's compute (fast code 5.9–6.3 h) so the comparison cannot be read as
"direct PPO needed more time". 1200 training episodes = 1200 + 15 × 18 validation = 1470 SUMO episodes ≈ 6.6 h
summed compute (1000 episodes: 1216 ≈ 5.4 h).
- `run.py extend-direct --seeds 0 1 2 --from-budget 1000 --to-budget 1200`: copies each finished run to
  `direct_ppo_1200ee_s<seed>` with its final model as the latest checkpoint (step 120000), its validation history
  and its ledger (`m14_direct_1200_s<seed>`), then `sumo-ppo --budgets 1200` resumes it to 144000 steps; the three
  seeds run concurrently; then evaluate/tables/report with budgets 1000 and 1200. Same-seed continuation with a
  constant learning rate and kept optimizer state is the same run trained longer; checkpoint selection covers
  steps 0–144k, so the 1200 run may keep a ≤ 120k checkpoint if later ones do not validate better.
- Fix in `train_ppo`: a resumed session's callbacks count steps from the restored step count, so checkpoints and
  evaluations stay on the `eval_freq` grid (120000 is off the 9600 grid; before the fix the evaluations would have
  come at 129.6k and 139.2k and none at 144k). Earlier resumes started from on-grid checkpoints and were unaffected.
- Tables: SUMO-PPO = largest budget; each smaller budget gets a `SUMO-PPO (<n> ep.)` row and its own headline
  reductions.
- Checks: local SUMO run 360 steps (eval every 240) then continued from the final model to 720 steps: checkpoints
  240/360/480/720, evaluations 240/480/720, ledger 6 training + 3 validation episodes, nothing lost. New tests
  (CLI dry run, run copy, multi-budget table); m14 suite 91 passed.
- Colab: `colab/m14_direct_extend.ipynb` (keeps the current tables in `tables_direct1000/`). Expected ≈ 1 h
  training + 15–20 min evaluation.

## 10. Results: direct SUMO-PPO at 1200 episodes (2026-09-25)
Colab run 02:27–04:15 UTC (commit 39bf227): training 02:28–03:33 (three seeds at once, ≈ 7 steps/s each),
evaluation of the three new policies 03:55–04:15 (all other results reused). Checkpoints and evaluations landed on
the 9600-step grid (124.8k, 134.4k, 144k), no ledger rows lost. The tables before the run are in
`tables_direct1000/`.

Checkpoint selection (validation mean return; higher is better):

| seed | 1000-ep. run selected | 1200-ep. run selected |
|---|---|---|
| 0 | 86.4k (−48.7) | 144k (−46.7) |
| 1 | 48k (−64.2) | 144k (−49.9) |
| 2 | 96k (−67.7) | 96k (−67.7), same policy (later checkpoints −69 to −72) |

Test / OOD TTS (veh h):

| | seed 0 | seed 1 | seed 2 | mean ± sd | SUMO episodes | compute h |
|---|---|---|---|---|---|---|
| SUMO-PPO 1000 ep., ID | 46.57 | 62.19 | 65.79 | 58.19 ± 10.22 | 1215 | 5.4 |
| **SUMO-PPO 1200 ep., ID** | 45.03 | 49.58 | 65.79 | **53.47 ± 10.92** | 1469 | 6.6 |
| Surrogate-PPO, ID | 41.34 | 46.55 | 44.40 | 44.10 ± 2.61 | 944 | 5.9–8.1 |
| SUMO-PPO 1000 ep., OOD | 51.55 | 78.20 | 78.33 | 69.36 ± 15.42 | | |
| **SUMO-PPO 1200 ep., OOD** | 49.71 | 53.89 | 78.33 | **60.64 ± 15.46** | | |
| Surrogate-PPO, OOD | 50.17 | 51.86 | 47.44 | 49.82 ± 2.23 | | |

PI-ALINEA 47.34 ID / 53.19 OOD. Surrogate-PPO's TTS reduction (hierarchical bootstrap CI):

| vs | test | OOD |
|---|---|---|
| SUMO-PPO 1200 ep. | **17.5 % [4.7, 31.9]** (8.2 / 6.1 / 32.5) | **17.8 % [−2.4, 38.8]** (−0.9 / 3.8 / 39.4) |
| SUMO-PPO 1000 ep. | 24.2 % [11.8, 32.7] | 28.2 % [4.0, 39.9] |
| PI-ALINEA (unchanged) | 6.8 % [0.6, 12.8] | 6.3 % [0.5, 11.2] |

Per seed, Surrogate-PPO vs SUMO-PPO 1200 ep. (paired bootstrap within the seed; `tables/seed_<s>/tables.md`):

| seed | SUMO-PPO 1200 ID TTS / max q | OOD TTS / max q | SUMO episodes | compute h | test reduction | OOD reduction |
|---|---|---|---|---|---|---|
| 0 | 45.03 / 38.8 | 49.71 / 46.3 | 1470 | 6.7 | 8.2 % [4.3, 11.5] | −0.9 % [−10.0, 6.0] |
| 1 | 49.58 / 49.4 | 53.89 / 53.4 | 1470 | 6.6 | 6.1 % [−1.1, 12.5] | 3.8 % [−4.0, 11.9] |
| 2 | 65.79 / 72.6 | 78.33 / 63.1 | 1466 | 6.6 | 32.5 % [27.2, 37.4] | 39.4 % [33.8, 43.4] |

### Training time (all runs, Colab L4, 12 CPUs)
Summed compute = SUMO episode wall times from the ledgers + DeepONet member training + surrogate PPO process time
(direct PPO's gradient updates are not in its ledger). Elapsed = wall-clock on Colab.

| run | SUMO episodes | summed compute h | elapsed | notes |
|---|---|---|---|---|
| Surrogate-PPO seed 0 (old GRU code) | 962 | 8.1 | ≈ 3.1 h (E0/data/ensemble 27 min + loop 2.7 h) | PPO 26–35 min per round |
| Surrogate-PPO seed 1 (fast code) | 962 (selected r3: 854) | 6.3 | loop 58 min (+ shared 27 min) | PPO 5.7–7.6 min per round |
| Surrogate-PPO seed 2 (fast code) | 908 | 5.9 | loop 48 min (+ shared 27 min) | PPO 6.1–7.2 min per round |
| SUMO-PPO 1000 ep., seed 0 | 1216 | 5.5 | ≈ 5 h (one simulation at a time) | |
| SUMO-PPO 1000 ep., seeds 1, 2 | 1216 / 1212 | 5.4 / 5.4 | 4.6 h each (concurrent with the loops) | ≈ 7 steps/s |
| SUMO-PPO 1200 ep., seeds 0/1/2 | 1470 / 1470 / 1466 | 6.7 / 6.6 / 6.6 | continuation 65 min (02:28–03:33, three seeds at once) | +254 episodes each |
| ALINEA / PI-ALINEA tuning | 666 / 882 | 3.8 / 4.9 | | seed-independent |

- Surrogate-PPO with the fast code (5.9–6.3 h) costs about as much compute as direct PPO at 1000 episodes (5.4 h)
  and less than the compute-matched 1200-episode run (6.6 h), while using 35–38 % fewer SUMO episodes.
- The 1000 → 1200 continuation took 65 min elapsed (≈ 13.5 s per SUMO episode incl. PPO updates) plus 20 min of
  final evaluation (03:55–04:15; all older results reused after re-hashing, ≈ 20 min 03:33–03:55). A fresh 1200-episode
  run would have taken ≈ 5.5 h.

Reading:
- More training helps direct PPO: seed 1 recovered (ID 62.2 → 49.6, OOD 78.2 → 53.9), seed 0 improved a little,
  seed 2 found nothing better than its 96k checkpoint. Seeds 0 and 1 selected their last checkpoint, so direct
  PPO is not converged at 1200 episodes; a still longer budget could narrow the gap further.
- With more compute (6.6 h vs 5.9–6.3 h fast-code Surrogate-PPO) and 1.55× the SUMO episodes, direct PPO is still
  worse on average than both Surrogate-PPO and PI-ALINEA (only seed 0 beats PI-ALINEA, ID 45.0 < 47.3).
- Surrogate-PPO vs compute-matched direct PPO: significant on test (17.5 % [4.7, 31.9]); on OOD the mean is 17.8 %
  but the CI includes 0, and seed 0 is a tie (−0.9 %). Much of the average gap comes from seed 2's failed direct
  run. The robust statements: Surrogate-PPO is better on test at equal compute with ≈ 35 % fewer SUMO episodes,
  and it is far more reliable across seeds (sd 2.6 vs 10.9 ID, 2.2 vs 15.5 OOD).
- For the paper: report SUMO-PPO at 1200 episodes as the compute-matched baseline (with the 1000-episode row or a
  sentence), and phrase the OOD comparison with direct PPO as not significant.

## Open
- Paper edits with the three-seed numbers (§8): Table II as mean ± sd, headline vs PI-ALINEA 6.8 % / 6.3 %, the
  reliability argument vs direct PPO; 60 km/h setting, Eq. 4 Q_{k+1}, Theorem 1, cost column (report per seed or
  re-measure seed-0 compute with the fast code). Direct-PPO baseline = 1200 episodes (§10: vs Surrogate-PPO
  17.5 % test, OOD n.s.), 1000-episode row as a supplement.
- Optionally: direct PPO still improving at 1200 episodes (seeds 0, 1 selected their last checkpoint); a longer
  continuation (e.g. 1500) would show where it saturates, at ≈ 1.2 h Colab per +300 episodes.
- Explanation of the Surrogate-MPC negative result.
- Per-profile look: seed 1's selected policy (validation-best, test-worst) and seed 2's 6/90 breakdowns; whether a
  different selection rule (e.g. worst-case or p10 on validation) would have picked better rounds.
- Download the results archives (m14_seeds.ipynb and m14_direct_extend.ipynb cell 8) and copy the key files into the local repo; rebuild
  the tables once with the `_pm` fix to drop the "± 0.00" artefacts.
