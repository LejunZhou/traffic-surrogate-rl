# M10 progress — Surrogate RL and the aggregation loop (2026-09-13)

Plan: `_plans/m10_surrogate_rl_plan.md`. Study `demo_s0` (`runs/aggregation/demo_s0`,
ledger `runs/ledger/demo_s0.jsonl`): demo budget on the Mac CPU — 300k PPO steps per
round (≈ 2 500 surrogate episodes, 9 min at ≈ 600 fps on 16 batched envs), top-3
checkpoints rolled on the 18 V profiles per round (54 EE), 3 rounds, one PPO seed.
Paper scale (1M steps, 4 rounds, 5 seeds) runs the same script with the env vars of
`scripts/run_study.sh`.

## 1. Implementation checks
- `SurrogateVecEnv` vs `SumoEnv` parity test (`tests/test_plant_surrogate.py::test_sumo_and_surrogate_env_parity`):
  identical observation layout (19 densities + d/2500 + r/1000 + look-ahead + k/K + Q/100),
  identical demand / time features, the same offered q_ref, the same warm-up mask, queue
  recursions within the 1-vehicle integer rounding of SUMO's meter.
- Smoke loop (`smoke_agg`, 4 800-step rounds on a 2-epoch ensemble) exercised PPO →
  ranking → 36 SUMO rollouts → store append → fine-tune → stop rule → A0 / A1 outputs.
- PPO statistics on the surrogate: approx_kl 0.004–0.01 at target 0.02, one early stop in
  the first update, ep_rew_mean −70 → −58 across rounds.

## 2. Aggregation rounds (demo_s0)
| round | PPO steps | surrogate V return (top-1) | SUMO V return of the selected ckpt | gap (SUMO − surrogate) | breakdown rate on V | return-pred. error on the 54 new rollouts, before → after fine-tune | cumulative EE |
|---|---|---|---|---|---|---|---|
| 1 | 300k (from action_init_u 0.3) | −71.3 | **−68.5** (ckpt 264k; p10 −160, worst −212) | +2.8 | 0.11 | 0.185 → 0.128 | 692 + 54 = 746 |
| 2 | 300k (warm start) | −58.2 | **−58.4** (ckpt 288k; p10 −100, worst −195) | −0.2 | 0.00 | 0.101 → 0.099 | 800 |
| 3 | 300k (warm start) | −55.7 | **−49.5** (ckpt 288k; p10 −87, worst −134) | +6.3 | 0.00 | 0.101 → 0.079 | 854 |

References on the same 18 V episodes: PI-ALINEA (tuned, 522 EE) −57.0; best constant
u = 0.3 −90.1 (see `_progress/m11_reference_arms_progress.md`).

Reading: the zero-shot policy (A0 = round-1 top checkpoint by surrogate return) transfers
with a +2.8 gap and beats every constant by 20+ veh h but is 11 veh h behind tuned
PI-ALINEA and jams on 2 of 18 profiles; one aggregation round (54 EE of on-policy data,
20 fine-tune epochs) closes the gap to −0.2 and the warm-started round-2 policy matches
PI-ALINEA (−58.4 vs −57.0, 0/18 breakdowns, TTS 53.1 vs 52.0 veh h) at 800 EE; round 3
reaches **−49.5 on V (TTS 44.9 veh h, 0/18 breakdowns, worst episode −134)**, 7.5 veh h
better than PI-ALINEA, at 854 EE, 45 min wall-clock for the whole loop (three 9-min PPO
phases, three 54-episode SUMO batches, three 2–3 min fine-tunes). The surrogate's own
V-return ranking picked the right checkpoint in every round (top-1 by surrogate return
was top-1 in SUMO), and the fine-tuned ensembles' return-prediction error on the
new on-policy rollouts fell from 0.185 (round-0 model on round-1 data) to 0.079.
The round-3 gap (+6.3: SUMO better than the surrogate) says the fine-tuned model is
now pessimistic on the policy's own distribution; the stop rule (< 2 veh h improvement)
never fired in three rounds, so a fourth round is warranted at paper scale.

## 3. Ledger (runs/ledger)
`m8_e0` 404 dataset lines, `m9_round0` 288 dataset lines, `demo_s0` 54 aggregation lines
per round; `demo_alinea` 522 tuning lines; direct-PPO studies log every training and
eval_val episode. `Ledger.budget()` excludes eval_test / eval_ood.
