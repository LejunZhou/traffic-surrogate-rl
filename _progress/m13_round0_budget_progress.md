# M13 progress — round-0 budget and mixture (2026-09-14)

Plan: `_plans/m13_round0_budget_plan.md`. Driver `scripts/run_m13.sh`, run
2026-09-14 01:19–03:47 with both mixes concurrent (4 SUMO workers each, 11-round cap, stop rule
δ = 2 with patience 2). Log: `runs/logs/run_m13.log`, per-stage logs `runs/logs/m13_*.log`.

## 0. Implementation (done, tested)
- Seven controller types in `src/rl/behaviour_controllers.py` (three new: `alinea_wide`,
  `alinea_dither`, `random_policy`; `feedforward` now explicit), per-entry seeding,
  storage-mandatory re-draw; 8 new tests in `tests/test_behaviour_controllers.py`
  (24 passed with the plant and baseline-controller suites).
- Smoke run (32 rollouts, 2-member ensemble, two 4 800-step rounds) exercised every
  stage including the patience stop rule and the per-point density normalisation in
  the final evaluator; artefacts deleted afterwards.
- Plan dry check: mix1 = constant 48 / alinea 96 / store_flush 32 / feedforward 16 /
  random_signal 48; mix2 = alinea_wide 84 / alinea_dither 36 / random_policy 24 /
  store_flush 24 / feedforward 12 / random_signal 36 / constant 24. Storage-mandatory
  fraction after the re-draw: 0.775 in both (120 forced, at most 11 attempts).

## 1. Round-0 stores and gates
Both stores: 240 rollouts in ≈ 13 min (4 workers each, concurrent), 77.5 % storage-mandatory
profiles, breakdown rate 0.55 (mix1) / 0.54 (mix2); splits ≈ 166 / 37 / 37.
mix1 = store_flush 32 / alinea 96 / constant 48 / random_signal 48 / feedforward 16;
mix2 = alinea_wide 84 / alinea_dither 36 / random_policy 24 / store_flush 24 / feedforward 12 /
random_signal 36 / constant 24. Ensembles: 5 members × 300 epochs, ≈ 20 min each (concurrent).

| store | split | rel-L2 ρ (free / band / jam) | rel-L2 q_exit | return-pred. error | false / missed breakdown | calib. slope | gate |
|---|---|---|---|---|---|---|---|
| mix1 | val (37) | 0.192 (0.245 / 0.208 / 0.165) | 0.061 | **0.113** | 0.056 / 0.211 | 1.46 | return error above 10 % → fail |
| mix1 | test (37) | 0.235 (0.268 / 0.230 / 0.221) | 0.062 | **0.104** | 0.053 / 0.278 | 1.93 | return error above 10 % → fail |
| mix2 | val (38) | 0.207 (0.254 / 0.232 / 0.183) | 0.060 | **0.078** | 0.000 / 0.091 | 1.47 | passed |
| mix2 | test (37) | 0.223 (0.267 / 0.238 / 0.226) | 0.061 | **0.081** | 0.077 / 0.208 | 1.23 | passed |
| 692 (M9, reference) | val (106) | 0.177 (0.201 / 0.194 / 0.208) | 0.053 | 0.114 | 0.018 / 0.102 | 1.52 | marginal fail |
| 692 (M9, reference) | test (100) | 0.163 (0.174 / 0.206 / 0.215) | 0.053 | 0.090 | 0.000 / 0.067 | 1.78 | passed |

First finding (H2, model side): with 240 rollouts the closed-loop-heavy mix2 ensemble passes
the round-0 gate on both splits (return error 0.078 / 0.081) and is *better* than the 692-rollout
reference, while mix1 (the original mixture without run-7) sits at 0.113 / 0.104, i.e. where the
692 store was. The 240 stores contain more storage-mandatory profiles (77 % vs 55 %), so the
splits are harder, not easier, than the reference. Each store's own val/test split is used, so
the three numbers are not on identical rollouts; the T/O policy results in §3 are the comparable ones.

## 2. Aggregation rounds

**mix1** — 5 rounds, stopped by the rule (patience 2), A1 = round 4 (-60.9 on V at 456 EE), 57 min wall (concurrent with the other mix).

| round | cum. EE | selected step | surrogate V | SUMO V | gap | breakdown (sel / top-3) | catastrophic (top-3) | model return err on new rollouts, before → after fine-tune | improvement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 294 | 288k | -82.0 | **-84.1** | -2.1 | 0.06 / 0.06, 0.00, 0.00 | 4, 4, 4 | 0.173 → 0.095 | – |
| 2 | 348 | 288k | -65.6 | **-64.3** | +1.3 | 0.00 / 0.00, 0.00, 0.00 | 2, 2, 2 | 0.235 → 0.172 | +19.8 |
| 3 | 402 | 72k | -59.0 | **-62.1** | -3.2 | 0.00 / 0.00, 0.00, 0.00 | 2, 2, 2 | 0.205 → 0.160 | +2.1 |
| 4 | 456 | 144k | -53.0 | **-60.9** | -7.9 | 0.00 / 0.00, 0.00, 0.00 | 2, 2, 2 | 0.204 → 0.171 | +1.2 |
| 5 | 510 | 96k | -56.4 | **-64.3** | -7.9 | 0.00 / 0.00, 0.00, 0.00 | 2, 2, 2 | 0.154 → 0.158 | -3.4 |

**mix2** — 6 rounds, stopped by the rule (patience 2), A1 = round 4 (-61.6 on V at 456 EE), 66 min wall (concurrent with the other mix).

| round | cum. EE | selected step | surrogate V | SUMO V | gap | breakdown (sel / top-3) | catastrophic (top-3) | model return err on new rollouts, before → after fine-tune | improvement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 294 | 288k | -76.2 | **-77.5** | -1.3 | 0.17 / 0.17, 0.06, 0.11 | 3, 4, 4 | 0.149 → 0.136 | – |
| 2 | 348 | 264k | -63.7 | **-64.2** | -0.5 | 0.00 / 0.00, 0.00, 0.00 | 3, 3, 2 | 0.175 → 0.259 | +13.4 |
| 3 | 402 | 168k | -69.8 | **-64.0** | +5.8 | 0.00 / 0.00, 0.00, 0.00 | 2, 2, 2 | 0.293 → 0.112 | +0.2 |
| 4 | 456 | 192k | -57.0 | **-61.6** | -4.6 | 0.06 / 0.06, 0.06, 0.00 | 2, 2, 2 | 0.162 → 0.160 | +2.4 |
| 5 | 510 | 24k | -61.7 | **-72.3** | -10.6 | 0.00 / 0.06, 0.00, 0.06 | 2, 3, 3 | 0.138 → 0.117 | -10.7 |
| 6 | 564 | 288k | -61.0 | **-70.5** | -9.5 | 0.11 / 0.11, 0.06, 0.11 | 3, 3, 2 | 0.153 → 0.121 | -8.9 |

**Reference (692 store, `demo_s0`):** V returns −68.5 / −58.4 / −49.5 at 746 / 800 / 854 EE, gaps +2.8 / −0.2 / +6.3, round-1 breakdown 0.11 (see `_progress/m12_study_progress.md`).

Observations on the loop itself (policy results on T/O in §3):
- Neither 240 model was "too weak": round-1 breakdown rates on V were 0.06 (mix1) and 0.17 (mix2),
  against 0.11 for the 692 store, and the round-1 return error on the policy's own rollouts was
  0.17 / 0.15 (692: 0.185). The surrogate-to-SUMO gap was within ±2 in round 1 for both.
- Both loops reach ≈ −61 on V at 456 EE (round 4), which the 692 arm only passes at 800 EE.
- Both loops then *regress*: rounds 5–6 select policies that are 3–11 veh h worse in SUMO
  than round 4, with the surrogate over-optimistic (gap −8 to −11). The fine-tuned ensembles'
  return error on the newest rollouts stays at 0.12–0.17 rather than falling as in the demo
  (0.185 → 0.128 → 0.099 → 0.079), i.e. 20 fine-tune epochs on a store growing by 20 % per round
  no longer track the policy. The stop rule (δ = 2, patience 2) ended both runs; the A1 policy
  is round 4 in both cases.
- Every round after the first has 2 catastrophic V episodes for every top-3 checkpoint
  (the same two hardest V profiles), so the `feasible` filter never bites and selection is by mean.

## 3. Final evaluation on T and O

### Test set T (30 profiles × 3 seeds = 90 episodes), ledger `m13_final`
| arm | round | EE | mean | p10 | worst | breakdown | TTS (veh h) | Δ vs PI-ALINEA [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| A1 aggregation (N0=692, as run) | 0 (zero-shot) | 692 | **-70.0** | -165.7 | -379.0 | 0.13 | 61.5 | -9.5 [-15.8, -3.2] |
| A1 aggregation (N0=692, as run) | 2 | 800 | **-55.0** | -78.4 | -325.1 | 0.00 | 49.9 | +5.6 [+1.4, +9.0] |
| A1 aggregation (N0=692, as run) | 3 | 854 | **-54.7** | -87.7 | -294.7 | 0.02 | 48.8 | +5.9 [+1.6, +9.8] |
| B direct SUMO PPO |  | 290 | **-79.7** | -164.3 | -208.4 | 0.17 | 71.9 | -19.2 [-24.9, -13.9] |
| B direct SUMO PPO |  | 844 | **-64.1** | -121.1 | -315.3 | 0.19 | 56.7 | -3.5 [-7.9, +1.0] |
| ALINEA |  | 522 | **-60.6** | -108.7 | -222.5 | 0.02 | 55.2 | +0.0 [+0.0, +0.0] |
| constant u |  | 198 | **-95.7** | -197.8 | -402.6 | 0.43 | 79.7 | -35.1 [-43.4, -27.3] |
| A1 aggregation (N0=240, mix1) | 0 (zero-shot) | 240 | **-83.4** | -166.7 | -360.8 | 0.01 | 70.6 | -22.8 [-29.9, -16.3] |
| A1 aggregation (N0=240, mix1) | 0/1 (zero-shot = round-1 top) | 294 | **-83.4** | -166.7 | -360.8 | 0.01 | 70.6 | -22.8 [-29.9, -16.3] |
| A1 aggregation (N0=240, mix1) | 2 | 348 | **-68.0** | -137.2 | -296.3 | 0.00 | 61.4 | -7.4 [-11.0, -4.0] |
| A1 aggregation (N0=240, mix1) | 3 | 402 | **-66.1** | -130.2 | -280.1 | 0.00 | 60.2 | -5.5 [-9.3, -1.9] |
| A1 aggregation (N0=240, mix1) | 4 | 456 | **-70.1** | -130.6 | -279.7 | 0.00 | 63.4 | -9.5 [-14.1, -5.3] |
| A1 aggregation (N0=240, mix1) | 5 | 510 | **-69.5** | -136.2 | -275.6 | 0.00 | 63.1 | -9.0 [-12.9, -5.4] |
| A1 aggregation (N0=240, mix2) | 0 (zero-shot) | 240 | **-83.7** | -183.0 | -385.8 | 0.20 | 71.8 | -23.2 [-31.2, -16.0] |
| A1 aggregation (N0=240, mix2) | 0/1 (zero-shot = round-1 top) | 294 | **-83.7** | -183.0 | -385.8 | 0.20 | 71.8 | -23.2 [-31.2, -16.0] |
| A1 aggregation (N0=240, mix2) | 2 | 348 | **-67.8** | -138.7 | -332.6 | 0.00 | 59.7 | -7.2 [-12.8, -2.2] |
| A1 aggregation (N0=240, mix2) | 3 | 402 | **-68.4** | -141.2 | -300.5 | 0.00 | 60.4 | -7.8 [-12.1, -3.5] |
| A1 aggregation (N0=240, mix2) | 4 | 456 | **-66.7** | -121.2 | -299.3 | 0.00 | 58.7 | -6.1 [-10.9, -1.6] |
| A1 aggregation (N0=240, mix2) | 5 | 510 | **-68.5** | -141.7 | -304.1 | 0.00 | 59.9 | -7.9 [-12.8, -3.4] |
| A1 aggregation (N0=240, mix2) | 6 | 564 | **-71.3** | -148.3 | -345.5 | 0.07 | 61.3 | -10.8 [-16.9, -5.4] |

### OOD set O (12 × 3 = 36 episodes), ledger `m13_final`
| arm | round | EE | mean | p10 | worst | breakdown | TTS (veh h) | Δ vs PI-ALINEA [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| A1 aggregation (N0=692, as run) | 0 (zero-shot) | 692 | **-91.4** | -217.0 | -261.2 | 0.08 | 79.6 | -19.8 [-30.1, -10.9] |
| A1 aggregation (N0=692, as run) | 2 | 800 | **-72.0** | -156.9 | -232.8 | 0.00 | 64.6 | -0.4 [-6.3, +4.9] |
| A1 aggregation (N0=692, as run) | 3 | 854 | **-70.4** | -133.7 | -216.1 | 0.00 | 63.0 | +1.2 [-4.8, +7.0] |
| B direct SUMO PPO |  | 290 | **-111.4** | -262.6 | -276.1 | 0.44 | 97.8 | -39.8 [-51.3, -29.5] |
| B direct SUMO PPO |  | 844 | **-85.0** | -188.6 | -231.7 | 0.42 | 74.8 | -13.5 [-20.4, -7.1] |
| ALINEA |  | 522 | **-71.6** | -151.1 | -176.9 | 0.00 | 66.2 | +0.0 [+0.0, +0.0] |
| constant u |  | 198 | **-110.0** | -257.9 | -282.0 | 0.39 | 94.3 | -38.4 [-51.7, -27.0] |
| A1 aggregation (N0=240, mix1) | 0 (zero-shot) | 240 | **-93.2** | -228.5 | -253.0 | 0.03 | 80.1 | -21.6 [-31.5, -13.2] |
| A1 aggregation (N0=240, mix1) | 0/1 (zero-shot = round-1 top) | 294 | **-93.2** | -228.5 | -253.0 | 0.03 | 80.1 | -21.6 [-31.5, -13.2] |
| A1 aggregation (N0=240, mix1) | 2 | 348 | **-79.4** | -189.8 | -213.0 | 0.00 | 71.4 | -7.8 [-12.8, -3.5] |
| A1 aggregation (N0=240, mix1) | 3 | 402 | **-72.8** | -152.3 | -197.5 | 0.00 | 66.8 | -1.2 [-3.8, +1.1] |
| A1 aggregation (N0=240, mix1) | 4 | 456 | **-73.1** | -174.3 | -195.6 | 0.00 | 66.7 | -1.6 [-4.9, +1.5] |
| A1 aggregation (N0=240, mix1) | 5 | 510 | **-74.4** | -146.7 | -195.7 | 0.00 | 67.6 | -2.8 [-6.8, +1.0] |
| A1 aggregation (N0=240, mix2) | 0 (zero-shot) | 240 | **-98.9** | -243.9 | -269.3 | 0.25 | 85.4 | -27.3 [-39.2, -17.4] |
| A1 aggregation (N0=240, mix2) | 0/1 (zero-shot = round-1 top) | 294 | **-98.9** | -243.9 | -269.3 | 0.25 | 85.4 | -27.3 [-39.2, -17.4] |
| A1 aggregation (N0=240, mix2) | 2 | 348 | **-82.3** | -209.9 | -230.1 | 0.00 | 72.3 | -10.7 [-18.0, -4.6] |
| A1 aggregation (N0=240, mix2) | 3 | 402 | **-75.0** | -168.1 | -208.4 | 0.00 | 66.7 | -3.4 [-7.1, -0.2] |
| A1 aggregation (N0=240, mix2) | 4 | 456 | **-76.8** | -167.2 | -210.1 | 0.00 | 67.9 | -5.2 [-9.3, -1.6] |
| A1 aggregation (N0=240, mix2) | 5 | 510 | **-78.9** | -163.6 | -212.3 | 0.00 | 69.0 | -7.3 [-12.1, -3.0] |
| A1 aggregation (N0=240, mix2) | 6 | 564 | **-83.7** | -188.0 | -238.0 | 0.06 | 71.6 | -12.1 [-19.0, -5.8] |

## 4. Reading

**H1 (a 240-rollout round 0 is enough to start the loop): supported for reaching the
direct-PPO level, not for reaching the 692 arm's final level within the rounds the stop rule allowed.**
- Zero-shot from 240 rollouts is worse than from 692 (−83 vs −70 on T), as expected.
- Two rounds later (348 EE) both 240 arms are at −68 on T with 0 % breakdowns, i.e. they
  match direct SUMO PPO at 844 EE (−64.1, 19 % breakdowns; CI of the difference overlaps zero)
  with 41 % of its budget, and beat direct PPO at 290 EE (−79.7) by 12–16 veh h. On the OOD set
  they reach −73 to −75 at 402 EE, a tie with PI-ALINEA (−71.6) and 10 veh h ahead of direct PPO
  at 844 EE (−85.0).
- They then plateau at −66 to −70 on T and never reach PI-ALINEA (−60.6) or the 692 arm's −55.
  The 692 arm passes −60 only at 800 EE (its own round 2), so the two families never overlap in
  budget: the stop rule (δ = 2, patience 2, on the *running best* V return) ended the 240 runs at
  510 / 564 EE after two regressing rounds. Whether they would have continued to −55 by 800 EE
  is not known from this run.
- Sample-efficiency reading (fig 1): below ≈ 700 EE the 240 arms dominate every other point on
  the curve; above 800 EE the 692 arm is 12 veh h better. The cheapest way to the direct-PPO
  quality is a small round 0 plus two rounds (≈ 350 EE); the best quality seen still needs the
  larger round 0.

**H2 (closed-loop-heavy mixture helps): supported for the model, not for the policy.**
- The mix2 ensemble passes the round-0 gate (return error 0.078 / 0.081) where mix1 fails
  (0.113 / 0.104), and is better than the 692 reference.
- The policies are indistinguishable on T (mix1 −66.1 … −70.1, mix2 −66.7 … −71.3 across rounds
  2–6; every paired CI between the two overlaps) and mix1 is slightly better on O. mix2's
  zero-shot policy breaks down more (0.20 vs 0.01 on T): the random-policy and dithered inputs
  make the model better at predicting *any* controller, but the first PPO policy exploits the
  remaining error more aggressively. No evidence that the mixture matters once the loop runs.

**Negative finding on the loop (both mixes): after round 4 the fine-tuned ensemble stops
tracking the policy.** Return error on the newest rollouts stays at 0.12–0.17 (the demo's went
0.185 → 0.079), the surrogate becomes over-optimistic (gap −8 to −11 in rounds 5–6), and the
selected policies regress in SUMO. The fixed 20 fine-tune epochs at lr 3e-4 were tuned for
a store where 54 new rollouts are 8 % of the data; here they are 20 % and the members drift.
The stop rule then fires on noise from the regression rather than on convergence.

**What to change next (for the paper-scale runs):**
1. Fine-tune budget scaled with the new-data fraction (e.g. epochs ∝ new/total, or retrain
   from scratch every second round), and select the A1 policy on V *after* the loop rather
   than letting a regressing round end it: stop on the running best with patience 3, or run
   to a fixed budget (the 240 arms need 11 rounds to reach 834 EE for a matched comparison).
2. Keep round 0 at 240 rollouts with the closed-loop-heavy mixture (better model, same policy,
   no legacy checkpoint needed) and re-run to a fixed 834 EE cap — one 3-hour run — to close
   the missing overlap with the 692 arm.
3. The two hardest V profiles are catastrophic for every checkpoint from round 2 on; the
   `feasible` filter never bites. Selection by mean is fine, but those two profiles should be
   looked at (are they solvable at all under the queue constraint?).

Figures: `_progress/figures/m13/fig1_return_vs_ee.png` (return vs EE, all arms),
`fig2_return_vs_wallclock.png` (wall-clock inflated by the concurrent runs),
`fig4_transfer_gap.png` (both loops; the round ≥ 4 cluster sits below the diagonal),
`fig5_breakdown_rates.png`, `fig7_ood.png`.

## 5. Plant-model figures for the two 240-rollout ensembles
`scripts/plot_plant_eval.py --ensemble runs/surrogate/plant_v2_r0s240_<mix> --split test --out
_progress/figures/m13_plant_<mix>` (representative picker now covers the M13 controller types).
mix2's model reproduces the jam pulse trains of the oscillating wide-ALINEA controllers and
the dithered ones, and the noisy near-constant random actors; return error 0.081 (median 0.051),
calibration slope 1.23 with 55 % coverage — the best-calibrated ensemble so far. mix1's misses
come from the store-and-flush regime (return error 0.20, half the breakdowns missed). Field
rel-L2 (0.22–0.24) is above the 692 reference (0.163) because these test splits are 77 %
storage-mandatory with ≈ 50 % breakdowns. Per-regime tables in `figures/m13_plant_<mix>/summary.json`.
