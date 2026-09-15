# M9 progress — Plant-model DeepONet v2 (2026-09-13)

Plan: `_plans/m9_plant_model_v2_plan.md`.

## 1. Round-0 dataset (`data/plant_v2/round0`, ledgers `m8_e0`, `m9_round0`)
692 rollouts of the training family, `speed_dev` 0.03, occupancy density, one
behaviour controller each: 152 constant, 180 store-and-flush (108 from E0 + 72 of the
mixture, one third of which are capacity-tracking feedforward schedules), 120 ALINEA /
PI-ALINEA with random gains, 96 random open-loop signals, 72 run-7 policy with action
noise (the run-7 checkpoint is on this Mac, so D6's share was kept). E0's 404 rollouts
are part of the set (the draft's "reused as data"), the mixture generator only filled
the remaining shares (`--target-total 480`, 288 new rollouts, 8 min on 8 workers).
Breakdown rate 45 %; split 486 / 106 / 100 stratified by controller type and peak-total
tertile; train-split density mean 20.9 / std 16.6 veh/km (the normalisation both
environments z-score with).

## 2. Ensemble training (`runs/surrogate/plant_v2_round0`, 5 members, 300 epochs)
GRU branch (2 layers × 128, p = 256), trunk 3 × 512, bootstrap resample per member;
31 min wall-clock for the five members in parallel (2 threads each). Validation curve
of member 0: rel-L2(ρ) 0.77 → 0.27 (epoch 60) → 0.19 (epoch 260); exit-flow rel-L2 0.22 →
0.055. Best epochs 200–290.

## 3. Round-0 gate (`scripts/eval_surrogate_regimes.py`, ensemble mean)
| split | rel-L2 ρ (free / band / jam) | rel-L2 q_exit | return-pred. error | false / missed breakdown | onset err (min) | calib. slope | gate |
|---|---|---|---|---|---|---|---|
| val (106) | 0.177 (0.201 / 0.194 / 0.208) | 0.053 | **0.114** | 0.018 / 0.102 | 1.6 | 1.52 | return error above 10 % → marginal fail |
| test (100) | 0.163 (0.174 / 0.206 / 0.215) | 0.053 | 0.090 | 0.000 / 0.067 | 1.9 | 1.78 | passed |

Per regime (val) the return-prediction error is 5 % for store-and-flush, 9–10 % for
constants and feedforward, 16 % for ALINEA and 26 % for the run-7 policy rollouts: the
closed-loop, edge-riding controllers are the hard part, as D6 anticipated. Calibration:
ensemble std vs |error| slope 1.5–1.8 (errors ≈ 1.5–1.8 × spread; 48 % of cells inside
± 2 std), so disagreement is a usable, slightly over-confident uncertainty signal.
Decision: proceed to RL with this ensemble (test gate passed, val marginal); the
aggregation loop targets exactly the on-policy regime where the error is largest.

## 4. E2 reward parity (`_progress/m9_e2_parity.json`)
384 E0 sweep rollouts replayed through the ensemble mean (0 EE):

| term | SUMO range | surrogate range | ratio | corr |
|---|---|---|---|---|
| S_out (lost outflow) | 51.9 | 51.2 | 0.99 | 0.997 |
| S_que | 109.4 | 109.4 | 1.00 | 1.000 |
| S_std | 364.3 | 358.2 | 0.98 | 0.996 |
| S_road (veh h on the road) | 51.1 | 51.2 | 1.00 | 0.997 |
| S_backlog | 157.9 | 142.8 | 0.90 | 0.987 |
| S_tts | 330.7 | 330.6 | 1.00 | 0.994 |

Every range within 20 % → parity **PASSED**; TTS-return mean relative error 7.7 %,
correlation 0.994. The best schedule per E0 profile agrees with SUMO's in 4/12 profiles
only: the candidate schedules are often within a few veh h of each other, so the
ranking is sensitive to the 8 % error even though the scale is right.

## 5. E1 surrogate study
Launched by `scripts/run_study.sh` after the RL arms (`scripts/run_e1_surrogate_study.py`,
3 members × 150 epochs per variant, results in `_progress/m9_e1_surrogate_study.json`,
Fig. 3): N_0 ∈ {120, 240, 486}; random-only (constant + random signals) vs mixture;
GRU vs causal conv vs padded-MLP branch; M = 1 vs 5; DeepONet vs one-step MLP.
Results are appended to `_progress/m12_study_progress.md` when the run finishes.

## 6. Deviations from the draft, and why
- GRU branch by default instead of the dilated conv (CPU speed, 20×; conv kept as ablation).
- Occupancy divisor 5 m + jam clip instead of 7 m (see M8 §plan).
- 300 epochs instead of 150 (3.5–6.5 s/epoch made it cheap; best epochs were 200–290).
- The E0 rollouts count as round-0 data, so N_0 = 692 rather than 480 (all in the ledger).

## 7. Field-level figures (`scripts/plot_plant_eval.py` → `_progress/figures/m9_plant_eval/`)
Round-0 ensemble on the test split (100 rollouts). Representative rollouts are the median
return-error case inside each (regime, breakdown) cell, not the best case.
- `fig_a_fields`: SUMO density / ensemble mean / |error| / ensemble std as (x, t) maps for a
  store-and-flush breakdown, an ALINEA breakdown, a random-signal breakdown and a run-7 policy
  rollout. The jam wedge upstream of the merge (position, onset, extent) is reproduced; the
  errors sit on the wedge's edges (the shock front is a few cells too smooth) and on the
  within-jam speckle that SUMO's 30-s occupancy carries; the ensemble std lights up on the same
  edges, which is the calibration signal the loop uses.
- `fig_b_exit_flow`: exit flow rel-L2 0.04–0.09 per rollout; the capacity drop after breakdown
  and the recovery are timed correctly; the last 5 min of some rollouts are under-predicted.
- `fig_c_cells`: detector traces at 1000 / 1300 / 1700 m; the merge cell is the best predicted,
  the run-7 policy's short 25–35 veh/km episodes at 1000–1300 m are smoothed away (rel-L2 jam
  0.45 for that regime is these near-threshold events, not a missed breakdown).
- `fig_d_error_map`: mean |error| over the split is 2.5 veh/km/lane (mean density 21.9) and
  75 veh/h for the exit flow (mean 1874 veh/h); error grows with time from 1 to 3 veh/km/lane
  and concentrates in the 200–1250 m jam region.
- `fig_f_case_studies`: six test rollouts (one per regime), one column each: demand d(t), r(t);
  metering rate u(t); metered ramp inflow; SUMO density; DeepONet density; exit flow. This is the
  end-to-end view of what the operator maps (d, metered inflow) → (ρ field, q_exit).
- `fig_e_return_scatter`: predicted vs true return, mean rel. error 0.090 (median 0.045); the
  four outliers (> 20 veh h) are ALINEA and feedforward rollouts on which the member spread is
  small, i.e. the ensemble is confidently wrong there — the argument for on-policy aggregation.

## 8. Is N_0 = 692 enough? SUMO seed-noise floor (`_progress/m9_seed_noise_floor.json`)
Diagnostic run (not in the store or the ledger): the first 10 *test-set* profiles (never in
any plant store) × constant u ∈ {0.3, 0.6} × 3 SUMO seeds, 60 one-hour rollouts. The
seed-to-seed rel-L2 of the density field (same profile, same control, different seed) is the
floor no deterministic model can beat.

| quantity (mean over 20 profile × u cases) | density rel-L2 | exit-flow rel-L2 |
|---|---|---|
| seed-to-seed (noise floor) | 0.149 | 0.048 |
| round-0 ensemble vs one seed | 0.157 | 0.044 |
| round-0 ensemble vs the 3-seed mean | 0.125 | – |

On open-loop constant control the ensemble is *at* the SUMO noise floor on unseen demand
profiles; breakdown status never differed across seeds and the ensemble got it right on all
20 cases. The exception is one profile (late 780-vph ramp surge after the mainline peak):
error 0.30–0.42 against a floor of 0.10, a level bias of ≈ 4 veh/km/lane in the tail with a
member spread up to 29 veh/km/lane, i.e. flagged as uncertain. Together with §3 and E1:
the data-size curve on val (rel-L2 0.33 → 0.28 → 0.18 for 120 → 240 → 486 train rollouts)
has not flattened, and the closed-loop regimes (ALINEA 19 % return error, run-7 policy)
carry the remaining error, so what 692 lacks is *coverage of closed-loop control inputs*,
not more open-loop rollouts. That is what the aggregation rounds add (54 EE closed the
transfer gap in round 1). Cost is not the constraint (2000 rollouts ≈ 30 min on 8 workers);
the episode budget accounting is.
