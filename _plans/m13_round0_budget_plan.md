# M13 plan — round-0 budget and mixture (2026-09-14)

## Why
The demo study (M12) showed the aggregation loop beats direct SUMO PPO at a matched
budget, but every surrogate arm pays a 692-episode entry fee (the round-0 dataset)
before its first point on the sample-efficiency curve, and that number was an
accident (E0 sweeps + a 480 target). Three diagnostics from the M9 review point the
same way: the plant model is at the SUMO seed-noise floor on open-loop control
(`_progress/m9_plant_model_v2_progress.md` §8), its error concentrates in the
closed-loop regimes (ALINEA rollouts 19 % return error), and 54 on-policy rollouts
in round 1 closed the surrogate-to-SUMO gap (+2.8 → −0.2). E1 showed 120 rollouts
already give a usable model.

## Questions
- **H1 (size):** can round 0 be 240 rollouts if the loop supplies the rest?
- **H2 (mixture):** does a closed-loop-heavy round 0 without the run-7 legacy policy help?

## Arms (one PPO seed, 300k steps per round, Mac CPU)
| arm | round 0 | rounds | budget |
|---|---|---|---|
| A1 (N0=692, as run) | `runs/aggregation/demo_s0` (reused) | 3 | 854 |
| A1 (N0=240, mix1) | original mixture, run-7 share → ALINEA, feedforward fold kept | ≤ 11, stop rule δ = 2 with patience 2 | 240 + 54·r ≤ 834 |
| A1 (N0=240, mix2) | alinea_wide 0.35, alinea_dither 0.15, random_policy 0.10, store_flush 0.10, feedforward 0.05, random_signal 0.15, constant 0.10 | same | same |

Both 240 stores force half of their profiles to be storage-mandatory (peak total
> 2500 vph; family base rate 49 %, realised ≈ 77 %). Reference arms (direct SUMO PPO,
PI-ALINEA, constant) are copied from the demo manifest, nothing is re-evaluated.

## New behaviour controllers (`src/rl/behaviour_controllers.py`)
- `alinea_wide`: ki U(5,60), kp 0 or U(0.5,12), ρ_set U(18,50), detector 10–13,
  u_init U(0.1,0.8), optional ALINEA/Q queue override (queue_max ∈ {none, 100, 200}).
- `alinea_dither`: alinea_wide plus held Gaussian action dither (σ U(0.05,0.2),
  hold 1/2/4 steps); the integrator continues from the applied rate.
- `random_policy`: a freshly initialised SB3 actor of the PPO architecture acting
  stochastically (log_std −2/−1.5/−1, init_u U(0.15,0.5), output gain 0.01/0.3/1.0).
- explicit `feedforward` type (disables the store_flush fold when present in the shares).
- `build_mixture_plan(per_entry_seeds=True)`: each draw seeded by (seed, type, j),
  so types can be added or resized without perturbing the others; the default path
  still reproduces the 692 store's plan (regression test).
- `enforce_storage_mandatory`: re-draws profiles (stride 10 000) until the peak
  total exceeds the threshold; draws recorded in `generation_plan.json`.

## Pipeline changes
- `scripts/generate_round0_dataset.py`: `dataset.per_entry_seeds`,
  `storage_mandatory_frac/vph`, `--store-dir/--network-dir`, `generation_plan_summary.json`.
- `scripts/run_aggregation_loop.py`: `--stop-patience` (default 1 = old rule);
  `bad_rounds`, `stopped_by_rule`, `rounds_cap` recorded.
- `scripts/run_final_evaluation.py`: per-point `ensemble_dir` so the density
  normalisation comes from the ensemble the policy was trained against (without it the
  240 arms would have been evaluated with the 692 store's statistics).
- `scripts/plot_sample_efficiency.py`: distinct colours for the three aggregation arms
  (longest-prefix match), transfer-gap figure with per-study markers, optional E1 figure.
- new `scripts/build_m13_manifest.py`, `scripts/run_m13.sh`,
  `configs/experiments/round0_small_mix{1,2}.yaml`, `tests/test_behaviour_controllers.py`.

## Driver
`sh scripts/run_m13.sh` (env: MIX, ROUNDS=11, STEPS_PER_ROUND=300000, WORKERS,
SEED=0, STOP_DELTA=2.0, STOP_PATIENCE=2, CONCURRENT=1). Per mix: dataset → 5-member
ensemble → gate (val, test) → aggregation loop; then manifest → final evaluation on
T and O of every round's policy → figures in `_progress/figures/m13/`. Outputs:
`data/plant_v2/round0_s240_<mix>`, `runs/surrogate/plant_v2_r0s240_<mix>`,
`runs/aggregation/m13_<mix>_s0`, `runs/study/m13/{arms.json, eval/}`; ledgers
`m13_r0_<mix>`, `m13_<mix>_s0`, `m13_final`.

## Success criteria
- H1 confirmed if a 240 arm reaches ≈ −60 on T (PI-ALINEA −60.6, demo A1 −55.0 at
  800 EE) by ≈ 500 EE and ≤ −57 by 834 EE with a T breakdown rate ≤ 0.05; refuted if at
  834 EE it sits below direct SUMO PPO at 844 EE (−64.1) or its round-1 rollouts stay
  catastrophic for three rounds.
- "Model too weak" signal: round-1 breakdown rate > 0.3 or all top-3 catastrophic
  on V with a pre-fine-tune return error > 0.25 (demo at 692: 0.06–0.11, 0.185).
- H2 confirmed if mix2 has a lower round-1 gap and breakdown rate than mix1 and
  reaches −60 at least one round earlier with non-overlapping CIs at a matched EE.
- Round-0 gate at N0 = 240: return error 0.15–0.25 expected on the ≈ 40-file val
  split; false-breakdown ≤ 0.10 is the criterion that matters.

## Verification done before launch
`pytest tests/test_behaviour_controllers.py tests/test_baseline_controllers.py
tests/test_plant_surrogate.py` (24 passed); a 32-rollout smoke run through every
stage (dataset with all seven controller types, 2-member ensemble, gate, two
aggregation rounds with the patience rule, manifest, final evaluation with the
per-point normalisation, figures), then deleted.
