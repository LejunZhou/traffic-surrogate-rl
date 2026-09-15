# M8 plan — Scenario v2: time-varying demand profiles (draft_pipeline.md step 1, E0)

Source: `draft_pipeline.md` §4, §12 step 1, §11 E0. Status log: `_progress/m8_scenario_v2_progress.md`.

## Goal
Make the SUMO scenario accept a per-episode, time-varying mainline demand d_k and
ramp arrival rate r_k (piecewise-constant on 5-min blocks), measure density with one
bounded estimator in both environments, freeze the validation / test / OOD profile
sets, and check (E0) that a dynamic metering policy has something to win.

## Deliverables
| Item | Location | Notes |
|---|---|---|
| Profile family + seeded sampler | `src/sumo_env/demand_profiles.py`, `configs/profiles/family_v1.yaml` | `ProfileFamily.sample_by_key(set, index)`; families of §4.1 (peak / step / const × surge / const), OOD families (double, plateau, early surge) |
| Frozen sets | `configs/profiles/{val,test,ood}.json` | V 18 (stratified peak-total tertile × surge-overlaps-peak, seeds 10000+i), T 30 × seeds 100–102, O 12 × 3 |
| Per-block routes | `network_builder._write_routes(path, cfg, mainline_blocks)` | one `<flow>` per block, `departSpeed="desired"`, `--extrapolate-departpos` unchanged |
| Time-varying ramp arrivals | `MeteredRampQueue.step(u, arrival_vph)`, `analytic_queue_step` | same recursion in SumoEnv (inline) and SurrogateVecEnv |
| Scenario v2 config | `configs/sumo/scenario_v2.yaml` | phase1_1 geometry + `speed_dev 0.03` + occupancy density |
| SumoEnv profile mode | `src/rl/sumo_env_wrapper.py` | `env.profiles` (family YAML / set JSON), fixed-constant normalisers, z-score clip [-3, 25], optional look-ahead, per-step d_k / r_k, offered q_ref, conservation backlog, TTS terms in info |
| Closed-loop rollout driver + npz contract | `src/sumo_env/rollout.py` | `rollout_episode`, `save_rollout_npz`, §10 metrics (`episode_metrics`, `breakdown_flags`), offline `rescore_return` |
| Behaviour controllers + mixture plan | `src/rl/behaviour_controllers.py` | constant grid, ALINEA/PI random gains, store-and-flush, capacity-tracking feedforward, run-7 policy adapter + noise, random signals |
| Parallel rollout runner | `src/sumo_env/parallel_rollouts.py` | spawn pool, one SumoEnv per worker, policy specs |
| Rollout store | `src/sumo_env/rollout_store.py` | index, stratified 70/15/15 split, `append_round`, `fork` (per-study store) |
| E0 script | `scripts/run_scenario_characterisation.py` | insertion check, capacity map, schedule family, gate; returns re-scored under the training reward |
| Tests | `tests/test_demand_profiles.py`, additions to `tests/test_reward.py` | determinism, bounds, block XML, frozen sets, queue with time-varying arrivals |

## Density estimator decision (D9)
`detectors.density_method: occupancy`, ρ = occ · 1000 / ℓ_eff per lane, clipped at the
jam density 1000/(ℓ + minGap) = 143 veh/km/lane, lane-averaged on the acceleration
segment. The draft's divisor ℓ + minGap = 7 m under-estimates moving traffic by 5/7
(a point loop is occupied for ℓ/v, not (ℓ + minGap)/v), so ℓ_eff = 5 m plus the clip
was chosen: unbiased in free flow, bounded at a standstill.

## E0 gate (as run)
1. insertion check: 20 family profiles at a pass-through rate (u = 0.65);
2. capacity map: constant u ∈ {0, 0.1, …, 1.0} on 12 profiles (drawn from the family with the `e0` seed set);
3. dynamic schedules on the same profiles: store-and-flush (u_low ∈ {0, 0.15, 0.3}, flush at u = 1 after
   {0, 5, 10} min), moderate flush (u_low ∈ {0.1, 0.2, 0.3} × u_high ∈ {0.5, 0.7}), capacity-tracking
   feedforward u_k = (C − d_k)/D for C ∈ {2300, 2400, 2500} and 0 / 2 step lag;
4. gate: best schedule beats best constant by ≥ 15 return units on ≥ 2/3 of the profiles where
   the peak total exceeds capacity (≈ 2500 vph). All 404 rollouts are kept as round-0 data.

## Exit criteria
- per-block insertion exact where no breakdown occurs (served / offered ≥ 0.98, 0 teleports);
- V / T / O frozen and committed; sampler determinism test passes;
- E0 gate passed under the training reward, with the reward form fixed (see progress §3).
