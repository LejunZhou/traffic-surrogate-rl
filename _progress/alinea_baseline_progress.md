# ALINEA / PI-ALINEA baseline — progress

Plan: `_plans/alinea_baseline_plan.md`. Goal: tuned classical feedback
baselines (ALINEA, PI-ALINEA) on the M7 grid benchmark, next to
run 7 @ 72k and the constant policies, on both the published seeds
(0/1/2) and held-out seeds (100/101/102).

## A.1 — Implementation (2026-08-31)

- `src/rl/baseline_controllers.py`: `PIALINEAController` (ALINEA = kp 0),
  density-based law per 30 s step, anti-windup via clamped-rate storage,
  optional ALINEA/Q queue-override lower bound, spec parser
  (`alinea:ki=35,rho=30,det=14`, `pialinea:kp=4,...`; extras
  `u0/umin/umax/cap/qmax`). Constants (n_detectors, dt_ctrl, discharge
  1600, density_mean/std, queue_scale, ramp levels) resolved from the
  eval config's env block + its sumo yaml; spec params win.
- `scripts/eval_sumo_baselines.py::_make_action_callback` accepts the
  specs and calls `.reset()` per episode, so
  `eval_policy_grid_sumo.evaluate_policies` and the analysis scripts work
  with controllers unchanged.
- `tests/test_baseline_controllers.py`: 8 numpy-only tests (integral
  sign/step size, anti-windup, P-damping, reset, z-score
  de-normalisation, queue override incl. ramp-demand-from-obs, spec
  parsing) — all pass.
- SUMO smoke (run-7 env config, seed 0): both controllers roll through a
  full episode; labels and per-episode stats flow into the standard
  summary.

## A.2 — Detector placement finding (2026-08-31)

First smoke put the measurement at det 14 (1500 m, downstream of the
accel lane) with rho_set 30: the episode jammed (mean density 134) while
the controller held u ~= 1. Cause: the merge bottleneck queues *upstream*
of ~1400 m; downstream of an active bottleneck the density stays near
critical (capacity-drop outflow ~1740 vph), so a downstream detector
never sees the breakdown. Classic ALINEA wants the detector where the
congestion caused by over-feeding appears — here that is the merge area
itself / immediately upstream. Detector grid for tuning chosen from the
per-detector jam profile (below).

(sections A.3 tuning, A.4 benchmark to follow)

## A.3 — Stage-1 ALINEA tuning (2026-08-31, `_progress/alinea_tune_stage1.jsonl`)

36 candidates (det {11,12,13} x rho_set {25,30,35,40} x K_I {20,35,70})
x 6 tuning cells ({1500,1800,2000} x {400,800}), seed 0, speed_dev 0.03,
run-7 env config. References on the same 6 cells / seed:
run 7 @ 72k mean -65.8 (worst -102.7), run 5 best -71.4, u=0.25 -79.4.

| spec | mean | worst | jams | u |
|---|---|---|---|---|
| **alinea:ki=20,rho=35,det=12 (winner)** | **-73.8** | **-102.7** | 2 | 0.53 |
| alinea:ki=20,rho=35,det=13 | -78.1 | -112.6 | 4 | 0.52 |
| alinea:ki=35,rho=35,det=12 | -80.3 | -106.4 | 3 | 0.51 |
| alinea:ki=20,rho=40,det=12 | -86.6 | -143.3 | 2 | 0.64 |
| worst 3 (rho 25, det 12/13) | -145 | -200 | 2 | 0.14 |

Findings:
- **Detector 12 (1300 m, merge nose) is the right measurement point.**
  det 14+ never sees the jam (A.2); det 11 reacts too late (means
  -101..-124); det 13 is close behind det 12.
- **rho_set 35 is the sweet spot**: 30 over-closes in free flow (u~0.34,
  throttles healthy traffic since the merge-nose free-flow density is
  ~29.5), 25 is far worse (u~0.14); 40 tolerates incipient jams too long.
- **Low gain wins** (K_I 20 beats 35 beats 70 at every det/rho) - the
  2000+800 knife edge punishes aggressive corrections. K_I 20 is the grid
  edge, so stage 2 probes K_I 10/15 (+ rho 33/37) and adds the PI term
  (kp {1,4,10,20}).
- Already beats u=0.25 (-79.4) and is within ~8 of run 7 (-65.8) on the
  tuning cells; behind run 7 on worst-case tail so far only via jams=2
  (same cells run 7 also jams transiently).

## A.3b — Stage-2 tuning (2026-08-31, `_progress/alinea_tune_stage2.jsonl`, `..._stage2b.jsonl`)

12 + 2 refinement candidates around the stage-1 winner (6 cells, seed 0):

| spec | mean | worst | jams | u |
|---|---|---|---|---|
| **alinea:ki=15,rho=37,det=12 (final ALINEA)** | **-69.2** | -102.9 | 3 | 0.59 |
| alinea:ki=10,rho=37,det=12 | -70.4 | -102.7 | 4 | 0.56 |
| **pialinea:kp=4,ki=15,rho=37,det=12 (final PI)** | **-69.0** | -102.7 | 3 | 0.53 |
| pialinea:kp=1,ki=20,rho=35,det=12 | -74.7 | -102.7 | 3 | 0.52 |
| (stage-1 winner ki=20,rho=35) | -73.8 | -102.7 | 2 | 0.53 |

- Lower gain + slightly higher set-point (K_I 15, rho_set 37) buys ~4.6
  return over the stage-1 winner; rho_set 33 is clearly worse (-85), so
  the optimum sits in the 35-38 band at the merge nose.
- **The P-term is a statistical tie with pure ALINEA** at every (K_I,
  rho_set) tried (kp 1..20 within ~2 of the matching ALINEA). Expected:
  PI-ALINEA's advantage is for *distant downstream* bottlenecks, and this
  scenario's bottleneck is the merge itself.
- Tuning-cell caveat: 6 cells x 1 seed => differences of ~2-3 return are
  selection noise; the benchmark (A.4) decides on 18 cells x 3 seeds.

## A.4 — Final benchmark (launched 2026-08-31)

Policies: `alinea:ki=15,rho=37,det=12`, `pialinea:kp=4,ki=15,rho=37,det=12`,
run 7 @ 72k (`best_model_multiseed.zip`), u=0.25.
- Published seeds 0/1/2 (controllers only; run 7 / constants already
  measured there) -> `_progress/alinea_benchmark_seeds012.jsonl`.
- Held-out seeds 100/101/102, all four policies (fixes the
  selection/test-seed leakage for the paper table) ->
  `_progress/benchmark_heldout_seeds100.jsonl`.

### A.4a — Published seeds 0/1/2 (complete, `_progress/alinea_benchmark_seeds012.jsonl`)

18 cells x 3 seeds (54 episodes):

| policy | mean | p10 | worst | eps < -150 | jams (rho_max>60) | u | queue_final | outflow |
|---|---|---|---|---|---|---|---|---|
| run 7 @ 72k | **-60.7** | -85.7 | -102.9 | 0 | 11 | 0.32 | 129 | 2179 |
| ALINEA ki15 rho37 d12 | -62.7 | -86.1 | -102.9 | 0 | 27 | 0.58 | 131 | 2176 |
| PI-ALINEA kp4 ki15 rho37 d12 | -62.8 | -83.8 | -102.9 | 0 | 27 | 0.58 | 132 | 2176 |
| run 5 best | -70.1 | -95.1 | -151.2 | 1 | 3 | 0.29 | 163 | 2141 |
| u = 0.25 | -85.6 | -137.7 | -189.5 | 2 | 3 | 0.25 | 199 | 2087 |

- **Well-tuned ALINEA is a strong baseline: within ~2 return of run 7**
  on grid mean, identical worst case (the 1500+400 floor), 0 catastrophic
  episodes. PI-ALINEA is a tie with ALINEA (as in tuning).
- **Where RL wins**: the high-ramp cells - 1500+800 (-13.7), 1600+800
  (-17.6), 1900+800 (-9.9), 1700+800 (-9.7): run 7 uses its queue/demand
  observations to meter earlier and avoid the merge turbulence ALINEA
  only reacts to after it registers at the detector. ALINEA takes 27/54
  transient jams vs run 7's 11 (all recovered in both).
- **Where ALINEA wins**: 2000+400 (+10.1) and 1900+400 (+5.3) - exactly
  the cells noted in M7 SS7.15 as run 7's remaining gap vs lucky policies;
  ALINEA's steady-state regulation extracts them without luck.
- Same throughput/queue operating point (outflow ~2176 vs 2179; final
  queue ~131 both) - the ~2-point margin comes from the transient-jam
  frequency, not a different trade-off.

### A.4b — Held-out seeds 100/101/102 (`_progress/benchmark_heldout_seeds100.jsonl`)

The leakage-free table: none of these seeds was used for tuning
(controllers: seed 0) or checkpoint selection (run 7: seeds 0/1/2).

| policy | mean | p10 | worst | eps < -150 | jams | u | queue_final | outflow |
|---|---|---|---|---|---|---|---|---|
| run 7 @ 72k | **-61.3** | -86.2 | -102.9 | 0 | 11 | 0.32 | 131 | 2177 |
| PI-ALINEA kp4 ki15 rho37 d12 | -62.8 | -86.1 | -102.9 | 0 | 26 | 0.58 | 131 | 2177 |
| ALINEA ki15 rho37 d12 | -63.5 | -84.7 | -102.9 | 0 | 29 | 0.58 | 134 | 2174 |
| u = 0.25 | -84.6 | -129.4 | -170.9 | 1 | 3 | 0.25 | 199 | 2091 |

- **Everything replicates on unseen seeds.** run 7 moves -60.7 -> -61.3
  (selection was not luck), controllers keep the ~1.5-2.2 gap, all three
  feedback policies have 0 catastrophic episodes and the identical
  -102.9 pass-through floor as worst case.
- Per-cell structure identical to A.4a: RL wins the high-ramp cells
  (1500+800 -18.9, 1600+800 -20.5, 1700+800 -12.9), ALINEA wins
  1900/2000+400 (+5.4/+6.2) and, on these seeds, 2000+800 (+11.7 - the
  knife-edge cell swings a few points either way per seed set).
- PI-ALINEA edges ALINEA (-62.8 vs -63.5): within seed noise; report
  both, claim no PI advantage (bottleneck is the merge, not distant).

## Verdict (A.5)

Tuned classical feedback control is a **strong, honest baseline** on this
scenario: ~2 return behind the RL policy on both seed sets, same worst
case, same throughput/queue operating point. The RL margin is real but
modest and has a mechanism: run 7 observes queue + demand and meters
pre-emptively (11 vs ~27 transient jams), while ALINEA reacts only after
density registers at the merge-nose detector. For the paper: RL vs
ALINEA is the fair headline comparison; u=0.25 remains only to anchor
the do-nothing-clever floor. Practitioner findings worth a subsection:
(1) textbook downstream detector placement fails in this geometry (blind
to the upstream-queuing merge jam -> drives into gridlock at u~1);
(2) low gain + set-point ~25% above free-flow merge density wins;
(3) PI extension is a tie without a distant downstream bottleneck.
