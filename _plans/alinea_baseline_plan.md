# ALINEA / PI-ALINEA baseline plan (paper baseline #1)

## Motivation

The M7 benchmark compares the learned policy only against constant-u
policies. For a paper, the standard local ramp-metering baselines are
**ALINEA** (Papageorgiou et al., TRR 1320, 1991) and **PI-ALINEA**
(Wang, Kosmatopoulos, Papageorgiou & Papamichail, IEEE T-ITS 2014).
Without them, "RL beats baselines" is not a defensible claim.

## Control laws (density-based, per 30 s control step)

```
ALINEA:     r(k) = r(k-1) + K_I * (rho_set - rho(k))
PI-ALINEA:  r(k) = r(k-1) - K_P * (rho(k) - rho(k-1)) + K_I * (rho_set - rho(k))
u(k) = clip(r(k) / ramp_discharge_vph, u_min, u_max)      # u = green fraction of 1600 vph
r(k-1) <- u(k) * ramp_discharge_vph                        # anti-windup: store the clamped rate
```

- `rho(k)`: physical density (veh/km) at one mainline detector downstream
  of the merge (merge nose 1300 m, accel lane to 1400 m) — detector index
  13 (1400 m) or 14 (1500 m), de-normalized from the env observation with
  the config's `density_mean` / `density_std`.
- `rho_set`: critical-density set-point (max-throughput operating point).
- Gains in veh/h per (veh/km) per step. Classic field value K_R = 70
  veh/h/% occupancy corresponds to roughly K_I = 35 veh/h per veh/km at
  5 m vehicles; tuned here by sweep. Wang et al. 2014 report K_P = 4,
  K_I = 100 (km*lane/h) robust in their setting; K_P tuned here too.
- Optional queue override (ALINEA/Q lower bound, Smaragdis &
  Papageorgiou 2003): when the ramp queue w exceeds w_max, release at
  least r_w = d_ramp + (w - w_max) * 3600 / dt_ctrl.
- Controllers are stateful; reset to `u_init` (default 0.5) per episode.

## Implementation

- `src/rl/baseline_controllers.py`: `PIALINEAController` (ALINEA =
  K_P 0), spec parser `alinea:ki=35,rho=30,det=14` /
  `pialinea:kp=4,ki=35,rho=30,det=14` (+ optional `u0,umin,umax,cap,qmax`).
  Reads geometry/normalisation from the eval config's env block + sumo
  yaml; no SUMO import (unit-testable).
- `scripts/eval_sumo_baselines.py::_make_action_callback` accepts the
  specs, so `eval_policy_grid_sumo.evaluate_policies` and every existing
  sweep/analysis script work with controller baselines unchanged.
- `tests/test_baseline_controllers.py`: numpy-only unit tests (integral
  sign, anti-windup, P-damping, reset, de-normalisation, queue override,
  spec parsing).

## Tasks

1. **Implement + unit tests + 1-episode SUMO smoke.**
2. **Tune ALINEA** on 6 tuning cells ({1500, 1800, 2000} x {400, 800}),
   seed 0, `speed_dev 0.03`: K_I in {20, 35, 70} x rho_set in {25, 30, 35}
   x det in {13, 14} (36 candidates, 216 episodes, background).
3. **Tune PI-ALINEA**: winner's (K_I, rho_set, det) with K_P in
   {1, 4, 10} (+ neighbourhood re-check if the winner is on a sweep edge).
4. **Benchmark** best ALINEA + best PI-ALINEA on the published grid
   (18 cells x seeds 0/1/2) next to run 7 @ 72k and u = 0.25; then all
   four on held-out seeds 100/101/102 (fixes the selection/test leakage
   for the paper table).
5. **Docs**: progress file `_progress/alinea_baseline_progress.md`,
   README baselines section, commit + push.

## Acceptance

- Unit tests pass; controllers run in SumoEnv with no code change to the
  grid evaluator beyond the callback hook.
- Tuned baselines reported with the same metrics as run 7 (grid mean,
  worst episode, breakdowns, per-cell table) on both seed sets.
- Verdict recorded either way (RL better / tied / worse) — this is a
  baseline, not a result to engineer.

## Outcome (2026-08-31) — COMPLETE

All acceptance criteria met; see `_progress/alinea_baseline_progress.md`.
Final specs: `alinea:ki=15,rho=37,det=12`,
`pialinea:kp=4,ki=15,rho=37,det=12` (detector = merge nose 1300 m).
Benchmark (18 cells x 3 seeds, both published 0/1/2 and held-out
100/101/102): run 7 @ 72k -60.7/-61.3, ALINEA -62.7/-63.5, PI-ALINEA
-62.8/-62.8, u=0.25 -85.6/-84.6; all feedback policies 0 catastrophic
episodes. Verdict: RL retains a real but modest ~2-point edge over
well-tuned ALINEA, won on high-ramp cells via pre-emptive metering.
