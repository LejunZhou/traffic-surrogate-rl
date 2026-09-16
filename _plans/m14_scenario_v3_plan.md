# M14 plan — scenario v3: a realistic metered ramp (2026-09-14)

## Why
The v2 scenario review found three things a reviewer would question: the ramp speed
limit equals the mainline's 120 km/h, released vehicles are inserted at the ramp start
so the whole ramp is an acceleration run, and the ramp queue has no physical storage
limit (M13 policies ended with 50–70 vehicles waiting; one E0 schedule reached 250).
The user asked for a 60 km/h ramp, a stop line as a real signal would have, and a hard
storage constraint. Nothing else changes: mainline, IDM, detectors, density estimator,
demand family, reward.

## Decisions (user, 2026-09-14)
- Queue limit as a **hard cap on the virtual queue** enforced by the actuator (ALINEA/Q
  rule, `u = max(u_policy, u_min)`), identical in SumoEnv and SurrogateVecEnv; not a
  physically simulated signal (that stays a possible later milestone).
- **Ramp 300 m**, stop line 100 m before the merge nose → storage (300 − 100) / 7 m = 28
  vehicles.
- **Implement and test only**; no round-0 / study re-run yet.

## Changes
- `configs/sumo/scenario_v3.yaml`: ramp_length 300, ramp speed 16.67 m/s, new demand keys
  `ramp_stopline_offset_m: 100`, `ramp_queue_max_veh: 28`, network dir `data/raw/network_v3`.
- `configs/rl/env_v3.yaml`: overlay for either env (sumo_config v3, density stats from
  `data/plant_v3/round0/metadata.json`, the two ramp keys, network dir).
- `configs/experiments/round0_v3.yaml`: round-0 mixture for v3 per the M13 reading
  (store_flush 0.25, feedforward 0.15, alinea_wide 0.25, alinea_dither 0.15, random 0.10,
  constant 0.10; 288 rollouts, half storage-mandatory); store `data/plant_v3/round0`.
- `src/sumo_env/ramp_queue.py`: `queue_override_rate(Q, r, Q_max, D, dt)` = smallest u
  that keeps Q_k ≤ Q_max (1.0 when arrivals exceed the discharge capacity).
- `src/rl/sumo_env_wrapper.py`: reads the two keys (env config overrides the scenario
  file); applies the override in `step` before the interval; inserts released vehicles at
  `ramp length − offset` from standstill; info keys `u_requested`, `u_override`,
  `queue_max_veh`, `queue_cap_exceeded`, per-episode counters.
- `src/rl/surrogate_vec_env.py`: same override before `analytic_queue_step`; reads the
  cap from the env config or the scenario file; same info keys.
- `src/sumo_env/rollout.py`: `action` = applied rate, new `action_requested`, metric
  `queue_override_frac`.
- Tests: `tests/test_ramp_queue.py` (override keeps the queue at the cap), new
  `tests/test_scenario_v3.py` (SUMO: ramp speed, stop-line position, cap binds with u = 0).

## Verification
1. Unit + SUMO tests above, plus the surrogate/plant suites (parity unchanged when no cap).
2. E0 characterisation on v3 (`run_scenario_characterisation.py --config
   configs/experiments/round0_v3.yaml --study m14_e0`, 12 profiles, stored in
   `data/plant_v3/round0` as future round-0 data): merge capacity, insertion check,
   storage-mandatory gate, and the fraction of steps where the cap binds.
3. Compare with v2's E0 (`_progress/m8_e0_characterisation.json`): breakdown-onset rate
   vs u, best constant, gate count.

## Not done (needs the user's go)
Round-0 dataset, ensemble, ALINEA tuning, direct PPO and the aggregation study on v3
(≈ 5 h at demo scale via `run_study.sh`-style driver with the v3 overlay).

## Deferred (user, 2026-09-14)
Desired-speed spread stays at 3 % (`speed_dev 0.03`). A 10 % spread (SUMO default) is
more realistic but pulls the merge capacity from ≈ 2500 to ≈ 2350 veh/h (M7 §speedDev
sweep), so adopting it means re-measuring capacity with E0 and rescaling the demand
family's ceilings, the storage-mandatory threshold, the feedforward capacity range and
the reward's flow cap together. Revisit before the paper-scale runs.

## v3b (user decisions, 2026-09-14, supersedes v3 for the study)
- Simulated ramp = acceleration segment only: 200 m at 120 km/h (no artificial limit on
  the acceleration run; the posted ramp limit belongs to the unsimulated approach; 300 m
  total acceleration distance with the 100 m lane, US design range).
- Meter signal at the start of the simulated ramp; released vehicles inserted at position 0
  from standstill, one vehicle per green; discharge D = 900 veh/h.
- Storage is the virtual queue, **uncapped**: its waiting time is already in the TTS reward.
  A blocked ramp (jam reaching the stop bar) is handled by the pending-vehicle bookkeeping
  and happens only when the release rate exceeds the jammed merge's share (≈ 860 veh/h),
  i.e. not on the training family (ramp ≤ 800 veh/h).
- Files: `configs/sumo/scenario_v3b.yaml`, `configs/rl/env_v3b.yaml`,
  `configs/experiments/round0_v3b.yaml`, `tests/test_scenario_v3b.py`,
  figure `_progress/figures/m14_ramp_layout_options.png` (`scripts/plot_scenario_layout.py`).
- Merge-station density = through lane only (`detectors.merge_station_lanes: mainline`).
- Demand family v2 (`configs/profiles/family_v2.yaml`, frozen sets `configs/profiles/v2/`):
  step-down and ramp surge over by minute 45, ramp surge ≤ 600 veh/h; v1 untouched.
- Terminal queue weight stays 0 (user). Rollouts record `pending_ramp` + `pending_ramp_max`.
- E0 characterisation on v3b run 2026-09-14 (progress §9, §11); single-constant gate added (§12).
- Study driver `scripts/run_v3b_study.sh` (scenario overlay + profile sets via environment
  variables, smoke mode); the v3b study itself runs on the user's Windows machine.
- Ramp entry angle 10° (user, 2026-09-15): at the 30° default netconvert caps the merge
  junction's internal link at 9.18 m/s, so ramp vehicles braked to 33 km/h at the nose on
  every scenario so far (progress §13). `network.ramp_entry_angle_deg` (builder default 30
  keeps v2/v3 nets bit-identical); v3b sets 10 → 97 km/h at the nose. The v3b E0 report
  (§11) predates this and must be re-run before any v3b study.
