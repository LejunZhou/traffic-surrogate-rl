# M14 progress — scenario v3: realistic metered ramp (2026-09-14)

Plan: `_plans/m14_scenario_v3_plan.md`. Scope agreed with the user: implement and test
the scenario; no round-0 / study re-run yet.

## 1. Implementation (done)
- `configs/sumo/scenario_v3.yaml` (ramp 300 m at 60 km/h, stop line 100 m before the
  nose, storage cap 28 veh), `configs/rl/env_v3.yaml` overlay, `configs/experiments/round0_v3.yaml`.
- `queue_override_rate` in `src/sumo_env/ramp_queue.py`; SumoEnv and SurrogateVecEnv apply
  `u = max(u_policy, u_min)` every control step, so the cap is identical in both
  environments and the DeepONet interface (metered inflow → fields) is unchanged.
- Released vehicles are inserted at `ramp length − 100 m` from standstill (SUMO
  `departPos` numeric, `departSpeed 0`); v2 configs keep `departPos free` at the ramp start.
- Rollouts record `action` (applied) and `action_requested`; metric `queue_override_frac`.
- Tests: `tests/test_ramp_queue.py::test_queue_override_rate_keeps_queue_at_cap`,
  `tests/test_scenario_v3.py` (SUMO: 60 km/h ramp lane, stop line at lane length − 100,
  cap binds with the meter closed and the queue stays ≤ 28 + 1); surrogate / plant /
  parity suites unchanged (16 passed, 7 legacy skips).

## 2. E0 characterisation on v3 (`_progress/m14_e0_v3_characterisation.json`, ledger `m14_e0`)
404 rollouts (20 insertion checks, 12 profiles × 11 constant rates, 12 × 21 dynamic
schedules), 11.5 min on 8 workers, stored in `data/plant_v3/round0` as future round-0 data.
Figure: `_progress/figures/m14_e0_v2_vs_v3.png`.

**Gate (storage-mandatory profiles = peak total demand > 2500 veh/h, i.e. above the merge
capacity; a profile passes when the best dynamic schedule beats the best constant rate by
≥ 15 veh h, the script's `GATE_MARGIN`): v3 4 / 7 (script verdict FAILED, needs ≥ 2/3),
v2 5 / 7 (PASSED).** Two v3 profiles sit just under the margin (profile 1 at 14.8, profile 8
at 13.1 veh h); with a 3 veh h margin the count would be 6 / 7, which is what the first draft
of this section reported. Corrected 2026-09-14. Insertion check at u = 0.65 unchanged (served fraction 0.90,
breakdown rate 0.60 on the 20 insertion profiles).

| profile | peak total | v2 best constant u / return | v2 best schedule | v2 margin | v3 best constant u / return | v3 best schedule | v3 margin |
|---|---|---|---|---|---|---|---|
| 0 | 2612 | 1.0 / -40.5 | -53.2 | -12.7 | 0.2 / -74.1 | -53.0 | +21.1 |
| 1 | 2786 | 0.2 / -167.4 | -109.3 | +58.1 | 0.5 / -171.3 | -156.5 | +14.8 |
| 2 | 2508 | 1.0 / -34.4 | -34.5 | -0.1 | 0.6 / -34.1 | -34.2 | -0.1 |
| 4 | 2677 | 0.3 / -192.9 | -120.0 | +72.9 | 1.0 / -208.5 | -177.4 | +31.1 |
| 5 | 2886 | 0.2 / -99.9 | -76.5 | +23.4 | 0.3 / -139.7 | -123.2 | +16.6 |
| 7 | 2542 | 0.2 / -112.6 | -85.9 | +26.6 | 0.6 / -149.4 | -112.3 | +37.1 |
| 8 | 2635 | 0.3 / -100.8 | -64.0 | +36.8 | 0.3 / -110.2 | -97.1 | +13.1 |

Constant-rate sweep, mean over the 12 profiles (v2 / v3):

| u | return | breakdown rate | end-of-hour ramp queue |
|---|---|---|---|
| 0.0 | -240.5 / -129.8 | 0.00 / 0.50 | 412 / 28 |
| 0.1 | -162.7 / -123.5 | 0.00 / 0.50 | 252 / 28 |
| 0.2 | -92.4 / -110.6 | 0.00 / 0.50 | 106 / 17 |
| 0.3 | -89.5 / -104.6 | 0.42 / 0.50 | 28 / 6 |
| 0.4 | -89.3 / -93.4 | 0.42 / 0.42 | 2 / 2 |
| 0.5 | -97.6 / -96.4 | 0.50 / 0.50 | 0 / 0 |
| 0.6 | -101.5 / -101.8 | 0.50 / 0.50 | 0 / 0 |
| 0.7 | -97.9 / -98.7 | 0.50 / 0.50 | 0 / 0 |
| 0.8 | -101.3 / -100.7 | 0.50 / 0.50 | 0 / 0 |
| 0.9 | -97.8 / -100.7 | 0.50 / 0.50 | 0 / 0 |
| 1.0 | -93.8 / -100.2 | 0.50 / 0.50 | 0 / 0 |

Cap statistics over the 404 v3 rollouts: the override is active in 15 % of all control
steps (up to 96 % for the closed-meter schedules); 46 % of the episodes reach the cap;
the queue never exceeded 29 vehicles (28 + one step of insertion lag).

## 3. Reading: what the realistic ramp changes
- **Merge capacity is unchanged.** For u ≥ 0.4 the returns, breakdown rates and queues of
  v2 and v3 agree to within noise (e.g. u = 0.5: −97.6 vs −96.4, breakdown 0.50 in both).
  The 60 km/h ramp and the stop line 100 m before the nose do not make the merge harder:
  ramp vehicles still reach ≈ 100 km/h by the lane drop.
- **Closing the ramp is no longer a strategy.** In v2, u = 0 to 0.2 parked 100 to 400
  vehicles on the ramp, which the TTS reward priced but nothing forbade. In v3 the cap
  forces the meter open once 28 vehicles wait, so at u = 0 half of the storage-mandatory
  profiles break down anyway and the mean return at u = 0 goes from −240 (queue cost) to
  −130 (breakdown cost). The best constant rates move up accordingly (profile 4: u 0.3 →
  1.0; profile 7: 0.2 → 0.6).
- **The storage lever is weaker, so the optimal returns are worse.** On the
  storage-mandatory profiles the best dynamic schedule loses 20 to 60 veh h against v2
  (profile 4: −120 → −177; profile 1: −109 → −157), because 28 vehicles buffer only 2 to 3
  minutes of a 600 to 800 veh/h ramp surge. The margins of a dynamic schedule over the best
  constant shrink (73 → 31, 58 → 15 veh h) and stay positive on 6 of 7 profiles, but only 4
  of 7 clear the 15 veh h gate margin (v2: 5 of 7), so the case that metering matters is
  weaker on v3 than on v2 and just below the script's 2/3 criterion; the gain now comes from *timing* the limited
  storage against the mainline peak rather than from holding the ramp.
- **For the learning problem** this removes the degenerate "store everything" solution the
  M13 policies drifted toward (50 to 70 vehicles parked at the end of the hour) and makes
  the policy's job harder in the right way. The E0 reference schedules and the ALINEA/Q
  rule in the behaviour controllers already cover the capped regime, and the cap binding
  in 15 % of steps means the round-0 data will contain it.
- Everything downstream (round-0 store, ensemble, ALINEA tuning, direct PPO, aggregation)
  still has to be regenerated on v3; the 404 E0 rollouts are the first part of that store.

## 4. Proposed v3b layout (2026-09-14, discussion, not implemented)
The user found the mid-ramp stop bar of v3 strange. Reading of the ramp as two segments
(storage upstream of the bar, acceleration downstream) shows that the 200 m of v3 ramp
upstream of the bar is never driven: it only gives the cap a physical number. Proposal:
simulate only the acceleration segment (ramp 100 m at 60 km/h), place the meter signal at
its start, insert released vehicles there from standstill (one vehicle per green), and keep
the storage entirely in the virtual queue with an explicit capacity (Q_max = 28 veh ≈ 200 m
of storage road). Config-only change (ramp_length 100, stopline offset 100 → insertion at
position 0). Figure: `_progress/figures/m14_ramp_layout_options.png` (panel a = v3 as
implemented, panel b = v3b), script `scripts/plot_scenario_layout.py`.
Kept virtual rather than a physical SUMO queue: a physical queue turns the storage cap into
SUMO insertion failure (pending backlog, cf. M7), makes the SUMO and surrogate queue
dynamics differ (extra transfer gap not attributable to the plant model), and needs
per-vehicle TraCI stops plus a queue detector. Open decision flagged: D = 1600 veh/h is a
queue saturation flow; a literal one-car-per-green US meter saturates near 900–1000 veh/h.

## 5. v3b files and the blocked-ramp check (2026-09-14)
User decisions: v3b = simulated ramp is the acceleration segment only, 120 km/h (no
artificial cap on the acceleration run), meter at the ramp start, virtual storage cap 28,
meter discharge D lowered to 900 veh/h (one vehicle per green). Files:
`configs/sumo/scenario_v3b.yaml`, `configs/rl/env_v3b.yaml`,
`configs/experiments/round0_v3b.yaml`, `tests/test_scenario_v3b.py` (tests not yet run).
Ramp length 100 m for now; 200 m recommended (300 m total acceleration distance).

**Can a released vehicle fail to enter the ramp?** Yes, when the jam at the lane drop backs
up over the acceleration lane and the 100 m ramp to the stop bar. SumoEnv handles it
correctly (released-but-not-inserted vehicles stay in the virtual queue, are charged in
TTS, block further releases, and the DeepONet input counts only vehicles that entered).
One-hour constant-demand test at u = 1, mainline jammed the whole hour (exit flow
≈ 1700 veh/h, zipper share to the ramp ≈ 860 veh/h):

| mainline + ramp (veh/h) | D = 1600: blocked at the meter after 1 h | D = 900 | max veh on the 100 m ramp |
|---|---|---|---|
| 2300 + 800 | 0 | 0 | 3 |
| 2600 + 800 | – | 0 | 3 |
| 2600 + 1000 | 135 | 34 | 11 |
| 2000 + 1200 | 332 | 33 | 11 |

Blocking occurs only when the release rate exceeds the jammed merge's share (≈ 860 veh/h);
with D = 900 the excess is ≈ 35 veh/h and starts ≈ 15 min into the jam once the
acceleration lane has filled. The training family's ramp demand tops out at 800 veh/h, so
on family demand the meter never blocks (v3 E0: max queue 29 over 404 rollouts). Caveats:
the 28-veh cap is unenforceable while the ramp is blocked (spillback, counted in TTS,
`queue_cap_exceeded` counts it); SurrogateVecEnv cannot reproduce blocking. To do before
the v3b data run: store the max blocked count as a rollout metric.

**Cap dropped (user, 2026-09-14).** Since the virtual queue is charged in the TTS reward
(Q_k·Δt every step), the storage limit is priced rather than constrained: v3b runs with
`ramp_queue_max_veh: null` (no forced-open rule; a number re-enables it). Only leak: vehicles
still queued at the end of the hour are not charged for their remaining wait
(`terminal_queue_weight` 0). v3b as it stands: ramp 100 m at 120 km/h, meter at the ramp
start, D = 900 veh/h, uncapped virtual queue. `tests/test_scenario_v3b.py` (geometry,
insertion at position 0, queue accumulates with no override) + `tests/test_ramp_queue.py`:
7 passed. Layout figure updated (panel b).

## 6. Merge-station density = through lane only (user, 2026-09-14)
`density_from_loops` gained `merge_station_lanes` ("mean" = v2/v3 lane average, "mainline" =
drop lane 0, the ramp/acceleration lane, at the two-loop station 12 at 1300 m); read from
`detectors.merge_station_lanes` in the scenario file (default "mean", so v2/v3 unchanged),
recorded in every rollout's metadata by `parallel_rollouts`. v3b sets "mainline".
Reason: every station then measures the through-lane density (one consistent field for the
DeepONet), ramp platoons stop leaking into station 12, and breakdown detection is unchanged
because the through lane jams whenever the merge fails. The TTS is unaffected: it is a
conservation count (offered − served, checked on the 404 v3 E0 rollouts: charged − conservation
= +0.09 veh h mean, +0.75 max, from the max(·,0) clip of the backlog in 5 % of steps), so
acceleration-lane vehicles move from the detector term into the backlog term.
Tests: `tests/test_scenario_v3b.py` (+ unit test of both rules) with v3, ramp-queue,
overrides, departures, reward suites: 33 passed. Final layout figure:
`_progress/figures/m14_v3b_road_layout.png` (`scripts/plot_scenario_layout.py --final`).

**v3b as configured (`configs/sumo/scenario_v3b.yaml`):** mainline 2000 m, 1 lane, 120 km/h,
IDM τ 1.0 s, speedDev 0.03; ramp 100 m at 120 km/h (200 m still under consideration), meter at
the ramp start, insertion from standstill, D = 900 veh/h; virtual queue, no cap; 19 stations,
occupancy density, merge station through-lane only; TTS reward, terminal queue weight still 0
(to be set, §5 leak).

## 7. Demand family v2 (user decisions, 2026-09-14)
`configs/profiles/family_v2.yaml` (version 2, new seed stream) = family v1 with three
constraints, implemented as optional keys in `ProfileFamily` (absent keys → v1 behaviour,
v1 sampler output unchanged): `step.step_down_end_max_min: 45` (the 10-min fall of a
step-down completes by 45; v1 allowed 55), `surge.end_max_min: 45` (start + length ≤ 45; v1
allowed 60), `surge.r_max: 600` and `ood.early_surge.r_max: 600` (ramp surge never above
2/3 of the 900 veh/h meter; v1: 1000). The constant ramp family keeps its [300, 800] range
(below the meter). Frozen sets redrawn into `configs/profiles/v2/{val,test,ood}.json`
(18 / 30 / 12; V 7 and T 13 storage-mandatory). v3b configs point at the v2 family and sets
(`round0_v3b.yaml`, `env_v3b.yaml`: profiles + eval_profiles). Figure
`_progress/figures/m14_family_v2.png`. Test `test_family_v2_constraints` (7 passed).

3000 training draws: max mainline peak/plateau end 45.0 min, max surge end 45.0 min, ramp
surge ≤ 600 (median max ramp 588); storage-mandatory rate 44 % (v1 49 %); surge overlaps the
mainline peak in 65 %; profiles with an unavoidable end queue 3.0 % (all constant-overload
profiles: const mainline + const ramp above capacity all hour, no peak to wait out).

## 8. v3b closed out for the E0 run (user decisions, 2026-09-14)
1. Ramp 200 m at 120 km/h (stop bar = ramp start, offset 200): 300 m of acceleration distance
   to the lane drop, ≈ 100 km/h at the nose. `scenario_v3b.yaml`, `env_v3b.yaml`, test bounds.
2. `terminal_queue_weight` stays 0 (user).
3. Rollouts now store `pending_ramp` (K,) = released-but-not-inserted vehicles per step and
   the metrics `pending_ramp_max`, `pending_ramp_steps` (`src/sumo_env/rollout.py`).
4. E0 characterisation on v3b launched: `run_scenario_characterisation.py --config
   configs/experiments/round0_v3b.yaml --out _progress/m14_e0_v3b_characterisation.json
   --study m14_e0_v3b` (family v2 E0 profiles, store `data/plant_v3b/round0`, log
   `runs/logs/m14_e0_v3b.log`).
5. The v3b study (round 0, ensemble, ALINEA, PPO, aggregation) waits for the user's go.
Tests after 1–3: `test_scenario_v3b.py`, `test_ramp_departures.py`, `test_plant_surrogate.py`:
18 passed. Layout figures regenerated for 200 m.

## 9. E0 characterisation on v3b (2026-09-14, `_progress/m14_e0_v3b_characterisation.json`, store `data/plant_v3b/round0`)
404 rollouts, 11.4 min, 12 family-v2 E0 profiles (peak totals 1981–2820, 4 storage-mandatory).
Figure `_progress/figures/m14_e0_v3b_capacity.png`.

**Gate: 0 / 4 storage-mandatory (0 / 11 peaked) with margin ≥ 15 veh h — FAILED** (v3 4/7, v2 5/7).
Ramp blocking: `pending_ramp_max` = 0 in all 404 rollouts (never blocks on family v2).
Insertion check (u = 0.65 = 585 veh/h with D = 900, no longer pass-through): served 0.936,
breakdown 0.35, mean end queue 28.

Constant sweep (return vs u·D): sharp optimum at 360–450 veh/h (u 0.4–0.5) on the loaded
profiles, breakdown from 540 veh/h up; mean return over the 12 profiles −53 at u = 0.5 vs −64
at u = 1 (v3: −93 best vs −101 pass-through, different profiles). Per-profile best constant
varies 0.4–1.0; over the 12 profiles: per-profile best constants −490, single constant u = 0.5
−634, pass-through −763.

Why the dynamic schedules lose: the E0 schedule grid is expressed in u (u_pre 0.6, flush at
u = 1, feedforward u_min 0.05), designed for D = 1600. With D = 900 a "flush" is only 900 veh/h
(net drain ≤ 900 − r ≈ 300–700 veh/h) and releasing 900 veh/h into a 1700–2000 veh/h mainline
breaks the merge, while a constant 450 veh/h already stores 150 veh/h of a 600 veh/h surge
and drains it afterwards — the constant *is* a mild store-and-release. Pilot of eight
meter-aware schedules (u_pre 1.0, u_low 0.35/0.45, lag 0/5; feedforward C 2450/2550,
u_min 0.3/0.4; 56 rollouts, scratchpad only) on the seven hardest profiles: beats the best
constant only on profile 0 (+29.6); elsewhere −0 to −34 (the u = 1 phases break the merge).

Reading: on v3b + family v2 (surge ≤ 600, D = 900) metering matters a lot (u = 1 breaks
down on 6/12 profiles; 273 veh h over 12 profiles vs the best constants) but the *anticipative*
lever (store early, flush fast) that made RL beat ALINEA on v2 is mostly gone: the meter
cannot flush faster than 900 veh/h and the surge amplitude above base is ≤ 400 veh/h. The
remaining problem is choosing the right level per profile (0.4–1.0), a feedback-control
problem where ALINEA is a strong baseline. Options put to the user: (a) accept and redefine the
E0 gate against a single constant / ALINEA; (b) D = 1200 (two-car-per-green) and/or surge cap
800 to restore the storage lever; (c) keep the schedule grid but express it in veh/h. Also to
fix in the script regardless: the E0 grid should be meter-aware (rates in veh/h, u = rate/D).

## 10. Storage lever restored: D = 1200, surge cap 800 (user, 2026-09-14)
`scenario_v3b.yaml` ramp_discharge_vph 1200 (two vehicles per green); `family_v2.yaml` surge
r_max 800 (training and OOD early surge); frozen sets `configs/profiles/v2/` redrawn (T 14 / V 9
storage-mandatory). The D = 900 / cap-600 E0 report is kept as
`_progress/m14_e0_v3b_d900_cap600_characterisation.json`; its store and ledger were deleted
(`data/plant_v3b/round0` regenerated). E0 script made meter-aware: schedule rates in veh/h
(u_pre 960, u_low 0/240/480, flush 1600, u_low2 160/320/480, u_high2 800/1120, ff u_min 80,
insertion 1040 veh/h), u = rate / D, identical fractions to the original grid at D = 1600;
D and the resolved grid are written into the report. Tests: profiles + v3b, 10 passed.
Figures regenerated (`m14_family_v2.png`, layouts with D = 1200).

## 11. E0 on v3b with D = 1200, surge cap 800 (2026-09-14, `_progress/m14_e0_v3b_characterisation.json`)
404 rollouts, 25 min (3.8 s/ep, machine busy), 12 family-v2 E0 profiles (peaks 1981–2834,
8 storage-mandatory; the E0 draws changed with the cap). Store `data/plant_v3b/round0`.
Figure `_progress/figures/m14_e0_v3b_capacity.png` (left: return vs constant rate + best
schedule per profile; right: mean curves of v2, v3, v3b-D900, v3b-D1200).

**Gate 2 / 8 storage-mandatory (3 / 11 peaked), margin ≥ 15 veh h — still FAILED** (v2 5/7,
v3 4/7, v3b-D900 0/4). Margins: +30.9, +35.6 (storage-mandatory passes), +28.4 (peak 2462),
+13.5 and +8.2 near, +1.2 / +0.1 / −0.2 / 0 flat, −11.1 / −18.6 on the two step/surge
profiles with a 2200–2300 plateau (every schedule that releases > C − d during the plateau
breaks down; the grid's flush timing (lag 0 / 10 around the plateau end) is wrong for a
plateau whose fall ends at 45).
Sums over the 12 profiles: best schedule −597, per-profile best constant −685 (dynamic +88,
13 %), best single constant −781, pass-through −1008. Constant sweep: optimum at 360–480
veh/h on loaded profiles, breakdown from 600 veh/h (u ≥ 0.5) on 9/12.
Ramp blocking is back: `pending_ramp_max` > 0 in 87/404 rollouts (max 89), *only* in the
store-and-flush schedules that flush a stored queue at 1200 veh/h into a still-jammed merge
(flush 69/108, flush2 18/72); constants (even u = 1, since arrivals ≤ 800), feedforward and
insertion rollouts never block. Insertion check (1040 veh/h): served 0.926, breakdown 0.40.

Reading: raising D and the cap brought back roughly half of the v2 storage lever (three
profiles gain 28–36 veh h from timing; the D = 900 run had none), but the family-v2 timing
rules and the 800 cap keep the lever below v2's (surges to 1000, flush at 1600). The
remaining failures are step-plateau profiles where the hand grid mistimes the flush, not
evidence that no dynamic policy can win there. Round-0 data will contain the blocked-ramp
regime through the flush schedules (SumoEnv handles it; SurrogateVecEnv cannot reproduce it).

## 12. Study driver for v3b and the single-constant gate (2026-09-15)
**Gate line 2.** `run_scenario_characterisation.py` now also reports the margin of the best
schedule over the best *single* constant (the one u maximising the summed return over the
12 profiles; recorded per profile as `single_constant_u/return`, `margin_single`,
`passed_single`, and in `gate_summary`). On v3b the single constant is u = 0.4 (480 veh/h):
4/11 peaked and 3/8 storage-mandatory pass (per-profile constant: 3/11, 2/8) — so the
earlier expectation that v3b "passes clearly" against a single constant was wrong: u = 0.4 is
already the best constant on 6 of the 8 storage-mandatory profiles; the single constant loses
mainly on the light profiles (best u 0.7–1.0), which are not storage-mandatory. Profile 5
(step/surge 2834) flips from −18.6 to +46.2 because its best constant is 0.3.

**Scenario-aware evaluation.** `rl.profile_eval.sumo_env_config` applies the overlays listed
in the environment variable `SCENARIO_OVERLAY` (":"-separated YAML paths) after
ppo_common + env_sumo; `load_set` resolves 'val'/'test'/'ood' under `PROFILE_SETS_DIR`
(default configs/profiles = family v1). `run_aggregation_loop.py` passes the same overlays to
its PPO runs and reward config. `build_arms_manifest.py`: `--r0-studies`, `--no-mpc`.
`SurrogateVecEnv` reads the meter discharge from the scenario file when the env config does
not set it (parity with SumoEnv; `env_v3b.yaml` also sets `ramp_discharge_vph: 1200`).
Unset variables reproduce the v2 behaviour exactly.

**Driver `scripts/run_v3b_study.sh`** (SCENARIO=v3b): E0 (if the report is missing) →
round-0 dataset (`round0_v3b.yaml`, on top of the E0 rollouts) → 5-member ensemble → gate
(val/test) → aggregation loop (patience 2, fine-tune 20 epochs) ‖ direct SUMO PPO (200, 700 EE)
→ ALINEA tuning + constants on the v2 V set → checkpoint selection → manifest (no MPC) → final
evaluation on the v2 T and O sets → figures in `_progress/figures/m14_v3b`. `SMOKE=1` runs
the whole chain on isolated paths (40 rollouts, 2×2-epoch ensemble, 1 round of 4800 steps,
4-EE direct arm, reduced ALINEA grid, final evaluation on V). Paper scale: `SEEDS="0 1 2"
STEPS_PER_ROUND=1000000 ROUNDS=4 DIRECT_EE="200 700 2000"`.
Smoke test (`SMOKE=1`, 2026-09-15, 14 min on the Mac): every stage ran end to end (40-rollout
store, 2-member ensemble, gate (fails as expected for a 2-epoch model), 1 aggregation round,
4-EE direct arm, reduced ALINEA grid, manifest with 5 arms, final evaluation on the v2 V set,
figures 1/2/4). Two fixes found by it: the study writes `runs/study/<study>/env_study.yaml`
(density stats from its own store) and appends it to SCENARIO_OVERLAY; the final evaluation
derives the output path of a set the manifest does not name (val). The untrained smoke
policies sit at the initial u = 0.3 (−97 on V) vs ALINEA −70 and constant 0.4 −80, as
expected at 4800 PPO steps. Full suite: 66 passed, 7 skipped. Smoke artefacts deleted.

## 13. v3b road configuration plotted from the compiled SUMO net (2026-09-15)
Figures regenerated on this checkout (gitignored, `_progress/figures/`): the schematic
`m14_v3b_road_layout.png` (`scripts/plot_scenario_layout.py --final`) and a new plot of the
true netconvert geometry `m14_v3b_sumo_network.png` (`scripts/plot_v3b_sumo_network.py`:
edges, internal lanes, the 19 E1 loops incl. both loops of station 12, insertion point).
The v3b network was rebuilt into `data/raw/network_v3b` (`build_network`).

**Finding: the merge junction caps ramp speed at 33 km/h.** netconvert assigns the
internal merge link (`:merge_0_0`, 3.8 m, 30° entry angle) a speed of 9.18 m/s
(`junctions.limit-turn-speed` default). TraCI check (empty mainline, IDM, departSpeed 0):
71 km/h max on the 200 m ramp, 33 km/h at the nose, 81 km/h at the end of the 100 m
acceleration lane, 88 km/h entering `highway_post`. The "≈ 100 km/h at the nose" reading
in §8 is therefore wrong: vehicles brake before the nose and do the real acceleration on
the 100 m lane. The same 9.18 m/s link exists in the v2 net (`data/raw/network`), so every
E0 / capacity number so far already includes it; it is inherited, not a v3b artefact.
Options (user decision, changes merge capacity → E0 re-run): (a) keep, document as a
"yield-at-nose" merge; (b) `netconvert --junctions.limit-turn-speed -1` (or a large value)
so the ramp joins at speed; (c) shallower ramp angle. Not changed yet.

Follow-up check (2026-09-15): the cap is a junction property (netconvert `junctions.limit-turn-speed`
from the 30° entry angle), independent of ramp length. Same TraCI test on four net variants:

| net | ramp max | at the nose | at the lane drop |
|---|---|---|---|
| v3b as is (200 m, 30°) | 71 | 33 | 81 km/h |
| 100 m ramp, 30° | 49 | 33 | 81 km/h |
| 200 m, `--junctions.limit-turn-speed -1` | 100 | 100 | 112 km/h |
| 200 m, 10° entry angle | 97 | 97 | 112 km/h |

Shortening the ramp does not remove the braking; disabling the limit or a ≤ 10° angle does,
and then the 200 m ramp delivers the intended ≈ 100 km/h at the nose. Either fix raises the
merge speed differential and must be followed by an E0 re-run (capacity will change).

**Adopted (user, 2026-09-15): 10° entry angle.** `network.ramp_entry_angle_deg` in the builder
(default 30 → v2/v3 nets bit-identical), `scenario_v3b.yaml` sets 10. Rebuilt v3b net: merge
link 33.33 m/s, ramp lane 179.8 m (the shallow junction absorbs ≈ 20 m; test bound relaxed to
175 m), nose speed 97 km/h, 112 km/h at the lane drop. Tests: scenario v3/v3b, ramp queue,
departures, profiles, plant surrogate — 32 passed (two fixtures now read YAML as UTF-8; they
failed on this machine's GBK locale before, unrelated to the angle).

Capacity effect, constant rates on E0 profiles 0 / 5 / 8 (same seeds), 30° vs 10° net:

| profile | u | R 30° | bd | R 10° | bd |
|---|---|---|---|---|---|
| 0 | 0.3 / 0.4 / 0.5 | −87 / −93 / −98 | 0 / 1 / 1 | −86 / −83 / −90 | 0 / 1 / 1 |
| 5 | 0.3 / 0.4 / 0.5 | −72 / −137 / −138 | 0 / 1 / 1 | −61 / −132 / −146 | 0 / 1 / 1 |
| 8 | 0.4 / 0.5 / 0.6 | −61 / −107 / −125 | 0 / 1 / 1 | −60 / −44 / −103 | 0 / 0 / 1 |

Reading: the first-breakdown rate is unchanged on profiles 0 and 5 and moves up one grid step
on profile 8 (u = 0.5 now survives, −107 → −44); returns at non-breaking rates improve by 1 to
11 veh h. So the merge capacity rises modestly rather than dramatically, but the E0 gate on
the step profiles can change. **The 30° E0 report is renamed to
`_progress/m14_e0_v3b_30deg_characterisation.json`** (its single-constant numbers in §12
stay valid for the 30° net) so that `scripts/run_v3b_study.sh` regenerates E0 on the 10° net
instead of reusing the stale report; do not copy the Mac's `data/plant_v3b/round0` (404
rollouts at 30°) onto this machine.

## 14. v3b study on the Windows machine (2026-09-15, complete: 5 rounds, stop rule fired at round 5)
**Readiness (handoff checklist).** Driver PATH fix: on Windows the conda env keeps `python.exe`
at the env root (no `Scripts/python.exe`), so `sh scripts/run_v3b_study.sh` resolved `python`
to the miniconda base 3.12; the driver now also prepends the env root (→ project 3.11, torch
2.11 CPU, SB3 2.8). Full suite: 65 passed, 1 skipped, 7 errors (legacy `test_surrogate_env.py`
loading an old M3 checkpoint under `runs/surrogate/` whose saved config points at a previous
checkout path; skipped on the Mac, unrelated to the study).
**Smoke run** (`SMOKE=1 WORKERS=8`, 02:24–02:54, 30 min vs 14 on the Mac): every stage ran,
exit 0: 40-rollout store (train 19 / val 13 / test 8), 2-member 2-epoch ensemble, gate FAILED
as expected (return err 11.0, calibration slope 640, false breakdown 0 but missed 100 %), one
4800-step aggregation round, 4-EE direct arm, reduced ALINEA grid (pialinea kp4 ki20 rho30
det12; constant u 0.5), manifest with 5 arms, final evaluation on V (B −94.6, ALINEA −62.7,
constant −70.0 at these untrained budgets), figures 1/2/4. Smoke artefacts deleted; logs kept
(`runs/logs/v3b_smoke_*.log`).
**Overnight run:** `WORKERS=10 sh scripts/run_v3b_study.sh` detached (sh pid in
`runs/logs/v3b_driver.pid`, driver log `runs/logs/v3b_driver.log`, stage logs
`runs/logs/v3b_*.log`), demo scale: seed 0, 3 rounds × 300k steps, direct 200/700 EE. Starts
with E0 on the 10° net (the 30° report was renamed, §13), then round 0 (288 + 404 rollouts),
5-member ensemble, gate, aggregation ‖ direct PPO ‖ ALINEA, final evaluation on T and O,
figures in `_progress/figures/m14_v3b/`. Expected ≈ 5–6 h. The driver does not stop on a
failed round-0 gate: read the gate line in the driver log before reading the arms table.
To record when done: the handoff's list (`_plans/m14_v3b_study_handoff.md`).
**E0 on the 10° net** (03:11, 404 rollouts, 17 min, `_progress/m14_e0_v3b_characterisation.json`):
gate 2/11 peaked, 2/8 storage-mandatory (same counts as at 30°); vs the best single constant,
now u = 0.5 (was 0.4: the merge takes 600 veh/h more often), 4/11 and 4/8 (30°: 4/11, 3/8).
Margins: +27.5, +15.9 pass; +14.7, +9.7, +8.0, +5.3 short; −28.9 on the step/surge profile 5 —
the grid-timing failure of §11 is unchanged by the angle. First-breakdown rate moved up one
step on 5 of the 8 loaded profiles (0.4–0.6, was 0.4–0.5).
**Round-0 store and gate** (03:11–05:23): 692 rollouts (404 E0 + 288 mixture), 5-member
ensemble 300 epochs, 2.0 h on CPU (≈ 7100 s per member, all five in parallel). Gate
**PASSED on both splits** (v2 at 692 rollouts: val 0.114 marginal fail / test pass):

| split | n | return-pred. error | false / missed breakdown | calib. slope | coverage ±2σ | rel-L2 ρ / q_exit |
|---|---|---|---|---|---|---|
| val | 105 | 0.053 | 0.019 / 0.151 | 1.44 | 0.47 | 0.183 / 0.050 |
| test | 101 | 0.055 | 0.017 / 0.116 | 1.41 | 0.47 | 0.186 / 0.052 |

The v3b plant model is better than v2's at the same store size (return error halved, false
breakdown at 2 %); the through-lane merge density and the family-v2 timing constraints
(everything over by minute 45) give a more regular field. Aggregation ‖ direct PPO ‖ ALINEA
started 05:23.

**Interrupted at 07:51 by a Windows Update forced restart** (KB5129195, three restarts by
TrustedInstaller; not a crash, nothing pending afterwards). State at the kill: aggregation
round 3 PPO 107 s in; direct 700 EE at 29 760 / 84 000 steps; everything else complete.
Resumed 12:17 without redoing finished work:
- `run_aggregation_loop.py --resume` (new): reopens the study store (rounds 1–2 already
  appended), keeps the rounds in `rounds.json` whose selected checkpoint and fine-tuned
  ensemble exist, clears the half-started round, continues (round 3 from `ensemble_r2` /
  `selected_r2.zip`).
- Direct 700 EE continued from `checkpoints/ppo_sumo_28800_steps.zip` with `--init-policy`
  (optimizer state restored, constant lr / clip, no entropy schedule) for the remaining 55 200
  steps into `direct_ppo_700ee_s0_part2` (seed 1000 so the profile sequence is not replayed),
  same ledger study, so the EE budget stays continuous (248 training episodes logged before
  the kill). To be merged into the 700 EE run dir (checkpoint steps offset by 28 800,
  `evaluations.npz` concatenated) before checkpoint selection.

**Results so far (V = family-v2 val set, 18 profiles):**

| arm | EE | SUMO V mean | p10 | worst | breakdown | note |
|---|---|---|---|---|---|---|
| A1 round 1 (step 288k) | 746 | −46.4 | −68.5 | −134.8 | 0 | surrogate −47.6, gap +1.2; PPO 72 min |
| A1 round 2 (step 96k, warm start) | 800 | −43.9 | −66.8 | −112.6 | 0 | surrogate −48.1, gap +4.2; top-3 SUMO −43.9 / −44.7 / −45.4 |
| B direct 200 EE | 200 | −70.5 (eval curve best, at 24k steps) | | | | still improving at the budget |
| B direct 700 EE | 248 so far | −82.0 at 28.8k steps | | | | eval curve −81.3 / −80.4 / −82.0 |
| ALINEA (tuned on V: pialinea kp 4, ki 20, ρ 26, det 13) | 236 | −46.7 | −81.6 | −101.1 | 0 | |
| constant u | 198 | u = 0.5 best | | | | |

Reading before the final evaluation: the aggregation policy at 800 EE is 2.8 veh h better than
the tuned PI-ALINEA on V (and 26.6 better than direct PPO at 200 EE, which has not converged);
the round-2 spread after fine-tuning went *up* (0.082 → 0.135), the M13 "fine-tune does not
track" symptom, so round 3's gap column is the one to watch. Final numbers on T and O follow.

**Round 3 (resumed, 12:17–13:12):** PPO 2401 s (warm start from `selected_r2`), top-3 by
surrogate 96k / 192k / 120k (−48.0 / −48.0 / −48.2), SUMO on V −45.9 / −40.4 / −40.9 →
selected 192k: **SUMO V −40.4** (p10 −57.6, worst −91.6, breakdown 0, no catastrophic), gap
+7.6 (surrogate pessimistic), improvement +3.5 ≥ δ = 2 so the stop rule never fired; spread on
the new rollouts 0.177 → 0.050 after fine-tuning. `study.json`: A1 = round 3 at 854 EE.

**Final evaluation (13:12–13:56, driver re-run with `DIRECT_EE=200`; `runs/study/v3b/eval/`,
figures `_progress/figures/m14_v3b/` figs 1, 2, 4, 5, 7).** T = family-v2 test set, 30 profiles
× 3 seeds; O = 12 OOD profiles × 3 seeds. "diff" = paired difference to tuned PI-ALINEA
(kp 4, ki 20, ρ̂ 26, det 13; 522 EE incl. tuning) with a 95 % CI over the paired episodes.

| arm | EE | T mean | T p10 | T worst | T bd | T diff vs ALINEA | O mean | O bd | O diff vs ALINEA |
|---|---|---|---|---|---|---|---|---|---|
| A0 zero-shot (= A1 r1 checkpoint) | 692 | −48.4 | −94.4 | −173.4 | 0.00 | −2.0 [−4.2, +0.3] | −51.9 | 0.00 | +2.4 [−0.6, +5.5] |
| A1 round 2 | 800 | −48.7 | −83.3 | −195.7 | 0.00 | −2.3 [−5.8, +1.2] | −51.7 | 0.00 | +2.6 [−1.4, +6.5] |
| **A1 round 3 (final)** | 854 | **−41.7** | −64.2 | −129.9 | 0.00 | **+4.7 [+2.9, +6.6]** | **−48.6** | 0.00 | **+5.8 [+2.5, +9.0]** |
| B direct SUMO PPO | 290 | −74.4 | −132.6 | −278.3 | 0.07 | −28.0 [−36.3, −19.7] | −70.9 | 0.00 | −16.6 [−25.3, −8.0] |
| B direct SUMO PPO, 700 EE arm | 855 | −50.2 | −101.6 | −164.1 | 0.00 | −3.8 [−6.4, −1.2] | −52.3 | 0.00 | +2.0 [−1.5, +5.6] |
| ALINEA (tuned on V) | 522 | −46.4 | −89.1 | −135.3 | 0.01 | reference | −54.3 | 0.00 | reference |
| constant u = 0.5 | 198 | −89.5 | −227.8 | −334.6 | 0.46 | −43.1 [−55.7, −30.5] | −105.6 | 0.50 | −51.3 [−68.5, −34.2] |

Reading:
- **The surrogate-trained policy beats tuned PI-ALINEA on both held-out sets**, +4.7 veh h on T
  and +5.8 on O, CIs excluding zero, with zero breakdowns and a better tail (p10 −64 vs −89,
  worst −130 vs −135). The v2 demo study had the same ordering with a similar margin (A1 −55.0
  vs ALINEA −60.6 on its T, +5.6); only rankings and margins are comparable across scenarios.
- The gain comes from round 3: A0 / rounds 1–2 sit at ALINEA's level (−2 ± 2 on T). Rounds 2 → 3
  moved T from −48.7 to −41.7, so the aggregation loop, not the zero-shot model, carries the
  result; the 3-round cap, not the stop rule, ended the loop (a 4th round is an open question).
- **Direct SUMO PPO at the same budget loses to the aggregation policy**: the 700 EE arm
  (855 EE with its V evaluations, selected checkpoint 76.8k steps, V −51.7) reaches −50.2 on T,
  8.6 [5.7, 11.4] worse than A1 round 3 at 854 EE and 3.8 [1.2, 6.4] worse than tuned ALINEA;
  on O it is −52.3, 3.7 [1.9, 5.5] behind A1 and level with ALINEA. Its V curve was still
  rising at the budget (−82 → −52 over the last 48k steps), so more SUMO episodes would close
  the gap, which is the sample-efficiency claim: at ≈ 850 EE the surrogate route is ahead by
  8.6 veh h on T with zero breakdowns for both. The 200 EE arm (−74.4) is far behind.
- Transfer gap: surrogate ranking picked the right checkpoint in rounds 1 and 3 (fig 4); the
  surrogate is *pessimistic* in rounds 2–3 (SUMO better than predicted by 4–8), the opposite sign
  of M13's late-round over-optimism.
- Ramp blocking (`pending_ramp` in the npz files): 80 / 692 round-0 rollouts (76 of the 252
  store-and-flush schedules, max 85 vehicles; 2 feedforward, 1 random, 1 ALINEA-wide) and **none**
  of the 162 aggregation rollouts or any final-evaluation episode. The learned policies never
  flush into a jam, so the SumoEnv-only blocking regime is in the data but not on-policy.
- Budget accounting: EE from the ledgers count evaluation-on-V episodes (B 200 → 290 EE,
  ALINEA tuning 522 EE), as in M12.

**Direct 700 EE, completed 16:00 and merged** (`merge_info.json` in the run dir; part 2 was `direct_ppo_700ee_s0_part2`, 55 200 steps); checkpoints 9.6k–76.8k continuous, `evaluations.npz`
concatenated (part-1 copy kept as `evaluations_part1_backup.npz`); driver re-run with
`DIRECT_EE="200 700"` 16:02–16:08 (selection, manifest with B at 290 / 855 EE, evaluation of
the B 700 arm on T and O, figures regenerated). **Study complete**; handoff items 1–5 recorded above.

**Round-3 ensemble vs SUMO (2026-09-15, `runs/aggregation/v3b_s0/ensemble_r3/eval_{test,policy_r3}*.json`;
figures `_progress/figures/m14_v3b_plant_r3_test/` and `.../m14_v3b_plant_r3_policy/`, figs a–f via
`scripts/plot_plant_eval.py`, which now picks error-quantile representatives for single-regime splits).**

| split | n | rel-L2 ρ (free / band / jam) | rel-L2 q_exit | return err mean / median | false / missed bd | ±2σ coverage | slope |
|---|---|---|---|---|---|---|---|
| held-out test (round-0 controllers) | 101 | 0.182 (0.196 / 0.215 / 0.209) | 0.051 | 0.058 / 0.041 | 0.017 / 0.140 | 0.44 | 1.36 |
| round-3 on-policy rollouts (A1 r3 on V) | 54 | 0.096 (0.089 / 0.237 / 0.673) | 0.037 | 0.050 / 0.048 | 0.000 / 0.000 | 0.48 | 1.35 |

Reading: after three fine-tunes the ensemble is unchanged on the held-out test split (round-0
ensemble: 0.186 / 0.052 / 0.055 / 0.017 / 0.116) — no forgetting of the open-loop regimes — and
on its own policy's rollouts the density error halves (0.096) and the return error is 2.5 veh h
absolute (5 %), with no breakdown in truth or prediction. On-policy the residual error sits in the
merge cell after minute 30 (mean |error| ≈ 3 veh/km there vs ≈ 1.2 elsewhere, fig d) where the
ensemble std is also largest; the worst on-policy case (14.5 % return error, ramp surge to 750 veh/h,
96-vehicle queue) is over-pessimistic by 10 veh h because the predicted exit flow sags ≈ 150 veh/h
below SUMO after the surge. Held-out breakdown cases are reproduced as the right shock geometry with
a smoother front and a lag of ≈ 1–2 min at the jam edge (fig a); ALINEA-wide's on/off platoons are
the hardest regime (rel-L2 0.25–0.32).

**Round 4 (user request, 16:40–17:12, `--resume --rounds 4`; T/O evaluation 17:13–17:18).** Does the
loop keep climbing, plateau, or regress as M13's did?

| round | PPO s | SUMO V | improvement | gap | V worst | top-3 SUMO | spread b→a | cum EE | stop rule |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4328 | −46.4 | | +1.2 | −134.8 | −46.4 / −50.0 / −53.1 | 0.056 → 0.040 | 746 | |
| 2 | 2651 | −43.9 | +2.6 | +4.2 | −112.6 | −43.9 / −44.7 / −45.4 | 0.082 → 0.135 | 800 | |
| 3 | 2401 | −40.4 | +3.5 | +7.6 | −91.6 | −45.9 / −40.4 / −40.9 | 0.177 → 0.050 | 854 | |
| 4 | 1258 | −39.9 | +0.5 | +2.1 | −84.8 | −39.9 / −40.3 / −40.4 | 0.128 → 0.065 | 908 | strike 1 / 2 |

Held-out (paired 95 % CI): A1 r4 on T −41.0 (p10 −73.0, worst −118.8, bd 0) = +5.4 [+3.5, +7.2]
vs ALINEA, +0.6 [−0.4, +1.7] vs r3, +9.2 [+6.0, +12.4] vs B 700; on O −47.0 (worst −97.4, bd 0) =
+7.3 [+4.3, +10.4] vs ALINEA, +1.6 [+0.1, +3.0] vs r3, +5.3 [+3.6, +7.0] vs B 700. `study.json`:
A1 = round 4 at 908 EE (best V); manifest / final eval / figures regenerated with the 4-point A1 curve.

Reading: **plateau with a closing gap, not M13's regression.** V improves 0.5 (< δ = 2, first stop
strike), T +0.6 (n.s.), O +1.6 (just significant); the three round-4 candidates agree in SUMO within
0.5, the surrogate's own V estimate rose from −48 to −42 (the fine-tuned ensemble caught up with the
policy's regime) and the gap fell to +2.1 while staying pessimistic — M13's late rounds had gaps of
−8 to −11 (over-optimistic) and selected policies 3–11 worse. The worst V episode keeps improving
(−135 → −113 → −92 → −85). Round 4 cost 32 min (PPO 21 min from the warm start). Conclusion for the
paper-scale settings: 4 rounds with patience 2 is the right cap; the remaining uncertainty is
across seeds, not rounds.

**Round 5 (user request, 17:36–18:06; T/O evaluation 18:06–18:12): the stop rule fired.** PPO 1141 s,
top-3 by surrogate 24k / 96k / 48k (−41.6 / −41.9 / −43.6), SUMO V −39.5 / −40.0 / −43.5 → selected
24k: **V −39.5** (p10 −59.9, worst −79.2, bd 0), gap +2.1, improvement +0.4 < δ = 2 → second strike,
`stopped_by_rule = true`, 962 EE, 30 min. Held-out: T −41.5 (p10 −74.0, worst −117.1) = +4.9 [+3.1, +6.6]
vs ALINEA, −0.5 [−1.6, +0.6] vs r4, +8.7 [+5.5, +11.8] vs B 700; O −47.6 = +6.7 [+3.4, +10.0] vs ALINEA,
−0.6 [−1.9, +0.7] vs r4, +4.7 [+2.8, +6.5] vs B 700.

Five-round curve on the held-out sets (paired vs the previous round):

| round | cum EE | V | T mean | T vs prev | T worst | O mean | O vs prev |
|---|---|---|---|---|---|---|---|
| 1 | 746 | −46.4 | −48.4 | | −173.4 | −51.9 | |
| 2 | 800 | −43.9 | −48.7 | −0.3 [−3.1, +2.5] | −195.7 | −51.7 | +0.2 |
| 3 | 854 | −40.4 | −41.7 | +7.0 [+3.2, +10.8] | −129.9 | −48.6 | +3.2 [+0.8, +5.5] |
| 4 | 908 | −39.9 | −41.0 | +0.6 [−0.4, +1.7] | −118.8 | −47.0 | +1.6 [+0.1, +3.0] |
| 5 | 962 | −39.5 | −41.5 | −0.5 [−1.6, +0.6] | −117.1 | −47.6 | −0.6 [−1.9, +0.7] |

Reading: the loop terminates by its own rule after two flat rounds, at a plateau that holds on T and O
(rounds 3–5 within ±0.6 of each other, all CIs spanning zero), with no late regression (M13: −3 to −11
in rounds 5–6) and the gap steady at +2.1. The `study.json` A1 is round 5 by the V criterion; on T/O
rounds 3–5 are statistically indistinguishable, so the honest headline is "A1 ≈ −41 on T from 854 EE
on, +5 over tuned ALINEA, +9 over direct PPO at the same budget". Paper-scale setting confirmed: 4–5
rounds with δ = 2, patience 2.

**Relative TTS reduction vs tuned PI-ALINEA (2026-09-16, for the paper sentence).** Paired over the
same (profile, SUMO seed) episodes of the final evaluation (`runs/study/v3b/eval/*.jsonl`; T = 90
episodes, O = 36), 1 − mean(TTS_A1) / mean(TTS_ALINEA), 95 % CI by paired bootstrap (5000
resamples). "return" is the training reward (−TTS, 90 s warm-up masked), "TTS" the full-episode
`tts_veh_h` metric; both agree within 0.7 points.

| arm | T, return | T, TTS | O, return | O, TTS |
|---|---|---|---|---|
| A1 round 3 (854 EE) | 10.2 % [6.8, 13.5] | 10.5 % [6.9, 14.0] | 10.6 % [5.3, 15.0] | 10.6 % [5.2, 15.1] |
| A1 round 4 (908 EE) | 11.6 % [8.5, 14.8] | 12.2 % [8.8, 15.6] | 13.5 % [9.2, 17.3] | 13.7 % [9.2, 17.4] |
| **A1 round 5 (962 EE, `study.json` A1 by the V rule)** | 10.5 % [7.4, 13.5] | **11.1 % [7.8, 14.2]** | 12.3 % [6.9, 16.8] | **12.5 % [7.2, 17.0]** |
| B direct SUMO PPO 700 EE (855 EE) | −8.2 % [−13.2, −3.1] | −10.2 % [−15.8, −4.5] | 3.7 % [−2.9, 8.9] | 2.7 % [−4.5, 8.4] |

Headline (selected arm, full-episode TTS): **11 % on T and 12 % on O**, seed 0, demo scale.

## 15. Seed sweep, seeds 1 and 2 (launched 2026-09-16, overnight)
`SEEDS="0 1 2" ROUNDS=5 WORKERS=10 sh scripts/run_v3b_study.sh`, detached (pid in
`runs/logs/v3b_driver_seeds12.pid`, driver log `runs/logs/v3b_driver_seeds12.log`, stage logs
`runs/logs/v3b_agg_s{1,2}.log`, `runs/logs/v3b_direct_{200,700}_s{1,2}.log`). Seed-0 stages
(E0, round-0 store, ensemble, gate, ALINEA tuning, seed-0 loop and direct arms) are skipped by
the driver's idempotence checks; seeds 1-2 run the 5-round loop (δ 2, patience 2, same round-0
ensemble) in parallel with their 200 / 700 EE direct arms, then the manifest, final evaluation
on T and O (existing seed-0 JSONLs reused) and figures are regenerated with all three seeds.
Expected: aggregation loops ≈ 6-8 h; the 700 EE direct arms ≈ 14 h each (seed 0's took ≈ 14 h),
so the driver's final stages may run into the next day. If a Windows restart interrupts it,
rerun the same command (aggregation resumes via `--resume`, direct arms restart from scratch).
Launched 02:59 (first attempt died at once: `sh.exe` started from PowerShell had no Git
`usr/bin` on PATH, so `dirname` / `mkdir` were missing; relaunched with that PATH prepended).
At 03:01: seed-0 stages skipped, gate line re-read (passed), loops s1 / s2 in round 1 of 5,
direct 200 / 700 arms for both seeds running. **To record tomorrow:** per-seed rounds table
(V, gap, stop round), T / O table with paired CIs vs PI-ALINEA for A1 seeds 0-2 and the
direct arms, the seed-to-seed spread of the A1 T/O means, the relative TTS reduction per seed
and pooled (method of the §14 addendum), and regenerated figures `_progress/figures/m14_v3b/`.
**Crash 2026-09-16 04:19** (System event 6008: unexpected shutdown, not a Windows Update restart;
machine back at 16:27). State at the crash: loops s1 / s2 at 292k / 300k PPO steps of round 1 (no
completed round, so nothing to resume), direct arms at 18k / 24k (200 EE) and 20k / 84k (700 EE)
steps, all ending with `FatalTraCIError: Connection closed by SUMO`. Partial run dirs and their
ledgers moved to `runs/crashed_0916/` so the restarted arms count their SUMO episodes once;
relaunched 17:03 with the same command (loops s1 / s2 in round 1, four direct arms). ≈ 1.3 h of compute lost.
**Stopped 2026-09-16 17:1x (user decision): the seed sweep moves to Colab.** Driver process tree
killed ≈ 10 min after the relaunch (loops s1 / s2 early in round 1, direct arms at a few k steps);
the partial dirs of this second attempt (`runs/aggregation/v3b_s{1,2}`, `runs/study/v3b/direct_ppo_*_s{1,2}`,
ledgers `runs/ledger/v3b_direct_*_s{1,2}.jsonl`) are throw-away and must be moved aside or deleted
before any local rerun. The Windows machine has now lost two overnight runs to restarts (07:51 on
09-15, 04:19 on 09-16).
**Colab hand-off (2026-09-16 evening).** `colab/v3b_seed_sweep.ipynb`: Drive-mounted clone, SUMO from
the `eclipse-sumo` wheel, the seed-0 bundle `runs/colab/v3b_seed0_bundle.zip` (231 MB, git-ignored:
round-0 store, ensemble, seed-0 loop / direct / eval files, ledgers, E0 + ALINEA reports), then
`SEEDS="0 1 2" ROUNDS=5 RESUME=1 WORKERS=$(nproc) sh scripts/run_v3b_study.sh`. New driver option
`RESUME=1` passes `--resume` to a loop whose study dir exists (Colab session limits). The second
local attempt's partial dirs are in `runs/crashed_0916_b/`. Results still to be recorded here.

**Round-5 ensemble accuracy on the held-out splits (2026-09-16, paper table; `runs/aggregation/v3b_s0/ensemble_r5/table_metrics.json`,
eval_val / eval_test.json).** Study-store splits (val 105, test 101 rollouts; test = round-0 controllers), mean over rollouts:

| metric | validation | test |
|---|---|---|
| density rel-L2 | 18.2 % | 17.9 % |
| density MAE (veh/km) | 2.36 | 2.53 |
| exit-flow rel-L2 | 5.0 % | 5.2 % |
| exit-flow MAE (veh/h) | 70.9 | 71.8 |
| return relative error (mean / median) | 5.8 % / 3.4 % | 6.4 % / 4.1 % |
| return MAE (veh h) | 4.21 | 4.68 |
| false / missed breakdown rate | 1.9 % / 18.9 % | 1.7 % / 14.0 % |
| true breakdown rate of the split | 50 % | 43 % |
Gate: passed on both (return err ≤ 10 %, false breakdown ≤ 10 %, slope 1.43 / 1.35).

**Controller-comparison table (2026-09-16, paper; `runs/study/v3b/eval/table_metrics.json`).** Seed 0, T = 90 / O = 36 episodes;
mean ramp queue from the 30-s samples of the stored rollouts (OOD arrays re-generated under study `v3b_table`, identical TTS to
the final evaluation), max queue = episode max, completed trips = network arrivals. Added arms: pure ALINEA (best of the joint
search, `alinea:ki=20,rho=26,det=13`) and Surrogate-MPC on the round-5 ensemble (default H 20, 30 iters, TTS objective):
TTS ID / OOD — u=0.5 77.1 / 92.6; ALINEA 42.6 / 51.1; PI-ALINEA 42.6 / 50.6; SUMO-PPO 700 EE 47.0 / 49.2; Surrogate-MPC 89.9 / 91.1
(54 / 58 % breakdowns: the M12 §6-7 solver negative result reproduces on v3b); Surrogate-PPO r5 37.9 / 44.3. Mean queue ID:
u=0.5 10.9, ALINEA 13.4, PI-ALINEA 13.5, SUMO-PPO 18.6, MPC 1.2, Surrogate-PPO 9.0. Costs (ledger, summed episode wall):
ALINEA search 324 ep / 2.66 h (constants excluded), SUMO-PPO 855 / 7.18 h, Surrogate-PPO 962 / 11.08 h (E0 2.74 + round-0 1.97 +
aggregation SUMO 2.01 + PPO 3.28 + ensemble ≈ 1.1), MPC = the ensemble's data + training (962 / ≈ 7.8 h without PPO).

## 16. Direct SUMO PPO continued to 1000 EE (launched 2026-09-16 evening, user request)
Does the direct arm catch up with more budget? `runs/logs/direct_1000_chain.sh`: warm start from
`direct_ppo_700ee_s0/final_model.zip` (value net and optimiser kept, seed 1001 so the profile
sequence differs), 36 000 more steps (700 → 1000 EE; the 700 EE arm was 84 000 steps), eval on V
every 9 600 steps, ledger `v3b_direct_1000_s0`; then checkpoint selection on V among the
continuation's checkpoints (`select_checkpoint_profiles.py`), then T / O evaluation paired against
tuned PI-ALINEA together with A1 round 5 (`runs/study/v3b/eval/B_1000_s0_{test,ood}.jsonl`).
Budget of the arm = 855 EE (700 EE run incl. its eval_val episodes) + this run's ledger.
Logs: `runs/logs/v3b_direct_1000_s0.log`, `..._select_...`, `..._eval_{test,ood}.log`.

**Single-case figures (2026-09-16).** `scripts/plot_plant_eval.py --case GROUP[:breakdown|nobreakdown]
[--case-pick median|busiest|worst]` writes one landscape 1x4 case study (inputs | SUMO density | DeepONet
density | exit flow; no std band, no metric annotations) as `fig_f_case_<group>_<bd>[_<pick>].png`:

| figure dir | ensemble | rollouts | case |
|---|---|---|---|
| `m14_v3b_plant_r3_test/` | r3 | held-out test | `alinea_wide:breakdown` median → `r0_alinea_wide_00041` |
| `m14_v3b_plant_r4_policy_A1/` | r4 (A1's training ensemble, out-of-sample) | A1 = `selected_r5` (ppo_r5 @ 24k) on the 18 V profiles, `rollouts_r5/aggregation_p00_val*` | median → `val003`; busiest → `val017` |
| `m14_v3b_plant_r5_policy_A1/` | r5 (fine-tuned on those rollouts, in-sample) | same | busiest → `val017` |

Rows: `ensemble_r{4,5}/eval_policy_r5_A1{,_rows}.jsonl`. r4 on A1's rollouts: rel-L2 ρ 0.342, return err
0.236, false breakdown 0.111 (gate FAILED; the busiest case `val017` is a predicted breakdown after min 45
that SUMO does not show). r5 on the same: 0.135 / 0.073 / 0.000 (PASSED) — this is the round-5 fine-tune
closing the on-policy gap, but those 18 rollouts are in r5's train split, so it is not a held-out number.
**Result (2026-09-16 22:45).** Training 20:06-22:25 (36 000 steps, 300 episodes + 54 eval_val), V evaluations −52.7 /
−51.0 / −46.9 (9.6k / 19.2k / 28.8k steps); selection picked 28 800 steps (V −46.9; the 700 EE arm's selected checkpoint
was ≈ −50). Total budget of the arm 855 + 354 = 1209 EE (nominal 1000). The first OOD run died on a transient TraCI
start-up error in one worker (rerun clean). Paired against tuned PI-ALINEA (`B_1000_s0_{test,ood}.jsonl`, 95 % bootstrap CI):

| arm | T mean | T diff vs PI-ALINEA | T TTS reduction | O mean | O diff vs PI-ALINEA | O TTS reduction |
|---|---|---|---|---|---|---|
| direct SUMO PPO, 700 EE (855) | −50.2 | −3.8 [−6.4, −1.2] | −10.2 % | −52.3 | +2.0 [−1.5, +5.6] | +2.7 % |
| **direct SUMO PPO, 1000 EE (1209)** | **−44.9** | +1.6 [−0.4, +3.6] | +3.6 % | **−51.4** | +2.9 [+0.1, +5.9] | +5.4 % |
| surrogate-PPO round 5 (962) | −41.5 | +4.9 [+3.2, +6.7] | +11.1 % | −47.6 | +6.7 [+3.5, +10.1] | +12.5 % |
| surrogate-PPO minus direct 1000 EE | | +3.3 [+1.5, +5.3] | | | +3.8 [+1.9, +5.8] | |

Reading: with 300 more SUMO episodes the direct arm improves by 5.3 on T and 0.9 on O, reaches PI-ALINEA's level on T
(CI spans zero) and edges past it on O, but stays 3.3-3.8 veh h behind the surrogate policy with a CI clear of zero, at a
budget 26 % larger than the surrogate arm's. Its V curve was still rising at the end (−46.9 at 28.8k of 36k steps), so a
longer run may close more of the gap; the sample-efficiency claim (surrogate-PPO at 962 EE > direct at 1209 EE) holds.
Table row for the 1000 EE arm (arrays re-generated, `runs/study/v3b/eval/table_b1000_{test,ood}.jsonl`): ID TTS 41.11,
mean queue 12.11, max queue 32.92, served 1974.5; OOD TTS 47.89, mean queue 13.11, max queue 39.19, served 1992.7;
budget 1209 episodes, 9.9 h (7.18 + 2.67 ledger).

## 17. Reward queue timing unified to Q_{k+1} in `m14/` (2026-09-22, user decision)
Paper audit before the 60 km/h-ramp rerun: paper Eq. 4 charges Q_k in both environments, but the
code charged the SUMO *interval-average* queue and the surrogate *end-of-interval* queue. The user chose Q_{k+1}
for both, which removes the paper's lagged terms δ^z_{k-1} + L_z b_{k-1} from Theorem 1. Q_k would have left the
final-interval queue uncharged (closing the meter in the last 30 s would be free).
- Code (`m14/` only; the parent `src/` keeps the old timing): `m14/src/rl/sumo_env_wrapper.py` charges
  `queue_end`; the interval mean is still in info as `interval_queue_mean`. The surrogate, offline rescoring
  (`rollout.py`, plant return error, gate) and the MPC cost already used the end queue: no change. Docstring in
  `reward.py`, `m14/docs/model.md` updated; `tests/test_scenario.py::test_m14_no_cap_queue_grows` asserts
  tts_step = (N + Q_after + P)·τ on a growing queue.
- Tests: 72 passed, 1 failed (`test_standalone.py` detached copy: `dateutil` missing under `python -I`; fails
  identically without the change, environment issue).
- Size of the change, rescored on the v3b final-eval test rollouts (10 arms × 90 episodes): mean returns move
  by −0.23 to +0.27 veh h (max 1.5 in one episode), paired differences vs PI-ALINEA by ≤ 0.3, rankings unchanged;
  `tts_veh_h` (the reported metric) does not use the reward and is unaffected.
- Paper edits pending (co-authors): Eq. 4 Q_k → Q_{k+1}, drop "Q_k denotes the beginning-of-interval queue",
  Theorem 1 / Eq. 21 without the lagged terms, proof step via N+Q+P = max(N_{k+1}+Q_{k+1}, C^in − C^out).

## 18. 60 km/h ramp in `m14/` (2026-09-22, user decision; rerun of the whole study pending)
`m14/configs/scenario.yaml` `ramp_speed_limit_mps: 16.67` (was 33.33); everything else in the v3b road unchanged
(10° entry, 200 m ramp, 100 m acceleration lane at the mainline 120 km/h, D = 1200, uncapped virtual queue).
Edited in place (m14/ had no generated data or runs; the 120 km/h version is in git at 357d5b7). `m14/README.md`,
`m14/docs/model.md`, `m14/tests/test_scenario.py` (ramp lane 16.67, acceleration lane 33.33) updated.
Tests: 72 passed, 1 failed (the pre-existing `test_standalone.py` `dateutil` failure of §17).

Probe vehicle from standstill on an empty road (scratchpad TraCI check, same method as §13):

| net | ramp lane / merge link limit | max on ramp | at the nose | at the lane drop | entering the main lane |
|---|---|---|---|---|---|
| 120 km/h ramp (study as published) | 120 / 120 | 94 | 100 | 111 | 113 km/h |
| **60 km/h ramp** | 60 / 90 (netconvert averages the two edges) | 61 | 61 | **92** | 103 km/h |
| 30° net, 120 km/h ramp (§13, for reference) | 120 / 33 | 71 | 33 | 81 | 88 km/h |

Reading: ramp vehicles now merge ≈ 28 km/h below mainline speed (was ≈ 9). The lane-drop speed sits between
the two nets already compared (30°: 81 km/h, 10°: 111 km/h; going from 30° to 10° moved the first-breakdown rate up
one 120 veh/h step on 5 of the 8 loaded E0 profiles, §14), so the capacity change should be at most about one
step, but it is not measured yet.
Next: E0 on the 60 km/h net as the capacity check before any round-0 data; if the merge capacity moves by more
than ≈ 50 veh/h, rescale the capacity-tied constants (storage-mandatory 2500 veh/h, feedforward range 2250–2550,
E0 feedforward capacities, demand-family ceilings) before generating round 0.
