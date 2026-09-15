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
