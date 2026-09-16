# M15 progress — scenario v4: three-lane mainline

Plan: `_plans/m15_scenario_v4_three_lane_plan.md`. Decision pending: merge observation and
lane-change model (plan §"design decisions"). Tests below run on a scratch v4 config =
`scenario_v3b.yaml` with `network.num_lanes: 3` (nothing else changed), LC2013 lane changing,
1-hour constant-demand episodes through `SumoEnv`, meter open (u = 1), ramp 700 veh/h.
Scripts kept in `scripts/m15_scratch/`; figures in `_progress/figures/m15/` (git-ignored).
Code additions: `SumoEnv.last_per_lane_density` (diagnostic, per-lane occupancy density of the
last control interval) and `vehicle.lane_change: {<SUMO lcXxx attr>: value}` in the scenario
file → extra vType attributes (absent = SUMO defaults, routes file unchanged).

## 1. Three-lane merge at SUMO defaults (2026-09-15, `merge_observation_test.png`)
Builder, detectors (3 loops per station, 4 at the merge station), density estimator all work
unchanged; no teleports and no insertion failures at any load; ≈ 35 s per 1-hour episode
(v3b: ≈ 4 s → the 3-lane budget is ≈ 9× the single-lane one, not 3×).

| mainline + ramp | exit flow, last 30 min | station 11 (1200 m), minutes 30–60: lane 0 / lanes 1–2 / through-mean | onset (lane 0 > 40 for 2 min) | onset (through-mean > 25) |
|---|---|---|---|---|
| 4800 + 700 | 5400 | 16 / 16 / 16 | none | none |
| 5400 + 700 | 5818 | 77 / 19 / 38 | 12.8 min | 12.8 min |
| 6000 + 700 | 5808 | 75 / 19 / 37 | 3.8 min | 3.8 min |
| 6600 + 700 | 5726 | ≈ 80 / 19 / 40 | 3.2 min | 3.2 min |
| 6000 + 700, u = 0.5 | 5722 | 75 / 19 / 37 | 4.8 min | 4.8 min |

**Merge observation (decision input).** The breakdown is a lane-0 phenomenon: lane 0 queues at
75–80 veh/km/lane from the nose back to 200 m while lanes 1–2 stay in free flow at 19. The
through-lane mean dilutes the jam 2× (38 vs 77) but does not hide it: the step from 15 to 38 is
unambiguous, the onset time is identical to lane 0's at every station, and the spatial
signature (plateau at ≈ 38 upstream of 1300 m, ≈ 15 downstream) is the same shock geometry the
v3b DeepONet learned. Consequence for the pipeline: the lane-averaged observation and DeepONet
targets work **provided the breakdown threshold is re-calibrated** (v3b: 40 veh/km/lane on one
lane; v4: ≈ 25–30 on the through-lane mean) — the breakdown detector, the E0 gate and the
plant-model "jam band" all use it. What the average loses is the *magnitude* (a mean of 38 could
be one jammed lane or three moderately loaded ones); the single-lane-jam pattern makes the two
distinguishable only through the spatial plateau. Recommendation stands: lane-averaged, plus
the threshold change; add a lane-0 channel only if the plant model's merge onset error is worse
than v3b's.

## 2. Lane-change model (2026-09-15, `lane_change_variants.png`)
LC2013 (SUMO default since 0.20; Erdmann 2015) with five parameter sets, station 11 at
6000 + 700 and 5400 + 700:

| vType lane-change attrs | 5400 + 700: exit / lane 0 / lanes 1–2 | 6000 + 700: exit / lane 0 / lanes 1–2 |
|---|---|---|
| defaults (lcSpeedGain 1, lcKeepRight 1, lcCooperative 1, lcAssertive 1) | 5824 / 77 / 19 | 5804 / 74 / 19 |
| lcKeepRight 0 (+ cooperative 1) | 5824 / 77 / 19 (identical) | 5826 / 77 / 19 |
| lcTimeToImpatience 30 s | 5824 / 77 / 19 (identical) | 5804 / 74 / 19 |
| lcSpeedGain 3, lcKeepRight 0 | 5948 / 26 / 17 (no breakdown) | 5806 / 76 / 19 |
| lcAssertive 2, lcSpeedGain 3, lcKeepRight 0 | 6090 / 13 / 18 (no breakdown) | 6424 / 71 / 29 |

Reading:
- The keep-right bias and impatience do nothing here (impatience only relaxes gap acceptance
  for *strategic* changes). The single-lane queue with free neighbours is LC2013's speed-gain
  gap acceptance: a stopped vehicle never finds an acceptable gap into a 100 km/h lane.
- Raising the speed-gain motive and the gap assertiveness moves the merge capacity from ≈ 5800
  to ≈ 6400 veh/h and lets the jam leak into lanes 1–2 (29 vs 19), but the lane-0 queue with
  flowing neighbours remains the breakdown mode at every setting tested.
- **Capacity drop is small on three lanes**: throughput in the jammed state (5800–5820) is within
  2–3 % of the free-flow maximum (≈ 5950 with speed-gain 3), against ≈ 30 % on the single lane
  (v3b: ≈ 2500 free vs ≈ 1700 jammed). Ramp metering's lever on v4 is therefore mostly the
  *delay* trade (ramp queue vs lane-0 mainline queue, both priced by TTS) rather than
  throughput protection — E0 (plan step 3) has to show whether any metering beats pass-through
  by the gate margin before a plant model is trained.

## 3. Decisions, capacity sweep, configs (2026-09-15, plan steps 1-2)
**Decisions (user, 2026-09-15 evening):** observation lane-averaged (through-lane mean at
every station, no lane-0 channel); lane-change model LC2013 with SUMO defaults (the
speed-gain / assertive variant of §2 is a documented sensitivity check). "Run E0 on v4 first."

**Capacity sweep** (`scripts/m15_scratch/capacity_sweep.py`, `_progress/m15_capacity_sweep_v4.json`,
17 constant-demand cells × 2 seeds, u = 1, 1 h, 239 s on 10 workers):

| ramp | mainline (offered) → result |
|---|---|
| 0 | 5800, 6200, 6600: no jam at any load; served flow saturates at ≈ 6150 (insertion-limited, ≈ 2050 veh/h/lane) |
| 400 | 5200-5800 (5600-6200): no breakdown, exit = offered |
| 700 | 4800-5200 (5500-5900): no breakdown; 5300 (6000): breakdown both seeds (onset 16.5 / 28 min); 5400, 5600: breakdown, jammed discharge 5700-5820 |
| 800 | 4800, 5000 (5600, 5800): no breakdown; 5200 (6000): breakdown (onset 11 / 24.5 min) |

Reading: the mainline alone never jams (the accel-lane zipper drop is not a bottleneck); the
merge breaks down when the **lane-0 load d/3 + r exceeds ≈ 2450 veh/h** (5200+700 = 2433 and
5800+400 = 2333 pass; 5300+700 = 2467 and 5200+800 = 2533 break), i.e. the same per-lane merge
capacity as v3b's 2500 on one lane. Total capacity is therefore ramp-dependent (≈ 5950 at
r = 700, > 6200 at r = 400), which is exactly the lever metering has: holding the ramp raises
the total the merge passes. Densities on the through-lane mean: free flow at capacity 16-19 at
1200 m, jam 34-37 at the nose station, episode max 38-40.5 → **breakdown threshold 30 veh/km**
(35 never fires; 25 fires early on some cells). Wall time 35-70 s per episode (jammed episodes
slower); no teleports, no insertion failures.

**Lane-aware generalisation (code, defaults reproduce v2/v3/v3b exactly).**
- `DemandProfile.peak_merge_load_vph(n_lanes)` = max_k(d_k/n + r_k) (n = 1: peak total).
- `enforce_storage_mandatory(..., n_lanes)` and the E0 storage label use it; the round-0
  generator and E0 read `network.num_lanes` from the scenario file.
- `feedforward_schedule(..., n_lanes)`: u = clip((C_lane − d/n)/D): C is a per-lane merge
  capacity; `make_controller_from_spec` passes the env's lane count.
- `run_scenario_characterisation.py`: optional `e0:` block of the round-0 config
  (`storage_needed_vph`, `ff_capacity_vph`, `insertion_vph`); the report records them and
  `peak_merge_load` per profile. `dataset.feedforward_capacity_vph` sets the round-0 draw range.
- `sumo_env.rollout.breakdown_flags(..., threshold)` / `episode_metrics(..., breakdown_density)`:
  the threshold is `detectors.breakdown_density_veh_km` of the scenario (default 60), carried
  by `SumoEnv`, `SurrogateVecEnv` and recorded in every rollout's metrics; `eval_plant` uses
  each rollout's own threshold. Driver: `MILESTONE` prefix (m14 default) for the E0 / ALINEA
  reports, ledgers and figure dir.

**Configs (plan step 1) and their deviations from the plan's audit table.**
- `configs/sumo/scenario_v4.yaml`: v3b + `num_lanes: 3`, `breakdown_density_veh_km: 30`,
  LC2013 defaults, `network_dir data/raw/network_v4`.
- `configs/profiles/family_v3.yaml` (version 3, new seed stream): mainline ranges = v2's as
  fractions of 2500 applied to 6000, peaks capped at 5750 (mainline alone never jams / never
  insertion-limited), peak amplitude 1200-2700 and step plateau 4600-5750 (raised at the
  bottom). **Ramp ranges changed** (plan said unchanged): with v2's ramp (200-500 base, surge
  to ≤ 800 anywhere in 5-40 min) only 12 % of the family exceeds the lane-0 rule and 1 of the
  12 E0 draws — a 10 % ramp share on three lanes rarely tips lane 0. v3: base 400-600, surge
  +200-400 (capped 800 = 2/3 of D, unchanged), start 10-30 min (overlaps the peak), constant
  ramp 500-800. Storage-mandatory share 0.39 (train stream), 4/12 E0 draws; frozen sets
  `configs/profiles/v3/`: val 8/18, test 18/30, ood 7/12 storage-mandatory; ramp max 800.
- `configs/experiments/round0_v4.yaml`: `storage_mandatory_vph 2450` (lane-0 rule),
  `feedforward_capacity_vph [2200, 2500]`, `e0: {storage_needed_vph 2450, ff_capacity_vph
  [2350, 2400, 2450], insertion_vph 1040}`, `demand_norm 6000`, q_ref/q_cap 5950 (logged only),
  store `data/plant_v4/round0`, 10 workers.
- `configs/rl/env_v4.yaml` (`demand_norm 6000`, family v3, `v3/val.json`),
  `configs/surrogate/plant_v4.yaml` (normalisers mainline / flow 6000, band `rho_min 25`).
- `tests/test_scenario_v4.py`: 8 tests (3-lane geometry and 4-loop merge station in SUMO,
  through-lane averaging, threshold parameter, scenario / round-0 / env / family consistency,
  frozen sets). Full v3b tests still pass (defaults unchanged).
- **Open for steps 4-5** (not in the plan's table): the ALINEA set-point grid (`--rhos 26 30 34
  38`, per lane) and the round-0 `alinea_wide` draw (18-50) are calibrated for a single-lane
  density; on the v4 lane mean the critical density is ≈ 17-20 and the jam reads 35-38, so a
  set-point ≥ 30 sits inside the jam. Use `--rhos 16 19 22 25` for the tuner and a lower
  `alinea_wide` range (config knob to add before round 0).

**E0 on v4 launched 2026-09-15** (`runs/logs/m15_e0_v4.log`, pid in `runs/logs/m15_e0_v4.pid`,
report `_progress/m15_e0_v4_characterisation.json`, rollouts into `data/plant_v4/round0`):
20 insertion + 12 × 11 constants + 12 × 21 schedules = 404 rollouts, ≈ 40 min on 10 workers.

## 4. E0 on v4 (2026-09-15, plan step 3, gate G2) — `_progress/m15_e0_v4_characterisation.json`, `e0_v4.png`
404 rollouts (20 insertion + 12 × 11 constants + 12 × 21 schedules), 2608 s on 10 workers
(≈ 65 s of wall per episode), all kept in `data/plant_v4/round0` (round-0 seed) and ledger
`m15_e0_v4`. Returns re-scored under the TTS reward.

**(i) Insertion check** (u = 0.87 = 1040 veh/h): served/offered 0.983-0.987 on all 20
profiles (v3b ≈ 0.98), pending mainline insertions ≤ 2 at the end, no teleports; 4/20 broke
down at pass-through (the four with lane-0 loads above 2450). G1 satisfied.

**(ii)+(iii) Capacity map and hand schedules** (profile: mainline/ramp family, peak lane-0
load d/3 + r; `*` = storage-mandatory > 2450; bd = first constant u with a breakdown):

| # | family | load | bd u | R(u = 1) | best constant u / R | best schedule / R | margin vs best / vs single u = 0.9 |
|---|---|---|---|---|---|---|---|
| 7 | peak/surge | 2110 | – | −67 | 1.0 / −66.9 | ff C2450 / −67.0 | −0.1 / 0 |
| 10 | peak/const | 2124 | – | −63 | 0.9 / −63.3 | ff C2350 / −63.2 | +0.1 / +0.1 |
| 11 | step/surge | 2136 | – | −67 | 0.8 / −66.7 | ff C2400 / −66.7 | +0.1 / 0 |
| 6 | const/const | 2150 | – | −96 | 0.8 / −95.5 | ff C2450 / −95.6 | −0.2 / 0 |
| 5 | const/const | 2209 | – | −99 | 0.9 / −98.6 | ff C2400 / −98.8 | −0.2 / 0 |
| 1 | peak/surge | 2311 | – | −69 | 1.0 / −69.4 | ff C2450 / −69.3 | 0 / 0 |
| 0 | step/surge | 2366 | – | −66 | 0.8 / −66.5 | ff C2450 / −66.4 | +0.1 / +0.5 |
| 8 | peak/surge | 2414 | – | −78 | 0.9 / −78.2 | ff C2450 / −78.3 | −0.1 / 0 |
| 2* | step/surge | 2506 | – | −81 | 1.0 / −81.0 | ff C2450 / −81.5 | −0.6 / −0.6 |
| 9* | step/surge | 2580 | 0.6 | −125 | 0.5 / −100.2 | ff C2400 lag 2 / −92.4 | +7.8 / +27.6 |
| 4* | step/surge | 2584 | 0.8 | −90 | 0.6 / −77.4 | ff C2450 lag 2 / −77.6 | −0.1 / +11.4 |
| 3* | step/const | 2589 | 0.6 | −110 | 0.9 / −96.0 (bd) | ff C2450 / −83.7 | +12.3 / +12.3 |

Gate lines: peaked profiles 0/10 (per-profile constant), storage-mandatory 0/4; vs the
single best constant (u = 0.9 = 1080 veh/h, near pass-through): 1/10 peaked, 1/4
storage-mandatory. **E0 gate failed on both lines.**

**The metering lever, v3b vs v4** (E0 reports, best hand schedule minus pass-through u = 1,
and minus the best single constant, mean over profiles):

| | storage-mandatory: n, lever vs u = 1 (% of TTS), vs single constant | other: n, lever vs u = 1, vs single constant | single constant |
|---|---|---|---|
| v3b (M14 §11-12) | 8/12, 53 veh h (121 %), 22 | 4/12, 13 veh h (47 %), 2 | u = 0.5 |
| v4 | 4/12, 18 veh h (21 %), 13 | 8/12, 0.0 veh h (0 %), 0 | u = 0.9 |

**Reading.**
1. On 8 of the 12 profiles nothing happens at any u ≥ 0.6: no breakdown, return flat within
   1 veh h from u = 0.7 to 1.0, every schedule ties. Metering is irrelevant for two thirds of
   the family (loads ≤ 2414 never reach the lane-0 conflict).
2. Of the 4 storage-mandatory profiles one (load 2506, a 5-min exceedance) never breaks down —
   the practical threshold for short peaks is ≈ 2550. The other three break down at u ≥ 0.6-0.8
   and the breakdown costs 12-33 veh h against the best schedule (10-25 % of the episode TTS).
   On v3b a breakdown cost 53 veh h, more than the whole no-breakdown TTS (121 %).
3. The best single constant is u = 0.9 (v3b: 0.5): "do nothing" is near-optimal on the family.
   The best schedules are the capacity-tracking feedforwards with per-lane C 2400-2450 (the
   lane-aware rule works as a schedule), and they beat the best per-profile constant by ≤ 12 and
   the single constant by ≤ 28 veh h, only on the loaded profiles.
4. This is the §2 prediction confirmed in the reward: under LC2013 the lane-0 queue leaves
   lanes 1-2 free, throughput drops 2-3 %, and the ramp queue that holding creates costs about
   as much as the jam it prevents. What remains of the lever is the ramp being blocked by its
   own jam (profiles 3 and 9).

**Gate G2 (plan step 3).** (a) storage-mandatory regime ≥ 1/3 of profiles: 4/12 — met only by
construction of family v3 (frozen sets: 8/18, 18/30, 7/12). (b) "metering matters": best
constant beats pass-through by ≥ 15 veh h on the loaded profiles: 1/4 (25, 14, 13, 0) — failed.
(c) dynamic schedule vs constants: 0/4 per-profile, 1/4 single-constant — failed, and unlike
v3b (where the hand grid's timing hid a lever the RL policy found: −42 vs constant −90 on T)
there is no hidden lever here: the constant grid itself is flat.
**Consequence:** the largest gain any controller can make over u = 0.9 on this family is
≈ 13 veh h on a third of the profiles, i.e. ≈ 4 veh h per episode on average — the size of the
v3b paired-CI half-width (± 2). A v4 study as designed (9× the v3b SUMO cost) would give a null
result by construction, not because the method fails to transfer. Round 0 was **not** started.

**Options for the user (plan step 4 is on hold).**
- A. Keep v4 as is, run the study anyway: expected A1 ≈ ALINEA ≈ constant within noise; usable
  only as a "where the surrogate does not pay" appendix. Cost: 2 overnights, 9× v3b SUMO time.
- B. v4b, busier ramp: metered ramps in the ALINEA literature carry ≈ 600-1500 veh/h onto
  5000-6000 veh/h freeways (10-25 % share). Raise the ramp family to ≈ 500-1400 veh/h with
  D = 1800-2400 (two-lane meter or two vehicles per green): the lane-0 rule then loads most
  profiles, holding the ramp is expensive and the ramp-blocking cost of a jam scales with r.
  Config-only (family + D + E0 constants), capacity re-sweep 5 min, E0 45 min.
- C. v4c, a real downstream bottleneck (3 → 2 lane drop or speed-limit drop after the merge)
  so a breakdown spreads to all lanes and has a real capacity drop: the textbook setting, but
  builder work (a second lane drop) plus re-sweep and E0, ≈ ½ day, and the ramp is no longer
  the bottleneck.
- D. Family shift only (more loaded profiles): raises the share of profiles where metering
  matters, not the ≤ 20 % per-profile lever. Cheapest, weakest.
Recommendation: B first (E0 decides, gate: best schedule − single constant ≥ 15 veh h on ≥ 1/2
of the profiles), C if B's lever stays below ≈ 30 % of TTS; A only as an appendix after the
v3b seed sweep.
