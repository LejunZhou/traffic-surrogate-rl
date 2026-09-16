# M15 plan — scenario v4: three-lane mainline (drafted 2026-09-15, user decision: before the seed sweep)

## Why
The v3b result (M14 §14: A1 ≈ −41 on T, +5 over tuned PI-ALINEA, +9 over direct PPO at equal
budget, seed 0) is on a single-lane mainline where the ramp is ≈ 25 % of the merge flow. A
reviewer's first question is whether the surrogate-RL advantage survives a realistic freeway:
three lanes, lane changing, a ramp that is ≈ 10 % of the flow, and a merge bottleneck whose
capacity drop is what metering protects. The ramp-metering literature (ALINEA, PI-ALINEA,
the METANET benchmarks) is on 3-lane freeways, so v4 is also the setting that makes the
ALINEA comparison standard. The question M15 answers: **does the mechanism (plant DeepONet
→ aggregation → policy that beats tuned ALINEA at a fixed SUMO budget) transfer to v4?** The
seed sweep on v3b (seeds 1–2) is deferred, not dropped; if v4 transfers, the sweep is done
on v4.

## What transfers untouched (audited 2026-09-15)
- **Code paths are lane-agnostic.** `network_builder` takes `network.num_lanes` (mainline
  n lanes, acceleration edge n + 1, zipper drop; mainline `departLane random`); detectors
  place one loop per lane per station; `density_from_loops` averages lanes per station and,
  with `merge_station_lanes: mainline`, drops the acceleration lane at the merge station.
  Observation (19 station densities + demand features), DeepONet targets (19 station
  densities + exit flow) and the reward keep their shapes. M1 §1.1 already validated a 2-lane
  zipper merge (no teleports); the 3-lane net is the same builder with n = 3.
- **The study driver** (`run_v3b_study.sh` with `SCENARIO=v4`: `env_v4.yaml`,
  `round0_v4.yaml`, `scenario_v4.yaml`, `PROFILE_SETS_DIR`), the meter-aware E0 script, the
  ALINEA tuner (per-lane density targets), the resume logic, the evaluation and figures.
- **Nothing trained transfers**: plant ensemble, tuned ALINEA, E0 numbers, policies are all
  scenario-specific and are regenerated.

## What must change (all configuration, one audit list)
| item | v3b value | v4 | why |
|---|---|---|---|
| `network.num_lanes` | 1 | 3 | the scenario |
| merge station density | through lane only | mean of the 3 through lanes (code already does this for n > 1) | one field per station |
| demand family `family_v3.yaml` | mainline 1200–2300, ramp 200–800 | mainline scaled to the measured v4 capacity (expect base 3600–4800, peaks to ≈ 0.95 × C, C ≈ 6000–6600); ramp unchanged 200–800, surge cap 800 | same 45-min timing rules as family v2 |
| storage-mandatory threshold (`STORAGE_NEEDED_VPH`, `enforce_storage_mandatory`) | 2500 | ≈ C from E0 | gate definition |
| E0 feedforward capacities, insertion rate | 2300 / 2400 / 2500, 1040 | C − 200 / C − 100 / C, ramp-max ≥ any r_k (unchanged 1040 if ramp family unchanged) | meter-aware grid, rates in veh/h |
| observation `demand_norm` | 2500 | ≈ C | z-scores / normalised demand features |
| surrogate normalisers `mainline_demand`, `flow` | 2500, 2500 | ≈ C, ≈ C | inputs / outputs in [0, 1] |
| reward `q_ref` / `q_cap` | 2970 (1-lane IDM) | ≈ 3× | only if the outflow term is used (TTS form: unaffected) |
| ALINEA grid `--rhos` | 26–38 veh/km/lane | same (per-lane) | density is per lane |
| meter discharge D | 1200 | 1200 | one-lane ramp, unchanged |
| ramp entry angle | 10° | 10° | M14 §13 |

Two design decisions to take with the user before E0:
1. **Observation at the merge: lane-averaged only, or lane-averaged + lane-0 density at
   stations 11–13?** The ramp shock starts in lane 0; the lane average dilutes it by 3. The
   DeepONet targets would stay lane-averaged either way. Recommendation: start lane-averaged
   (nothing changes downstream), add the lane-0 channel only if E0 / the plant model show
   the merge onset is invisible in the average.
2. **Lane-change model**: SUMO default LC2013 with default parameters (no cooperative
   speed-gain tuning). State it and keep it; it is the stochasticity the study should face.

## Expected differences from v3b (hypotheses, to be checked by E0)
- Capacity ≈ 3 × 2100–2200 veh/h at the drop; the merge bottleneck is weaker relative to the
  ramp (lane changes away from lane 0 absorb inflow), so the storage-mandatory regime needs
  mainline peaks near capacity: the family's peaks must be drawn closer to C than v2's.
- The seed-to-seed noise floor rises (lane changing), so the plant model's error floor and
  the calibration slope change; the round-0 gate thresholds (10 % return error) may need the
  floor re-measured (M9 §8 procedure, 60 rollouts).
- Wall time ≈ 3× per SUMO episode (3× vehicles): E0 ≈ 1 h, round-0 store ≈ 1.5–2 h, an
  aggregation round's 54 evaluations ≈ 15 min, **direct PPO 700 EE ≈ 18 h** — start it in the
  background on day one.

## Steps, gates, budgets
1. **Scenario + configs + tests** (½ day): `scenario_v4.yaml`, `env_v4.yaml`,
   `round0_v4.yaml`, `family_v3.yaml` with placeholder capacity, `tests/test_scenario_v4.py`
   (3-lane geometry, 4-lane acceleration edge, one loop per lane, merge-station averaging,
   ramp insertion at position 0, no teleports in a 10-min loaded run). Layout figure.
   **Gate G1**: no insertion failures or teleports at mainline 5400 + ramp 800; nose speed ≈
   100 km/h; the E0 insertion check served fraction ≥ 0.9.
2. **Capacity + noise floor** (1–2 h): constant-demand sweep to locate C at the drop with
   the 10° merge; 20 profile × u × 3-seed rollouts for the noise floor. Then freeze
   `family_v3.yaml` (peaks in [0.8 C, 1.05 C]), the storage threshold, the E0 constants and
   the normalisers.
3. **E0 on v4** (≈ 1 h): 12 profiles × (11 constants + 21 schedules) + insertion check.
   **Gate G2**: a storage-mandatory regime exists (≥ 1/3 of profiles); metering matters (best
   constant beats pass-through by ≥ 15 veh h on the loaded profiles); the single-constant and
   per-profile margins reported as in M14 §12. If no dynamic schedule beats the constants,
   proceed anyway (v3b showed the RL policy finds what the hand grid does not) but record it.
4. **Round-0 store + ensemble + gate** (≈ 5 h): 288 mixture + E0 rollouts, 5 members × 300
   epochs. **Gate G3**: return error ≤ 10 %, false breakdown ≤ 10 %, slope in [0.5, 2] on val
   and test; if the noise floor from step 2 is above the v3b level, report the error relative
   to the floor as well.
5. **Study, seed 0** (overnight): aggregation 5 rounds with δ 2 / patience 2, direct PPO 200
   and 700 EE (700 started first), ALINEA tuning on V, final evaluation on T and O, figures.
   **Gate G4 (the transfer question)**: A1 vs tuned ALINEA and vs direct PPO at equal budget,
   paired CIs, on T and O. Success = the same ordering as v3b; a null result (A1 ≈ ALINEA) is
   reportable and would redirect the paper to "where the learned surrogate pays off".
6. **Then** the seed sweep (seeds 1–2) on whichever scenario carries the paper.

Total ≈ 2 working days plus two overnight runs; ≈ 3000 SUMO episodes.

## Out of scope / deferred
- On-ramp on the middle or a two-lane ramp; multiple ramps; HOV lanes.
- Per-lane DeepONet targets (3 × 19 fields): only if the lane-averaged model fails G3.
- `proposal.md`: v4 is a scope addition beyond the current proposal; a revision entry
  (scenario family, budgets) needs the user's approval before it is written (CLAUDE.md).
