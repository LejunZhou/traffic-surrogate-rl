# Milestones overview plan

Master roadmap for the Phase 1 pipeline. Individual milestones get their own
plan files under `_plans/` as they are scoped in detail; this file is the
index.

## Current milestone order
1. Minimal SUMO network + rollout script
2. Dataset generation pipeline
3. Baseline DeepONet training
4. Gymnasium-compatible surrogate environment
5. PPO training
6. Evaluation in SUMO
7. Outflow-based reward for SUMO+PPO (`milestone_7_plan.md`, 2026-08-27):
   replace the M5c density-ReLU term with a direct mainline-outflow term,
   balance the three terms from a constant-u sweep, retrain SUMO+PPO at
   the current 2800 vph total demand
8. Comparison study and plots (previously item 7)

## Next phase M8–M12 (draft 2026-09-09, implemented 2026-09-13)
`draft_pipeline.md` (repo root) proposes M8–M12: time-varying mainline /
ramp demand profiles, a plant-model DeepONet (inflows → density + outflow,
causal branch, ensemble), a budgeted data-aggregation loop, unified PPO
configs, and a SUMO-episode ledger for the sample-efficiency comparison
against direct SUMO PPO, ALINEA and a surrogate-based MPC.

8. M8 Scenario v2 — `m8_scenario_v2_plan.md`: profiles, occupancy density, E0 (done)
9. M9 Plant model v2 — `m9_plant_model_v2_plan.md`: round-0 mixture data, GRU-branch
   DeepONet ensemble, round-0 gate, E2 parity, E1 study (done 2026-09-13)
10. M10 Surrogate RL — `m10_surrogate_rl_plan.md`: SurrogateVecEnv, ppo_common,
    aggregation loop (done 2026-09-13, demo scale)
11. M11 Reference arms — `m11_reference_arms_plan.md`: direct SUMO PPO, A2 fine-tune,
    ALINEA tuning on profiles, Surrogate-MPC, single-surrogate, one-step (done 2026-09-13)
12. M12 Study — `m12_study_plan.md`: final evaluation on T and O, figures 1–7, README
    (done 2026-09-13 at demo scale; paper-scale seeds pending)
13. M13 Round-0 budget — `m13_round0_budget_plan.md`: 240-rollout round 0 with the
    original vs a closed-loop-heavy mixture (no run-7 policy), aggregation until the
    stop rule; answers how small the entry fee of the surrogate arm can be
    (done 2026-09-14; see progress §4 for the follow-up run)

Draft decisions taken without the user (flagged in the progress files):
TTS-form training reward (E0 finding), GRU branch by default (CPU speed),
occupancy divisor 5 m + jam clip. The proposal.md revision (Appendix D.1) still
needs approval.
14. M14 Scenario v3/v3b — `m14_scenario_v3_plan.md`: realistic metered ramp. v3 (60 km/h
    ramp, mid-ramp stop line, 28-veh cap) superseded by v3b: 200 m acceleration ramp at
    120 km/h joining at 10°, meter at its start (D = 1200 veh/h), uncapped virtual queue,
    through-lane merge density, demand family v2; implemented and tested 2026-09-14/15,
    E0 characterised (gate still fails on step profiles: grid timing). Study run 2026-09-15
    (10° ramp entry, seed 0, 3 rounds, demo scale): round-0 gate passed both splits; A1
    aggregation ≈ −41 on T from 854 EE on (5 rounds, stop rule fired at round 5, plateau holds on
    T/O) vs tuned PI-ALINEA −46.4 (+5 [+3, +7]), ≈ +7 on O;
    direct PPO −50.2 at 855 EE (8.6 behind A1 at equal budget) and −74.4 at 290 EE. Study
    complete, progress §14. Open: seeds 1–2, paper scale.
15. M15 Scenario v4 (three-lane mainline) — `m15_scenario_v4_three_lane_plan.md`: does the
    v3b result transfer to a 3-lane freeway (LC2013 defaults, lane-averaged observation)?
    2026-09-15: user decisions taken (observation lane-averaged, LC2013 defaults), v3b seed
    sweep deferred. Capacity sweep: mainline alone ≈ 6150 veh/h, merge breaks down at a
    lane-0 load d/3 + r ≈ 2450 (same per-lane capacity as v3b) → lane-aware storage rule and
    feedforward, breakdown threshold as a scenario parameter (30 on the lane mean); configs
    scenario_v4 / family_v3 (busier ramp: base 400–600, surge cap 800) / sets v3 / round0_v4 /
    env_v4 / plant_v4, tests. E0 on v4 (progress §4): gate failed — metering has almost no
    lever on v4 (constant grid flat on 8/12 profiles, ≤ 28 veh h on the loaded 4; v3b: 53 veh h
    on 8/12) because the lane-0 jam leaves the other lanes free. Round 0 not started; user to
    choose: busier ramp (v4b, recommended), downstream bottleneck (v4c), run anyway, or family shift.
