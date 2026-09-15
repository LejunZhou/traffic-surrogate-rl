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
14. M14 Scenario v3 — `m14_scenario_v3_plan.md`: realistic metered ramp (60 km/h, stop
    line 100 m before the nose, 28-vehicle storage cap enforced as an actuator constraint
    in both envs); implemented and tested 2026-09-14, E0 characterised; study re-run pending
    the user's go.
