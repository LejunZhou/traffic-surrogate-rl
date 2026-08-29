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
