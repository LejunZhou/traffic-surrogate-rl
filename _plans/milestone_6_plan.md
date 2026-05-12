# Milestone 6 plan — Direct SUMO+RL training (M5c reward)

**Status:** in progress. See `_progress/milestone_6_progress.md` for the running log.

## Goal
Train PPO directly against `SumoEnv` (live TraCI-driven SUMO simulation),
using the **same M5c shaped reward** that M5c PPO used. Produce a SUMO-
native policy that can be compared to the M5c surrogate-trained policy
on identical evaluation conditions, finally measuring the
**surrogate-vs-SUMO transfer gap** that motivated the whole project.

## Why M6 matters
Phase 1 of the proposal asks two questions:
1. Can a surrogate model accelerate RL training while remaining accurate?
2. Does a surrogate-trained policy transfer to SUMO?

M5c answered (1) by producing a non-trivial PPO policy on the surrogate
in ~3 minutes. M6 produces the comparable SUMO-trained baseline so we
can quantify the transfer gap.

## Scope
- Same scenario: `configs/sumo/phase1_1.yaml` (1-lane mainline, ramp at
  1300 m, 19 detectors, 1500 vph mainline / 800 vph ramp).
- Same reward: M5c ReLU+quadratic form, weights
  `alpha=1, beta=1, gamma=1, rho_freeflow=20, queue_norm=100` (the
  defaults baked in to `RewardWeights` and `reward.py`).
- Same PPO hyperparams as M5c (`MlpPolicy`, `n_steps=480`, `batch=120`,
  `lr=3e-4`, `gamma=0.99`, `gae=0.95`, `clip=0.2`).
- Env: `SumoEnv` from `src/rl/sumo_env_wrapper.py` (already wilson-
  implemented in commit `f2bb92f`; reward call site retrofitted with the
  shaped reward in `b021df7` and the new M5c formula flows through
  automatically via `RewardWeights.from_config`).
- Density normalization: `density_mean=18.730, density_std=5.971` from
  `data/splits/metadata.json` (so SumoEnv's z-score obs matches the
  M5c surrogate-trained policy's input distribution — needed for the
  transfer comparison).

Out of scope:
- Multi-demand training.
- New reward forms — M6 is purely the SumoEnv counterpart of M5c.
- Hyperparameter tuning beyond reasonable defaults.

## Deliverables
- `configs/rl/ppo_sumo.yaml` (new) — SUMO env config + PPO + reward.
- `runs/rl/ppo_sumo_<timestamp>/` — SB3 checkpoints, monitor CSV,
  TensorBoard logs, wall-clock JSON.
- `_plans/milestone_6_plan.md` (this file).
- `_progress/milestone_6_progress.md` — running log.
- Optional: small extension to `scripts/eval_constant_baselines.py` (or
  a sibling `eval_sumo_policies.py`) to evaluate learned/constant
  policies against SumoEnv for the transfer comparison. Decide during
  step 6 below — may not be needed if a one-off script suffices.

## Wall-clock budget
SUMO+RL is *much* slower than surrogate+RL:
- Each episode = 3600 s sim, 30 s control step, 120 control steps.
- TraCI overhead per simulation step (1 s) is roughly 1–10 ms, plus the
  per-control-step inserts/detector reads in SumoEnv's
  `_advance_control_interval` (30 sim steps per control step).
- Rough estimate: 5–30 seconds wall-clock per episode.
- 100k control timesteps = ~833 episodes = **1.2–7 hours wall-clock**.

This is too long for a single session. We will:
1. **Benchmark a single episode first** (step 5b.0 below) to get a
   real wall-clock number.
2. **Right-size `total_timesteps`** so the run fits in a reasonable
   window (~1–2 hours).
3. **Accept that M6 may not converge as deeply as M5c** at this budget.
   That's a real finding (surrogate is faster) and is exactly what the
   research question wants quantified.

## Sub-milestones
- **6.0 — Plan + progress + config.** This file + skeleton + `ppo_sumo.yaml`.
- **6.1 — Single-episode benchmark.** Run one SUMO episode at u=0.5
  via SumoEnv directly (no PPO), time it. Tells us the per-episode
  wall-clock and surfaces any TraCI setup issues before PPO is
  involved.
- **6.2 — PPO smoke.** Very small `--total-timesteps` (e.g. 600 = ~5
  episodes). Verifies the SumoEnv path through `train_ppo.py` wires
  up, checkpoints save, no TraCI session leaks.
- **6.3 — PPO full.** Right-sized `total_timesteps` per the
  benchmark. Run in background. Expected wall time will dominate this
  milestone.
- **6.4 — Native eval.** Deterministic eval of `best_model.zip`
  against SumoEnv. Records per-step density / queue / reward and
  action trace.
- **6.5 — Transfer eval.** Roll out the **M5c surrogate-trained
  policy** (`runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_*/best_model.zip`)
  against SumoEnv at the same seed. This measures whether the
  surrogate-trained policy actually controls SUMO traffic the way it
  controls the surrogate. **The headline number is the per-step reward
  gap between native-SUMO and transferred-from-surrogate policies.**
- **6.6 — Constant baselines on SUMO.** Run u=0.0/0.5/1.0 through
  SumoEnv (not the surrogate) for the same seed. The reward values
  may differ slightly from the surrogate baselines because SUMO and
  the DeepONet disagree on density.
- **6.7 — Comparison table + commit.**

## Acceptance criteria
- Pipeline-level: the M6 PPO loop runs end-to-end without TraCI
  session leaks, with `best_model.zip` and `final_model.zip` written.
- Native-eval policy at seed 0 has action mean strictly in (0.05, 0.95)
  (i.e. genuinely interior, not corner) AND total reward > best constant
  baseline at the same seed in SumoEnv.
- Transfer-eval table is populated with finite numbers for native,
  transferred-from-surrogate, and three constants — even if the
  transferred policy underperforms, that's still a recordable result.
- The progress note documents the wall-clock per episode and per
  iteration explicitly, so future readers know the cost.

## Critical files
- `src/rl/sumo_env_wrapper.py` — unchanged, already shaped-reward-aware.
- `src/rl/train_ppo.py` — unchanged, env-type dispatcher already
  supports `env.type=sumo`.
- `src/rl/reward.py` — unchanged after M5c (M5c defaults are what we want).
- `configs/sumo/phase1_1.yaml` — scenario source of truth, unchanged.
- New: `configs/rl/ppo_sumo.yaml`.
- Optional new: `scripts/eval_sumo_baselines.py` if we need a SUMO-side
  rollout helper (the existing `scripts/eval_constant_baselines.py` is
  SurrogateEnv-only).

## Verification
- `bash scripts/eval_in_sumo.sh` or its equivalent runs the SumoEnv
  side of the eval and produces a `metrics.json` per policy.
- Side-by-side table in `_progress/milestone_6_progress.md` with the
  three constant baselines, the SUMO-native learned policy, and the
  transferred M5c surrogate-learned policy.

## Open follow-ups
- **Multi-seed M6 sweep**: same machinery as M5b/M5c (the
  `run_m5b_sweep.py` driver already supports `env.type=sumo` since
  `train_ppo.py` does the dispatch). Hold off — single SUMO seed will
  already cost ~1–2 hours; multi-seed is many hours.
- **M7 — full comparison study**: density / throughput / queue / wall-
  clock trade-off plot across both training paths. Plus learning-curve
  comparison (per-iteration eval reward) so the "surrogate accelerates"
  hypothesis can be answered quantitatively.
- **M2c — higher demand**: same M6 setup at 2000 vph or higher would
  better stress-test ramp metering. M6 closure on 1500 vph is the
  prerequisite.
