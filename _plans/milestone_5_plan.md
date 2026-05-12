# Milestone 5 plan — PPO surrogate training (wilson's scenario + shaped reward)

**Status:** in progress. See `_progress/milestone_5_progress.md` for the running log.

## Goal
Train a PPO policy on `SurrogateEnv` (from M4) with the shaped reward
(`b021df7`) and confirm the eval reward curve **is no longer flat** —
the key failure of the prior MVP that motivated the reward redesign.

## Scope
- Env: `SurrogateEnv` loaded from the M3 checkpoint, single demand
  (1500 vph), shaped reward weights `alpha=1.0, beta=0.1, gamma=1.0`.
- Policy: SB3 `MlpPolicy`, defaults (n_steps=480, batch=120, n_epochs=10,
  lr=3e-4, gamma=0.99, gae=0.95, clip=0.2).
- 100,000 total timesteps, eval every 4,800, 5 eval episodes.
- Output to `runs/ppo/ppo_surrogate_constant_inflow_<timestamp>/`.

Out of scope:
- SumoEnv training (wilson's `train_ppo.py` already has the SUMO branch;
  comparison is M6 territory).
- Multi-demand sweeps.
- Hyperparameter tuning beyond reasonable defaults.

## Deliverables
- `src/rl/train_ppo.py` updated: support both `env.type=surrogate` and
  `env.type=sumo`. Keep wilson's SUMO branch intact.
- `configs/rl/ppo_surrogate.yaml` — env config + PPO hyperparams +
  reward weights.
- `runs/ppo/.../best_model.zip`, `final_model.zip`, `config_snapshot.yaml`,
  `wall_clock.json`, `evaluations.npz`, `monitor/*.csv`, TensorBoard logs.
- `_plans/milestone_5_plan.md` (this file).
- `_progress/milestone_5_progress.md` — running log.

## Sub-milestones
- **5.0 — Trainer refactor.** Split `_build_env` to dispatch on
  `env.type`. The SUMO branch keeps wilson's exact wiring (Monitor,
  CheckpointCallback, etc.); the surrogate branch builds SurrogateEnv,
  wraps with Monitor + DummyVecEnv, and registers EvalCallback to save
  best_model.zip.
- **5.1 — Smoke run.** `--total-timesteps 5000` to verify the surrogate
  branch wires through and EvalCallback fires.
- **5.2 — Full run.** 100k timesteps. Capture train + eval reward curves
  and wall time.
- **5.3 — Sanity diagnostics.** Deterministic eval rollout on
  `best_model.zip`: log per-step action, density (mean / std), queue,
  reward. Confirm the policy is no longer u≡0 (the old MVP's degenerate
  optimum); expect mid-range actions that balance density vs. queue.

## Acceptance criteria
- Smoke run completes without errors; checkpoints written.
- Full run completes; `evaluations.npz` shows a non-flat eval reward
  curve (improvement > 5% over first eval).
- Deterministic eval rollout produces an action sequence with mean ≠ 0
  and std > 0 (the policy actually modulates ramp metering).
- Queue and density stay in a physically plausible range throughout
  the eval rollout (density ≥ 0, queue ≥ 0).

## Open follow-ups
- **M6 — SUMO comparison.** Run `train_ppo.py` with `env.type=sumo`
  using the same shaped reward, then compare against M5's policy on
  the same demand. The two should converge to similar policies; if
  they diverge significantly, the gap localizes the distribution shift.
- **Reward-weight sweep.** With the shaped reward landing in M5, we
  can now sweep α/β/γ to understand the trade-off surface.
- **Initialization-aware eval.** Log a pre-training eval at iteration 0
  so the eval-reward curve always starts from the random-policy
  baseline. Avoids the "starts at optimum" diagnostic ambiguity from
  the old MVP.
