# Milestone 5c progress

Running log for the nonlinear reward retune (ReLU density + quadratic
queue). See `_plans/milestone_5c_plan.md` for scope, math, and acceptance.

## 2026-05-12 — Kickoff

- Plan file: `_plans/milestone_5c_plan.md` (refined by Ultraplan from
  the in-session design discussion).
- Todo list refreshed for M5c (11 tasks).
- Predecessor: M5b (commit `d7eb732`) — null result showing that
  linear `-β·queue` cannot break the u=1.0 corner regardless of β.
  M5c replaces the reward form, not just the weights.

## 5c.1 — Edit reward.py

- Added two fields to `RewardWeights`: `rho_freeflow=20.0` and
  `queue_norm=100.0`. Updated `from_config` to read them.
- Rewrote `compute_reward` body to:
  ```
  density_excess = max(0, mean(density) - rho_freeflow)
  q_scaled = queue / max(queue_norm, 1e-6)
  return -(alpha * density_excess + beta * q_scaled^2 + gamma * std(density))
  ```
- Module + function docstrings updated to document the M5b→M5c
  motivation (linear queue corner-trap broken by ReLU density +
  quadratic queue).
- All existing validation kept (1-D density, finite, queue ≥ 0).

## 5c.2 — Edit configs/rl/ppo_surrogate.yaml

- `env.reward` block now: `alpha=1.0, beta=1.0, gamma=1.0,
  rho_freeflow=20.0, queue_norm=100.0`. (M5b was alpha=1.0, beta=0.1,
  gamma=1.0; no `rho_freeflow` or `queue_norm` fields.)
- `env.queue_norm_scale=100.0` (obs-side normalizer) left untouched.

## 5c.3 — Edit proposal.md

- Rewrote §"Reward (Phase 1 shaped)" to:
  - Replace formula with the new ReLU + quadratic form.
  - Update default weights line to include the two new thresholds.
  - Replace the empirical-rationale bullet with the M5b → M5c
    corner-trap argument, cross-referencing `_progress/milestone_5b_progress.md`
    §5b.5 and this file.

## 5c.4 — Edit tests/test_surrogate_env.py

- Tightened `test_random_rollout`'s reward-mean bound from
  `(-200.0, 0.0)` to `(-50.0, 0.0)` per plan §4. New comment
  explains the expected magnitude under the M5c formula.
- All seven existing tests left otherwise unchanged.

## 5c.5 — Pre-PPO sanity (3 constant baselines at new defaults, seed=0)

| Policy | total_reward | predicted | match |
|---|---|---|---|
| u=0.0 | -3032.52 | -3084 | within 52 (queue² discrete-sum approx) |
| u=0.5 | **-1240.30** | -1261 | within 21 (winner — matches plan) |
| u=1.0 | -1377.17 | -1380 | within 3 |

**u=0.5 wins as predicted.** The interior basin exists. Plan §6's
acceptance to gate the PPO run is satisfied. Numerical drift vs the
plan table comes from the meshgrid/integration approximation in the
math; rankings are correct.

## 5c.6 — PPO smoke run (5k)

Command:
`python -m rl.train_ppo --config configs/rl/ppo_surrogate.yaml --total-timesteps 5000 --run-name-suffix m5c_smoke`

Run dir: `runs/ppo/ppo_surrogate_constant_inflow_m5c_smoke_20260512_021402/`
- 5280 timesteps in 8 s (~650 fps).
- `ep_rew_mean` ≈ -1770 (much smaller magnitude than M5b's -6110 — the
  M5c reward is intrinsically smaller because the alpha-term rarely
  activates under random actions).
- EvalCallback fired once; pipeline wires end-to-end.

## 5c.7 — PPO full run (100k)

Command:
`python -m rl.train_ppo --config configs/rl/ppo_surrogate.yaml --seed 0 --run-name-suffix m5c_seed0`

Run dir: **`runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_20260512_021425/`**
(background; log mirror at `_progress/m5c_full_run.log`).

- 100,320 timesteps in 185 s (~542 fps).
- `ep_rew_mean`: -1770 → -1530 (~14% better).
- Policy `std`: 0.995 → 0.899 (slow collapse — still actively
  exploring).
- `entropy_loss`: -1.41 → -1.31.

**Eval reward curve (deterministic, 5 episodes, every 4,800 timesteps,
20 evals total):**
- First eval: **-2593.0**
- Last eval: -1617.4
- Best eval: **-1347.8** (saved as `best_model.zip`)
- **Improvement: +975.7 reward, +37.6%** — clearly non-flat learning.

First 5 evals: `[-2593, -2894, -2597, -1494, -2629]` — exploring.
Last 5 evals: `[-1794, -1790, -1635, -1562, -1617]` — settled.

## 5c.8 — Verify learned policy (deterministic eval, seed=0)

Command:
`python scripts/eval_constant_baselines.py --policy runs/ppo/.../m5c_seed0/best_model.zip --seed 0`

Result:
- **total_reward = -1347.83**
- **action mean = 0.688, std = 0.426, min=0.000, max=1.000**
- density mean = 19.14, std = 6.72, min = -0.42, max = 40.24
- queue final = 249.5, max = 249.5

Comparison vs constants at the same weights:

| Policy | reward | gap vs learned |
|---|---|---|
| **learned (best_model)** | **-1347.83** | — |
| u=0.0 | -3032.52 | learned beats by +1685 |
| u=0.5 | -1240.30 | u=0.5 beats learned by -108 (8.7%) |
| u=1.0 | -1377.17 | learned beats by +29 |

## 5c.9 — Optional multi-seed robustness

Skipped per plan §10. Trigger condition was "within 5% of u=0.5";
seed=0 result is 8.7% below u=0.5, outside that threshold. The natural
follow-up is a small multi-seed sweep if we want to rule out
single-seed bad luck — straightforward to invoke as
`scripts/run_m5b_sweep.py --betas 1.0 --seeds 1,2,3,4` (the existing
sweep tool reuses cleanly here).

## Acceptance verdict

**Headline win: the corner trap is broken.** PPO finds a genuinely
interior policy (action mean 0.688, std 0.426, queue final 249.5),
unlike M5/M5b where the policy converged near a corner.

Strict acceptance against plan §"Acceptance":

- `action_mean ∈ (0.05, 0.95)`: **0.688 ✓ PASS**
- `total_reward > max(constant baselines)`: best constant is u=0.5 at
  -1240; learned is -1347.83 → **FAIL by ~8.7%**.

Mixed verdict — the formula change accomplished its main objective
(non-trivial metering policy), but the learned policy on seed=0 sits
slightly below u=0.5. The eval curve is unambiguously not flat
(+37.6%), so it's not a training failure; PPO is exploring a richer
strategy space than constant u=0.5 but hasn't yet converged to a
policy strictly better than that midpoint baseline. Likely causes:
- PPO exploration noise (std 0.426 at end of training is high).
- The deterministic eval of the Gaussian policy's mean isn't
  necessarily the best representable policy under that std.
- Single-seed variance.

The M5c **infrastructure** is sound and the **reward design** is
validated by the constant-baseline ranking. Single-seed PPO falling
slightly short doesn't invalidate either.

## Open follow-ups

- **M5c-seeds**: 3-seed PPO sweep at the M5c reward (one-line invocation
  of `scripts/run_m5b_sweep.py --betas 1.0 --seeds 1,2,3,4`) to see
  whether the u=0.5 gap is seed-specific.
- **M5c-longer-training**: ~250k-step PPO run; the std curve at 100k
  is still 0.899, suggesting room to converge tighter.
- **M5c-ent_coef**: bump `ppo.ent_coef` from 0 to 0.01 to encourage
  cleaner exploration in the mid-action region.
- **M2c — higher demand**: same formula, fresh dataset at 2000 vph.
  The current reward should still apply; expect the optimum action
  mean to shift downward.
