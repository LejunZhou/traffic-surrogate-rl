# Milestone 5b plan — Reward weight retune (beta sweep)

**Status:** in progress. See `_progress/milestone_5b_progress.md` for the running log.

## Goal
Pick a `beta` (queue-penalty weight in the shaped reward) such that the
PPO-learned policy reliably beats every constant-action baseline,
including `u≡1.0`, which currently wins on seed=0 at the M5 default
`beta=0.1`. Commit the chosen value as the new default in all three
locations that source-of-truth the weights.

## Why this milestone exists
M5 (`fd0e7c8`) shipped a working PPO loop on the shaped reward but
on a single-seed sanity rollout the constant-open policy `u≡1.0`
scored -3758 vs the learned policy's -4440. The corner solution
hasn't been priced out. Diagnostic: queue contribution to total
reward at `u=1.0` is exactly zero, and at the learned policy it's
only ~768 (beta × sum(Q) ≈ 0.1 × 7680). Raising beta makes interior
metering (small queue + slightly lower density) more attractive
relative to either corner.

## Scope
- Sweep **beta ∈ {0.3, 1.0, 3.0}** with **alpha=1.0, gamma=1.0 fixed**.
- 5 PPO seeds per beta = 15 training runs.
- 100,000 timesteps per run, same as M5.
- Evaluate each run's `best_model.zip` against three constant baselines
  (u=0.0, u=0.5, u=1.0) at *its own training beta* (apples-to-apples).
- Pick the smallest beta that meets the acceptance bar.

Out of scope:
- Sweeping gamma (density-std weight).
- Adding a throughput term to the reward.
- Multi-demand episodes.

## Critical files
- `src/rl/train_ppo.py` — add `--seed`, `--reward-beta`, `--run-name-suffix` CLI overrides.
- `src/rl/reward.py` — `RewardWeights` dataclass default; will be updated at the end.
- `configs/rl/ppo_surrogate.yaml` — `env.reward.beta` will be updated at the end.
- `proposal.md` — §"Reward (Phase 1 shaped)" weight line and rationale paragraph.
- `scripts/eval_constant_baselines.py` (NEW) — shared baseline helper.
- `scripts/run_m5b_sweep.py` (NEW) — sweep driver.

## Deliverables
- `_plans/milestone_5b_plan.md` (this file).
- `_progress/milestone_5b_progress.md` — running log.
- `scripts/run_m5b_sweep.py` + `scripts/eval_constant_baselines.py`.
- `runs/ppo/m5b_sweep_<timestamp>/results.csv` (gitignored) and a
  copy at `_progress/m5b_results.csv` for version control.
- Updated defaults in `configs/rl/ppo_surrogate.yaml`,
  `src/rl/reward.py`, `proposal.md`.
- Optional: a single new plot in the progress note showing eval reward
  per beta vs. the constant baselines.

## Acceptance criterion
A beta passes if **both**:

1. The learned policy beats `u≡1.0` on **at least 4 of 5 seeds** (a win =
   strictly higher total deterministic-rollout reward at the training beta).
2. Mean total reward of the learned policy across the 5 seeds is at least
   **5% better** than the best constant baseline at that beta.

If multiple betas pass, prefer the **smallest** (least reward-shaping
intervention; more headroom for Phase 2 generalization).

## Sub-milestones
- **5b.0 — Plan + progress + todos.** This file + skeleton.
- **5b.1 — CLI flags.** Add `--seed`, `--reward-beta`, `--run-name-suffix`
  to `src/rl/train_ppo.py:main()`. Smoke: 1k-timestep run with `--seed
  99 --reward-beta 0.5` and check `config.yaml` snapshot reflects
  the overrides.
- **5b.2 — Baseline helper.** Write `scripts/eval_constant_baselines.py`
  with a `rollout_policy()` function and a CLI. Verify it reproduces
  the M5 progress table (u=0/0.5/1.0 → -7353/-5248/-3758 at beta=0.1,
  within rounding).
- **5b.3 — Sweep driver.** Write `scripts/run_m5b_sweep.py`. Smoke
  with 1 beta × 1 seed × 2k timesteps (~30 s) to verify the subprocess
  loop, results.csv writer, and post-training evaluation against
  baselines all hang together.
- **5b.4 — Full sweep.** 3 betas × 5 seeds × 100k timesteps. Background.
  Expected ~45 min wall clock.
- **5b.5 — Analyze and pick.** Open `results.csv`, apply the acceptance
  bar, log the decision and table into the progress note.
- **5b.6 — Bake defaults.** Update `configs/rl/ppo_surrogate.yaml`,
  `src/rl/reward.py` `RewardWeights.beta`, and `proposal.md` to the
  chosen value.
- **5b.7 — Verification (V1, V2).** See below.
- **5b.8 — Commit.** One commit with everything.

## Verification
**V1.** Re-evaluate the M5-era policy
(`runs/ppo/ppo_surrogate_constant_inflow_20260512_001054/best_model.zip`)
at the new beta. Expected: it scores *worse* than the M5b policy at the
new reward. If not, the retune is doing no work — M5's policy already
optimizes this objective.

**V2.** Deterministic 120-step rollout of the chosen M5b best_model at
seed=0. Expected: action mean shifts **downward** from M5's 0.84 (ramp
held more closed) AND action std stays > 0.1 (still modulates).
Failures:
- If action collapses to u≡1.0: chosen beta is too small.
- If action collapses to u≡0.0: chosen beta is too large.

**Canonical re-run command** (for reproducibility, after defaults are
baked in):
```
python scripts/eval_constant_baselines.py \
  --policy <run_dir>/best_model.zip --seed 0
```

## Open follow-ups
- **M5c — gamma sweep.** Hold beta at the M5b choice, sweep gamma to
  understand the density-uniformity vs density-magnitude trade-off.
  Not in M5b.
- **M5d — throughput bonus.** Add `+delta * throughput` to the reward.
  Requires analytical throughput tracking in both envs (cumulative
  vehicles crossing the downstream boundary). Out of M5b scope.
- **Multi-demand sweeps.** Tied to M3b and Phase 2.
