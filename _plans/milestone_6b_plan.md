# Milestone 6b plan — Extended direct SUMO+RL training (100k timesteps)

**Status:** in progress. See `_progress/milestone_6b_progress.md` for the running log.

## Hypothesis (the thing we are testing)

M6 (`f5fdd21`, 20k SUMO timesteps) produced a policy that collapsed to
u≈0 — the closed-ramp corner. The training-log analysis in
`_progress/milestone_6_progress.md` and the side-by-side trajectory
comparison with M5c showed:

- M5c and M6 had **statistically indistinguishable first 42 iterations**
  (both drifted from -1770 → -2030 in ep_rew_mean).
- M5c continued for 209 iterations and started recovering around iter
  100–150, finishing at ep_rew_mean -1530, action mean 0.688.
- M6 was forced to stop at iter 42 by its 20k-timestep budget,
  exactly where M5c was still in the drift phase.

**Hypothesis:** if we give M6 a 100k-timestep budget (matching M5c's
PPO iteration count of ~209), it will escape the u=0 corner and
produce a policy close to M5c's quality. If true, the M5/M6 story
becomes "surrogate is N× faster wall-clock at matched policy
quality" rather than "surrogate produces a better policy than
direct training".

## Scope
- **One change vs M6**: `total_timesteps: 20000 → 100000`.
- **Everything else identical to M6**: same `configs/rl/ppo_sumo.yaml`
  reward block, same PPO hyperparameters, same SumoEnv, same seed=0.
- 100k SUMO timesteps at the benchmarked ~104 ms/step (uncontested)
  ≈ **2.9 hours wall clock**.

## Critical files
- `configs/rl/ppo_sumo.yaml` — used as-is; we only override
  `--total-timesteps 100000` on the CLI.
- `src/rl/train_ppo.py` — unchanged.
- Run dir: `runs/rl/ppo_sumo_constant_inflow_m6b_seed0_100k_<timestamp>/`.

## Deliverables
- `runs/rl/.../m6b_seed0_100k_*/final_model.zip` (and periodic
  checkpoints from CheckpointCallback every 2400 timesteps).
- `runs/rl/.../m6b_seed0_100k_*/monitor/train.csv` — per-rollout
  ep_rew_mean trajectory (matches M5c's data shape for direct overlay).
- `runs/rl/.../m6b_seed0_100k_*/wall_clock.json` — actual wall-clock
  for the headline "~3 hours" claim.
- `_plans/milestone_6b_plan.md` (this file).
- `_progress/milestone_6b_progress.md` — running log + analysis.

## Sub-milestones
- **6b.0 — Plan + progress.** This file + skeleton.
- **6b.1 — PPO 100k SUMO run.** Background. ~3 hours wall clock.
- **6b.2 — Native eval.** `scripts/eval_sumo_baselines.py --policy <run>/final_model.zip --seed 0`.
- **6b.3 — Side-by-side iteration trajectory.** Extract per-iteration
  `ep_rew_mean` from M5c (already saved), M6 (already saved), and
  M6b (new). Same axes. Confirms or refutes the "drift then escape"
  pattern at the same iteration count.
- **6b.4 — Refresh the M5/M6 comparison table** in
  `_progress/milestone_6_progress.md` §6.7 with the new M6b numbers
  alongside M6's 20k results.

## Acceptance criteria (hypothesis test)

**Strong pass** — extended M6 escapes the corner:
- Final policy action mean ∈ (0.2, 0.95) (genuinely interior, not the
  u=0 cluster M6 collapsed to).
- Final SUMO eval reward > -1700 (M5c-transferred-to-SUMO got -1525;
  M6@20k got -2919; an escape would land in the M5c neighborhood).
- ep_rew_mean trajectory shows the same "drift then recover" pattern
  M5c exhibited around iter 100–150.

**Weak pass** — M6 escapes partially:
- Action mean somewhere in (0.05, 0.5) — not corner but not yet at
  M5c quality.
- Reward improvement vs M6@20k is at least 30% (i.e., better than
  -2300 reward).

**Fail** — M6 stays stuck at u=0:
- Final action mean ≤ 0.05 OR reward ≤ -2700.
- Would mean the hypothesis is wrong and SUMO PPO has a deeper
  optimization issue than just "needs more updates". Calls for an
  M6c follow-up (entropy bonus, different init, or reward sweep on
  SUMO).

## Wall-clock budget and execution

- ~3 hours uncontested (104 ms/step × 100k = 10,400 s = 2.9 hours).
- The earlier M5c 3-seed sweep died after seed=1 (verified via
  empty stdout file + no python processes); CPU is free for M6b.
- Background launch. Will get notified when complete.
- No other runs queued.

## Verification (post-run)

1. `scripts/eval_sumo_baselines.py --policy <run>/final_model.zip --seed 0`
   → record total_reward, action mean/std, density mean, queue final.
2. Compare against the existing M6 (20k) and M5c-transferred numbers
   in `_progress/milestone_6_progress.md` §6.7.
3. Inspect `monitor/train.csv` for the ep_rew_mean trajectory; check
   whether the iter-100+ recovery actually happened.
4. Update M6 progress §"Open follow-ups" to either close the
   "extended M6" item (hypothesis confirmed) or open M6c (hypothesis
   refuted).

## Out of scope

- Multi-seed M6b (one seed first to test the hypothesis; multi-seed
  is M6c).
- Changing PPO hyperparams (would confound the comparison).
- Tuning the reward (we just spent M5b/M5c on that; want to keep it
  fixed for the cross-env comparison).
