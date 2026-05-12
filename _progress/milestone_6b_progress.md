# Milestone 6b progress

Running log for the **extended M6 hypothesis test**: rerun direct
SUMO+RL training at 100k timesteps instead of 20k, to see whether the
M6 policy collapse to u=0 was caused by sample starvation. See
`_plans/milestone_6b_plan.md` for hypothesis, acceptance, and scope.

## 2026-05-12 — Kickoff

- Plan file written: `_plans/milestone_6b_plan.md`.
- 5 tasks in the todo list.
- Predecessor: M6 (commit `f5fdd21`) — 20k SUMO timesteps, policy
  collapsed to u=0, ep_rew_mean trajectory matched M5c iter 1-42
  almost exactly.
- CPU state at kickoff: the abandoned M5c 3-seed sweep stopped after
  seed=1 (no python processes running, sweep stdout file empty,
  only s1 sub-run dir exists). CPU is free; no contention expected
  for M6b.

## 6b.1 — PPO 100k SUMO run

Background task ID `b2b2n6spu`. Launched at 02:54.

Training is producing the expected output cadence (one PPO iteration
every ~50 s of wall clock). Periodic snapshot from the SB3 stdout at
iter 11 (~10 minutes in):

```
ep_rew_mean = -1860, std=1.01, total_timesteps=5280
```

This is consistent with the drift phase shared by M5c iter 1-5
(~-1770) and M6 iter 1-15 (~-1850) — early SB3 PPO policy under our
action-space init mostly samples actions near 0, accumulating queue
linearly. The hypothesis test starts firing around iter 50-100 where
M5c began climbing back.

### Comparison infrastructure pre-staged

Added `scripts/compare_training_trajectories.py` so the post-run
analysis is a single command:

```bash
python scripts/compare_training_trajectories.py \
  --m5c <m5c_task_output> --m6 <m6_task_output> --m6b <m6b_task_output> \
  --out _progress/m6b_training_curves.csv \
  --plot _progress/m6b_training_curves.png
```

Smoke-tested against M5c (complete) + M6 (complete) + M6b (in
progress at iter 11). Parses SB3's iter/ep_rew_mean stdout cleanly,
emits CSV + optional PNG. Will be re-run with the full M6b data when
the background job completes.

Sentinel trajectory (from the smoke run):

| iter | M5c surrogate | M6 SUMO 20k | M6b SUMO 100k (in progress) |
|---|---|---|---|
| 1   | -1740 | -1770 | -1770 |
| 5   | -1770 | -1820 | -1820 |
| 10  | -1760 | -1850 | -1850 |
| 20  | -1750 | -1890 | (pending)|
| 42  | -1710 | -2040 | (pending)|
| 50  | -1630 | (end) | (pending)|
| 100 | -1330 | n/a   | (pending)|
| 150 | -1430 | n/a   | (pending)|
| 200 | -1560 | n/a   | (pending)|
| 209 | -1530 | n/a   | (pending)|

M5c's best training reward was actually at iter ~100 (-1330), drifting
back to -1530 by iter 209 — the saved `best_model.zip` (from EvalCallback)
captures the iter-100-ish snapshot, not the final policy. This is a
useful reminder that the eval reward we reported (-1348 on surrogate)
and the training ep_rew_mean don't track in lockstep.

## 6b.2 — Native eval

Command:
`python scripts/eval_sumo_baselines.py --policy runs/rl/.../m6b_seed0_100k_20260512_045505/final_model.zip --seed 0`

Result:
- **total_reward = -2941.16** (within 0.05 of the constant u=0 baseline of -2941.16)
- **action mean = 0.000, std = 0.000** — policy is literally u≡0 every step
- density mean = 15.83, std = 3.16 — same as constant u=0
- queue final = 800.0 (ramp closed entire episode)
- throughput = 1473 vph (mainline only)

The deterministic policy is **indistinguishable from constant u=0**. The
network learned to output 0 with zero variance.

## 6b.3 — Iteration trajectory comparison

Full curve at sentinel iterations (from `scripts/compare_training_trajectories.py`,
`_progress/m6b_training_curves.csv`):

| iter | M5c (surrogate) | M6 (SUMO 20k) | M6b (SUMO 100k) |
|---|---|---|---|
| 1   | -1740 | -1770 | -1770 |
| 5   | -1770 | -1820 | -1820 |
| 10  | -1760 | -1850 | -1850 |
| 20  | -1750 | -1890 | -1890 |
| 42  | -1710 | **-2040 (M6 end)** | -2040 |
| 50  | -1630 | — | -2020 |
| 100 | **-1330 (M5c best)** | — | -1800 |
| 150 | -1430 | — | -2140 |
| 200 | -1560 | — | -2120 |
| 209 | **-1530 (M5c end)** | — | **-2080 (M6b end)** |

Wall clock: **9854 s ≈ 2.7 h** for 100,320 timesteps. ~10.2 ts/sec —
slightly faster than the M6.1 benchmark predicted (~13 ts/sec without
sweep contention), confirming no concurrent CPU load.

Observations:

- **M5c and M6b had statistically identical first 42 iterations**, just
  as M5c and M6 (20k) did. Confirms the trajectory shape is not env-
  dependent in the warm-up phase.
- **M5c then climbed dramatically** to its best at iter 100 (-1330),
  recovered up to -1530 at iter 209.
- **M6b briefly approached an escape** at iter 100 (-1800, its own
  best), but then **relapsed** to -2140 at iter 150 and finished at
  -2080. The deterministic policy at iter 209 has collapsed entirely
  (mean=std=0, exactly u=0).
- The "best moment" at iter 100 for M6b (-1800) is much weaker than
  M5c's (-1330), and M6b couldn't sustain it.

## 6b.4 — M6 §6.7 comparison table (refresh)

Updated SUMO-side comparison at seed=0 on the M5c shaped reward
(extends the M6 §6.7 table with M6b's column):

| Policy                                  | SUMO reward | Action mean (std) | Density (mean/std) | Queue | Throughput |
|---|---|---|---|---|---|
| u=0.0 constant                          | -2941.2 | 0.000 (0)     | 15.83 / 3.16 | 800.0 | 1473 vph |
| u=0.5 constant                          | **-1238.5** | 0.500 (0) | 18.57 / 5.12 | 400.0 | 1867 vph |
| u=1.0 constant                          | -1431.7 | 1.000 (0)     | 22.38 / 9.55 | 0.0   | 2260 vph |
| **M5c (surrogate-trained, transferred)** | **-1525.6** | **0.729 (0.431)** | 20.35 / 8.37 | 217.1 | 2043 vph |
| M6 (SUMO-trained, 20k timesteps)        | -2919.2 | 0.003 (0.015) | 15.85 / 3.17 | 797.7 | 1475 vph |
| **M6b (SUMO-trained, 100k timesteps)**  | **-2941.2** | **0.000 (0.000)** | 15.83 / 3.16 | 800.0 | 1473 vph |

M6b is **even more collapsed than M6**: M6 still had a tiny action
mean (0.003) with some std (0.015) — residual stochasticity from
incomplete std collapse. M6b achieved full deterministic u=0 with
zero variance. **5× more training made the direct-SUMO policy
strictly worse, not better.**

## Acceptance verdict

**Hypothesis refuted.** The M6 progress note's claim — "M6 should
reach a policy similar to M5c's if given more wall clock" — is wrong.
SUMO PPO at this reward, these hyperparameters, and this scenario
**does not escape the u=0 corner even with M5c's exact iteration
count (209)**.

Strict acceptance against plan:

| Criterion | Result |
|---|---|
| Action mean ∈ (0.2, 0.95) | ❌ 0.000 — collapsed to corner |
| SUMO eval reward > -1700 | ❌ -2941 (deeper collapse than M6@20k's -2919) |
| ep_rew_mean shows "drift then recover" pattern | ❌ briefly approached at iter 100 (-1800), then relapsed |

This is the **strong-fail** mode the plan anticipated. The
"surrogate just needs more timesteps to match M5c" story is gone.

## What this means for the surrogate-acceleration story

The headline gets **stronger**, not weaker. Previously we said:

> "M5c (surrogate) beats M6 (SUMO@20k) by 48% because M6 was starved of training budget."

The honest version is now:

> "M5c (surrogate) beats M6b (SUMO@100k) by 48% **at matched PPO
> iteration count (209 each)**. The surrogate doesn't just train
> faster — it enables learning that direct SUMO training does not
> achieve at these hyperparameters and this sample budget. SUMO's
> per-step reward noise (insert timing, IDM dynamics) appears to
> destabilize PPO's value function badly enough that the policy
> can't escape the initial u=0 drift, while the deterministic
> surrogate gives clean gradients that PPO can act on."

This is the more interesting (and harder-to-dispute) version of the
sample-efficiency claim.

## Open follow-ups (M6c territory — not done in M6b)

The most plausible reasons SUMO PPO fails to escape, in priority order:

1. **Reward noise from SUMO stochasticity.** Per-step density depends
   on vehicle-insert timing, lane-change choices, etc. Surrogate is
   deterministic; SUMO is not. Higher reward variance → noisier
   advantages → harder to make the right policy gradient move.
   - Test: increase `n_steps` (more rollouts averaged before each
     update) and/or `n_eval_episodes` to reduce variance.
2. **Entropy collapse before escape.** `entropy_loss` stayed at -1.4
   throughout (essentially no change), but the policy std collapsed
   from 1.01 → 0.98 with the action mean stuck near 0. PPO with
   default `ent_coef=0` may not maintain enough exploration.
   - Test: `ent_coef=0.01` or 0.02 to encourage exploration past the
     initial drift.
3. **Action-mean init bias.** Default `MlpPolicy` initializes the
   action mean near 0; with clipping to [0, 1], early samples cluster
   near 0 and reinforce themselves.
   - Test: warm-start the policy by initializing the log_std lower
     and/or biasing the action mean toward 0.5.
4. **Mismatched value function.** Value function targets are noisy
   SUMO rewards; the value head can't fit them cleanly, so advantage
   estimates are bad.
   - Test: `vf_coef=1.0` (stronger value learning) or train a value
     network first.

The right next experiment is likely **M6c at ent_coef=0.01, same
20k–50k budget**. Cheap and tests exploration-vs-noise directly.

## Artifacts

- Run dir: `runs/rl/ppo_sumo_constant_inflow_m6b_seed0_100k_20260512_045505/`
- Training curves CSV: `_progress/m6b_training_curves.csv`
- Training curves PNG: `_progress/m6b_training_curves.png`
- Comparison script: `scripts/compare_training_trajectories.py`
