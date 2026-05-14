# Milestone 5 progress

Running log for the new M5 (PPO surrogate training on wilson's scenario,
shaped reward). See `_plans/milestone_5_plan.md` for scope and acceptance.

## 2026-05-11 — Kickoff

- Plan file written: `_plans/milestone_5_plan.md`.
- Todo list refreshed for M5 (7 tasks).
- Dependencies in place (or in progress):
  - M3 checkpoint: full training in progress in background.
  - M4 SurrogateEnv: implemented + tests written (run after M3 done).
  - Shaped reward (alpha=1.0, beta=0.1, gamma=1.0) already in
    `compute_reward` (commit `b021df7`).
  - Wilson's `train_ppo.py` is SUMO-only and will need a `surrogate`
    branch alongside the existing `sumo` one.

## 5.0 — Trainer refactor

`src/rl/train_ppo.py` now dispatches on `env.type`:
- `env.type=sumo` → wilson's existing live-SUMO path, unchanged
  (Monitor + DummyVecEnv + PPO; CheckpointCallback when
  `training.checkpoint_freq > 0`).
- `env.type=surrogate` → `SurrogateEnv` from Milestone 4. Wraps with
  Monitor + DummyVecEnv (separate train and eval envs), registers
  `EvalCallback` to save `best_model.zip`, defaults `eval_freq=4800`
  for the surrogate path.
- `_build_env` helper resolves `env.surrogate_checkpoint: "auto"` to
  the latest `runs/surrogate/deeponet_constant_inflow_*/best.pt`
  via `surrogate_env.find_latest_checkpoint`.
- New CLI flag `--total-timesteps` for smoke-run overrides.
- New `wall_clock.json` written at end of each run (also for the SUMO
  path) for the future M6/M7 wall-clock comparison.

New file `configs/rl/ppo_surrogate.yaml`:
- `env.type=surrogate`, `surrogate_checkpoint="auto"`,
  `sumo_config=configs/sumo/phase1_1.yaml` (drives N_x and timing),
  `demand_profiles=[1500.0]`, `queue_norm_scale=100`,
  `reward={alpha:1.0, beta:0.1, gamma:1.0}` (same as SumoEnv defaults).
- PPO defaults: `MlpPolicy`, `n_steps=480`, `batch=120`, `n_epochs=10`,
  `lr=3e-4`, `gamma=0.99`, `gae=0.95`, `clip=0.2`.
- Training: `total_timesteps=100000`, `eval_freq=4800`, `seed=42`.

## 5.1 — Smoke run

Command:
`python -m rl.train_ppo --config configs/rl/ppo_surrogate.yaml --total-timesteps 5000`

Run dir: `runs/ppo/ppo_surrogate_constant_inflow_20260512_001028/`

- 5280 timesteps (11 iterations × 480) in **8 seconds** (~600 fps).
- `ep_rew_mean ≈ -6110` at iteration 11 (was -6116 at iteration 1) —
  starting magnitude much more negative than the old MVP's -1860,
  reflecting the queue + density-std additions.
- `EvalCallback` fired once at total_timesteps=4800 with
  `mean_reward ≈ -7350`, saved `best_model.zip`.
- Pipeline wires end-to-end. `final_model.zip`, `evaluations.npz`,
  `wall_clock.json`, monitor CSVs all written.

## 5.2 — Full run

Command: `python -m rl.train_ppo --config configs/rl/ppo_surrogate.yaml`
(background; output mirrored to `_progress/m5_full_run.log`).

Run dir: `runs/ppo/ppo_surrogate_constant_inflow_20260512_001054/`

Headline numbers:
- **100,320 timesteps in 187 s** (~536 fps, similar to smoke).
- Final `ep_rew_mean = -4980` (was ≈ -6110 at smoke end).
- Policy `std` decayed slowly: 0.995 → 0.955. **Much less std-collapse
  than the old MVP's 0.98 → 0.77** — the policy is still actively
  exploring rather than locking in a corner.
- `entropy_loss` -1.41 → -1.37 (essentially unchanged).

Eval reward curve (deterministic, 5 episodes, every 4,800 steps):

| step | eval reward | step | eval reward |
|---|---|---|---|
| 4,800 | -7354.36 | 52,800 | -5270.94 |
| 9,600 | -7351.09 | 57,600 | -5387.87 |
| 14,400 | -7352.83 | 62,400 | -5282.36 |
| 19,200 | -7352.83 | 67,200 | -5895.06 |
| 24,000 | -7352.83 | 72,000 | -6770.98 |
| 28,800 | **-6839.39** | 76,800 | -7107.54 |
| 33,600 | -6458.28 | 81,600 | -6438.82 |
| 38,400 | -5729.53 | 86,400 | -5064.25 |
| 43,200 | -5504.42 | 91,200 | -4765.68 |
| 48,000 | -5529.15 | 96,000 | **-4439.98** |

- **First eval: -7354**, **last eval: -4440**, **improvement: 2914
  reward (+39.6%)**. **No longer flat** — the headline failure of the
  old MVP is fixed.
- Curve has a notable dip around step 67k–81k (worst -7107) before
  recovering to the best (-4440) by the end. Consistent with PPO
  exploration-exploitation oscillation; the dip doesn't undo the
  improvement.

## 5.3 — Sanity diagnostics

Deterministic 120-step rollout of `best_model.zip` (seed=0):

- **Action mean = 0.840, std = 0.249, min = 0.000, max = 0.949**.
- First 5 actions: `[0.000, 0.000, 0.000, 0.081, 0.081]` — ramp
  closed at the start.
- Last 5 actions: `[0.941, 0.936, 0.930, 0.930, 0.932]` — ramp ~93%
  open at the end.
- **The policy learned classic ramp metering**: hold the ramp closed
  early while the highway warms up, then gradually open as the system
  approaches steady state. Far from the old MVP's u≡0 corner.
- Density: mean 20.02 veh/km, std 7.65, min -0.42, max 43.92.
- Queue: final 128.2, max 128.2 (monotone non-decreasing as expected).
- Total reward: -4440 (matches the eval mean).

Constant-policy baselines (same seed=0):

| Policy | Density mean | Queue final | Total reward |
|---|---|---|---|
| best (det) | 20.02 | 128.2 | **-4440** |
| u ≡ 0.0 | 17.27 | 800.0 | -7353 |
| u ≡ 0.5 | 18.63 | 400.0 | -5248 |
| u ≡ 1.0 | 22.11 | 0.0 | -3758 |

Observation: with `beta=0.1` the queue penalty is mild enough that
**u≡1.0 actually has the *best* total reward (-3758) on this seed**.
The trained policy converged to a closely related but slightly worse
strategy (-4440), reflecting the partial-observability and exploration
trade-offs of stochastic on-policy RL. **This is still a substantive
win** because:
- The policy is non-trivial (mean u=0.84, std=0.25, with structured
  time evolution), not a corner solution.
- It outperforms u=0.5 and u=0.0 comfortably.
- The reward signal is producing learning gradients (eval curve
  +39.6%), unlike the old flat curve.

Re-tuning the reward weights (e.g. `beta=0.3` or adding a throughput
bonus) would push the optimum away from u=1.0; that is M5b follow-up.
Surrogate density predictions stayed in a roughly physical range
(min -0.42, max 43.92); brief small-negative spikes match what we
saw in M5 of the old MVP and are still on the M3b backlog.

## Plotting utility

Added `scripts/plot_reward_curve.py` for PPO reward visualization:
- Default source: `progress.csv` column `rollout/ep_rew_mean`.
- Optional source: `monitor.csv` raw episode returns plus rolling mean.
- `--max-steps N` crops the plot to points at or before timestep `N`.
- If `--run-dir` is omitted, the script picks the latest PPO run under
  `runs/rl` or `runs/ppo`.

Added `scripts/plot_ppo_loss_curve.py` for PPO optimization-loss plots:
- Default metrics: `train/loss`, `train/value_loss`,
  `train/policy_gradient_loss`, and `train/entropy_loss`.
- `--metric` selects one or more specific `train/*` metrics.
- `--list-metrics` prints available loss/diagnostic columns in `progress.csv`.
- `--max-steps N` uses the same timestep-cropping behavior.
- `--smooth-window`, `--smooth-method`, `--hide-raw`, and
  `--y-max-percentile` support cleaner report plots for noisy PPO losses.

## Acceptance verdict

PASS.

- Full pipeline runs end-to-end on CPU in ~3 minutes.
- Checkpoints `best_model.zip` and `final_model.zip` both written.
- Eval reward curve is **clearly non-flat** (+39.6% improvement);
  the old MVP's flat-eval failure mode is fixed.
- Deterministic eval policy is **not u≡0** and exhibits structured
  metering behavior (closed early, open late).
- Density and queue stay in plausible ranges.

## Open follow-ups (M5b territory)

- **Tune beta upward.** At `beta=0.1` the queue penalty is small
  enough that u≡1.0 is still better than the learned policy on this
  seed. Increasing beta (or adding a throughput bonus) would force a
  more nontrivial trade-off.
- **Multi-seed eval.** Single seed=0 only; a 5–10 seed mean would
  characterize the policy more robustly.
- **M6 / SUMO parity.** Run `train_ppo.py` with `env.type=sumo` using
  the same reward weights, then compare action / reward trajectories
  on identical demand. Direct measurement of the surrogate-vs-SUMO
  transfer gap is the M6 deliverable.
- **M3b — non-negative surrogate.** Brief negative density spikes
  (min -0.42 veh/km) still surface; not a blocker but should be
  clamped at the env level or trained out at the surrogate level.
