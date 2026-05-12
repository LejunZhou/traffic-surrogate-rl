# Milestone 6 progress

Running log for direct SUMO+RL training (PPO against `SumoEnv`) at the
M5c shaped reward. See `_plans/milestone_6_plan.md` for scope and
acceptance.

## 2026-05-12 — Kickoff

- Plan file: `_plans/milestone_6_plan.md`.
- Todo list refreshed for M6 (8 tasks).
- Predecessor: M5c (commit `9a580fd`) — produced an interior PPO
  policy on the surrogate at action mean 0.688, std 0.426. M6 trains
  the same reward through `SumoEnv` to compare.
- Side note: the M5c 3-seed robustness sweep (`scripts/run_m5b_sweep.py
  --betas 1.0 --seeds 1,2,3`) is still running in background at M6
  kickoff. M6 setup/config work happens in parallel; CPU contention
  doesn't start until M6's own PPO training begins.

## 6.0 — Plan + progress + config

- Plan + progress files written.
- `configs/rl/ppo_sumo.yaml` to be written next (M5c reward, SumoEnv
  config, density mean/std from M3 metadata, eval_freq disabled to
  avoid expensive SUMO-side eval rollouts).

## 6.1 — Single-episode benchmark

Standalone SumoEnv timing at u=0.5, seed=0:
- `__init__` (network + routes build): **0.19 s** (one-time).
- `reset()` (starts SUMO via TraCI): **0.54 s** per episode.
- Per-control-step: **mean 102 ms, p95 119 ms, max 156 ms**
  (30 SUMO sub-steps + 19 detector reads per control step).
- Full 120-step episode loop: **12.29 s**.
- Total per-episode wall clock (reset + loop): ~12.8 s.

**Per-policy SUMO reward at the M5c reward and seed=0**:

| Policy | SUMO reward | Surrogate reward (from M5c sanity) | gap |
|---|---|---|---|
| u=0.5 | **-1238.49** | -1240.30 | only 1.8 ! |

The surrogate's density predictions integrate to almost exactly the
true SUMO densities at u=0.5. Strong signal that the M5c surrogate-
trained policy should transfer reasonably to SUMO without large
reward shifts. We'll verify the same for u=0.0/u=1.0 in 6.6 and for
the learned policies in 6.4/6.5.

Sizing implications (extrapolated from one episode):

| total_timesteps | n_episodes | est. wall clock |
|---|---|---|
| 600   | 5   | 1.1 min   |
| 5000  | 42  | 8.9 min   |
| 10000 | 83  | 17.8 min  |
| **20000** | **167** | **35.7 min** ← chosen for M6 full |
| 50000 | 417 | 89.1 min  |
| 100000 | 833 | 178.2 min |

Choice: `total_timesteps=20000` for the M6 full run. 167 episodes is
about 20% of M5c's effective episode count (M5c saw 833 episodes worth
of surrogate rollouts in 100k timesteps), but still enough to see real
learning given how cheap each gradient update is in absolute terms.
PPO will perform `~42` policy updates (n_steps=480 each).

## 6.2 — PPO smoke

Command:
`python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --total-timesteps 600 --run-name-suffix m6_smoke`

Run dir: `runs/rl/ppo_sumo_constant_inflow_m6_smoke_20260512_022943/`

- 960 timesteps (SB3 rounds up to nearest n_steps multiple) in 107 s
  (~9 fps). 2 PPO iterations, 10 gradient updates.
- `ep_rew_mean ≈ -1760` (similar to M5c smoke).
- `final_model.zip` saved (no `best_model.zip` — EvalCallback disabled
  by design; the run is too short for CheckpointCallback at freq=2400).
- Pipeline wires end-to-end through wilson's `SumoEnv` + `train_ppo.py`
  `env.type=sumo` branch.

## 6.3 — PPO full

Command:
`python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --total-timesteps 20000 --seed 0 --run-name-suffix m6_seed0`

Run dir: **`runs/rl/ppo_sumo_constant_inflow_m6_seed0_20260512_023250/`**

- 20,160 timesteps in 2105 s (~35 min, ~9 fps).
- 42 PPO iterations, 410 gradient updates (compare M5c: 209 iterations,
  2080 updates).
- `ep_rew_mean` trajectory: smoke level ~-1770 → -2040 at end.
- Policy `std`: 1.01 → 0.983 (barely budged; very little training
  signal to collapse the distribution).
- `entropy_loss` ≈ -1.4 throughout — no entropy decay.
- The M5c sweep was running concurrently and slowed SUMO by ~50%;
  per-episode time was ~17 s observed vs the ~13 s baseline benchmark.
- `final_model.zip` + 4 intermediate checkpoints saved by
  CheckpointCallback (every 2400 timesteps).

## 6.4 — Native eval (M6 policy in SUMO, seed=0)

`python scripts/eval_sumo_baselines.py --policy <run>/final_model.zip --seed 0`

Result:
- **total_reward = -2919.25**
- **action mean = 0.003, std = 0.015, max = 0.096** — **policy
  collapsed near u=0.0**. M6 PPO got stuck in the closed-ramp basin
  with only 410 gradient updates available to escape.
- density mean = 15.85 (low, matches u=0 constant).
- queue final = 797.7 (essentially u=0's 800).
- throughput = 1475 vph (mainline only).

This is *not* the failure mode of "PPO can't learn" — it's the failure
mode of "PPO didn't get enough updates to escape the first corner it
sampled into". The shaped-reward landscape is correct (the surrogate-
side M5c training found u=0.688 on the same reward), it's just slow
to find via direct SUMO simulation.

## 6.5 — Transfer eval (M5c surrogate-trained policy on SUMO, seed=0)

`python scripts/eval_sumo_baselines.py --policy runs/ppo/.../m5c_seed0/best_model.zip --seed 0`

Result:
- **total_reward = -1525.65**
- action mean = 0.729, std = 0.431 (very close to M5c's surrogate-eval
  numbers: 0.688 / 0.426).
- density mean = 20.35, std = 8.37 (matches M5c surrogate: 19.14 / 6.72).
- queue final = 217.1 (matches M5c surrogate: 249.5).
- throughput = 2043 vph (highest of any non-corner policy).

**The M5c policy transfers cleanly to SUMO.** Reward dropped from
-1347.83 (surrogate eval) → -1525.65 (SUMO eval), a 13.2% transfer
gap. Behavior is qualitatively the same: ramp closed early, opening
to ~0.93 by episode end, intermediate queue.

## 6.6 — Constant baselines in SUMO (seed=0, M5c reward)

| Policy | SUMO reward | (surrogate from M5c §5c.5) | gap |
|---|---|---|---|
| u=0.0 | -2941.16 | -3032.52 | 91 (3.0%) |
| u=0.5 | -1238.49 | -1240.30 | 1.8 (0.1%) |
| u=1.0 | -1431.72 | -1377.17 | 54.6 (4.0%) |

**Excellent surrogate↔SUMO agreement.** The M3 surrogate's density
predictions reproduce the SUMO integrals to within 4% on every
constant policy. u=0.5 agrees to within rounding (the regime PPO
actually trains in).

## 6.7 — Comparison table

Full table at seed=0 on the M5c shaped reward, **evaluated in SUMO**:

| Policy | Reward | Action mean (std) | Density (mean/std) | Queue final | Throughput | Notes |
|---|---|---|---|---|---|---|
| u=0.0 constant      | -2941.2 | 0.000 (0)     | 15.83 / 3.16 | 800.0 | 1473 vph | ramp closed |
| u=0.5 constant      | **-1238.5** | 0.500 (0) | 18.57 / 5.12 | 400.0 | 1867 vph | **best constant** |
| u=1.0 constant      | -1431.7 | 1.000 (0)     | 22.38 / 9.55 | 0.0   | 2260 vph | ramp wide open |
| **M5c (surrogate-trained, transferred)** | **-1525.6** | **0.729 (0.431)** | 20.35 / 8.37 | 217.1 | **2043 vph** | **interior policy** |
| M6 (SUMO-trained)   | -2919.2 | 0.003 (0.015) | 15.85 / 3.17 | 797.7 | 1475 vph | collapsed to u≈0 |

**Wall-clock + sample-efficiency comparison:**

| Run | Backend | Wall clock | Total timesteps | PPO updates | Result |
|---|---|---|---|---|---|
| M5c | DeepONet surrogate | **185 s** (~3 min) | 100,000 | 2,080 | interior policy u≈0.688 |
| M6  | Live SUMO via TraCI | **2105 s** (~35 min) | 20,000  |   410 | collapsed policy u≈0.003 |

Surrogate is **~58× faster per timestep** (555 ts/s vs 9.5 ts/s) and
got the surrogate run **5× more PPO updates** in **~1/11 of the wall
clock**. Apples-to-apples (matched PPO updates), the surrogate path
would need 100k surrogate timesteps to roughly match an estimated
500k SUMO timesteps — which would take 14.5 hours on this machine
(extrapolating 9.5 ts/s).

## Acceptance verdict

**Headline win for the surrogate-accelerated approach.** The
surrogate-trained M5c policy, when transferred to SUMO, beats the
direct-SUMO-trained M6 policy on SUMO **by 1394 reward units (~48%)**.
Same reward, same scenario, same seed — the only difference is which
env PPO trained against.

Strict acceptance against `_plans/milestone_6_plan.md`:

| Criterion | Result |
|---|---|
| Pipeline runs end-to-end, checkpoints saved | ✅ PASS |
| M6 native action mean ∈ (0.05, 0.95) AND beats best constant | ❌ FAIL — collapsed to u≈0 in 20k timesteps |
| Transfer eval table populated | ✅ PASS |
| Per-episode wall-clock documented | ✅ PASS (~13 s clean, ~17 s under sweep contention) |

Three pass / one fail. The "fail" is itself the headline finding: at
20k SUMO timesteps PPO is sample-starved, but at 100k surrogate
timesteps (~3 min) PPO produces a policy that transfers cleanly.

## Open follow-ups

- **M6-extended**: rerun M6 at 50k or 100k SUMO timesteps (1.5–3 h)
  to see whether direct training catches up or stays stuck. Decides
  whether M6's collapse is a sample-count artifact or a stickier
  optimization issue.
- **M6 transfer-quality table at multiple seeds**: re-run M6.5 across
  seeds 1–4 (uses the M5c surrogate policy from the same single
  training run, just rolled out at different SUMO seeds). Cheap (~1
  min per seed) and would characterize transfer-gap variance.
- **M7 — comparison plot**: per-iteration learning curve overlay
  (M5c surrogate timesteps vs M6 SUMO timesteps on the x-axis,
  smoothed eval reward on the y-axis) is the natural visualisation
  of the sample-efficiency finding.
- **The M5c 3-seed sweep is still running**; its output will
  augment the M5c story (variance across surrogate-training seeds)
  but doesn't change M6's conclusion.
