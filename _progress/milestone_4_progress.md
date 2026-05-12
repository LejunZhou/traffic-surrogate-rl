# Milestone 4 progress

Running log for the new M4 (SurrogateEnv against wilson's scenario,
shaped reward, analytical queue parity with SumoEnv). See
`_plans/milestone_4_plan.md` for scope and acceptance.

## 2026-05-11 — Kickoff

- Plan file written: `_plans/milestone_4_plan.md`.
- Todo list refreshed for M4 (6 tasks).
- Dependencies:
  - M3 checkpoint — full run currently in progress (background job).
    Implementation can proceed without waiting; smoke tests block on
    a valid `best.pt` and use `pytest.skip` if none is found.
  - SumoEnv (wilson's, with shaped reward from `b021df7`) is the
    parity reference for observation / reward / queue contracts.

## 4.0 — reward and queue plumbing

- `compute_reward(density, queue_length, weights)` already lives in
  `src/rl/reward.py` from commit `b021df7` (the shaped-reward overhaul).
  M4 only needs to call it with the surrogate's denormalized density
  and the env's internal `_analytical_queue`.
- `RewardWeights.from_config(env_config["reward"])` reads the
  `reward` block from the env config; defaults `alpha=1.0, beta=0.1,
  gamma=1.0` apply when no block is provided.
- Analytical queue formula:
  `queue[k+1] = max(0, queue[k] + (1 − u_k) · ramp_demand · dt_ctrl / 3600)`.
  Resets to 0 on `reset()`. Same formula as SumoEnv (see
  `sumo_env_wrapper.py`).

## 4.1 — __init__ and reset

Implementation choices baked into `src/rl/surrogate_env.py`:

- `_resolve_scenario()` reads `data.sumo_config` (or the env config's
  `sumo_config` key) and auto-fills `N_x`, `T_ctrl`, `highway_length_m`,
  `duration_s`, `dt_ctrl_s`, `detector_spacing_m`,
  `detector_start_position_m`, and `ramp_demand_vph`. Mirrors how
  `surrogate/train.py` resolves scenario constants — same source of
  truth.
- Checkpoint load: `torch.load(map_location="cpu", weights_only=False)`
  because the checkpoint carries the full training config, not just
  weights.
- `_build_model_from_checkpoint` resolves `branch_input_dim="auto"`
  using the checkpoint's saved `data.duration_s / data.dt_ctrl_s` so
  the model rebuild matches training exactly.
- Trunk `x_grid` derived from `(start_position_m + i * spacing_m)` for
  i in 0..N_x-1 → `[100, 200, ..., 1900]` on the current scenario.
- `observation_space = Box(-inf, inf, (N_x+3,), float32)`. Shape `(22,)`
  for N_x=19.
- `action_space = Box(0, 1, (1,), float32)`.
- Demand normalization fallback is **0.0** when span ≤ 1e-6 (matches
  SumoEnv exactly; the old M4 used 0.5).
- Reset seeds an internal `_rng`, samples one demand from
  `demand_profiles`, zeroes `_u_history` and `_analytical_queue`, runs
  one forward pass at k=0 for the initial obs.

## 4.2 — step

- Action clipping: `np.clip(action.reshape(-1)[0], 0.0, 1.0)` → `u_k`.
- Branch input: full `_u_history` (length 120) with positions ≤ k
  filled; tensor shape `(1, 120)` for BranchNet.
- Trunk input: `(1, 19, 2)` where each row is `(x_i_norm, t_k_norm)`.
- Output: `(1, 19)` z-score-normalized density → denormalize to
  physical units before reward.
- Queue updated *after* the surrogate call so the reward sees the
  queue at the end of the control interval.
- Post-transition obs runs one extra forward pass at `t = t_{k+1}` so
  the policy gets the next state's density for its next decision (the
  same convention SumoEnv uses by reading detectors after stepping).
- `info` carries `density_phys`, `mean_density`, `std_density`, `u`,
  `k`, `demand_vph`, `analytical_queue` — enough for downstream
  logging without re-running inference.

## 4.3 — Gym validation

Tests in `tests/test_surrogate_env.py` (7 tests):

- `test_spaces` — N_x=19, observation_space shape `(22,)`, action low/high.
- `test_reset_shape_and_queue` — obs shape, time and queue components
  are exactly 0 at reset.
- `test_step_clipping_and_termination` — out-of-range actions clip to
  `[0, 1]`; episode terminates exactly at `k == T_ctrl`; queue is
  monotone under `u ≤ 1`.
- `test_queue_growth_under_closed_ramp` — analytical queue grows by
  exactly `ramp_demand * dt / 3600` per step under `u=0`.
- `test_queue_constant_under_open_ramp` — queue stays at 0 under `u=1`
  (queue starts at 0 and `(1 - 1.0) * growth = 0`).
- `test_random_rollout` — 120-step random rollout completes with
  reward magnitudes in the expected `O(10–100)` range.
- `test_check_env` — gymnasium `check_env` runs without raising.

Command: `python -m pytest tests/test_surrogate_env.py -v`

Result:
```
test_spaces                        PASSED
test_reset_shape_and_queue         PASSED
test_step_clipping_and_termination PASSED
test_queue_growth_under_closed_ramp PASSED
test_queue_constant_under_open_ramp PASSED
test_random_rollout                PASSED
test_check_env                     PASSED
7 passed, 2 warnings in 1.74s
```

The two warnings are `check_env` complaining about the `-inf` / `+inf`
Box bounds (z-score normalized density is unbounded by design).
Cosmetic; not blocking.

## Acceptance verdict

PASS.

- All 7 smoke tests green.
- 120-step random rollout completes in ≪ 1 s on CPU.
- `check_env` passes (warnings only).
- Observation shape exactly `(22,) = (N_x + 3,)`; action `(1,)`.
- Reward values finite; queue grows / stays constant exactly as
  predicted by the analytical model.
- SurrogateEnv ↔ SumoEnv parity holds for observation shape, action
  shape, queue formula, and reward formula (verified by code review;
  side-by-side numerical comparison is M6 territory).

M5 (PPO surrogate training) can now load `SurrogateEnv` directly via
`configs/rl/ppo_surrogate.yaml`'s `env.surrogate_checkpoint: "auto"`,
which resolves to the M3 best.pt at run time.
