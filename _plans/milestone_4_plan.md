# Milestone 4 plan — SurrogateEnv with shaped reward (wilson's scenario)

**Status:** in progress. See `_progress/milestone_4_progress.md` for the running log.

## Goal
Implement `SurrogateEnv` (currently a stub) so PPO can train against the
DeepONet surrogate. The env must expose **the same observation, action,
and reward contract as `SumoEnv`** so M5 (surrogate-PPO) and a later M6
(SUMO-PPO comparison) can use environment-agnostic training code.

## Scope
Implementation must match the contract documented in `proposal.md`
§"RL interface contract" and `src/rl/sumo_env_wrapper.py`:

- **Observation** (shape `(N_x + 3,) = (22,)` with N_x=19):
  - `density[0:19]` — z-score normalized density (mean=18.730,
    std=5.971 from the M2 metadata).
  - `demand[19]` — min-max normalized; fallback **0.0** (not 0.5) when
    `max_demand == min_demand` — matches SumoEnv exactly.
  - `time[20]` — `min(k, T_ctrl) / T_ctrl`.
  - `queue[21]` — `analytical_queue / queue_norm_scale` (default 100).
- **Action**: `Box(0, 1, (1,), float32)`. Clipped on step input.
- **Reward**: shared `compute_reward(density, queue_length, weights)`
  with `RewardWeights.from_config(env_config["reward"])`. Density passed
  in physical units (denormalized from surrogate's z-score output).
- **Analytical queue**: same formula as `SumoEnv` —
  `queue[k+1] = max(0, queue[k] + (1 - u_k) * ramp_demand * dt_ctrl / 3600)`.
  Resets to 0 each episode.
- **Rollout strategy**: at step k, branch input is `u_history[:T_ctrl]`
  with positions ≤ k filled in and positions > k zero-padded. Trunk
  queries the 19 detector positions at `t = t_k`. DeepONet re-evaluated
  from scratch each step (not autoregressive).
- **Episode length**: `T_ctrl = duration_s / dt_ctrl_s = 120`.
- **Reset**: sample one demand value from configured `demand_levels`
  (single-value list in MVP).

## Deliverables
- `src/rl/surrogate_env.py` — replaces the `NotImplementedError` stub.
- `tests/test_surrogate_env.py` (new) — pytest covering spaces, reset,
  step (clipping + termination), random rollout, `check_env`, and
  queue-update correctness.
- `_plans/milestone_4_plan.md` (this file).
- `_progress/milestone_4_progress.md` — running log.

## Implementation notes
- Load checkpoint with `torch.load(map_location="cpu", weights_only=False)`
  because wilson's `surrogate/train.py` saves `model_state_dict`,
  `config`, `normalization`, and `optimizer_state_dict` together — the
  default `weights_only=True` since PyTorch 2.5 would reject this file.
- Resolve scenario constants the same way `surrogate/train.py` does:
  read `data.sumo_config` if provided, otherwise expect explicit
  `N_x` / `T_ctrl` / `highway_length_m` / `duration_s` /
  `detector_spacing_m` / `start_position_m` in the env config.
- Build trunk grid once in `__init__`. Per-step, only `t_k_norm` changes
  in the trunk; `x_grid_norm` is constant.
- Disable gradients globally on the loaded model (`requires_grad_(False)`,
  `model.eval()`), wrap inference in `torch.no_grad()`.
- Demand normalization fallback is **0.0 when span ≤ 1e-6** to match
  SumoEnv. The old M4 used 0.5; flipped to 0.0 for parity.

## Sub-milestones
- **4.0 — reward and queue plumbing.** Confirm `RewardWeights.from_config`
  reads the env config's `reward` block; confirm queue resets each
  episode.
- **4.1 — `__init__` + `reset`.** Load checkpoint, build spaces, do one
  forward pass at k=0 with all-zero branch input, return a well-shaped
  initial obs.
- **4.2 — `step`.** Drive a 120-step rollout. Verify shapes, queue
  growth when u<1, reward sign.
- **4.3 — Gym validation.** Run `gymnasium.utils.env_checker.check_env`.

## Acceptance criteria
- `pytest tests/test_surrogate_env.py` passes.
- `check_env(SurrogateEnv(...))` returns without raising.
- 120-step random rollout completes well under 1 second on CPU.
- Reward values are finite and follow the shaped-reward sign convention
  (negative for non-trivial density / queue / variance).
- Observation shape exactly `(22,)`; action `(1,)`; queue grows
  monotonically when actions ≤ 0.5 and stays constant when action ≡ 1.

## Open follow-ups
- **`SumoEnv` parity check.** Wilson already implemented SumoEnv with
  the analytical queue + shaped reward (from `b021df7`). A direct
  side-by-side test (same action sequence in both envs, compare reward
  trace) is the natural M6 deliverable.
- **Multi-demand obs.** Currently the demand component is 0.0 (degenerate
  span). Tied to M3b — only matters if demand becomes variable.
- **Queue formula realism.** Current model is "cumulative blocked
  ramp arrivals"; doesn't drain when ramp is reopened. Reasonable for
  MVP; revisit if M5 results suggest the policy can't recover from a
  large queue.
