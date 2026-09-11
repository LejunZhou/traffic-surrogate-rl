# Reward-term decomposition analysis: M5c vs M6 vs M6b

## Context

The current shaped reward has three penalty terms with very different per-step
peak magnitudes:

| Term | Typical peak per step |
|---|---|
| `α · max(0, mean(ρ) − 20)` | ~25 |
| `β · (queue / 100)²` | ~64 (full closure) |
| `γ · std(ρ)` | ~15 |

The quadratic queue blow-up was an intentional M5c design choice (it was the
lever that broke M5b's u=1 corner trap). Surrogate-trained PPO (M5c) worked;
direct SUMO PPO at 20k (M6) and 100k (M6b) timesteps collapsed to u ≈ 0.

**Open question:** did the queue penalty dominate and crush the learning
signal in M6 / M6b, or was the failure exploratory? The answer determines
whether rescaling the three terms to ~[0, 1] is worth doing, or whether the
`ent_coef = 0.01` + 3×512 net we just shipped on `main` is already the right
fix.

The runs predate the W&B integration (commit `1672424`), so the data does
not exist on W&B — we have to generate it locally by replaying the saved
checkpoints.

## Approach

One new analysis script that replays each saved policy through its training
env and captures the per-step `info["density_excess_penalty"]`,
`info["queue_penalty"]`, `info["std_penalty"]` already produced by the env.
No env changes; no policy retraining.

Three conditions, **5 deterministic episodes each**:

| Condition | Env | Checkpoint |
|---|---|---|
| M5c (success) | `SurrogateEnv` | `runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_20260512_021425/best_model.zip` |
| M6 (fail, 20k) | `SumoEnv` | `runs/rl/ppo_sumo_constant_inflow_m6_seed0_20260512_023250/final_model.zip` |
| M6b (fail, 100k) | `SumoEnv` | `runs/rl/ppo_sumo_constant_inflow_m6b_seed0_100k_20260512_045505/final_model.zip` |

Note: only M5c has a `best_model.zip` (EvalCallback was disabled for M6 /
M6b to avoid expensive live-SUMO evals; final_model.zip is the canonical
checkpoint there).

All three runs trained on identical reward weights (`α = β = γ = 1.0`,
`ρ_freeflow = 20.0`, `queue_norm = 100.0`), so per-term magnitudes are
directly comparable across runs.

## Files

**New:** `scripts/analyze_reward_terms.py`.

**Existing code to reuse (read-only):**

- `scripts/eval_constant_baselines.py:_resolve_env_config` and `:rollout_policy` — pattern for `SurrogateEnv` replay
- `scripts/eval_sumo_baselines.py:_load_sumo_env_config` and `:rollout_policy_sumo` — same pattern for `SumoEnv`
- `src/rl/sumo_env_wrapper.py:step` (lines 293–328), `:_reward_terms` (lines 486–499) — confirms info-dict keys
- `src/rl/surrogate_env.py:step`, `:_reward_terms` — same keys on the surrogate side
- `src/rl/reward.py:compute_reward` — sign convention check

**Run-time config sources:** each run dir's saved `config.yaml` (pinned with
the correct surrogate_checkpoint for M5c and SUMO settings for M6 / M6b).

## Output

```
runs/analysis/<ts>_reward_term_comparison/
├── m5c/{per_episode_terms.npz, summary.json}
├── m6/{per_episode_terms.npz, summary.json}
├── m6b/{per_episode_terms.npz, summary.json}
├── stacked_bar.png             # episode-mean integral per term, 3 bars
├── timeseries.png              # 3-panel mean±std time series
└── comparison_summary.json     # headline cross-condition table
```

## Verification

1. **Internal consistency:** for every (run, episode, step) assert
   `density_excess_penalty + queue_penalty + std_penalty ≈ -reward` to
   within `1e-5`.
2. **Cross-check against milestone progress:**
   - M5c action mean ≈ 0.688
   - M6 / M6b action mean ≈ 0.0
3. **Expected order-of-magnitude on collapse runs:** if M6 / M6b sit at
   u ≈ 0, queue grows to ≈ 800 → quadratic term ≈ 64 per step → episode
   integral ≈ 7680. Density excess and std should both be small.
4. **Decision rule for the headline question:**
   - If M6 / M6b queue_penalty integral ≥ ~5× M5c queue_penalty integral
     → queue term dominated → rescaling is worth doing.
   - Otherwise → collapse was exploratory → keep the M5c reward shape and
     rely on the entropy bonus + bigger net + sample budget.

## Notes

- This script is read-only with respect to the codebase: it only writes
  to `runs/analysis/`.
- SUMO is required for the M6 / M6b replay (TraCI must be importable).
