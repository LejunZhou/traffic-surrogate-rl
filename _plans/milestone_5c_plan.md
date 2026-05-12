# Milestone 5c plan — Nonlinear shaped reward (ReLU density + quadratic queue)

**Status:** in progress. See `_progress/milestone_5c_progress.md` for the running log.

## Context

M5b proved a structural ceiling: at mainline demand 1500 vph, no linear
β can make a non-trivial policy beat `u≡1.0`, because the linear
`-β · queue` term is exactly 0 at the corner (queue never grows from 0).
See `_progress/milestone_5b_progress.md` §5b.5 for the full diagnostic.

The fix the user proposed in-session is a **nonlinear** reward whose
shape matches the physics:

- mean-density penalty only **activates above free-flow density**
  (free-flow operation is free);
- queue penalty is **quadratic** so cost accelerates as the queue grows
  past nuisance length;
- std-of-density term stays linear (already a good uniformity proxy).

With the proposed defaults and per-baseline numbers measured in M5b's
seed=0 rollouts (see `_progress/milestone_5_progress.md` lines 117–121
and `_progress/m5b_results.csv`), `u=0.5` becomes the global optimum on
seed=0 — a real interior basin for PPO to find.

## New reward

```
r(t) = - alpha * max(0, mean(rho(t)) - rho_freeflow)        # ReLU on density excess
       - beta  * (queue(t) / queue_norm)^2                   # quadratic in queue
       - gamma * std(rho(t))                                  # unchanged
```

Defaults (rationale below): `alpha = 1.0, beta = 1.0, gamma = 1.0,
rho_freeflow = 20.0, queue_norm = 100.0`.

## Why these defaults — empirical decomposition at seed=0

Per-step constants from the M5b rollouts (seed=0):

| Policy | mean(rho) | std(rho) | Q_final |
|--------|-----------|----------|---------|
| u=0    | 17.27     | 4.28     | 800     |
| u=0.5  | 18.63     | 5.15     | 400     |
| u=1.0  | 22.11     | 9.39     | 0       |

`rho_freeflow = 20.0` puts the ReLU between u=0.5's 18.63 and u=1.0's
22.11 — only the corner activates. `queue_norm = 100.0` matches the
existing obs-side `queue_norm_scale` so reward and obs see the same
magnitude.

With queue growing linearly to `Q_final`, the per-episode quadratic
queue term sums to `(Q_final/100)^2 * sum_{k=0..119}(k/119)^2 ≈
(Q_final/100)^2 * 40.17`. Episode totals at `alpha=beta=gamma=1`:

| Policy | alpha-term | beta-term | gamma-term | total      |
|--------|-----------|----------|------------|------------|
| u=0    |     0     |  2570.8  |   513.6    | **-3084**  |
| u=0.5  |     0     |   642.7  |   618.0    | **-1261**  ← winner (margin 119 over u=1.0) |
| u=1.0  |   253.2   |     0    |  1126.8    | **-1380**  |

Sensitivity (selected from the math table):

| alpha | beta | gamma | r(0)  | r(0.5)   | r(1.0)   | winner          |
|-------|------|-------|-------|----------|----------|-----------------|
| 1     | 1    | 1     | -3084 | **-1261**| -1380    | u=0.5 (margin 119) |
| 0     | 1    | 1     | -3084 | -1261    | **-1127**| u=1.0 — alpha too weak |
| 1     | 0.5  | 1     | -1799 | **-939** | -1380    | u=0.5            |
| 1     | 2    | 1     | -5655 | **-1902**| -1380    | u=1.0 — beta too strong |
| 1     | 1    | 0     | -2571 |  -643    | **-253** | u=1.0 — gamma matters |
| 1     | 1    | 2     | -3598 | **-1879**| -2507    | u=0.5            |

The interior-wins band at `rho_freeflow=20, queue_norm=100` is roughly
alpha ∈ [0.5, ∞), beta ∈ [0.3, 1.5], gamma ∈ [0.5, ∞). Defaults sit in
the middle.

## Shape of the change

```
                                                 ┌─────────────────────┐
                                                 │ proposal.md         │
                                                 │ (rewrite reward §)  │
                                                 └─────────────────────┘

src/rl/reward.py ──┬──► configs/rl/ppo_surrogate.yaml ──► PPO smoke (5k) ──► PPO full (100k)
   (formula +      │      (alpha/beta/gamma/                                       │
    2 new fields)  │       rho_freeflow/queue_norm)                                ▼
                   │                                                  eval_constant_baselines.py
                   │                                                  • u=0.0   → ≈ -3080
                   │                                                  • u=0.5   → ≈ -1260  (target)
                   │                                                  • u=1.0   → ≈ -1380
                   │                                                  • learned → > -1260, action mean ∈ (.05,.95)
                   │
                   └──► tests/test_surrogate_env.py (loosen reward-magnitude bound)

env code (sumo_env_wrapper.py, surrogate_env.py): NO change.
   Both already do RewardWeights.from_config(env_cfg["reward"]) and the new
   fields flow through transparently as long as from_config reads them.
```

## File-by-file changes

### 1. `src/rl/reward.py` — core formula

- Add two fields to `RewardWeights`: `rho_freeflow: float = 20.0` and
  `queue_norm: float = 100.0`. Update `from_config` to read them. Keep
  `alpha`, `beta`, `gamma` (re-interpreted under the new formula).
- Rewrite the module docstring and `compute_reward` docstring to match
  the new formula. Mark M5b's linear corner-trap as the empirical
  motivation.
- Replace the final reward expression with:
  ```python
  mean_d = float(np.mean(density_arr))
  std_d  = float(np.std(density_arr))
  density_excess = max(0.0, mean_d - w.rho_freeflow)
  q_scaled = queue / max(w.queue_norm, 1e-6)
  return -(w.alpha * density_excess
           + w.beta  * q_scaled * q_scaled
           + w.gamma * std_d)
  ```
- Keep all existing validation (1-D density, finite, queue >= 0).
- Note on naming: the new `queue_norm` is conceptually independent of
  the env's existing obs-side `queue_norm_scale` (which scales the
  observation vector). They default to the same 100.0 but live in
  different configs so they can be tuned separately later if needed.

### 2. `configs/rl/ppo_surrogate.yaml` — defaults

Replace the `env.reward` block with:
```yaml
  reward:
    alpha: 1.0          # weight on -max(0, mean(rho) - rho_freeflow)
    beta: 1.0           # weight on -(queue/queue_norm)^2
    gamma: 1.0          # weight on -std(rho)
    rho_freeflow: 20.0  # below this mean density, alpha-term is 0
    queue_norm: 100.0   # quadratic queue scale (veh)
```
Leave `env.queue_norm_scale: 100.0` (obs-side normalizer) untouched.

### 3. `proposal.md` — §"Reward (Phase 1 shaped)"

Rewrite the bullets to:
- Replace the formula with the new ReLU/quadratic form.
- Replace the default weights line: `alpha=1.0, beta=1.0, gamma=1.0,
  rho_freeflow=20.0, queue_norm=100.0`.
- Replace the "Empirical rationale" bullet: cite M5b's linear-corner
  null result (`_progress/milestone_5b_progress.md`) and explain why the
  ReLU+quadratic shape breaks the corner trap.

### 4. `tests/test_surrogate_env.py` — loosen reward bound

In `test_random_rollout`, the random-action expected per-step reward
shrinks under the new formula (alpha-term near 0, queue<400 → beta-term
≤ ~10, gamma-term ≈ 5). The existing bound `-200.0 < mean < 0.0` still
holds with margin, but tighten to `-50.0 < mean < 0.0` to catch real
regressions. No other tests need updating — queue-growth and spaces
tests are formula-agnostic.

### 5. `_plans/milestone_5c_plan.md` and `_progress/milestone_5c_progress.md`

Per CLAUDE.md convention (one plan + one progress file per milestone).
This plan file; the progress file is a fresh skeleton to be filled in
during execution.

## Execution order

1. Write `_plans/milestone_5c_plan.md` and `_progress/milestone_5c_progress.md`.
2. Edit `src/rl/reward.py` (formula + 2 fields).
3. Edit `configs/rl/ppo_surrogate.yaml` (defaults).
4. Edit `proposal.md` (reward section).
5. Edit `tests/test_surrogate_env.py` (loosen bound).
6. **Pre-PPO sanity** — run the three constant baselines and confirm the
   table above (within ±5 reward):
   ```
   for p in u=0.0 u=0.5 u=1.0; do
     python scripts/eval_constant_baselines.py --policy "$p" --seed 0
   done
   ```
   Expect total_reward ≈ -3080 / -1260 / -1380. If u=0.5 is **not** the
   winner, stop and re-tune before any PPO run.
7. **Smoke PPO** — `python -m rl.train_ppo --config configs/rl/ppo_surrogate.yaml
   --total-timesteps 5000 --run-name-suffix m5c_smoke`. Confirms wiring;
   ignore policy quality.
8. **Full PPO** — same command, default 100k timesteps, ~3 min on CPU.
   Use `--run-name-suffix m5c_seed0` and `--seed 0`.
9. **Verify** — deterministic eval of `best_model.zip` at seed=0:
   ```
   python scripts/eval_constant_baselines.py \
     --policy runs/ppo/<m5c_seed0_dir>/best_model.zip --seed 0
   ```
   Acceptance: action mean strictly in (0.05, 0.95) AND total reward
   strictly greater than max constant-baseline reward (≈ -1261).
10. **Optional robustness** — only if (9) lands within 5% of u=0.5:
    rerun seeds 1–4 via `scripts/run_m5b_sweep.py` (reusable for one
    beta at five seeds) or a one-off loop. Document outcome in the
    progress file.
11. Update `_progress/milestone_5c_progress.md` with results, then
    commit.

## Critical files

- Edited: `src/rl/reward.py`, `configs/rl/ppo_surrogate.yaml`,
  `proposal.md`, `tests/test_surrogate_env.py`.
- New: `_plans/milestone_5c_plan.md`, `_progress/milestone_5c_progress.md`.
- Untouched but exercised: `src/rl/surrogate_env.py`,
  `src/rl/sumo_env_wrapper.py` (both already route the reward block
  through `RewardWeights.from_config`),
  `scripts/eval_constant_baselines.py`, `src/rl/train_ppo.py`.

## Verification (end-to-end)

| Step | Command | Pass condition |
|------|---------|----------------|
| Unit math | `pytest tests/test_surrogate_env.py -q` | All tests pass; check_env still clean. |
| Baseline ranking | `eval_constant_baselines.py --policy u=0.5 --seed 0` | total_reward ≈ -1261 (±5). |
| Baseline ranking | `eval_constant_baselines.py --policy u=1.0 --seed 0` | total_reward ≈ -1380 (±5). |
| Baseline ranking | `eval_constant_baselines.py --policy u=0.0 --seed 0` | total_reward ≈ -3084 (±10). |
| PPO seed 0       | `eval_constant_baselines.py --policy <run>/best_model.zip --seed 0` | action_mean ∈ (0.05, 0.95); total_reward > -1261. |

## Risks and follow-ups

- **Seed sensitivity unknown**: defaults were sized on seed=0 numbers.
  Other seeds' density/std could shift the optimum's location; M5b
  showed only mild seed variance, but we should re-check on 1–2 extra
  seeds if step (9) is borderline.
- **Surrogate prediction noise**: mean(rho) for u=1.0 hovers around
  22.1 ± ε from surrogate noise; alpha-term is `max(0, rho_bar - 20)`,
  so small noise can push the active region up or down. Acceptable:
  still positive at u=1.0 in expectation; small numerical jitter
  doesn't change ranking.
- **Deferred**: quadratic mean-density `-alpha · max(0, rho_bar-rho_ff)^2`
  (sharper) and piecewise queue `-lambda · max(0, Q-Q*)^2` (hard cap).
  Not needed if (9) passes; revisit if higher-demand work (M2c) shows
  the linear ReLU doesn't bite hard enough.
- **M2c interaction**: at higher mainline demand the optimum action
  mean moves down; same formula, same defaults — only retraining
  needed.
