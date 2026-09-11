# Reward-term decomposition analysis — progress

Plan: `_plans/reward_term_analysis_plan.md`.

## Status

**Complete.** Recommendation: **do NOT rescale the three reward terms.** The
collapse in M6 / M6b is exploratory, not driven by reward-shape dominance.

## Todos

- [x] Set up `_plans/` and `_progress/` docs.
- [x] Write `scripts/analyze_reward_terms.py` (replay + collection + plotting).
- [x] Run end-to-end and produce figures + summary JSON.
- [x] Record the headline finding and recommendation.

## Run dirs and checkpoints

- M5c (success):
  `runs/ppo/ppo_surrogate_constant_inflow_m5c_seed0_20260512_021425/best_model.zip`
- M6 (fail, 20k):
  `runs/rl/ppo_sumo_constant_inflow_m6_seed0_20260512_023250/final_model.zip`
- M6b (fail, 100k):
  `runs/rl/ppo_sumo_constant_inflow_m6b_seed0_100k_20260512_045505/final_model.zip`

Notes:

- All three runs used `α = β = γ = 1.0`, `ρ_freeflow = 20.0`, `queue_norm = 100.0`.
- M5c surrogate resolved via `auto` → `runs/surrogate/deeponet_constant_inflow_20260511_234849/best.pt`.
- The replay is deterministic and demand is single-element `[1500]`/`[2000]`,
  so all 5 episodes per condition produced identical trajectories (per-term
  std = 0). The episode-mean numbers below are exact.

## Sanity checks

- M5c action mean = **0.686** (matches recorded 0.688 in `_progress/milestone_5c_progress.md`).
- M6 action mean = **0.003** (matches recorded 0.003 in `_progress/milestone_6_progress.md`).
- M6b action mean = **0.000** (matches recorded 0.000 in `_progress/milestone_6b_progress.md`).
- Penalty-sum vs `-reward` consistency error ≤ **3.8e-6** across all (condition, episode, step).

## Headline numbers

Episode integrals of each reward term (sum over 120 control steps; positive
quantities — the negative sign in the reward goes onto the sum):

| Condition | density-excess | queue | std(ρ) | Total | Queue share |
|---|---:|---:|---:|---:|---:|
| **M5c (success, u≈0.69)** | 68 (5.0%) | 593 (43.6%) | 698 (51.4%) | 1 359 | 44% |
| **M6 (fail, u≈0)**  | 0 (0.0%) | 2 538 (87.9%) | 350 (12.1%) | 2 887 | **88%** |
| **M6b (fail, u≡0)** | 0 (0.0%) | 2 557 (88.0%) | 349 (12.0%) | 2 906 | **88%** |

Plots:
- `runs/analysis/20260512_230924_reward_term_comparison/stacked_bar.png`
- `runs/analysis/20260512_230924_reward_term_comparison/timeseries.png`
- `runs/analysis/20260512_230924_reward_term_comparison/comparison_summary.json`

## Findings

1. **The queue term dominates M6 / M6b reward integrals (88%) and is ~4.3× the
   M5c queue integral.** M5c's reward is split roughly 44% queue / 51% std /
   5% density-excess — a balanced landscape. M6 / M6b's reward is almost
   entirely the quadratic queue blow-up.

2. **But the queue term is dominating *because* the policy collapsed, not
   driving the collapse.** Reading the timeseries panels:
   - **Queue panel**: at u ≡ 0, queue grows monotonically; the quadratic
     penalty rises smoothly from 0 to 64 per step over the episode. This is
     a *clean, monotone gradient signal* that should push the policy toward
     larger u. The signal exists; the policy just can't see it.
   - **Std(ρ) panel**: M6 / M6b sits at the trivial ~2.9 per-step floor
     (uniform free-flow density when there's no ramp inflow). M5c's policy
     drives std up to ~7-8 by mid-episode by admitting ramp vehicles —
     creating exactly the spatial nonuniformity that `γ · std(ρ)` is
     designed to penalize.
   - **Density-excess panel**: identically 0 for M6 / M6b (mainline never
     exceeds the 20 veh/km threshold without ramp inflow). M5c's policy
     drives mean density above the threshold for ~80 of 120 steps,
     incurring 1-2 per step of penalty — this is the M5c-by-design tradeoff
     (small density-excess cost in exchange for much lower queue cost).

3. **The action statistics make the diagnosis concrete:**
   - M6 action: `mean = 0.003, std = 0.015, max = 0.096`
   - M6b action: `mean = 0.000, std = 0.000` (literally constant u = 0)
   - With action std ≈ 0.015 (M6) or 0.0 (M6b), PPO's training samples
     never explored beyond u ≈ 0.1. The clean queue-penalty gradient is
     unreachable from inside that bubble.

4. **Implication for the rescaling question.** Rescaling the three terms to
   ~[0, 1] would change the relative magnitudes that the policy sees, but
   would not change:
   - the *sign* of the gradient (still pulls away from u = 0),
   - the *exploration radius* of a collapsed policy (still stuck at u = 0),
   - the *signal-to-noise* of a single-direction trajectory (same one curve,
     just rescaled).

   In other words, the diagnostic data say the M5c reward shape produces
   the *right* signal — and we already shipped the right counter-measures
   for the *exploration* failure: `ent_coef = 0.01` (entropy bonus to
   prevent action-std collapse) + 3×512 actor/critic (more capacity to
   represent a non-degenerate Gaussian over a 22-dim observation).

## Recommendation

**Keep the M5c reward shape unchanged (`α = β = γ = 1.0`, `ρ_freeflow = 20`,
`queue_norm = 100`).**

The next SUMO PPO training should:
1. Use the current architecture + entropy settings (already on `main` as of
   commit `9925b62`).
2. Optionally: increase initial policy log-std slightly to widen early
   exploration, or briefly anneal `ent_coef` from a higher starting value
   (e.g. 0.05 → 0.01). Both are SB3-supported and avoid touching the reward.
3. Re-run M6 / M6b at the new settings and check whether action mean lands
   in the interior; if it does, the analysis above is corroborated.

If the new SUMO PPO still collapses with the entropy / network changes, then
re-open the rescaling discussion — but with a *different* hypothesis
(e.g. value-function variance is being driven by one term's larger magnitude,
hurting PPO's advantage estimation). That would call for `(queue/Q_max)² → min(1, ...)`-style soft saturation, not a uniform rescale.

## Reproducing

```
.venv-traffic-rl/python.exe scripts/analyze_reward_terms.py
```

Outputs land in `runs/analysis/<timestamp>_reward_term_comparison/`. Wall
clock: ~2.5 minutes (6 s surrogate + 72 s × 2 SUMO).
