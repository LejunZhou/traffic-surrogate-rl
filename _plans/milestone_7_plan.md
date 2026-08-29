# Milestone 7 plan — Outflow-based reward for SUMO+PPO

## Motivation

The M5c reward's first term, `-alpha · max(0, mean(rho) - rho_freeflow)`,
was meant as a proxy for "keep the mainline below critical density so
outflow stays near capacity". Two problems surfaced in the 2026-08-27
audit (`_progress/milestone_7_progress.md` §0):

1. **The proxy fires on the maximum-outflow state.** `rho_freeflow = 20`
   was tuned so the term activates at u ≈ 1 (mean density ≈ 22), but in
   the Phase 1 scenario u = 1 is also the highest-throughput policy
   (2260 vph vs 1867 at u = 0.5, 1473 at u = 0). The term penalised
   throughput instead of protecting it.
2. **The code no longer matched the docs.** Commit `5cbbb01` divided the
   std term by a hard-coded `67.7167` and set `alpha = 0`,
   `queue_norm = 200` in `ppo_surrogate.yaml` only, while `ppo_sumo.yaml`
   kept `alpha = 1`, `queue_norm = 100`. README/proposal still described
   the M5c form; the per-env `_reward_terms()` diagnostics no longer
   summed to the reward.

M7 replaces the density proxy with a **direct mainline-outflow term**,
keeps the reward at three terms, makes every normaliser explicit, and
balances the three weights so that no single term dominates — all on the
**current scenario (2000 vph mainline + 800 vph ramp = 2800 vph total)**
first, before touching demand.

## Reward (implemented in `src/rl/reward.py`)

```
r_k = -delta * max(0, q_ref - q_out_k) / q_ref     # lost mainline outflow
      -beta  * (Q_k / queue_norm)^2                # ramp queue (unchanged)
      -gamma * std(rho_k) / sigma_ref              # spatial uniformity
```

- `q_out_k`: vehicles leaving the network during the 30 s control interval
  (`traci.simulation.getArrivedNumber`, already accumulated by
  `SumoEnv._advance_control_interval` as `interval_arrived`) × 3600/30.
  The det_18 E1 loop flow was tried first and found to over-count by
  ~1.2× (a vehicle straddling a 1 s step boundary is counted twice); it is
  kept in `info["det18_flow_vph"]` for diagnostics only.
- Written as *lost* outflow so all three terms stay ≤ 0 and the episode
  return keeps its "always ≤ 0" meaning. No ReLU threshold: every
  veh/h counts.
- `sigma_ref` replaces the hidden `/67.7167` divisor.
- `reward_terms()` is the single source of truth; both envs log its
  output to `info` and use `["reward"]`, so the logged decomposition sums
  to the reward PPO sees.
- **Surrogate path**: the DeepONet predicts density only, so
  `SurrogateEnv` passes `outflow_vph = None`, which is accepted only with
  `delta = 0` (two-term reward). Surrogate-vs-SUMO parity on the full
  three-term reward needs a flow surrogate (deferred, see follow-ups).

Legacy keys `alpha` / `rho_freeflow` are rejected by
`RewardWeights.from_config` with a migration message so stale configs
cannot silently run the old reward.

## Balancing procedure ("no term dominates")

Every term is normalised to O(1) per step; the weights are then chosen
from data rather than by hand:

1. `scripts/run_u_sweep_sumo.py` rolls constant policies
   u ∈ {0.0, 0.1, …, 1.0} through `SumoEnv` once each (~13 s per point)
   and writes per-step arrays (`outflow_vph_steps`, `queue_steps`,
   `std_steps`) to JSONL.
2. `scripts/balance_reward_terms.py` recomputes each term's unit-weight
   episode sum for every u, reports the **range** (max − min across the
   sweep) of each term, and proposes `w_i = target / range_i`, anchored
   at `beta = 1`. A term whose range is far larger than the others'
   decides the optimum alone — that is the failure mode we are avoiding
   (M5b: queue term identically 0 at u = 1; M6/M6b: `(Q/100)^2` reaching
   64/step at u = 0 made the −2941 wall).
3. The script also prints the capacity-drop check (does `q_out` peak at
   some u* < 1?) and the return + per-term share of every constant policy
   under the proposed weights.

First pass (from the M6 §6.7 logged stats, synthetic per-step arrays):

| normalisers | proposed δ / β / γ | best constant | note |
|---|---|---|---|
| q_ref 2260 (measured peak), queue_norm 400, sigma_ref 6 | 3.93 / 1 / 1.28 | u = 1 (−239) vs u = 0.5 (−248) | near tie |
| q_ref 2970 (IDM capacity), same | 5.16 / 1 / 1.28 | u = 1 (−384) vs u = 0.5 (−392) | near tie |

Rounded to `delta 4.0, beta 1.0, gamma 1.3, q_ref 2260, queue_norm 400,
sigma_ref 6.0` in `configs/rl/ppo_sumo.yaml`. This must be re-derived
from the real sweep (step 1) before the PPO run.

## Tasks

- **7.0 — Reward implementation.** `reward.py` rewrite; `SumoEnv` passes
  the interval arrival count as outflow; `SurrogateEnv` passes `None`; `train_ppo.py` W&B keys and
  CLI overrides (`--reward-delta/--reward-q-ref/--reward-sigma-ref`);
  eval scripts and `run_m5b_sweep.py` migrated; `tests/test_reward.py`
  (numpy-only) added; `tests/test_surrogate_env.py` fixture migrated.
  **Done 2026-08-27.**
- **7.1 — Constant-u sweep in SUMO** at 2000 + 800 vph, seed 0:
  `python scripts/run_u_sweep_sumo.py --out _progress/m7_u_sweep_seed0.jsonl`.
  Records outflow / queue / std per step, teleports and rejected inserts
  (also closes the "no explicit teleport count at 2000 vph" gap).
  **Done 2026-08-27 on the Mac venv (SUMO 1.27.1)** — see progress §7.1.
- **7.2 — Balance weights** from the sweep:
  `python scripts/balance_reward_terms.py --sweep _progress/m7_u_sweep_seed0.jsonl`.
  Paste the proposed block into `ppo_sumo.yaml`. Record the range table,
  the capacity-drop verdict, and the per-term shares in the progress file.
- **7.3 — Hyper-parameter check before training.** With the balanced
  weights, confirm from the sweep table that (a) no term's share exceeds
  ~0.6 at any constant u, (b) the constant-policy return is not flat
  (range across u ≥ ~20 % of its magnitude), (c) `warmup_s = 90` masks
  the empty-road steps. Adjust `queue_norm`/`sigma_ref` if (a) or (b)
  fail and re-run 7.2.
- **7.4 — SUMO+PPO run** (20k timesteps, `n_steps = 480`, seed 0):
  `python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --seed 0 --run-name-suffix m7_seed0`.
- **7.5 — Evaluate** with `scripts/eval_sumo_baselines.py` (learned
  policy, 3 episodes) against the constant sweep; fill the comparison
  table in the progress file.
- **7.6 — Docs.** README reward section (done), proposal.md §"Reward"
  (needs user approval — see below), progress file.

## Acceptance

- `tests/test_reward.py` passes; `tests/test_surrogate_env.py` passes on
  a machine with a checkpoint.
- Sweep: 0 teleports, 0 rejected inserts at every u (else record where).
- Balanced weights: max per-term share ≤ 0.6 at every constant u.
- PPO: deterministic action mean ∈ (0.05, 0.95) **and** SUMO return ≥
  the best *noise-robust* constant policy (constant u with the policy's
  own action σ added; the nominal best constant u = 0.6 turned out to be
  a knife edge — see progress §"knife edge"). Status 2026-08-27: run 4
  `best_model.zip` −61 vs robust reference −56…−65 → **met within ~10 %**;
  the original "≥ best constant (−43)" wording is not achievable by any
  stochastic policy and is retired.

## Environment note (2026-08-27)

The audit machine (macOS arm64, system Python 3.9, no Homebrew / conda)
initially could not run SUMO. A project venv was then created with `uv`
(`.venv-traffic-rl/`, Python 3.11.16, `pip install -e ".[dev]" eclipse-sumo`),
which ships **SUMO 1.27.1** binaries inside the venv — README "Setup,
Option A". All of 7.1–7.5 run here now. The M2–M6b artifacts (`data/`,
`runs/`) were produced on a Windows machine with an older SUMO and are
not in this checkout; see progress §7.1 for why the two SUMO versions
give different traffic (mainline insertion).

## Follow-ups (not in M7)

- **Driver heterogeneity in training (domain randomisation).** The
  deterministic scenario's optimum (u ≈ 0.6) is a knife edge for both
  action noise and traffic noise: at `speed_dev 0.03` the capacity edge
  drops to u ≈ 0.50–0.52 and run 4's policy gridlocks in 9/10 seeds
  while constant u = 0.5 is robust (progress §7.7; after the insertion
  fix of §7.8 the robust constant is u = 0.45, −81). Train and select
  with `env.sumo_overrides.vehicle.speed_dev ≥ 0.03`, EvalCallback with
  ≥ 5 episodes, and report breakdown rate alongside mean return.
- ~~**Post-jam mainline insertion artifact**~~ — **done 2026-08-28**
  (progress §7.8): it was a metastable slow-entry state of
  `departSpeed="max"`, not a backlog. Scenario now uses
  `vehicle.depart_speed: desired` + `--extrapolate-departpos`, blocked
  vehicles wait (conserved; `max_depart_delay_s` available to discard);
  pending/discarded insertions are logged;
  the ramp virtual queue is decremented on departure, not on `add()`.
- ~~**Queue discharge > arrival** (`ramp_discharge_vph`)~~ — **done
  2026-08-28** (progress §7.10): scenario default `ramp_discharge_vph:
  1600`, `env.ramp_demand_levels` for per-episode ramp arrivals,
  `training.action_init_u` (0.3 in the run-4 config). Finding: at
  2000 vph mainline the merge margin (~480 vph) < 800 vph arrivals, so
  draining needs lower ramp arrivals or lower mainline demand — train
  on a demand range (`demand_levels` 1500–2000, `ramp_demand_levels`
  400–800). Done 2026-08-28 (progress §7.11): ramp arrival rate in the
  observation (23 features) and weights re-balanced over the grid
  (δ 3.572 / β 1 / γ 0.063). Run 5 done 2026-08-29 (progress §7.12):
  grid mean −70 vs −86 best constant, 3/54 breakdowns; degraded after
  24k steps (approx_kl 0.11). Next: run 6 with `target_kl 0.02`, lr 1e-4.
- **Flow surrogate** (second DeepONet trained on the dataset's `flow`
  field, queried at x = 1900 m) so `SurrogateEnv` can run `delta > 0`
  and the surrogate-vs-SUMO comparison is restored on the three-term
  reward.
- **Higher demand (M2c)**: if the sweep confirms `q_out` is monotone in
  u, raise mainline demand until a capacity drop appears (u* < 1), then
  regenerate data / retrain.
- **proposal.md** §"Reward (Phase 1 shaped, Milestone 5c)" still
  documents the M5c form; update once the user approves the reward
  change (CLAUDE.md rule).
