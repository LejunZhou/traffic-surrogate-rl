# Milestone 7 progress — Outflow-based reward for SUMO+PPO

Running log for `_plans/milestone_7_plan.md`.

## 2026-08-27 — §0 Audit that motivated M7

Read-only review of the repo state at commit `647993a` (fresh clone on a
macOS machine without SUMO/conda). Findings that fed the plan:

**Highway setting (unchanged, kept simple):** 1-lane mainline, 2000 m,
on-ramp at 1300 m into a 100 m acceleration lane (`highway_accel`,
2 lanes) with a zipper lane-drop at 1400 m; 19 detectors at 100…1900 m;
2000 vph mainline + 800 vph ramp cap = **2800 vph total at u = 1**;
30 s control step, 120 steps. All logged SUMO runs on this geometry show
0 teleports and 0 rejected inserts (M1.1 rerun, M2 smoke + full 120
rollouts). The "very bad queue" episodes in the logs were the reward's
non-draining virtual queue under a linear/heavy β (M5b β = 3 → queue
517–663, returns −100k…−123k), not a physical merge problem.

**Reward drift found:**

| item | docs (README / proposal / docstring) | code at `647993a` |
|---|---|---|
| std term | `-γ·std(ρ)` | `-γ·std(ρ)/67.7167` (commit `5cbbb01`, comment "std of the surrogate training data"; the dataset std is 5.97) |
| `ppo_surrogate.yaml` | α = 1, queue_norm = 100 ("identical to SUMO") | α = **0**, queue_norm = **200** |
| `ppo_sumo.yaml` | same | α = 1, queue_norm = 100 |
| per-env `_reward_terms()` (info / W&B) | sums to reward | `std_penalty` lacked the divisor → decomposition ≠ reward |

Effect: with the current code, u ≡ 1 is the best constant policy under
both configs (≈ −303 vs −657 at u = 0.5 for the SUMO config; ≈ −17 vs
−171 for the surrogate config) — the u = 1 corner that M5c was built to
break is back. Every logged M5c/M6/M6b number predates these changes.

**First term is a mis-calibrated outflow proxy:** `rho_freeflow = 20`
fires at u ≈ 1 (mean ρ ≈ 22), but u = 1 is the maximum-outflow state
here (M6 §6.7: 1473 / 1867 / 2260 vph at u = 0 / 0.5 / 1); IDM 1-lane
capacity at τ = 1.0 s is ≈ 2970 vph > 2800 vph demand, so there is no
capacity drop to protect against. Decision (user, 2026-08-27): replace
the term with a direct mainline-outflow term, keep three terms, try at
the current 2800 vph first, and balance the terms before training.

## 7.0 — Reward implementation (done)

Files changed:

- `src/rl/reward.py` — rewritten. `RewardWeights(delta, beta, gamma,
  q_ref, queue_norm, sigma_ref)`; `reward_terms()` returns the full
  decomposition (`outflow_vph`, `lost_outflow_frac`, `outflow_penalty`,
  `queue_penalty`, `std_penalty`, `reward`) and `compute_reward()` is
  `reward_terms()[...]["reward"]`. `from_config` **raises** on legacy
  `alpha` / `rho_freeflow` with a migration message; unknown keys warn.
  `outflow_vph=None` accepted only when `delta == 0`.
- `src/rl/sumo_env_wrapper.py` — `step()` passes `outflow_vph` = network
  arrivals per interval × 3600/30 (initially det_18 loop flow; see §7.1 for
  why that was replaced); `det18_flow_vph` kept in info; `_reward_terms`
  delegates to `reward.reward_terms`; info keys `density_excess*` →
  `outflow_vph / lost_outflow_frac / outflow_penalty`, `reward_alpha` →
  `reward_delta`, `reward_rho_freeflow` → `reward_q_ref / reward_sigma_ref`.
- `src/rl/surrogate_env.py` — same info/terms changes; passes
  `outflow_vph=None` (no flow prediction → surrogate path is delta = 0).
- `src/rl/train_ppo.py` — W&B info keys updated; CLI `--reward-alpha` →
  `--reward-delta`, `--reward-rho-freeflow` → `--reward-q-ref`, new
  `--reward-sigma-ref`.
- `scripts/eval_sumo_baselines.py` — rewritten: new override flags
  (`--delta/--q-ref/--sigma-ref`, `--config`), per-term episode sums,
  outflow stats, teleport + rejected-insert counts, and per-step arrays
  in `--json` output; accepts an external `env` for sweeps.
- `scripts/run_u_sweep_sumo.py` — **new**: constant-u sweep → JSONL.
- `scripts/balance_reward_terms.py` — **new**: numpy-only; capacity-drop
  check, unit-weight sums, ranges, proposed weights (equal ranges,
  β = 1 anchor), return + per-term shares per u, YAML block to paste.
- `scripts/eval_constant_baselines.py`, `scripts/run_m5b_sweep.py` —
  `alpha` → `delta` (surrogate path passes `delta = 0`).
- `configs/rl/ppo_sumo.yaml` — new reward block (see 7.2), `warmup_s: 90`,
  `n_steps: 480`, run name `ppo_sumo_outflow_reward`.
- `configs/rl/ppo_surrogate.yaml` — `alpha: 0` → `delta: 0`,
  `sigma_ref: 67.7167` (explicit legacy divisor so the "Working PPO
  Surrogate" run stays reproducible), `rho_freeflow` removed.
- `tests/test_reward.py` — **new**, 8 numpy-only tests (decomposition sums
  to reward, component values, monotone/saturating outflow term,
  independent weight scaling, `None` outflow ⇒ `delta = 0`, legacy-key
  rejection, config parsing, input validation).
- `tests/test_surrogate_env.py` — fixture migrated to `delta: 0`.
- `README.md` — reward section rewritten (formula, per-config weight
  table, balancing procedure, M5→M7 history). `proposal.md` **not**
  edited (needs approval; see plan follow-ups).

Verification on this machine (system Python 3.9 + scratch venv with
numpy/pytest/pyyaml; no torch/SUMO available):

```
python -m py_compile  <all 11 touched .py files>      -> OK
PYTHONPATH=src pytest tests/test_reward.py -q           -> 8 passed in 0.03s
RewardWeights.from_config(ppo_sumo.yaml env.reward)     -> delta=4.0 beta=1.0 gamma=1.3 q_ref=2260 queue_norm=400 sigma_ref=6.0
RewardWeights.from_config(ppo_surrogate.yaml env.reward)-> delta=0.0 beta=1.0 gamma=1.0 queue_norm=200 sigma_ref=67.7167
grep for density_excess|reward_alpha|rho_freeflow       -> only matplotlib alpha= and the legacy-rejection test remain
```

Not verified here: `tests/test_surrogate_env.py` (needs torch + an M3
checkpoint) and any `SumoEnv` run (needs SUMO). The env edits are
string-exact patches of the two call sites plus the shared
`_reward_terms` body, and both files byte-compile.

## Environment unblock (2026-08-27, same day)

Created a project venv on the Mac: `uv` → Python 3.11.16 →
`.venv-traffic-rl/` with `pip install -e ".[dev]" eclipse-sumo`
(SUMO **1.27.1** binaries inside the venv; README "Setup, Option A").
Full test suite: `8 passed, 7 skipped` (surrogate-env tests skip without
an M3 checkpoint). While doing this, found and fixed a pre-existing
breakage: `find_latest_checkpoint` was imported by
`scripts/eval_constant_baselines.py` and `tests/test_surrogate_env.py`
but had been deleted from `rl/surrogate_env.py` in commit `1672424`
("train ppo") — restored from `feb19d1`. Smoke rollout
(`scripts/run_rollout.py`, u = 0.5): 0 teleports, 400/400 inserts.

## 7.1 — Constant-u sweep in SUMO (done)

`python scripts/run_u_sweep_sumo.py --out _progress/m7_u_sweep_seed0.jsonl`
(11 points, seed 0, ~15 s per episode on the Mac).

**First attempt used the det_18 E1 loop flow as `q_out` and was wrong**
(archived as `_progress/m7_u_sweep_seed0_det18loopflow_SUPERSEDED.jsonl`):
loop flow read 2399 vph at u = 0 for a 2000 vph input and 3032 vph at
u = 0.6 for 2480 vph demand — a consistent ~1.2× over-count. Cause: E1
`getLastStepVehicleNumber` counts a vehicle in every 1 s step it touches
the loop, so a vehicle straddling a step boundary is counted twice
(≈ 1 + L/(v·dt) = 1 + 5/27.7). Fixed in `SumoEnv.step`: `q_out` is now the
exact **network arrival count per interval** (`interval_arrived` ×
3600/30); loop flow is kept in `info["det18_flow_vph"]`. (This bias also
sits in the dataset's `flow` field and hence in ρ = q/v — noted as an
open item.)

**Corrected sweep (arrival-based outflow):**

| u | q_out vph | ρ mean | ρ std | ρ max | Q final | teleports | rejected |
|---|---|---|---|---|---|---|---|
| 0.0 | 2000 | 21.6 | 3.0 | 28 | 796 | 0 | 0 |
| 0.1 | 2079 | 23.1 | 4.3 | 34 | 717 | 0 | 0 |
| 0.2 | 2159 | 23.6 | 4.8 | 37 | 638 | 0 | 0 |
| 0.3 | 2238 | 24.1 | 4.9 | 33 | 558 | 0 | 0 |
| 0.4 | 2318 | 25.2 | 6.5 | 41 | 478 | 0 | 0 |
| 0.5 | 2397 | 24.9 | 6.7 | 42 | 398 | 0 | 0 |
| **0.6** | **2476** | 27.7 | 9.3 | 46 | 319 | 0 | 0 |
| 0.7 | 1804 | 138.9 | 92.3 | 305 | 239 | 0 | 0 |
| 0.8 | 1742 | 149.5 | 98.5 | 308 | 159 | 0 | 0 |
| 0.9 | 1687 | 191.0 | 128.9 | 341 | 80 | 0 | 0 |
| 1.0 | 1695 | 195.3 | 132.4 | 409 | 0 | 0 | 0 |

Findings:

1. **There IS a capacity drop at 2800 vph total.** Served flow is exactly
   2000 + 800·u up to u = 0.6 (2476 vph), then the merge breaks down and
   the mainline gridlocks: outflow falls to ~1700 vph, mean density
   140–195 veh/km. Merge capacity ≈ 2500 vph. The outflow term is
   well-posed here — no demand increase needed.
2. **This contradicts the logged M6 table, and the reason is SUMO
   version.** M6 (Windows, older SUMO) reported throughput 1473 / 1867 /
   2260 vph at u = 0 / 0.5 / 1 — a constant ≈ 530 vph deficit vs demand,
   i.e. its mainline flow only delivered ≈ 1470 of the requested 2000
   vph (insertion-limited). SUMO 1.27.1 delivers the full 2000 vph
   (u = 0 → 2000 exactly). So every logged M5c/M6/M6b result describes
   an effectively ~1500 vph mainline with no breakdown; at the real
   2000 + 800 vph the scenario is over-saturated above u = 0.6.
3. **Density-estimator ceiling.** ρ up to 409 veh/km is unphysical (jam
   density ≈ 143 veh/km/lane for 5 m + 2 m): the occupancy fallback
   (`occ × 1000 / 5 m`, ignores minGap) saturates at 200 per lane, and
   the two lanes of `highway_accel` are summed. It only affects the
   breakdown regime, but it inflates `std` there by ~2×.
4. 0 teleports, 0 rejected inserts at every u, including full gridlock.

## 7.2 — Balance weights (done, from the real sweep)

`python scripts/balance_reward_terms.py --sweep _progress/m7_u_sweep_seed0.jsonl`
(q_ref auto = 2476, queue_norm 400, sigma_ref 6, skip 3 warmup steps):

```
Unit-weight sums   S_outflow  S_queue   S_std        range: outflow 35.9
  u=0.0               22.5     159.8     44.9               queue  159.8
  u=0.6                1.4      25.6    175.5               std   2388.1
  u=1.0               36.9       0.0   2433.0
Proposed (equal ranges, beta=1):  delta=4.45  beta=1.00  gamma=0.067
```

Return and per-term shares under the proposal:

| u | return | outflow | queue | std | max share |
|---|---|---|---|---|---|
| 0.0 | −263 | −100 | −160 | −3 | 0.61 (queue) |
| 0.3 | −134 | −50 | −78 | −6 | 0.58 |
| 0.5 | −65 | −17 | −40 | −8 | 0.62 (queue) |
| **0.6** | **−43** | −6 | −26 | −12 | 0.59 |
| 0.7 | −267 | −141 | −14 | −111 | 0.53 (outflow) |
| 1.0 | −327 | −164 | 0 | −163 | 0.50 |

Adopted in `configs/rl/ppo_sumo.yaml`: **delta 4.45, beta 1.0,
gamma 0.067, q_ref 2476, queue_norm 400, sigma_ref 6.0, warmup_s 90**.

Plan §7.3 check: (a) max share 0.62 — marginally above the 0.6 target at
u ≤ 0.5, where the queue term leads; accepted because it is the only
term that distinguishes u = 0.5 from u = 0.6 in free flow. (b) Return
spans −43 … −327 (not flat). (c) `warmup_s = 90` masks steps 0–2; first
arrivals at ~65 s.

Reading: with the terms balanced, the reward now says "meter at the
capacity edge (u ≈ 0.6)": above it outflow and std both fire (breakdown
≈ as bad as a closed ramp), below it the queue term climbs. The std term
is essentially a breakdown indicator at γ = 0.067 (share ≤ 0.27 in free
flow); if you want it to shape free-flow uniformity too, γ ≈ 0.2 keeps
u = 0.6 optimal while raising the breakdown cost to ≈ −600.

## 7.3 — Reward-weight sensitivity (corner check)

Offline grid over the real sweep (numpy only): δ ∈ {1, 2, 4.45, 8, 16} ×
γ ∈ {0, 0.067, 0.2, 0.5, 1} × queue_norm ∈ {200, 400, 800}, β = 1,
q_ref 2476, σ_ref 6 → best constant policy:

- 72 / 75 combinations interior (55× u = 0.6, 15× u = 0.5, 2× u = 0.3).
- 3 corner cases only at pathological settings: (δ ≤ 2, γ = 0, qn = 200)
  → u = 1; (δ = 1, γ = 1, qn = 800) → u = 0.
- Adopted weights: u = 0.6 at −43 beats u = 0 (−263) and u = 1 (−327) by
  219–284 — a 5–6× margin. Return is monotone from u = 0 to 0.6 (clear
  gradient for PPO), then a cliff at u ≥ 0.7 (breakdown).
- A gentler alternative (δ 2, γ 0.2, qn 800) keeps corners dominated but
  flattens the curve (−60 … −44 for u = 0.3 … 0.6) — weaker signal. Kept
  the adopted weights.

Caveats: constant policies only; single SUMO seed and demand. Re-run
the sweep + balancer after any scenario change.

## 7.4 – 7.5 — SUMO+PPO run and evaluation

### Run 1 — M6 hyper-parameters + balanced outflow reward (seed 0, 20k steps)

`WANDB_MODE=offline python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --seed 0 --run-name-suffix m7_seed0`
→ `runs/rl/ppo_sumo_outflow_reward_m7_seed0_20260827_143547/` (config as
of 7.2: δ 4.45 / β 1 / γ 0.067, `n_steps 480`, lr 1e-4, `symmetric_action`
**not yet present**).

Training trajectory (`ep_rew_mean`, stochastic policy): −213 (iter 1) →
−198 (iter 5) → −194 (iter 10–14) → −193 (iter 17). PPO stats throughout:
`approx_kl ≈ 8e-5`, `clip_fraction 0`, policy `std 1.00` (never moved),
`explained_variance −0.25`.

Deterministic eval of the 4800-step checkpoint in SUMO (seed 0):
**action mean 0.007** (std 0.048, max 0.41), return **−259.3**
[outflow −99 | queue −157 | std −3] — indistinguishable from constant
u = 0 (−262.9). The stochastic `ep_rew_mean ≈ −194` is better only because
`std = 1` sampling opens the ramp at random.

**Diagnosis (same mechanism as M6 / M6b, now explained):** SB3 initialises
the Gaussian mean at ≈ 0 (action-net gain 0.01) with std 1. On the [0, 1]
Box the initial *deterministic* action is u = 0 and half of every sampled
batch clips to 0. With lr 1e-4 and 5 epochs the update per iteration is
≈ 1e-4 in KL — the policy never leaves the corner in 20k steps. The reward
is not the problem: under it u = 0 is the second-worst constant policy
(−263) and u = 0.6 the best (−43).

**Final policy (20,160 steps, 42 PPO iterations, 17 min wall clock):**
deterministic eval in SUMO → return **−196.3** [outflow −71 | queue −120 |
std −5], action mean **0.183** (std 0.167, max 0.77), q_out 2102 vph,
queue final 651, 0 teleports. So by the end PPO had crept off the u = 0
corner to a low-u policy (between constant u = 0.1 at −218 and u = 0.2
at −175), but it is far from the u = 0.6 optimum (−43). Note:
`--n-episodes 3` returned three bit-identical episodes — the SUMO seed
is pinned in `phase1_1.yaml` (`seed: 42`) and IDM is deterministic
(σ = 0), so episode-to-episode variance needs a seed sweep on the env
side, not repeated episodes.

**Fix implemented for run 2:** `env.symmetric_action: true` →
`gymnasium.wrappers.RescaleAction(env, -1, 1)` in `train_ppo._make_env`,
so PPO's initial mean maps to u = 0.5 (return −65, near the optimum) and
clipping is symmetric. `eval_sumo_baselines.py` /
`eval_constant_baselines.py` detect a model trained on [−1, 1] and map
back. One change at a time: lr, `n_steps`, entropy untouched.

```bash
# 7.4: train (≈40–50 min at 20k timesteps on the Mac; W&B offline)
WANDB_MODE=offline python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --seed 0 --run-name-suffix m7_seed0
# 7.5: evaluate against the sweep
python scripts/eval_sumo_baselines.py --policy runs/rl/<run>/final_model.zip --seed 0 --n-episodes 3
```

### Run 2 — same config + `symmetric_action: true` (seed 0) — STOPPED at 4800 steps by user

`WANDB_MODE=offline python -m rl.train_ppo --config configs/rl/ppo_sumo.yaml --seed 0 --run-name-suffix m7_seed0_symact`
→ `runs/rl/ppo_sumo_outflow_reward_m7_seed0_symact_20260827_145247/`.
Stopped on request after 10 iterations (4800 / 20160 steps, ~5 min);
checkpoints `ppo_sumo_2400_steps.zip` and `ppo_sumo_4800_steps.zip` saved.

Stochastic `ep_rew_mean` per iteration: −229, −222, −217, −212, −209,
−206, −201, −198, −196, −194 — **monotonically rising** (run 1 was flat at
−193 from iteration 5 onward). Policy `std` still ≈ 1.0 at the stop.

Deterministic eval of the 2400-step checkpoint (seed 0): **return −131.8**
[outflow −49 | queue −77 | std −6], action mean **0.312** (std 0.064,
max 0.79), q_out 2205 vph, queue final 549, 0 teleports — better than
run 1's *final* policy (−196) after one-eighth of the budget. The mean
moved down from the 0.5 init (risk-averse to the u ≥ 0.7 breakdown
while std is wide); expected to return toward u ≈ 0.6 as std shrinks.

**To continue:** rerun the command above (fresh run, same config), or
warm-start from the 4800-step checkpoint if a `--resume` path is added to
`train_ppo.py` (not implemented yet). Evaluate with
`python scripts/eval_sumo_baselines.py --policy <run>/final_model.zip --seed 0`
— the script detects the [−1, 1] action box and maps back to u.

### Run 2 (full) — `symmetric_action: true`, fresh 20k run (seed 0)

`… --run-name-suffix m7_seed0_symact_full` →
`runs/rl/ppo_sumo_outflow_reward_m7_seed0_symact_full_20260827_161639/`.
Much slower than run 1 (~2.8 min per iteration vs 0.4): exploring across
the whole u range means many breakdown episodes, which SUMO simulates
slowly.

Stochastic `ep_rew_mean` rises monotonically (−229 → −172 by iteration
29) but policy `std` never leaves 1.0 and `explained_variance` drifts to
−0.7. Deterministic checkpoint evals tell the real story:

| checkpoint | stochastic `ep_rew_mean` | deterministic return | mean u |
|---|---|---|---|
| 2,400 | −209 | **−132** | 0.31 |
| 12,000 | −179 | **−209** | 0.13 |

**The deterministic policy regresses while the stochastic objective
improves.** With σ ≈ 1 (≈ 0.5 in u-units) PPO optimises the expected
return of a very wide action distribution beside the u ≥ 0.7 cliff; the
best *mean* for that distribution is far below 0.6, so μ slides toward
the closed ramp. σ does not shrink because `ent_coef = 0.01` rewards
keeping it wide and lr 1e-4 barely moves `log_std`. This is an
exploration-schedule problem, not a reward problem (the constant-policy
curve still peaks at u = 0.6).

**Run 2 final (20,160 steps, 42 iterations, 4012 s ≈ 67 min):** stochastic
`ep_rew_mean` plateaued at −170 from iteration 31; `std` 0.997 at the end.
Deterministic eval (seed 0): **return −144.2** [outflow −52 | queue −86 |
std −6], action mean **0.297** (std 0.128, min 0.01, max 1.0), q_out
2192 vph, queue final 561, 0 teleports. Action profile per 10-step block:
u = 0.31, 0.16, 0.19, 0.22, 0.25, 0.27, 0.30, 0.33, 0.35, 0.37, 0.39, 0.41
— a real time-varying policy that opens the ramp progressively as the
queue grows. Recovered from the 12k-step dip (−209) and beats run 1
(−196), but still far from the constant u = 0.6 baseline (−43).

| run | action box | exploration | final deterministic return | mean u |
|---|---|---|---|---|
| constant u = 0.6 | — | — | **−43** | 0.6 |
| run 1 | [0, 1] | σ₀ 1, ent 0.01, lr 1e-4 | −196 | 0.18 |
| run 2 | [−1, 1] | σ₀ 1, ent 0.01, lr 1e-4 | −144 | 0.30 |
| run 3 | [−1, 1] | σ₀ 0.37, ent 0, lr 3e-4 | −112 (best ckpt −70 @ 2.4k) | 0.38 (0.49) |

**Run 3 (in progress) checkpoints, deterministic eval (seed 0):**

| checkpoint | return | u mean | u profile (10-step blocks) | notes |
|---|---|---|---|---|
| 2,400 | −70.3 | 0.487 | 0.48 … 0.49 flat | best learned policy so far |
| 9,600 | −95.5 | 0.419 | 0.39, 0.36 → 0.48 rising | mean drifting down as in run 2 (milder); stochastic `ep_rew_mean` −201, `std` 0.36 |
| 4,800 | −70.0 | 0.492 | 0.42, 0.48 … 0.51 | unchanged; `approx_kl 0.026`, `clip_fraction 0.14`, `std` 0.365 (not shrinking), `explained_variance` −0.4 |

Reading: the mean parks at u ≈ 0.5 because with σ_u ≈ 0.18 any move
toward 0.6 raises the probability of a sample ≥ 0.65 that gridlocks the
rest of the episode; the stochastic objective therefore peaks below the
cliff. σ does not shrink despite `ent_coef 0`, plausibly because the
critic cannot fit the jam-vs-free value gap and the log-std gradient is
noisy. Next lever if the final policy stays here: `log_std_init −2`
(σ_u ≈ 0.07) so the mean can sit near 0.6 without sampling over the
edge; alternatively a decaying-σ schedule.

**Run 3 launched:** `configs/rl/ppo_sumo_narrow_explore.yaml` — identical
except `policy_kwargs.log_std_init: -1.0` (σ₀ 0.37, ≈ 0.18 in u),
`ent_coef: 0.0`, `learning_rate: 3e-4`. Launch:
`WANDB_MODE=offline python -m rl.train_ppo --config configs/rl/ppo_sumo_narrow_explore.yaml --seed 0 --run-name-suffix m7_seed0`.

**Run 3 final (20,160 steps, 1019 s ≈ 17 min — faster than run 2 because
fewer breakdown episodes are sampled):** stochastic `ep_rew_mean` −226 →
−172; `std` 0.367 → 0.350 (barely moved). Deterministic evals (seed 0):

| checkpoint | return | u mean | u profile (10-step blocks) |
|---|---|---|---|
| 2,400 | **−70.3** | 0.487 | flat 0.48–0.49 |
| 4,800 | −70.0 | 0.492 | 0.42, 0.48 → 0.51 |
| 9,600 | −95.5 | 0.419 | 0.36 → 0.48 |
| 14,400 | −107.8 | 0.382 | 0.33 → 0.44 |
| 19,200 | −114.8 | 0.364 | 0.30 → 0.43 |
| final | −111.5 | 0.375 | 0.30 → 0.44 |

### Run 4 — run 3 + `log_std_init −2` + EvalCallback (seed 0, launched 2026-08-27 evening)

Config `configs/rl/ppo_sumo_m7_run4.yaml`. Code additions for this run:
- `SumoEnv` now owns a **labelled TraCI connection** per instance
  (`traci.start(..., label=…)` + `traci.switch` before every TraCI use), so
  a training env and an eval env can coexist in one process. Smoke-tested
  with two interleaved envs.
- `train_ppo.py`: `training.eval_freq > 0` adds an SB3 `EvalCallback`
  (deterministic, `n_eval_episodes`, separate `SumoEnv`) that writes
  `best_model.zip` to the run dir whenever `eval/mean_reward` improves.
  The deployed policy is now `best_model.zip`, not `final_model.zip`.
- `log_std_init −2.0` → σ 0.135 (≈ 0.07 in u) so the mean can sit near
  u = 0.6 without sampling over the breakdown edge.

**Run 4 deterministic evals (EvalCallback, seed 0):** 2,400 → −62.7,
4,800 → **−61.3** (best, saved), 7,200 → −61.8, 9,600 → −61.4. Stable —
no downward drift this time. Stochastic `ep_rew_mean` −135 … −160 with
σ = 0.134, i.e. still far below the deterministic value.

### The u = 0.6 optimum is a knife edge (noise-robustness test)

Constant policies with small Gaussian action noise (seed 0 SUMO, noise
seed 123), same reward:

| mean u | σ | return | q_out | ρ max | breakdown |
|---|---|---|---|---|---|
| 0.60 | 0.00 | −43.4 | 2476 | 28 | no |
| 0.60 | 0.03 | −243.8 | 1874 | 146 | **yes** |
| 0.60 | 0.07 | −236.1 | 1895 | 142 | **yes** |
| 0.55 | 0.03 | −55.7 | 2431 | 31 | no |
| 0.55 | 0.07 | −228.9 | 1926 | 138 | **yes** |
| 0.50 | 0.07 | −92.5 | 2323 | 101 | yes (late) |

±0.03 jitter around u = 0.6 is enough to tip the merge into gridlock,
and the gridlock is irreversible within the episode (the 2000 vph
mainline alone is close to the ~2500 vph merge capacity, so the jam
never clears). Consequences:

- **"Beat the best constant (−43)" is the wrong yardstick**: no
  stochastic policy can sit on that edge. The best *noise-robust*
  constant is ≈ u = 0.55 at σ 0.03 (−56); with σ 0.07 even u = 0.5 is
  marginal.
- **PPO's behaviour in runs 2–4 is correct risk management** under its
  own exploration noise, not a learning failure: the mean parks where
  the sampled actions stay off the edge. Run 4's deterministic −61 is
  within ~10 % of the robust optimum.
- The M7 acceptance criterion should be: deterministic return ≥ best
  noise-robust constant at the policy's own σ, evaluated across SUMO
  seeds. Run 4 meets the spirit of that; a formal check needs the
  seed sweep (env seed, not `--n-episodes`).
- Engineering levers if a higher deterministic return is wanted anyway:
  make the breakdown recoverable (queue discharge > arrival, or lower
  mainline demand so the jam can drain), or a slower control step so a
  single noisy action cannot trigger it.

**Run 4 final (20,160 steps, 1058 s ≈ 17.6 min).** EvalCallback trace:
2,400 → −62.7 · 4,800 → **−61.3 (best)** · 7,200 → −61.8 · 9,600 → −61.4 ·
12,000 → −77.1 · 14,400 → −71.9 · 16,800 → −73.5 · 19,200 → −70.4.
Stochastic `ep_rew_mean` −146 → −109 (still improving at the end);
`std` 0.135 → 0.131; `explained_variance` reached 0.9+ from iteration 26.

| model | return | u mean | u profile | q_out | Q final |
|---|---|---|---|---|---|
| `best_model.zip` (4,800) | **−61.3** [out −15 \| que −37 \| std −9] | 0.518 | flat 0.50–0.52 | 2367 | 384 |
| `final_model.zip` | −74.3 [out −21 \| que −45 \| std −9] | 0.477 | 0.45 → 0.50 | 2334 | 417 |

`best_model.zip` is the deployable policy: a steady u ≈ 0.52 that serves
2367 vph with no breakdown (ρ max 45).

## M7 verdict (2026-08-27)

| policy | deterministic return (seed 0) | u | robust to σ=0.03 noise? |
|---|---|---|---|
| constant u = 0.6 (best constant, knife edge) | −43 | 0.60 | **no** (−244) |
| constant u = 0.55 | −50 (interp.) | 0.55 | yes (−56) |
| **run 4 `best_model.zip`** (σ₀ 0.135 + EvalCallback) | **−61** | 0.52 flat | policy's own σ 0.13 |
| constant u = 0.5 | −65 | 0.50 | marginal at σ 0.07 |
| run 3 best checkpoint / final | −70 / −112 | 0.49 / 0.38 | — |
| run 4 `final_model.zip` | −74 | 0.48 | — |
| run 2 final | −144 | 0.30 rising | — |
| run 1 final | −196 | 0.18 | — |
| constant u = 0 / u = 1 | −263 / −327 | — | — |

What M7 established:

1. **The reward works.** The outflow term is measured correctly (arrival
   count), the three terms are balanced (equal ranges, max share 0.62),
   and the constant-policy curve peaks at an interior u = 0.6 with both
   corners 5–6× worse across a 75-point weight grid.
2. **The u = 0 collapse of M6/M6b/run 1 was an action-space artifact**
   (zero-initialised Gaussian on a [0, 1] box). `symmetric_action` fixes
   it: runs 2 and 3 start at u = 0.5 and never touch the corner.
3. **The remaining blocker is PPO's exploration next to a capacity
   cliff.** With a fixed-width Gaussian, the expected *stochastic* return
   is maximised by a mean well below u = 0.6 (samples above ~0.65
   gridlock the rest of the episode), so the deterministic policy drifts
   *down* over training in both run 2 (0.31 → 0.13 → 0.30) and run 3
   (0.49 → 0.38) even as `ep_rew_mean` improves. σ does not self-adapt
   (`ent_coef 0`, lr 3e-4 did not help), and the critic never fits the
   jam-vs-free value gap (`explained_variance` ≈ 0).
4. **The nominal best constant (u = 0.6, −43) is a knife edge**: ±0.03
   action noise gridlocks the merge irreversibly. No stochastic policy
   can sit there, so "return ≥ best constant" was the wrong acceptance
   test. Against the noise-robust reference (u ≈ 0.55 at σ 0.03 → −56;
   u = 0.5 → −65), run 4's `best_model.zip` (−61, u ≈ 0.52) is at the
   robust optimum within ~10 %. "Action mean ∈ (0.05, 0.95)" met by
   runs 2–4.
5. **Narrow exploration + best-checkpoint saving was the combination
   that worked** (run 4): σ₀ 0.135 keeps samples off the edge so the
   mean stops drifting, the critic finally fits (EV 0.9+), and the
   EvalCallback preserves the −61 policy that the final iteration
   (−74) would have lost.

Recommended next steps, in order of cost:

- ~~EvalCallback best-checkpoint saving~~ and ~~`log_std_init −2`~~ — done in
  run 4 (`configs/rl/ppo_sumo_m7_run4.yaml`); adopt as the SUMO default.
- **Seed sweep**: evaluate `best_model.zip` and the constant references
  across SUMO seeds (env `seed`, not `--n-episodes`) to put error bars on
  the −61 vs −56/−65 comparison.
- **Make the breakdown recoverable** if a higher deterministic return
  matters: queue discharge > arrival, or slightly lower mainline demand,
  so a single noisy action does not lock the episode.
- **Soften the cliff for learning only**: clip the per-step reward at
  e.g. −3 (breakdown steps are −2…−3 anyway) or use `VecNormalize`
  reward scaling so the critic can fit; evaluate on the unclipped reward.
- **State-dependent exploration** (`use_sde: true`) or a smaller
  network / larger `n_steps` (960) if the critic still fails.
- Longer term: flow surrogate for the surrogate path (delta > 0),
  regenerate M2/M3 on SUMO 1.27.1 with the loop-count fix, multi-demand
  (M2c).

## 7.6 — Is the breakdown recoverable? (2026-08-28)

Forced-breakdown experiment (seed 0, deterministic scenario): u = 1 for
10 min to gridlock the merge, then close the ramp.

| run | return | q_out last 30 min | ρ last 30 min | pending mainline insertions at end | arrived |
|---|---|---|---|---|---|
| u = 0 all hour | −263 | 2000 | 21.8 | 0 | 1962 |
| u = 1 ×10 min → u = 0 | −296 | **1550** | 16.8 | **455** | 1648 |
| u = 0.6 all hour | −43 | 2480 | 28.0 | 0 | 2430 |
| u = 0.7 ×10 min → u = 0.5 | −239 | 1892 | 117 (still jammed) | 438 | 1859 |

Findings:

1. **The jam itself clears** once the ramp is closed: ρ 200 → 17 veh/km
   within ~5 min, at 2000 and at 1800 vph. Lower demand is not needed for
   recoverability. Half-open (u = 0.5) is not enough to clear it.
2. **SUMO insertion artifact after a jam.** Mainline vehicles that could
   not be inserted during the jam stay *pending*; afterwards SUMO
   re-inserts them one per step under a safe-gap check at 120 km/h
   (`departSpeed="max"`), which caps insertion at ≈ 1550 vph < 2000 vph
   scheduled. The backlog therefore never drains (455 pending at the end)
   and the mainline runs at reduced demand for the rest of the hour —
   the episode silently becomes a different scenario. Likely the same
   mechanism throttled the older Windows SUMO to ≈ 1470 vph from t = 0.
   Fix: `departSpeed="desired"` / `"speedLimit"` or `--max-depart-delay`,
   and log pending insertions (`traci.simulation.getPendingVehicles`).
3. **Ramp backlog is permanent by design** (virtual-queue discharge ≤
   arrival), so a recovery closure is penalised for the rest of the
   episode. `ramp_discharge_vph > 800` would let a policy work it off.

So the cliff is a *slope* physically but a *cliff* in episode economics:
jam penalty + permanent ramp backlog + a quietly reduced mainline. Levers,
in priority: fix the insertion artifact (a scenario bug), then allow
queue discharge > arrival; demand reduction is unnecessary.

## 7.7 — Seed sweep for error bars (2026-08-28)

**The Phase 1 scenario is fully deterministic.** `SumoEnv.reset(seed=s)`
does forward `s` to SUMO `--seed`, but with IDM σ = 0, `speedDev = 0`, a
fixed-rate mainline flow and the deterministic ramp accumulator there is
nothing for SUMO's RNG to act on: seeds 0 / 1 / 42 / 999 / 31337 give
byte-identical density trajectories (same md5) and identical returns
(u = 0.5 → −65.01, u = 0.6 → −43.40 at every seed). A SUMO-seed sweep on
the training scenario therefore has zero-width error bars — which also
explains the bit-identical `--n-episodes 3` evals in §7.4.

To obtain variability, added `vehicle.speed_dev` (SUMO `speedDev`,
per-vehicle desired-speed spread) to `network_builder._write_routes`,
default 0.0 (no change to any existing run), settable per env via
`env.sumo_overrides`. New `scripts/run_seed_sweep_sumo.py` runs a list of
policies × seeds on one env with a chosen `--speed-dev`, writes JSONL +
summary (mean ± std, min/max, breakdown count).

First probe: `speedDev = 0.1` (SUMO's default) is a **harsher scenario** —
constant u = 0.5 gridlocks at every seed (−229, q_out ≈ 1890 vph), i.e. a
10 % speed spread pulls the merge capacity below 2400 vph. Two sweeps were
therefore run: 0.1 (robustness to realistic heterogeneity) and 0.03
(mild perturbation of the training scenario). Results below.

**`speedDev = 0.1` (SUMO default heterogeneity) — over-saturated, every
policy gridlocks:**

| policy | mean ± std (10 seeds) | min … max | q_out | breakdowns |
|---|---|---|---|---|
| constant u = 0.45 | −218.8 ± 2.9 | −223 … −213 | 1950 | 10/10 |
| constant u = 0.50 | −226.6 ± 2.9 | −231 … −222 | 1900 | 10/10 |
| run 4 `best_model.zip` (u ≈ 0.52) | −231.2 ± 2.2 | −235 … −228 | 1879 | 10/10 |
| constant u = 0.55 | −240.3 ± 3.4 | −245 … −234 | 1848 | 10/10 |
| constant u = 0.60 | −251.5 ± 1.4 | −254 … −249 | 1807 | 10/10 |

A 10 % speed spread pulls merge capacity below ~2350 vph, so at 2000 +
800 vph nothing short of a nearly closed ramp survives. Not a
perturbation of the training scenario — a different one.

**`speedDev = 0.03` (mild) — the informative sweep:**

| policy | mean ± std (10 seeds) | min … max | q_out | breakdowns |
|---|---|---|---|---|
| constant u = 0.50 | **−65.9 ± 0.1** | −66.1 … −65.8 | 2352 | **0/10** |
| constant u = 0.45 | −81.0 ± 0.1 | −81.2 … −80.8 | 2313 | 0/10 |
| run 4 `best_model.zip` (u ≈ 0.52) | **−159.2 ± 56.3** | −224 … −61 | 2088 | **9/10** |
| constant u = 0.55 | −189.8 ± 43.1 | −234 … −121 | 1997 | 10/10 |
| constant u = 0.60 | −247.6 ± 2.7 | −250 … −243 | 1828 | 10/10 |

Per-seed, `best_model.zip`: −214, −61, −224, −162, −121, −165, −90, −191,
−140, −223 — bimodal: ≈ −61 when the merge holds (seed 1), −90 … −224
when it breaks down.

Interpretation:

- With 3 % heterogeneity the capacity edge moves from u ≈ 0.6 down to
  between 0.50 and 0.52. The learned policy (flat u ≈ 0.52, trained on
  the deterministic scenario) sits 0.02 above it and gridlocks in 9 of
  10 seeds; **constant u = 0.5 is robust and beats it** (−66 vs −159).
- So the deterministic-scenario optimum is a knife edge for the
  *traffic*, not only for action noise (§"knife edge"). Any policy that
  is to be deployed must be trained and selected under heterogeneity.
- Error bars on the deterministic scenario are exactly zero; the
  meaningful comparison is at `speed_dev` ≥ 0.03 with ≥ 10 seeds, and
  the metric should include the breakdown rate, not just the mean.

Next steps that follow:

1. **Train with driver heterogeneity** (domain randomisation):
   `env.sumo_overrides.vehicle.speed_dev: 0.03` (or sample 0–0.05 per
   episode) in the SUMO PPO config, EvalCallback on the same
   distribution with `n_eval_episodes ≥ 5`. Expect the learned edge to
   settle near u ≈ 0.48–0.50 with a much lower breakdown rate.
2. Fix the post-jam insertion artifact (§7.6) first, otherwise every
   breakdown episode also silently lowers mainline demand.
3. Re-run the balance sweep at `speed_dev 0.03` — `q_ref` and the
   constant-policy curve shift with heterogeneity.

Artifacts: `_progress/m7_seed_sweep_sd0.1.jsonl` / `.summary.json`,
`_progress/m7_seed_sweep_sd0.03.jsonl` / `.summary.json`.

## 7.8 — Post-jam insertion artifact: diagnosis and fix (2026-08-28)

Follow-up to §7.6 item 2. Tooling added first, then the fix was found by
experiment — the fix I had pencilled in (`--max-depart-delay` alone) was
**not** sufficient.

### Instrumentation (`SumoEnv`, `run_simulation.py`, eval scripts)

- `info["pending_mainline"]`, `["pending_ramp"]`, `["episode_pending_mainline_max"]`,
  `["discarded_mainline"]`, `["discarded_ramp"]`, `["max_depart_delay_s"]`
  from `traci.simulation.getPendingVehicles()` / `getDepartedIDList()`
  each sub-step (a vehicle that was pending and is now neither pending nor
  departed was discarded by `--max-depart-delay`). Logged to W&B and to
  `eval_sumo_baselines.py` / `run_seed_sweep_sumo.py` output.
- **Ramp accounting hole closed.** `traci.vehicle.add()` does not fail when
  the ramp edge is full — the vehicle becomes *pending* — but the virtual
  queue was decremented at the `add()` call. The queue is now decremented
  only when SUMO reports the vehicle as departed, pending ramp vehicles
  are not released a second time, and a discarded ramp vehicle simply
  stays in the virtual queue (conserved). The queue sample is taken after
  the sub-step so that no-jam episodes are **byte-identical** to before
  (checked: u = 0.5 and u = 0.6 per-step reward/queue/outflow arrays equal
  to the pre-change baseline with `max_depart_delay_s = −1` *and* `5`).
- `simulation.max_depart_delay_s` (→ `--max-depart-delay`),
  `simulation.sumo_extra_args` (list of extra SUMO options),
  `vehicle.depart_speed` (mainline flow `departSpeed`, was hard-coded
  `"max"`), all with defaults that reproduce the old behaviour when absent.
  `eval_sumo_baselines.py` / `run_forced_jam_sumo.py` accept
  `--sumo-override section.key=value` (+ `--network-dir`) so variants can
  be tested without touching the training network files
  (`tests/test_sumo_overrides.py`).
- `scripts/run_forced_jam_sumo.py`: the §7.6 protocol (u = 1 for 10 min,
  then u = 0) as a script, printing 5-min windows of outflow, mean
  density, entry-detector speed, pending / discarded counts.

### What actually happens after a jam

Forced jam, seed 0, deterministic scenario, 2000 + 800 vph
(`_progress/m7_forced_jam_insertion_fix.json`; "post-jam" = last 40 min):

| mainline `departSpeed` | `--extrapolate-departpos` | `--max-depart-delay` | post-jam q_out | post-jam entry speed | pending max / final | discarded | arrived | return |
|---|---|---|---|---|---|---|---|---|
| max (old default) | no | −1 (old default) | 1551 | 76 km/h | 455 / 455 | 0 | 1648 | −295.7 |
| max | no | 5 | 1551 | 76 km/h | 3 / 3 | **452** | 1648 | −295.7 |
| max | yes | 5 | 1548 | 75 km/h | 3 / 2 | 456 | 1644 | −296.9 |
| desired | no | −1 | 1800 | 110 km/h | 274 / 274 | 0 | 1826 | −259.5 |
| desired / speedLimit | no | 5 | 1800 | 110 km/h | 3 / 3 | 271 | 1826 | −259.5 |
| desired | yes | 5 | 2000 | 108 km/h | 3 / 0 | 120 (all during the jam) | 1975 | −224.1 |
| **desired** | **yes** | **−1 (adopted default)** | **2088** (2000 + backlog drain) | **107 km/h** | **123 / 56** | **0** | **2037** | **−209.7** |

1. **It is not a backlog effect.** Discarding the backlog
   (`--max-depart-delay 5`) leaves post-jam insertion at 1551 vph; SUMO
   now throws away ≈ 450 vph *continuously* in free flow (ρ ≈ 17).
2. **It is a metastable slow-entry state.** With `departSpeed="max"` SUMO
   inserts each vehicle at the fastest *safe* speed behind its leader.
   Once a jam has left a slow vehicle at the entry, every new vehicle is
   inserted ≈ 1 s behind a 76 km/h leader at ≈ 76 km/h, and the entry
   never speeds up again (pre-jam entry speed 106 km/h). Insertion at
   that speed/gap fits ≈ 1550 vph.
3. **`departSpeed="desired"` alone caps insertion at exactly 1800 vph.**
   Entry speed recovers (110 km/h) but with a 1 s step the safe gap at
   120 km/h (≈ 40 m > 33 m travelled per step) needs 2 steps, so
   back-to-back attempts succeed every 2 s. The 1.8 s schedule is only
   met while the flow is in sync; once perturbed it stays at 1800 vph.
4. **`--extrapolate-departpos` fixes 3.** A vehicle inserted a fraction of
   a step late is placed downstream by that fraction × speed, preserving
   the 1.8 s spacing → 2000 vph after the jam, pending 0, discards
   confined to the jam itself (120 vehicles ≈ what the blocked entry
   could not take).

### Scenario change (`configs/sumo/phase1_1.yaml`)

`vehicle.depart_speed: desired`, `simulation.sumo_extra_args:
["--extrapolate-departpos"]` are now the scenario defaults, with
`simulation.max_depart_delay_s: -1` (SUMO's wait-forever) — **option 1,
chosen by the user 2026-08-28 to conserve vehicles**. With the slow-entry
state gone, waiting works: the backlog of a 10-min gridlock (123
vehicles) drains at ≈ 2090 vph (2000 + ≈ 90 vph) with 0 discards, i.e. the
pending list is a *virtual upstream queue*, logged as
`info["pending_mainline"]`. Its discharge margin is unrealistically small
(a physical queue would discharge at merge capacity, ~2500 vph; a 10-min
jam takes ~80 min to work off here) and it is not in the observation —
both recorded as open items. `max_depart_delay_s: 5` (discard + count)
remains available; the delay-5 tables below were measured before the
switch and are kept because the free-flow branch is identical. `run_simulation.py` (dataset generation) applies
the same three settings, so a regenerated M2 dataset will be consistent.

Consequences:

- Free flow is unaffected in substance (0 discards, 0 pending, same
  outflow; verified byte-identical between delay 5 and −1) but **not
  byte-identical to the old insertion** — insertion positions change, so
  the deterministic returns move slightly: constant u = 0.5 −65.0 → −67.2,
  u = 0.6 −43.4 → −42.7, run 4 `best_model.zip` −61.3 → −60.5
  (u ≈ 0.52, 2368 vph, 0 discards). All M7 numbers above this section
  were measured on the old insertion; the re-measured constant-u sweep
  and speedDev-0.03 seed sweep are below.
- A breakdown now costs what it should: the jam + the ramp backlog, not
  a silently lowered mainline for the rest of the hour. The "recovered"
  episode improves from −296 to −224 with the same policy.
- `discarded_mainline` is the honest count of demand the jammed entry
  could not admit; report it next to outflow whenever a breakdown occurs.
  Vehicle conservation is deliberately given up during a jam (they would
  have queued off-network); the conservation-preserving alternative — a
  ~2 km source edge upstream of the study section — is recorded as an
  option, not needed now.
- The delay must be ≥ `step_length_s` for the discard counter to see the
  vehicle (warned in `SumoEnv.__init__`).

### Constant-u sweep re-measured (`_progress/m7_u_sweep_seed0_fixed_insertion.jsonl`)

Same weights (δ 4.45, β 1, γ 0.067, q_ref 2476, queue_norm 400, σ_ref 6);
"old" = §7.1 sweep re-scored with these weights.

| u | return old | return new | q_out (both) | ρ_max new | discarded new |
|---|---|---|---|---|---|
| 0.0 | −262.9 | −262.5 | 1962 | 24 | 0 |
| 0.3 | −134.3 | −133.5 | 2197 | 33 | 0 |
| 0.4 | −98.8 | −97.2 | 2275 | 38 | 0 |
| 0.5 | −65.0 | −67.2 | 2354 | 41 | 0 |
| **0.6** | −43.4 | **−42.7** | 2430 | 47 | 0 |
| 0.7 | −266.8 | −270.6 | 1773 | 301 | 654 |
| 0.8 | −281.5 | −286.2 | 1714 | 307 | 789 |
| 1.0 | −327.2 | −331.0 | 1672 | 409 | 979 |

- Free-flow branch (u ≤ 0.6): identical outflow, returns within ±2.
  The capacity edge is still between u = 0.6 and 0.7.
- Breakdown branch: outflow unchanged (the merge, not the entry, is the
  bottleneck while jammed) but now 650–980 vehicles/h of demand are
  *discarded* instead of quietly replayed — the honest cost of the jam.
- Balance check on the new sweep: proposed δ 4.461 / β 1 / γ 0.065 vs the
  adopted 4.45 / 1 / 0.067; max per-term share 0.61 → **weights kept**.

- **Wait-forever vs discard (`_progress/m7_u_sweep_seed0_fixed_insertion_wait.jsonl`):
  all 11 constant-u episodes are byte-identical** between
  `max_depart_delay_s: 5` and `-1`. In free flow nothing is ever pending;
  in the u ≥ 0.7 branch the entry stays blocked for the whole hour, so the
  656–981 pending vehicles never re-enter either way (they are the
  discarded count of the delay-5 run). The two modes differ only in
  episodes where a jam *clears* — exactly the recovery cases that matter
  for a policy, which is why the seed sweep below is re-run.

### speedDev 0.03 seed sweep re-measured (`_progress/m7_seed_sweep_sd0.03_fixed_insertion.jsonl`)

10 seeds, same protocol as §7.7; "old" = §7.7 numbers on the old insertion.

| policy | return old | return new | breakdowns old → new | discarded new (mean) | q_out new |
|---|---|---|---|---|---|
| constant u = 0.45 | −81.0 ± 0.1 | **−81.0 ± 0.1** | 0/10 → **0/10** | 0 | 2313 |
| constant u = 0.50 | −65.9 ± 0.1 | −89.5 ± 48.4 (−66 … −213) | 0/10 → **3/10** | 46 | 2286 |
| run 4 `best_model.zip` (u ≈ 0.52) | −159 ± 56 | −153 ± 54 (−61 … −233) | 9/10 → 9/10 | 195 | 2108 |
| constant u = 0.55 | −190 ± 43 | −224 ± 13 | 10/10 → 10/10 | 409 | 1907 |

Interpretation:

1. **The fixed scenario is harsher, not easier.** With `departSpeed="max"`
   SUMO inserted tight-gap vehicles at *reduced* speed — an accidental
   upstream smoothing of platoons. Inserting at the desired speed sends
   denser platoons to the merge, so under 3 % driver heterogeneity the
   capacity edge moves from u ≈ 0.50–0.52 down to **u ≈ 0.45–0.50**.
   Constant u = 0.45 is now the robust reference (−81, 0/10).
2. Breakdown episodes are now costed honestly (100–400 discarded
   vehicles each) and are **recoverable in principle**: best_model seed 6
   jams (ρ_max 206) yet finishes at −79 with 0 discards — the jam cleared
   before reaching the entry and the mainline came back at 2000 vph,
   which the old insertion made impossible.
3. run 4's policy remains the wrong operating point for a heterogeneous
   scenario (9/10 breakdowns); the next-step recommendation is
   unchanged — train and select under `speed_dev ≥ 0.03` — but the
   robust constant to beat is u = 0.45 (−81), not u = 0.5.

4. **Re-run under the adopted wait-forever default**
   (`_progress/m7_seed_sweep_sd0.03_fixed_insertion_wait.jsonl`): u = 0.45
   −81.0 ± 0.1 (0/10), u = 0.50 −89.7 ± 49.0 (3/10), best_model −152.8 ± 54.1
   (9/10), u = 0.55 −224.8 ± 13.1 (10/10) — every episode within ±2.5 of
   the discard run, same breakdown counts. In all 32 breakdown episodes
   `pending_final ≈ pending_max` (110–448 vehicles): none of these jams
   clears within the hour under these policies, so the conserved backlog
   never re-enters. Wait vs discard therefore only matters for policies
   that actually recover (e.g. a closure after breakdown), which is the
   behaviour we want training to discover.

All M7 numbers in §7.1–7.7 are on the old insertion; everything from
§7.8 on uses the fixed scenario (wait-forever default). Sweep network dirs were removed after
the runs; the training network dir was regenerated with the new routes.

## 7.9 — Scenario check across mainline demand 1500–2000 vph (2026-08-28)

Before training, the user asked whether the fixed SUMO settings hold at
other mainline demands. `scripts/check_demand_range_sumo.py` (new) runs
constant u ∈ {0, 0.5, 0.7, 1.0} plus a forced jam (u = 1 ×10 min → u = 0)
at each demand level; data in `_progress/m7_demand_range_check.jsonl`.

| demand | u = 0 | u = 0.5 | u = 0.7 | u = 1.0 | forced jam → recovery |
|---|---|---|---|---|---|
| 1500 | 1500 ✓ | 1900 ✓ | 2060 ✓ | 2299 ✓ (−49, best) | no jam at u = 1 |
| 1600 | 1600 ✓ | 2000 ✓ | 2160 ✓ | **2400 → jam** (1690) | recovered, post-jam 1599 (1.00×), pending 0 |
| 1700 | 1699 ✓ | 2100 ✓ | 2260 ✓ | jam (1688) | recovered, 1731 (1.02×), pending 0 |
| 1800 | 1800 ✓ | 2201 ✓ | 2359 ✓ | jam (1690) | recovered, 1875 (1.04×), pending 0 |
| 1900 | 1900 ✓ | 2299 ✓ | **2460 → jam** (1789) | jam (1688) | recovered, 2026 (1.07×), pending 0 |
| 2000 | 2000 ✓ | 2400 ✓ | jam (1789) | jam (1690) | recovered, 2090 (1.05×), pending 56 |

(cells: delivered outflow in the steady state; ✓ = within 0.1 % of
demand + 800·u, 0 pending, 0 discarded, 0 teleports, entry speed
108–115 km/h.)

Findings:

1. **Insertion is exact at every demand level** — delivery ratio 1.000
   in all 18 free-flow episodes, nothing pending, no teleports. The
   fix is not tuned to 2000 vph.
2. **Forced jams recover at every level** with the wait-forever
   default: post-jam insertion returns to demand plus the draining
   backlog (1.00–1.07×), and the backlog is fully absorbed within the
   hour except at 2000 vph (56 left of 122). Lower demand → faster
   recovery, as expected.
3. **The merge capacity depends on the ramp share, not only the total.**
   2400 vph total is fine as 2000 + 400 (u = 0.5) but breaks down as
   1600 + 800 (u = 1); 2480 is fine as 2000 + 480 (u = 0.6, §7.1) but
   2460 breaks down as 1900 + 560. Rule of thumb from this grid:
   total ≤ ~2480 with ramp ≤ ~500 vph, total ≤ ~2360 with ramp ≈ 560,
   total ≤ ~2300 with ramp = 800. Gridlock discharge is ≈ 1690 vph
   regardless of demand (≈ 1790 when the ramp is at 0.7).
4. **The optimal constant u moves with demand**: u = 1 at 1500
   (−49), u ≈ 0.7 at 1800 (−53), u ≈ 0.6 at 2000 (−43). A demand range
   therefore makes the (currently dead) demand observation input
   informative and gives PPO something to learn beyond a single
   set-point. Reward weights were balanced at 2000 vph; at 1500 the
   outflow term dominates u = 0 (lost fraction 0.39 vs q_ref 2476), which
   is the intended direction (open the ramp) but worth a re-balance if a
   demand range is trained.

Verdict: the scenario is fit for a `demand_levels` range of 1500–2000
(and the ramp cap of 800 vph) without further SUMO changes.

## 7.10 — Ramp meter can now drain its queue; variable ramp arrivals (2026-08-28)

User request: (1) fix "departure rate ≤ arrival rate even at u = 1" for the
ramp, (2) allow a variable ramp arrival rate, (3) test in SUMO before
training anything.

### Changes

- **`ramp_discharge_vph` (scenario default 1600, `configs/sumo/phase1_1.yaml`).**
  `SumoEnv` release capacity is now `u · ramp_discharge_vph` (capped by
  the queue) instead of `u · ramp_demand_vph`. u is the green fraction of
  a 1600 vph saturation flow (a single-lane ramp with the meter off):
  u = 0.5 passes the full 800 vph demand, u = 1 drains a backlog at
  +800 vph. Lookup order mirrors `SurrogateEnv` (`env.ramp_discharge_vph`
  → `queue.discharge_vph` → `demand.ramp_discharge_vph` → arrival rate),
  and `SurrogateEnv` now also falls back to the scenario value. Default
  when absent = arrival rate = the old behaviour. `run_simulation.py`
  (open-loop dataset, no queue) is unchanged: there `ramp_control` is a
  fraction of the ramp *demand*.
- **`ramp_demand_levels`** (env config): per-episode ramp arrival rate,
  sampled at `reset()` like `demand_levels`; `reset(options={"ramp_demand_vph": …})`
  forces it. `info["ramp_demand_vph"]`, `info["ramp_discharge_vph"]`.
  Not in the observation (the queue length reflects it); flagged below.
- **`training.action_init_u`** (`train_ppo._set_initial_action_mean`):
  sets the Gaussian head's bias so the untrained policy starts at u₀
  instead of SB3's accidental u = 0.5 (zero-bias `action_net`, which with
  discharge 1600 = 800 vph ramp flow = immediate breakdown). Verified:
  u 0.500 → 0.300 for any observation; weights untouched. Run-4 config
  sets 0.3 = the best constant of the re-indexed sweep.
- Scripts: `check_demand_range_sumo.py --ramp-demands …`,
  `run_forced_jam_sumo.py --phases "u:s,u:s,…" --ramp-demand …`.

### Re-indexed constant-u sweep (`_progress/m7_u_sweep_seed0_discharge1600.jsonl`)

| u (discharge 1600) | ramp capacity | return | q_out | ρ_max | queue final | old u | identical to old |
|---|---|---|---|---|---|---|---|
| 0.0 | 0 | −262.5 | 1962 | 24 | 796 | 0.0 | yes |
| 0.1 | 160 | −173.0 | 2118 | 34 | 638 | 0.2 | yes |
| 0.2 | 320 | −97.2 | 2275 | 38 | 478 | 0.4 | yes |
| **0.3** | **480** | **−42.7** | 2430 | 46 | 319 | 0.6 | yes |
| 0.4 | 640 | −286.2 | 1714 | 307 | 159 | 0.8 | yes |
| 0.5 | 800 (= arrivals) | −331.0 | 1672 | 409 | 0 | 1.0 | yes |
| 0.6 – 1.0 | 960 – 1600 | −324 … −336 | ~1680 | ~390 | 0 | — | (all breakdown) |

Every u_new = u_old/2 episode is **byte-identical** to the old sweep, so
no earlier conclusion changes — it is a pure re-labelling of the action.
The half of the action range above 0.5 is a breakdown plateau at 2000 vph
mainline (only distinguishable from 0.5 when a queue exists). Balance
check on the new sweep: proposed δ 4.43 / β 1 / γ 0.064 → weights kept.

### Jam → close → flush test (`_progress/m7_ramp_flush_test.json`, 2000 + 800 vph)

| schedule | ramp out after re-opening | ramp queue | merge | return |
|---|---|---|---|---|
| u = 1 ×10 min (jam) → u = 0 ×5 min → **u = 1 flush** | **1176 vph** first 5 min, then ≈ 850 | 63 → 4 (drained) | re-jams immediately (ρ 208, q_out 1692), mainline backlog 879 | −343 |
| same → **u = 0.3** (480 vph) | 480 | 63 → 305 (+320 vph) | **re-jams** at t ≈ 20 min (ρ 120): mainline backlog drains at ≈ 2090 vph and 2090 + 480 > capacity | −256 |
| constant u = 0.3 (no jam) | 480 | 0 → 319 | free flow | −42.7 |

1. The mechanism works: with a queue, u = 1 releases 1176 > 800 vph and
   the queue is gone within a control interval or two.
2. **At 2000 vph mainline the queue can never be drained safely.** The
   merge's margin above the mainline is ≈ 480 vph (§7.9: 2000 + 480 ok,
   2000 + 560 jams), which is *below* the 800 vph arrival rate, so any
   release ≥ arrivals breaks the merge. Queue growth of ≥ 320 vph at the
   best safe set-point is a property of the scenario, not of the meter.
3. After a jam the situation is worse: the conserved mainline backlog
   drains at ≈ 2090 vph, so even the pre-jam set-point u = 0.3 re-jams;
   the safe post-jam rate is ≈ 0.2 (320 vph) until the backlog is gone.
   This is what a recovery policy has to learn now that the backlog is
   conserved (§7.8, option 1).
4. Therefore **draining requires ramp arrivals below the merge margin**
   — i.e. the variable ramp arrival rate of (2), or lower mainline
   demand (§7.9). The drain demonstrations and the mainline × ramp grid
   follow.

### Drain demonstrations (`_progress/m7_ramp_drain_test.json`)

| mainline | ramp arrivals | schedule | queue | merge | return |
|---|---|---|---|---|---|
| 2000 | 400 | close 10 min → u = 0.3 (480 vph) | 65 → **0** in 50 min (+80 vph net) | free flow throughout, 2480 vph served | −32.4 |
| 1500 | 400 | close 10 min → u = 0.5 (800 vph) | 65 → **0** in ~10 min (+400 vph net) | free flow (2300 vph) | −126.5 |
| 1500 | 600 | u = 1 ×10 min → close 5 min → u = 0.5 | 48 → **0** in 15 min | no jam at any point (1500 + 600 = 2100; flush 2300) | −87.9 |

So the meter now does what a meter should: a queue built by a closure is
worked off as soon as the merge has margin above the arrivals — slowly at
2000 + 400 (margin 80 vph), quickly at 1500. (Two first attempts at
"1500" silently ran at 2000 because `demand_levels` in the RL config
overrode the scenario demand; `run_forced_jam_sumo.py --demand` added.)
Note the 1500 vph returns are dominated by the outflow term (lost
fraction vs `q_ref` 2476 ≈ 0.2–0.4 per step) — a re-balance is needed
before training on a demand range (also noted in §7.9).

### Mainline × ramp-arrival grid (`_progress/m7_ramp_demand_range_check.jsonl`)

Constant u ∈ {0, 0.25, 0.5, 1} (= 0 / 400 / 800 / 1600 vph green capacity)
per cell plus a forced jam (u = 1 ×10 min → u = 0). Cells: delivered
outflow (✓ = within 0.1 % of mainline + min(u·1600, arrivals), 0 pending,
0 discards, 0 teleports) and the ramp queue at the end of the hour.

| mainline | ramp arrivals | u = 0 | u = 0.25 (400) | u = 0.5 (800) | u = 1 (1600) | forced jam |
|---|---|---|---|---|---|---|
| 1500 | 400 | 1500 ✓, Q 398 | 1900 ✓, Q 0 (−127) | 1900 ✓, Q 0 | 1900 ✓, Q 0 | no jam |
| 1500 | 600 | 1500 ✓, Q 597 | 1900 ✓, Q 199 | 2100 ✓, Q 0 (−87) | 2100 ✓, Q 0 | no jam |
| 1500 | 800 | 1500 ✓, Q 796 | 1900 ✓, Q 398 | 2300 ✓, Q 0 (−49) | 2300 ✓, Q 0 (−48) | no jam |
| 2000 | 400 | 2000 ✓, Q 398 | 2400 ✓, Q 0 (**−27**) | 2400 ✓, Q 0 | 2400 ✓, Q 0 | no jam |
| 2000 | 600 | 2000 ✓, Q 597 | 2400 ✓, Q 199 (−37) | **jam** (1756) | jam (1800) | jam → recovered, pending 4 |
| 2000 | 800 | 2000 ✓, Q 796 | 2400 ✓, Q 398 (−67) | **jam** (1690) | jam (1690) | jam → recovered, pending 57 |

Findings:

1. **Insertion and the ramp model are exact in every cell**: delivery
   ratio 1.000 in all 22 free-flow episodes; the end-of-hour queue at
   u = 0 equals the sampled arrival rate × 1 h (398 / 597 / 796), so
   `ramp_demand_levels` and `reset(options={"ramp_demand_vph"})` work;
   pass-through (queue 0) whenever u·1600 ≥ arrivals.
2. **Breakdown is governed by total flow ≈ 2480 with the ramp share
   caveat of §7.9**: 2400 is safe for every split tested; 2600 jams
   (2000 + 600); 2300 (1500 + 800) is safe. Every jam recovers after a
   closure under the wait-forever insertion.
3. **The optimum moves with both demands**, which is what makes a
   demand range worth training on: u ≥ 0.25 at 2000 + 400 (−27),
   u ≥ 0.5 at 1500 + 800 (−48), u = 0.25 at 2000 + 800 (−67, queue still
   growing). With `demand_levels` and `ramp_demand_levels` sampled per
   episode the currently dead demand input becomes informative — and a
   ramp-arrival observation is probably needed too (the queue length
   alone lags the arrival rate); recorded as an open item.

**Verdict:** the ramp fix and variable ramp arrivals work in SUMO across
1500–2000 × 400–800 vph. Before training: re-balance the reward weights
on a sweep over the demand grid (the outflow term's `q_ref` 2476 is a
2000 + 800 number), decide the training ranges, and whether to add the
ramp arrival rate to the observation.

## 7.11 — Ramp arrival rate in the observation; reward re-balance over the demand grid (2026-08-28)

User decisions: re-balance the reward on a sweep over the demand grid
before training; put the ramp arrival rate into the observation.

### Observation (`env.observe_ramp_demand`)

- `SumoEnv` and `SurrogateEnv` append the min-max-normalised ramp arrival
  rate (over `ramp_demand_levels`) after the mainline-demand feature:
  obs = [ρ₁…ρ₁₉ (z-scored), demand, **ramp demand**, k/T, queue] → 23
  features. Off by default (old 22-dim policies still load); **on** in
  `ppo_sumo.yaml`, `ppo_sumo_m7_run4.yaml`, `ppo_surrogate.yaml`.
  Verified: `SumoEnv` obs shape (23,), feature = 0.5 for 600 vph with
  levels [400, 600, 800].
- `SurrogateEnv` gained `ramp_demand_levels` / `reset(options={"ramp_demand_vph"})`
  for parity (its virtual queue uses the sampled rate). Caveat recorded
  in the config: the DeepONet was trained at a fixed 800 vph ramp demand
  with `ramp_control` as a fraction of that demand, so other arrival rates
  and u > arrivals/discharge are extrapolation for the surrogate branch.
- Consequence: run-4 `best_model.zip` (22-dim, discharge-800 semantics) is
  no longer directly usable as a warm start; it remains the deterministic
  reference in the pre-§7.10 tables.

### Tooling

- `run_u_sweep_sumo.py --demands … --ramp-demands …` sweeps every
  (mainline, ramp) cell; `rollout_policy_sumo(reset_options=…)`; per
  episode `demand_vph` / `ramp_demand_vph` in the JSON.
- `balance_reward_terms.py` is grid-aware: with several cells it takes
  each term's range across **all** grid episodes (no term may dominate
  anywhere), proposes weights, and prints per-cell best u and shares.
  `--q-ref-mode offered` evaluates the alternative normalisation
  q_ref_episode = min(mainline + ramp arrivals, q_ref), i.e. "unserved
  offered demand" instead of "distance to one capacity number".

### Grid sweep (`_progress/m7_u_sweep_grid_seed0.jsonl`, 3 × 3 cells × 11 u = 99 episodes, seed 0)

Best constant u per cell (current reward form, fixed q_ref 2476) and the
first u that breaks the merge:

| mainline \ ramp | 400 | 600 | 800 |
|---|---|---|---|
| 1500 | u ≥ 0.25 pass-through (−103 … −127), no edge | u = 0.5 (−71), no edge | u = 1.0 (−40), edge 0.9 |
| 1750 | u ≥ 0.25 (−61), no edge | u = 0.9 (−31), no edge | u = 0.4 (−31), edge 0.5 |
| 2000 | u = 0.3 (−21), no edge | u = 0.3 (−19), edge 0.4 | u = 0.3 (−41), edge 0.4 |

### Balance over the grid

Ranges of the unit-weight episode sums across **all 99 episodes**
(the "no term dominates anywhere" criterion): outflow 44.7, queue 159.8,
std 2522 → proposed **δ 3.572 / β 1 / γ 0.063** (fixed q_ref 2476), vs
the 2000 + 800 values 4.45 / 1 / 0.067. With `--q-ref-mode offered`
(q_ref = min(offered demand, 2476)): 3.939 / 1 / 0.063.

The per-cell *share* table printed by the script is misleading for the
low-demand cells: with a fixed q_ref the outflow term is a constant
offset there (1500 + 400 can never reach 2476; lost fraction 0.23 every
step), which inflates its share to 0.9 but cannot influence which u is
best. What decides the optimum is each term's **range across u inside a
cell**; the dominant term's share of the summed ranges:

| cell | fixed q_ref (3.57/1/0.063) | offered q_ref (3.94/1/0.063) | note |
|---|---|---|---|
| 1500+400 / 1750+400 / 2000+400 | outflow 0.58–0.60 | 0.61–0.67 | no breakdown: outflow and queue both say "open"; std tiny |
| 1500+600 / 1750+600 | outflow 0.51 | 0.53–0.56 | no breakdown |
| 1500+800 | queue 0.52 | 0.48 | edge at u = 0.9 |
| 1750+800 / 2000+600 / 2000+800 | 0.36–0.38 (three-way) | 0.35–0.40 | cells with a capacity edge: all three terms matter |

Both modes keep every term's decision share ≤ 0.6–0.67 and pick the same
safe optimum (below the edge) in every cell. **Adopted: fixed q_ref 2476
with δ 3.572 / β 1 / γ 0.063** — slightly better balanced by the range
criterion and no change to the reward definition; the "offered" variant
only changes per-cell constant offsets (it makes returns ≈ 0 when all
demand is served, which is nicer to read but not needed for learning,
since the critic sees the demand features). Written into
`ppo_sumo.yaml` and `ppo_sumo_m7_run4.yaml`. Under the new weights the
2000 + 800 curve is unchanged in shape (best u = 0.3, −40.9; breakdown
plateau −290).

**Ready to train** on `demand_levels` 1500–2000 × `ramp_demand_levels`
400–800 with the 23-dim observation, `action_init_u 0.3`, and (recommended)
`speed_dev 0.03`. Note that the reward is now balanced for the *range*;
single-cell runs at 2000 + 800 remain comparable to run 4 up to the
weight change (δ 4.45 → 3.57 scales the outflow term by 0.8).

## 7.12 — Run 5: demand-range PPO on the fixed scenario (launched 2026-08-29 14:10)

Config `configs/rl/ppo_sumo_m7_run5_range.yaml` (run-4 PPO hyper-parameters):
mainline `demand_levels` [1500 … 2000 step 100] × `ramp_demand_levels`
[400, 600, 800] sampled per episode, `speed_dev 0.03`, 23-dim observation
(ramp arrival rate included), `ramp_discharge_vph` 1600, reward
δ 3.572 / β 1 / γ 0.063 (§7.11), `action_init_u 0.3`, `log_std_init −2`,
60k steps (≈ 500 episodes, ≈ 28 per cell). Evaluation: deterministic policy
every 2400 steps on all 18 cells with fixed seeds (`CycleDemandCells`
wrapper in `train_ppo.py`, `eval_seed` 10000); `best_model.zip` on the
best 18-cell mean. Run dir `runs/rl/ppo_sumo_m7_run5_range_m7_seed0_20260829_141042`,
launched detached (`nohup`), log in the session scratchpad.

Smoke test beforehand (480 steps, 2-cell eval): obs (23,), init u 0.300,
eval −98 (1500 + 400 / 1500 + 600 at u = 0.3 pass-through, consistent
with the grid sweep), `final_model.zip` written.

References to beat (from the §7.11 grid, deterministic): per-cell best
constant, mean over the 9 measured cells ≈ −46; single global constant
u = 0.3 ≈ −45 at 2000 + 400/600, −41 at 2000 + 800 but only −103 … −127
in the 1500 cells (queue never released faster than 480 vph). Criteria:
18-cell eval mean better than the best single constant, approaching the
per-cell oracle, **0 breakdowns** in evaluation under `speed_dev 0.03`.

Results: _pending_.

## Open items

- ~~Ramp arrival rate not observed~~ — done in §7.11 (`observe_ramp_demand`, obs 22 → 23).
- ~~Reward weights balanced for 2000 + 800 only~~ — re-balanced over the grid in §7.11 (δ 3.572 / β 1 / γ 0.063).
- **Virtual upstream mainline queue (post-jam backlog) is invisible to the
  agent and drains slowly.** With `max_depart_delay_s: -1` the vehicles a
  jam blocks wait in SUMO's pending list and re-enter at only ≈ 2090 vph
  (≈ 90 vph above demand), so a 10-min gridlock takes ~80 min to work off
  and the observation never sees it. Options: add `pending_mainline` to
  the observation (like the ramp queue), or a ~3 km physical source edge
  upstream of the study section so the queue discharges at merge capacity.
- **E1 loop-count bias in the dataset.** `run_simulation.py` derives
  `flow` (and hence ρ = q/v) from `getLastStepVehicleNumber`, which
  over-counts by ≈ 1 + L/(v·dt) (~1.2× at 100 km/h). The M2 dataset and the
  M3 surrogate therefore carry ~20 % inflated density/flow in free flow.
  Fix candidates: `getLastIntervalVehicleNumber` / E1 `nVehEntered`, or
  `getLastStepVehicleIDs` with per-vehicle de-duplication. Needs an M2/M3
  regeneration to propagate.
- **Density-estimator ceiling in gridlock.** Occupancy fallback ignores
  minGap (saturates at 200 veh/km/lane) and sums lanes on `highway_accel`
  (→ 400). Only matters in the breakdown regime.
- **SUMO version dependence.** SUMO 1.27.1 inserts the full 2000 vph
  mainline; the Windows SUMO used for M2–M6b delivered ≈ 1470 vph. Pin
  the SUMO version in `pyproject.toml` (`eclipse-sumo==1.27.1`) and
  re-generate M2/M3 on it before any surrogate comparison.
- `proposal.md` §"Reward (Phase 1 shaped, Milestone 5c)" still describes
  the M5c reward — awaiting approval to update.
- `_progress/milestone_2_progress.md` says the dataset was generated at
  1500 vph while `dataset_constant_inflow.yaml` now says 2000; the
  M5c/M6 numbers are consistent with 2000. Worth a one-line
  confirmation from whoever reran M2/M3.
- `configs/sumo/phase1_1.yaml` header comments still describe the M1.1
  two-lane zipper geometry and label 33.33 m/s as "60 km/h".
