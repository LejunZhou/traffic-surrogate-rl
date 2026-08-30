# Sample-Efficient Surrogate Traffic Model for RL-Based Traffic Flow Control

This repository implements a surrogate-accelerated reinforcement learning pipeline for ramp metering control on a highway segment.

## Pipeline

1. Build a SUMO traffic simulation (1-lane mainline + 100 m acceleration lane + one on-ramp)
2. Generate training data by sweeping over ramp metering signals at a fixed mainline demand
3. Train a DeepONet surrogate to predict density trajectories from control inputs
4. Wrap the surrogate as a Gymnasium environment for fast RL training
5. Train PPO in both the surrogate environment and directly in SUMO
6. Evaluate both policies in SUMO and compare performance

## Setup

**Option A — self-contained venv (macOS / Linux, no system SUMO needed).** The
[`eclipse-sumo`](https://pypi.org/project/eclipse-sumo/) wheel ships the SUMO
binaries (`sumo`, `netconvert`, `sumo-gui`) plus `traci`/`sumolib`, so the whole
toolchain lives inside the project venv. Python >= 3.11 is required; if the
system Python is older, [`uv`](https://docs.astral.sh/uv/) can fetch one:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # one-time, installs to ~/.local/bin
uv python install 3.11
uv venv .venv-traffic-rl --python 3.11
uv pip install --python .venv-traffic-rl/bin/python -e ".[dev,sumo]"   # pins SUMO 1.27.1

source .venv-traffic-rl/bin/activate                  # puts sumo/netconvert on PATH
export PYTHONPATH=src
sumo --version && python -m pytest tests/test_reward.py -q
```

`.venv-traffic-rl/` is gitignored. With the venv activated no `SUMO_HOME` export
is needed (the `sumo` package resolves it to its own `site-packages/sumo`).

**Option B — system SUMO.** Install [SUMO >= 1.18](https://sumo.dlr.de/docs/Downloads.php)
(Windows MSI, macOS framework, or `apt`), make sure `sumo` is on your PATH, then:

```bash
pip install -e ".[dev]"
```

## Repository structure

```
configs/          YAML config files for SUMO, surrogate, RL, and experiments
data/             Raw simulation outputs, processed datasets, and train/val/test splits
src/
  sumo_env/       SUMO network construction, simulation runner, detectors, dataset generation
  surrogate/      DeepONet model, dataset loader, loss, training loop, evaluation
  rl/             Gymnasium environments (surrogate + SUMO), reward, PPO training, evaluation
  utils/          Config loading, logging, plotting
scripts/          Shell scripts to run each pipeline stage end-to-end
notebooks/        Exploratory notebooks
```

## Running the pipeline

```bash
# 1. Generate dataset
bash scripts/make_dataset.sh configs/experiments/phase1.yaml

# 2. Train surrogate
bash scripts/train_surrogate.sh configs/surrogate/baseline.yaml

# 3. Evaluate surrogate model
PYTHONPATH=src python -m surrogate.eval --help

# 4. Train PPO in surrogate environment
bash scripts/train_ppo_surrogate.sh configs/rl/ppo_surrogate.yaml

# 5. Evaluate in SUMO
bash scripts/eval_in_sumo.sh configs/rl/ppo_surrogate.yaml runs/rl/<run_name>/final_model.zip
```

### Quick start: run a single simulation

```bash
# macOS only: set SUMO environment (framework install).
# Windows: SUMO_HOME and PATH are set by the Eclipse SUMO MSI installer
# system-wide; no exports needed.
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO"
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export PATH="$SUMO_HOME/bin:$PATH"

# Run one rollout with the current Phase 1 config (1-lane mainline + accel lane)
python scripts/run_rollout.py \
  --config configs/sumo/phase1_1.yaml \
  --ramp-rate 0.5 \
  --output-index test
```

## Phase 1 scope (as built)

- 1-lane mainline + 100 m acceleration lane downstream of the on-ramp, 2000 m total, one on-ramp at 1300 m
- DeepONet trained on density trajectories only (speed/flow logged for diagnostics)
- PPO observation: density at 19 detectors + current demand + normalized time index + analytical queue (22 features)
- Reward (Milestone 7, SUMO path): `-delta · max(0, q_ref - q_out)/q_ref - beta · (queue / queue_norm)^2 - gamma · std(rho)/sigma_ref` — lost mainline outflow + ramp queue + spatial uniformity (see "Reward setup" below). The surrogate path runs the same reward with `delta = 0` (no flow prediction).
- Mainline demand: single constant 2000 vph, ramp demand cap 800 vph
- Multi-demand family training (low / medium / high constant + mild peak profile) is a deferred follow-up (Milestone 2c)

## Reward setup

Both PPO paths optimize a reward computed each control step by `src/rl/reward.py::reward_terms` and called identically from `SurrogateEnv` and `SumoEnv`. The reward is a sum of three non-positive, O(1)-per-step penalty terms:

```
r(t) = -delta * max(0, q_ref - q_out(t)) / q_ref     # lost mainline outflow (network arrivals)
       -beta  * (queue(t) / queue_norm)^2              # quadratic in on-ramp queue
       -gamma * std(rho(t)) / sigma_ref                # spatial uniformity of density
```

with `rho` the 19-detector density vector (veh/km), `queue` the virtual ramp queue (vehicles) and `q_out` the outflow downstream of the merge (veh/h): in `SumoEnv` the exact count of vehicles leaving the network per control interval. (The det_18 loop flow is also logged as `det18_flow_vph`, but E1 `getLastStepVehicleNumber` double-counts vehicles that straddle a 1 s step boundary — ~1.2× at 100 km/h — so it is not used for the reward.) The episode return is the undiscounted sum over the 120 control steps (PPO itself discounts with `gamma_ppo = 0.99`).

**Weights** — `RewardWeights` in `src/rl/reward.py`, overridden per experiment via the `env.reward` block of the PPO config:

| Symbol       | `ppo_sumo.yaml` | `ppo_surrogate.yaml` | What it controls |
| ------------ | --------------- | -------------------- | ---------------- |
| `delta`      | 4.0             | **0.0**              | Weight on lost outflow. Must be 0 on the surrogate path: the DeepONet predicts density only, so no outflow is available there. |
| `beta`       | 1.0             | 1.0                  | Quadratic ramp-queue penalty. |
| `gamma`      | 1.3             | 1.0                  | Density-std (hotspot) penalty. |
| `q_ref`      | 2260 veh/h      | —                    | Outflow reference. Measured peak from the M6 constant-policy table; the IDM 1-lane capacity (~2970 veh/h) is the alternative. |
| `queue_norm` | 400 veh         | 200 veh              | Queue normaliser. |
| `sigma_ref`  | 6.0 veh/km      | 67.7167              | Std normaliser (~dataset density std on the SUMO path; the surrogate value is the legacy divisor from the "Working PPO Surrogate" run). |
| `warmup_s`   | 90 s            | 0                    | Reward masked to 0 before this time (first vehicle reaches det_18 after ~60 s). |

**Balancing the three terms.** The SUMO weights are chosen so that no single term decides the optimum: `scripts/run_u_sweep_sumo.py` rolls constant policies u = 0.0 … 1.0 through SUMO and `scripts/balance_reward_terms.py` proposes `delta/beta/gamma` that equalise each term's episode-sum range across the sweep (anchored at `beta = 1`). The values above are a first pass from the M6 logged stats; re-run the sweep before training on a new scenario.

**Virtual queue model** (same formula in both envs — `src/rl/sumo_env_wrapper.py` and `src/rl/surrogate_env.py`):

```
queue[k+1] = max(0, queue[k] + arrivals - min(queue[k] + arrivals, u_k * discharge))
arrivals = discharge = ramp_demand_vph * dt_ctrl_s / 3600 = 6.67 veh / step
```

At `ramp_demand_vph = 800` and `dt_ctrl_s = 30 s`, the queue grows by `(1 - u_k) * 6.67` vehicles per step (~800 over a closed-ramp episode, 0 over an open-ramp episode) and never drains faster than arrivals. **Important for SumoEnv:** this virtual queue is what feeds the reward; SUMO's *measured* ramp occupancy (`traci.edge.getLastStepVehicleNumber("ramp")`) is recorded in the `info` dict for diagnostics only.

### Generating training data on the current SUMO scenario (read before regenerating M2)

The scenario every M7 result uses is `configs/sumo/phase1_1.yaml` on **SUMO 1.27.1**
(`pip install -e ".[dev,sumo]"` pins it; other SUMO versions insert traffic differently,
see `_progress/milestone_7_progress.md` §7.1). Dataset generation
(`scripts/make_dataset.sh configs/experiments/dataset_constant_inflow.yaml`) picks the
scenario up automatically — routes are written with `departSpeed="desired"`, SUMO runs with
`--extrapolate-departpos`, and blocked vehicles wait — verified end-to-end on 2026-08-29.
Three things differ from the M2 dataset generated on the old Windows SUMO:

1. **The regime is different.** The old SUMO delivered only ≈ 1470 vph of the scheduled
   2000 vph mainline, so M2 was free flow throughout (density mean 18.7 veh/km). SUMO 1.27.1
   delivers the full 2000 vph, and the merge breaks down once ramp flow exceeds ≈ 480–560 vph
   at that demand (§7.9). The generator draws `ramp_control` uniformly in [0, 1] as a fraction
   of `ramp_demand_vph` = 800, so with `demand_levels: [2000]` **most samples gridlock for the
   rest of the hour** (smoke test: density mean 128 veh/km). If you want a mix of regimes,
   spread `dataset.demand_levels` over 1500–2000 (1500 + 800 never breaks down; 2000 + 800 does
   above u ≈ 0.6); if you want free flow only, cap the control range or use ≤ 1600 vph.
2. **Ramp semantics (metered queue, same as the RL env).** With `demand.ramp_model:
   metered_queue` (the `phase1_1.yaml` default since 2026-08-29) the generator runs the same
   virtual queue as `SumoEnv`: vehicles arrive at `ramp_demand_vph`, wait upstream of the
   meter, and the control signal u releases `min(u · ramp_discharge_vph, queue)` per step — so
   the ramp inflow exceeds the arrival rate exactly when a queue exists and u > 0.5 (up to
   1600 vph; smoke test: 1560 vph at u ≈ 1 after a low-u period). Each npz stores
   `ramp_control` = **physical inflow / 1600** (the DeepONet branch input), `ramp_control_cmd`
   = the command u, `ramp_inflow_vph`, and `ramp_queue`. A surrogate trained on such data must
   be driven with `env.surrogate_branch_input: inflow_frac` in `SurrogateEnv` (the released
   flow, not the raw action); the M2/M3 open-loop dataset and its checkpoint keep `"action"`.
   `ramp_model: open_loop` restores the M2 behaviour (inflow = u · ramp_demand_vph, no queue).
3. **Known measurement caveats, unchanged:** detector flow (and hence density = flow/speed)
   from E1 `getLastStepVehicleNumber` over-counts by ≈ 1.2× in free flow, and in gridlock
   the occupancy fallback saturates (summed lanes on the acceleration segment → up to
   400 veh/km). Both are listed under "Open items" in the M7 progress file.

For a demand-range dataset use `configs/experiments/dataset_demand_range.yaml` (mainline 1500–2000 × ramp 400/600/800; `dataset.ramp_demand_levels` is cycled per sample together with `demand_levels`, and each sample's `ramp_demand_vph` is stored in its npz). `vehicle.speed_dev` is 0 (fully deterministic; the SUMO seed then changes nothing) — set it
to e.g. 0.03 via the dataset config's SUMO overrides if seed-to-seed variability is wanted.

**Why this form — M5 → M5b → M5c → M7 history:**

- **M5** used `-mean(density)` only. PPO converged to action mean ≈ 0.84 — essentially the ramp-open corner.
- **M5b** added a linear `-beta * queue` term and swept β. No β broke the u = 1.0 corner: at u ≡ 1.0 the queue is identically 0, so a linear queue term is free there.
- **M5c** replaced the density term with a ReLU `-alpha * max(0, mean(rho) - 20)` and made the queue quadratic. PPO found an interior policy (action mean 0.688) on the surrogate; transferred to SUMO it beat direct SUMO training (M6/M6b), which collapsed to u ≡ 0.
- **M7 (current)** replaces the ReLU-on-mean-density proxy with a direct **outflow** term. The proxy's threshold (20 veh/km) was tuned to fire at u ≈ 1, but in this scenario u = 1 is also the *maximum-outflow* state (2260 vph vs 1867 at u = 0.5): the term penalised throughput rather than protecting it. The outflow term rewards served flow directly; `std` keeps the hotspot penalty; the quadratic queue is unchanged. Caveat: at 2000 + 800 vph (< ~2970 vph capacity) SUMO shows no capacity drop, so outflow is monotone in u and the interior optimum still comes from the std/queue trade — see `_plans/milestone_7_plan.md`.

Details: `_plans/milestone_7_plan.md`, `_progress/milestone_7_progress.md`, and `_progress/milestone_5b_progress.md` / `_progress/milestone_5c_progress.md` for the earlier iterations.

## Milestones completed

- **M1** — Built the SUMO simulation pipeline (programmatic network construction, 19 detectors at `[100, 200, …, 1900]` m, single-rollout TraCI runner) on the 1-lane mainline + 100 m acceleration lane geometry pinned in `configs/sumo/phase1_1.yaml`.
- **M2** — Generated 120 SUMO rollouts at fixed 1500 vph mainline demand by sweeping ramp metering signals across 4 control families (constant, piecewise constant, smooth, ramp-step); 0 teleports, train/val/test split 84/18/18 in `data/raw/training_data/`.
- **M3** — Trained a DeepONet surrogate (`u(t) → density(x, t)`) for 1000 epochs on the M2 dataset; achieved **test rel-L2 = 0.074**, well under the 0.15 acceptance gate.
- **M4** — Implemented `SurrogateEnv`, a Gymnasium environment that runs the M3 checkpoint with the same observation / action / shared-reward contract as `SumoEnv` so PPO training code is environment-agnostic; 7 pytest smoke tests pass.
- **M5 / M5b / M5c** — Trained PPO on the surrogate (~3 min per 100k timesteps); iterated through three reward forms — baseline `-mean(density)` collapsed to u≈0.84, M5b's linear queue penalty sweep showed no β can break the u=1.0 corner trap, and **M5c's nonlinear `-α·ReLU(mean(ρ)−ρ_freeflow) − β·(queue/queue_norm)² − γ·std(ρ)`** finally produced a genuinely interior metering policy (action mean 0.688, std 0.426).
- **M6** — Trained PPO directly in SUMO at 20k timesteps (~35 min, M5c reward); the SUMO-trained policy collapsed to u≈0 (sample starvation — only 42 PPO iterations vs M5c's 209) while the M5c surrogate-trained policy transferred to SUMO with only a 13% reward drop and **beat the SUMO-trained policy in SUMO by ~48%** — the headline surrogate-acceleration result. M6b at 100k SUMO timesteps (~3 hours, currently running) is testing whether the corner collapse goes away with more sample budget.
- **M7** — Replaced the density-ReLU reward term with a direct mainline-**outflow** term, balanced the three terms from a constant-u sweep (`scripts/run_u_sweep_sumo.py` + `scripts/balance_reward_terms.py`), and fixed two SUMO+PPO blockers: a zero-initialised Gaussian on a [0, 1] action box (→ `env.symmetric_action`) and drifting exploration beside the merge's capacity cliff (→ `log_std_init −2` + `EvalCallback` best-checkpoint saving). Result on SUMO 1.27.1 at 2000 + 800 vph: `best_model.zip` return −61 (u ≈ 0.52, 2367 vph served, no breakdown) vs −65 for constant u = 0.5; the nominal best constant u = 0.6 (−43) is a knife edge that ±0.03 action noise tips into gridlock. The scenario itself is fully deterministic (SUMO seed has no effect); with a mild 3 % driver-speed spread (`vehicle.speed_dev`, new knob) the capacity edge drops to u ≈ 0.5, constant u = 0.5 is robust (−66 ± 0.1 over 10 seeds) and the deterministic-trained policy gridlocks in 9/10 seeds — so the next step is training under heterogeneity (`scripts/run_seed_sweep_sumo.py`). A SUMO scenario bug found on the way: after any merge breakdown, SUMO's `departSpeed="max"` left the mainline entry in a self-sustaining slow-insertion state (~76 km/h, ~1550 vph) for the rest of the episode, so every post-jam episode silently ran at reduced demand. Fixed in `configs/sumo/phase1_1.yaml` (`vehicle.depart_speed: desired` + `--extrapolate-departpos`; blocked vehicles wait and are conserved — the backlog drains after the jam at ≈2090 vph and is logged as `pending_mainline`, or set `max_depart_delay_s` to discard and count them instead; the ramp virtual queue is decremented on actual departure) — post-jam insertion is back to 2000 vph and a cleared jam no longer poisons the rest of the episode, verified with `scripts/run_forced_jam_sumo.py`; `scripts/check_demand_range_sumo.py` confirms exact insertion and jam recovery at every mainline demand 1500–2000 vph, with the merge capacity depending on the ramp share (2000 + 480 ok, 1600 + 800 breaks down). **Run 5 (demand-range PPO, 2026-08-29):** trained on mainline 1500–2000 × ramp 400–800 vph with `speed_dev 0.03`; `best_model.zip` (24k steps) learned a demand-conditioned throttle (u 0.36 → 0.23 from 1500 to 2000 vph), grid mean −70 over 18 cells × 3 seeds vs −86 for the best constant, 3/54 breakdowns (`scripts/eval_policy_grid_sumo.py`); the run later degraded from a PPO trust-region blow-up (`approx_kl` 0.11, no `target_kl`). **Run 6** (same setup + `target_kl 0.02`, lr 1e-4, 80k steps) fixed the collapse (best deterministic eval −56 at 57.6k, no late degradation) and is 10–30 return units better in every cell below 1900 vph, but its best checkpoint acts ~0.03 higher at 2000 vph and gridlocks 2000 + 800 in all 3 seeds (grid mean −70.5, 9/54 breakdowns vs run 5's −70.1, 3/54) — single-seed checkpoint selection rewards luck on knife-edge cells (re-scoring earlier run-6 checkpoints gives −70.3 with 3/54, i.e. run-5 parity); multi-seed checkpoint selection is the next fix, and run 5 @ 24k remains the reference robust policy. The ramp meter can now actually drain its queue: `ramp_discharge_vph: 1600` decouples the meter's saturation flow from the 800 vph arrival rate (u is the green fraction of 1600 vph, so u = 0.5 passes the full demand and u = 1 flushes a backlog at +800 vph; earlier constant-u results re-index as u_new = u_old/2, byte-identically), `env.ramp_demand_levels` samples the ramp arrival rate per episode, and `training.action_init_u` starts PPO at a chosen metering rate instead of SB3's accidental u = 0.5. SUMO tests before training (`_progress/milestone_7_progress.md` §7.10): at 2000 vph mainline the merge margin (~480 vph) is below the 800 vph arrivals, so the queue can only be drained when arrivals drop (400 vph → drained) or mainline demand is lower (1500 → an 800 vph release clears a 65-vehicle queue in 10 min). Ahead of a demand-range run the ramp arrival rate was added to the observation (`env.observe_ramp_demand`, 22 → 23 features) and the reward re-balanced over a 3 × 3 demand grid (99 constant-u episodes; δ 3.57 / β 1 / γ 0.063, `scripts/balance_reward_terms.py` is grid-aware). Deterministic no-jam episodes are unchanged in substance (returns move by ≤ 2), but the fixed scenario is harsher under driver heterogeneity — the old `"max"` insertion had been smoothing platoons — so at `speed_dev 0.03` the robust constant is now u = 0.45 (−81, 0/10 breakdowns; u = 0.5 breaks down in 3/10 seeds). Details: `_progress/milestone_7_progress.md`.
