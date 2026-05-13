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

**Prerequisites:** Install [SUMO >= 1.18](https://sumo.dlr.de/docs/Downloads.php) and ensure `sumo` is on your PATH.

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
- Reward: shaped Phase 1 form — `-alpha · max(0, mean(rho) - rho_freeflow) - beta · (queue / queue_norm)^2 - gamma · std(rho)` (see `proposal.md` §"Reward (Phase 1 shaped)")
- Mainline demand: single constant 2000 vph, ramp demand cap 800 vph
- Multi-demand family training (low / medium / high constant + mild peak profile) is a deferred follow-up (Milestone 2c)

## Reward setup

Both PPO paths (surrogate-trained in M5/M5c and SUMO-trained in M6/M6b) optimize the **same** shaped reward, computed each control step by `src/rl/reward.py::compute_reward` and called identically from `SurrogateEnv` and `SumoEnv`. Using one reward function across both envs is what lets us compare a surrogate-trained policy against a SUMO-trained one on the same yardstick.

**Formula (per control step, with `rho` the 19-detector density vector in physical units veh/km):**

```
r(t) = -alpha * max(0, mean(rho(t)) - rho_freeflow)    # ReLU on mean-density excess
       -beta  * (queue(t) / queue_norm)^2               # quadratic in on-ramp queue
       -gamma * std(rho(t))                             # linear spatial-uniformity
```

Each term is a non-negative penalty, so the reward is always ≤ 0.

**Default weights** — defined in `src/rl/reward.py` `RewardWeights` and overridden per experiment via the `env.reward` block in `configs/rl/ppo_surrogate.yaml` and `configs/rl/ppo_sumo.yaml`:

| Symbol         | Value | What it controls                                                                 |
| -------------- | ----- | -------------------------------------------------------------------------------- |
| `alpha`        | 1.0   | Penalty on **mean density above free-flow**. Term is 0 below the threshold.       |
| `beta`         | 1.0   | **Quadratic** penalty on the ramp queue. Cost grows fast as queue builds.        |
| `gamma`        | 1.0   | Penalty on **spatial density variation** (discourages localized hotspots).        |
| `rho_freeflow` | 20.0  | Free-flow threshold in veh/km; below this, the alpha-term contributes 0.        |
| `queue_norm`   | 100.0 | Queue normalizer in vehicles; squared queue is divided by `queue_norm^2`.        |

**Analytical queue model** (same formula in both envs — `src/rl/sumo_env_wrapper.py` and `src/rl/surrogate_env.py`):

```
queue[k+1] = max(0, queue[k] + (1 - u_k) * ramp_demand_vph * dt_ctrl_s / 3600)
queue[0]   = 0   (reset every episode)
```

At `ramp_demand_vph = 800` and `dt_ctrl_s = 30 s`, queue grows by `(1 - u_k) * 6.67` vehicles per step (so ~800 over a closed-ramp episode, 0 over an open-ramp episode). **Important for SumoEnv:** the analytical queue above is what feeds the reward; SUMO's *measured* ramp queue (`traci.edge.getLastStepVehicleNumber("ramp")`) is recorded in the `info` dict for diagnostics but is **not** the reward signal. Sharing the analytical formula keeps M5c (surrogate-trained) and M6 (SUMO-trained) policies comparable.

**Why this form — M5 → M5b → M5c iteration history:**

- **M5** used `-mean(density)` only. PPO converged to action mean ≈ 0.84 — essentially the ramp-open corner, because that minimizes density when there's no penalty for the resulting downstream spike.
- **M5b** added a linear `-beta * queue_length` term and swept `beta ∈ {0.1, 0.3, 1.0, 3.0}` × 5 seeds. **No β broke the u=1.0 corner trap**: at u ≡ 1.0 the queue is identically 0, so a *linear* queue-weighted term contributes 0 regardless of β — it only penalizes interior policies that build queue, not the corner itself.
- **M5c (current)** replaced the linear queue with the **ReLU-on-density + quadratic-on-queue** form above. The ReLU's `-alpha * max(0, mean(rho) - 20)` activates specifically at u ≈ 1.0 (where mean density crosses ~22 in this scenario), and the quadratic queue makes closer-to-closure policies expensive enough to keep u ≡ 0 from winning either. PPO finds a genuinely interior policy with action mean 0.688, std 0.426 — classic ramp metering (closed early, open late).

Full rationale and decomposition: `proposal.md` §"Reward (Phase 1 shaped, Milestone 5c)", `_progress/milestone_5b_progress.md` (null sweep diagnostic), `_progress/milestone_5c_progress.md` (M5c validation run).

## Milestones completed

- **M1** — Built the SUMO simulation pipeline (programmatic network construction, 19 detectors at `[100, 200, …, 1900]` m, single-rollout TraCI runner) on the 1-lane mainline + 100 m acceleration lane geometry pinned in `configs/sumo/phase1_1.yaml`.
- **M2** — Generated 120 SUMO rollouts at fixed 1500 vph mainline demand by sweeping ramp metering signals across 4 control families (constant, piecewise constant, smooth, ramp-step); 0 teleports, train/val/test split 84/18/18 in `data/raw/training_data/`.
- **M3** — Trained a DeepONet surrogate (`u(t) → density(x, t)`) for 1000 epochs on the M2 dataset; achieved **test rel-L2 = 0.074**, well under the 0.15 acceptance gate.
- **M4** — Implemented `SurrogateEnv`, a Gymnasium environment that runs the M3 checkpoint with the same observation / action / shared-reward contract as `SumoEnv` so PPO training code is environment-agnostic; 7 pytest smoke tests pass.
- **M5 / M5b / M5c** — Trained PPO on the surrogate (~3 min per 100k timesteps); iterated through three reward forms — baseline `-mean(density)` collapsed to u≈0.84, M5b's linear queue penalty sweep showed no β can break the u=1.0 corner trap, and **M5c's nonlinear `-α·ReLU(mean(ρ)−ρ_freeflow) − β·(queue/queue_norm)² − γ·std(ρ)`** finally produced a genuinely interior metering policy (action mean 0.688, std 0.426).
- **M6** — Trained PPO directly in SUMO at 20k timesteps (~35 min, M5c reward); the SUMO-trained policy collapsed to u≈0 (sample starvation — only 42 PPO iterations vs M5c's 209) while the M5c surrogate-trained policy transferred to SUMO with only a 13% reward drop and **beat the SUMO-trained policy in SUMO by ~48%** — the headline surrogate-acceleration result. M6b at 100k SUMO timesteps (~3 hours, currently running) is testing whether the corner collapse goes away with more sample budget.
