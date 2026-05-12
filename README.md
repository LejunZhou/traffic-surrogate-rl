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
# 1. Generate dataset (M2: 120 SUMO rollouts at constant 1500 vph)
bash scripts/make_dataset.sh configs/experiments/dataset_constant_inflow.yaml

# 2. Train surrogate (M3: DeepONet on the dataset above)
bash scripts/train_surrogate.sh configs/surrogate/baseline.yaml

# 3. Train PPO in surrogate environment (M5)
bash scripts/train_ppo_surrogate.sh configs/rl/ppo_surrogate.yaml

# 4. Evaluate a trained PPO policy in live SUMO (M6 native eval, or M6.5 transfer)
bash scripts/eval_in_sumo.sh configs/rl/ppo_sumo.yaml runs/rl/<run_dir>/best_model.zip
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
- Mainline demand: single constant 1500 vph, ramp demand cap 800 vph
- Multi-demand family training (low / medium / high constant + mild peak profile) is a deferred follow-up (Milestone 2c)
