# Sample-Efficient Surrogate Traffic Model for RL-Based Traffic Flow Control

This repository implements a surrogate-accelerated reinforcement learning pipeline for ramp metering control on a highway segment.

## Pipeline

1. Build a SUMO traffic simulation (single-lane highway + one on-ramp)
2. Generate training data by sweeping over demand profiles and ramp metering signals
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

# 3. Train PPO in surrogate environment
bash scripts/train_ppo_surrogate.sh configs/rl/ppo_surrogate.yaml

# 4. Evaluate in SUMO
bash scripts/eval_in_sumo.sh configs/experiments/phase1.yaml
```

## Phase 1 scope

- Single-lane highway, 2000 m, one on-ramp
- DeepONet trained on density trajectories only (speed/flow logged for diagnostics)
- PPO observation: density at 20 detectors + current demand + normalized time index (22 features)
- Reward: negative mean density (Phase 1 baseline)
- Demand family: low / medium / high constant + mild peak profile
