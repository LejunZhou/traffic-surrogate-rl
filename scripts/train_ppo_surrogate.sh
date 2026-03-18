#!/usr/bin/env bash
# Train PPO in the surrogate environment.
#
# Usage:
#   bash scripts/train_ppo_surrogate.sh configs/rl/ppo_surrogate.yaml
#
# Reads all PPO and environment parameters from the config file.
# Saves policy checkpoints and training curves to the configured output directory.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "[train_ppo_surrogate] Config: $CONFIG"
# TODO: invoke src/rl/train_ppo.py with env.type=surrogate
python -m rl.train_ppo --config "$CONFIG"
