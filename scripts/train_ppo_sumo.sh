#!/usr/bin/env bash
# Train PPO directly in the live SUMO environment.
#
# Usage:
#   bash scripts/train_ppo_sumo.sh configs/rl/ppo_sumo.yaml

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "[train_ppo_sumo] Config: $CONFIG"
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m rl.train_ppo --config "$CONFIG"
