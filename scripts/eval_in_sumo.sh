#!/usr/bin/env bash
# Evaluate a trained PPO policy (surrogate-trained or SUMO-trained) in SUMO.
#
# Usage:
#   bash scripts/eval_in_sumo.sh configs/experiments/phase1.yaml
#
# Reads policy paths and eval parameters from the config file.
# Outputs metrics JSON and comparison plots to the configured output directory.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "[eval_in_sumo] Config: $CONFIG"
# TODO: invoke src/rl/evaluate.py
python -m rl.evaluate --config "$CONFIG"
