#!/usr/bin/env bash
# Evaluate a trained PPO policy (surrogate-trained or SUMO-trained) in SUMO.
#
# Usage:
#   bash scripts/eval_in_sumo.sh configs/rl/ppo_sumo.yaml runs/rl/<run>/final_model.zip
#
# Reads eval parameters from the config file. The policy path can be passed as
# the second argument or set as evaluation.policy_path in the config.
# Outputs metrics JSON and comparison plots to the configured output directory.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"
POLICY="${2:-}"

echo "[eval_in_sumo] Config: $CONFIG"
if [[ -n "$POLICY" ]]; then
  echo "[eval_in_sumo] Policy: $POLICY"
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m rl.evaluate --config "$CONFIG" --policy "$POLICY"
else
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m rl.evaluate --config "$CONFIG"
fi
