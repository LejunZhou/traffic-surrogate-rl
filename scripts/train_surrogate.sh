#!/usr/bin/env bash
# Train the DeepONet surrogate model.
#
# Usage:
#   bash scripts/train_surrogate.sh configs/surrogate/baseline.yaml
#
# Reads all model and training parameters from the config file.
# Saves checkpoints and TensorBoard logs to the configured output directory.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "[train_surrogate] Config: $CONFIG"
# TODO: invoke src/surrogate/train.py
python -m surrogate.train --config "$CONFIG"
