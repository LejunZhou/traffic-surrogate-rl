#!/usr/bin/env bash
# Generate the SUMO dataset (raw trajectories + train/val/test splits).
#
# Usage:
#   bash scripts/make_dataset.sh configs/experiments/phase1.yaml
#
# Reads all dataset parameters from the config file.
# Outputs raw simulation files to data/raw/ and splits to data/splits/.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "[make_dataset] Config: $CONFIG"
# TODO: invoke src/sumo_env/dataset_generation.py
python -m sumo_env.dataset_generation --config "$CONFIG"
