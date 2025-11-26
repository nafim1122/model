#!/usr/bin/env bash
# Bash launch script for 1B LoRA fine-tune
# Usage: ./launch_1b.sh
TRAIN_CONFIG="training/train_config_1b.json"

echo "Using config: $TRAIN_CONFIG"
python -m pip install -r requirements.txt || true
accelerate launch training/train.py --config "$TRAIN_CONFIG"
