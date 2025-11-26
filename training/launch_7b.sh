#!/usr/bin/env bash
TRAIN_CONFIG="training/train_config_7b.json"

echo "Using config: $TRAIN_CONFIG"
python -m pip install -r requirements.txt || true
accelerate launch training/train.py --config "$TRAIN_CONFIG"
