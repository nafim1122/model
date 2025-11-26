# PowerShell launch script for 1B LoRA fine-tune
# Edit TRAIN_CONFIG to point to the correct config file and ensure `accelerate` is configured.
$TRAIN_CONFIG = "training\train_config_1b.json"

# Optionally set environment variables
# $env:WANDB_API_KEY = "<your-wandb-key>"

Write-Host "Launching training with config: $TRAIN_CONFIG"
python -m pip install -r requirements.txt 2>$null
accelerate launch training\train.py --config $TRAIN_CONFIG
