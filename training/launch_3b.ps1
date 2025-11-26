# PowerShell launch script for 3B LoRA fine-tune
$TRAIN_CONFIG = "training\train_config_3b.json"
Write-Host "Launching training with config: $TRAIN_CONFIG"
python -m pip install -r requirements.txt 2>$null
accelerate launch training\train.py --config $TRAIN_CONFIG
