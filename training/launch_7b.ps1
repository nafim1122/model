# PowerShell launcher for 7B QLoRA
$TRAIN_CONFIG = "training\train_config_7b.json"
Write-Host "Launching training with config: $TRAIN_CONFIG"
python -m pip install -r requirements.txt 2>$null
accelerate launch training\train.py --config $TRAIN_CONFIG
