# C-Language LLM Model - Windows Installation Script
# This script installs all dependencies and sets up the environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "C-Language LLM Model Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host "[2/6] Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✓ Found pip" -ForegroundColor Green
} catch {
    Write-Host "✗ pip not found. Installing pip..." -ForegroundColor Red
    python -m ensurepip --upgrade
}

# Upgrade pip
Write-Host "[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies
Write-Host "[4/6] Installing core dependencies..." -ForegroundColor Yellow
pip install -r docker\requirements.txt

# Install additional RLHF dependencies
Write-Host "[5/6] Installing RLHF dependencies..." -ForegroundColor Yellow
pip install trl
pip install peft
pip install bitsandbytes
pip install wandb
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy

# Install development tools (optional but recommended)
Write-Host "[6/6] Installing development tools..." -ForegroundColor Yellow
pip install ipython
pip install jupyter
pip install black
pip install flake8

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Prepare your dataset: python training\convert_to_hfdataset.py --jsonl dataset_tools\retrain_prompts_sample20.jsonl --out_dir data\hf_dataset" -ForegroundColor White
Write-Host "2. Train SFT model: python training\train_peft.py --model_name_or_path gpt2 --dataset_dir data\hf_dataset --output_dir runs\sft --use_lora true" -ForegroundColor White
Write-Host "3. Generate outputs: python training\generate_save.py --model_dir runs\sft --prompts dataset_tools\retrain_prompts_sample20.jsonl --out training\generated.jsonl" -ForegroundColor White
Write-Host "4. Score outputs: python training\score_reward.py --in training\generated.jsonl --out training\rewards.jsonl" -ForegroundColor White
Write-Host "5. Train PPO: python training\ppo_train.py --sft_model runs\sft --rm_model runs\rm --dataset_dir data\hf_dataset --out runs\ppo" -ForegroundColor White
Write-Host ""
Write-Host "For more details, see README_RLHF.md" -ForegroundColor Cyan
