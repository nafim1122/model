@echo off
REM C-Language LLM Model - Windows Installation Script
REM This script installs all dependencies and sets up the environment

echo ========================================
echo C-Language LLM Model Installation
echo ========================================
echo.

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+ from https://www.python.org/
    exit /b 1
)
python --version
echo.

REM Upgrade pip
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install core dependencies
echo [3/6] Installing core dependencies from docker\requirements.txt...
pip install -r docker\requirements.txt
echo.

REM Install additional RLHF dependencies
echo [4/6] Installing RLHF dependencies...
pip install trl peft scipy scikit-learn matplotlib seaborn pandas numpy
echo.

REM Install optional bitsandbytes (may fail on Windows, that's okay)
echo [5/6] Installing optional dependencies...
pip install wandb ipython jupyter black flake8
echo.

REM Try to install bitsandbytes (Windows compatible version)
echo [6/6] Attempting to install bitsandbytes (optional)...
pip install bitsandbytes-windows 2>nul
if %errorlevel% neq 0 (
    echo WARNING: bitsandbytes not available on Windows, skipping...
    echo You can still use the model without quantization.
)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Prepare dataset: python training\convert_to_hfdataset.py --jsonl dataset_tools\retrain_prompts_sample20.jsonl --out_dir data\hf_dataset
echo 2. Train SFT model: python training\train_peft.py --model_name_or_path gpt2 --dataset_dir data\hf_dataset --output_dir runs\sft --use_lora true
echo 3. Run full pipeline: run_full_pipeline.bat
echo.
echo For more details, see README_RLHF.md
echo.
pause
