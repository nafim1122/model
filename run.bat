@echo off
REM C-Language LLM Model - Quick Start Script

echo ========================================
echo C-Language LLM Model - Quick Start
echo ========================================
echo.

REM Verify installation
echo [1/3] Verifying installation...
python -c "import transformers; print('✓ transformers installed')" 2>nul || echo WARNING: transformers not found
python -c "import torch; print('✓ torch installed')" 2>nul || echo WARNING: torch not found
python -c "import datasets; print('✓ datasets installed')" 2>nul || echo WARNING: datasets not found
python -c "import accelerate; print('✓ accelerate installed')" 2>nul || echo WARNING: accelerate not found
python -c "import trl; print('✓ trl installed')" 2>nul || echo WARNING: trl not found
python -c "import peft; print('✓ peft installed')" 2>nul || echo WARNING: peft not found
echo.

REM Create directories
echo [2/3] Creating directories...
if not exist "data" mkdir data
if not exist "runs" mkdir runs
if not exist "runs\sft" mkdir runs\sft
if not exist "runs\rm" mkdir runs\rm
if not exist "runs\ppo" mkdir runs\ppo
echo ✓ Directories created
echo.

REM Display system info
echo [3/3] System Information:
echo Python version:
python --version
echo PyTorch version:
python -c "import torch; print(torch.__version__)"
echo CUDA available:
python -c "import torch; print('Yes - GPU available' if torch.cuda.is_available() else 'No - CPU only')"
echo.

echo ========================================
echo Setup Complete - Ready to Train!
echo ========================================
echo.
echo Available Scripts:
echo - install.bat              : Install all dependencies
echo - run_full_pipeline.bat    : Run complete RLHF pipeline
echo - run_sft.bat              : Train SFT model only
echo.
pause
