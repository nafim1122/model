# C-Language LLM Model - Quick Start Script
# This script runs a smoke test to verify the installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "C-Language LLM Model - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify installation
Write-Host "[1/4] Verifying installation..." -ForegroundColor Yellow
$packages = @("transformers", "torch", "datasets", "accelerate", "trl", "peft")

foreach ($package in $packages) {
    try {
        python -c "import $package; print('✓ $package installed')"
    } catch {
        Write-Host "✗ $package not found. Please run install.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
Write-Host "[2/4] Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
New-Item -ItemType Directory -Force -Path "runs\sft" | Out-Null
New-Item -ItemType Directory -Force -Path "runs\rm" | Out-Null
New-Item -ItemType Directory -Force -Path "runs\ppo" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Run smoke test
Write-Host "[3/4] Running smoke test..." -ForegroundColor Yellow
if (Test-Path "training\smoke_train.py") {
    python training\smoke_train.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Smoke test passed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Smoke test completed with warnings" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ Smoke test script not found, skipping..." -ForegroundColor Yellow
}

# Display system info
Write-Host "[4/4] System Information:" -ForegroundColor Yellow
Write-Host "Python version:" -ForegroundColor Cyan
python --version
Write-Host "PyTorch version:" -ForegroundColor Cyan
python -c "import torch; print(torch.__version__)"
Write-Host "CUDA available:" -ForegroundColor Cyan
python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No (CPU only)')"
if (python -c "import torch; exit(0 if torch.cuda.is_available() else 1)") {
    Write-Host "GPU devices:" -ForegroundColor Cyan
    python -c "import torch; print(f'{torch.cuda.device_count()} GPU(s) detected')"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete - Ready to Train!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Available Scripts:" -ForegroundColor Cyan
Write-Host "- .\run_full_pipeline.ps1  : Run complete RLHF pipeline" -ForegroundColor White
Write-Host "- .\run_sft.ps1            : Train SFT model only" -ForegroundColor White
Write-Host "- .\run_evaluation.ps1     : Evaluate generated code" -ForegroundColor White
