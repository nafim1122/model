# C-Language LLM Model - Full RLHF Pipeline
# This script runs the complete training pipeline from start to finish

param(
    [string]$ModelName = "gpt2",
    [int]$NumEpochs = 3,
    [int]$BatchSize = 8,
    [int]$NumSamples = 500,
    [switch]$SkipSFT,
    [switch]$SkipGenerate,
    [switch]$SkipScore,
    [switch]$SkipRM,
    [switch]$SkipPPO
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "C-Language LLM - Full RLHF Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model: $ModelName" -ForegroundColor White
Write-Host "  Epochs: $NumEpochs" -ForegroundColor White
Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
Write-Host "  Samples: $NumSamples" -ForegroundColor White
Write-Host ""

$ErrorActionPreference = "Stop"

# Stage 0: Prepare dataset
Write-Host "[Stage 0] Preparing dataset..." -ForegroundColor Cyan
if (Test-Path "dataset_tools\retrain_prompts_sample20.jsonl") {
    if (!(Test-Path "data\hf_dataset")) {
        python training\convert_to_hfdataset.py --jsonl dataset_tools\retrain_prompts_sample20.jsonl --out_dir data\hf_dataset
        Write-Host "✓ Dataset prepared" -ForegroundColor Green
    } else {
        Write-Host "✓ Dataset already exists" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ Sample dataset not found. Using default..." -ForegroundColor Yellow
}

# Stage 1: SFT Training
if (!$SkipSFT) {
    Write-Host ""
    Write-Host "[Stage 1] Training SFT model with LoRA..." -ForegroundColor Cyan
    python training\train_peft.py `
        --model_name_or_path $ModelName `
        --dataset_dir data\hf_dataset `
        --output_dir runs\sft `
        --per_device_train_batch_size $BatchSize `
        --num_train_epochs $NumEpochs `
        --use_lora true `
        --lora_r 8 `
        --lora_alpha 32 `
        --lora_dropout 0.05 `
        --learning_rate 2e-4 `
        --save_strategy epoch
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ SFT training complete" -ForegroundColor Green
    } else {
        Write-Host "✗ SFT training failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Stage 1] Skipping SFT training" -ForegroundColor Yellow
}

# Stage 2: Generate candidates
if (!$SkipGenerate) {
    Write-Host ""
    Write-Host "[Stage 2] Generating code samples..." -ForegroundColor Cyan
    if (Test-Path "dataset_tools\retrain_prompts_sample20.jsonl") {
        $promptFile = "dataset_tools\retrain_prompts_sample20.jsonl"
    } else {
        Write-Host "⚠ Using synthetic prompts" -ForegroundColor Yellow
        $promptFile = "training\generated.jsonl"
    }
    
    python training\generate_save.py `
        --model_dir runs\sft `
        --prompts $promptFile `
        --out training\generated.jsonl `
        --n $NumSamples `
        --do_sample `
        --temperature 0.7 `
        --max_length 512
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Generation complete" -ForegroundColor Green
    } else {
        Write-Host "✗ Generation failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Stage 2] Skipping generation" -ForegroundColor Yellow
}

# Stage 3: Score outputs
if (!$SkipScore) {
    Write-Host ""
    Write-Host "[Stage 3] Scoring generated code..." -ForegroundColor Cyan
    python training\score_reward.py `
        --in training\generated.jsonl `
        --out training\rewards.jsonl `
        --timeout 10 `
        --max $NumSamples
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Scoring complete" -ForegroundColor Green
    } else {
        Write-Host "✗ Scoring failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Stage 3] Skipping scoring" -ForegroundColor Yellow
}

# Stage 4: Train Reward Model
if (!$SkipRM) {
    Write-Host ""
    Write-Host "[Stage 4] Training Reward Model..." -ForegroundColor Cyan
    python training\rm_train_local.py `
        --in training\rewards.jsonl `
        --out runs\rm `
        --model_name distilbert-base-uncased `
        --batch 32 `
        --epochs 3
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ RM training complete" -ForegroundColor Green
    } else {
        Write-Host "✗ RM training failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Stage 4] Skipping RM training" -ForegroundColor Yellow
}

# Stage 5: PPO Training
if (!$SkipPPO) {
    Write-Host ""
    Write-Host "[Stage 5] Running PPO training..." -ForegroundColor Cyan
    python training\ppo_train.py `
        --sft_model runs\sft `
        --rm_model runs\rm `
        --dataset_dir data\hf_dataset `
        --out runs\ppo `
        --steps 2000 `
        --batch_size $BatchSize
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PPO training complete" -ForegroundColor Green
    } else {
        Write-Host "✗ PPO training failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Stage 5] Skipping PPO training" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Pipeline Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Output locations:" -ForegroundColor Cyan
Write-Host "  SFT Model: runs\sft" -ForegroundColor White
Write-Host "  Generated Code: training\generated.jsonl" -ForegroundColor White
Write-Host "  Rewards: training\rewards.jsonl" -ForegroundColor White
Write-Host "  Reward Model: runs\rm" -ForegroundColor White
Write-Host "  PPO Model: runs\ppo" -ForegroundColor White
Write-Host ""
Write-Host "To evaluate the final model:" -ForegroundColor Yellow
Write-Host "  python training\evaluate_codegen.py --model runs\ppo" -ForegroundColor White
