@echo off
REM C-Language LLM Model - Full RLHF Pipeline

echo ========================================
echo C-Language LLM - Full RLHF Pipeline
echo ========================================
echo.

REM Configuration
set MODEL_NAME=gpt2
set NUM_EPOCHS=3
set BATCH_SIZE=8
set NUM_SAMPLES=500

echo Configuration:
echo   Model: %MODEL_NAME%
echo   Epochs: %NUM_EPOCHS%
echo   Batch Size: %BATCH_SIZE%
echo   Samples: %NUM_SAMPLES%
echo.

REM Stage 0: Prepare dataset
echo [Stage 0] Preparing dataset...
if exist "dataset_tools\retrain_prompts_sample20.jsonl" (
    if not exist "data\hf_dataset" (
        python training\convert_to_hfdataset.py --jsonl dataset_tools\retrain_prompts_sample20.jsonl --out_dir data\hf_dataset
        if %errorlevel% neq 0 exit /b 1
        echo ✓ Dataset prepared
    ) else (
        echo ✓ Dataset already exists
    )
) else (
    echo WARNING: Sample dataset not found
)
echo.

REM Stage 1: SFT Training
echo [Stage 1] Training SFT model with LoRA...
python training\train_peft.py --model_name_or_path %MODEL_NAME% --dataset_dir data\hf_dataset --output_dir runs\sft --per_device_train_batch_size %BATCH_SIZE% --num_train_epochs %NUM_EPOCHS% --use_lora true --lora_r 8 --lora_alpha 32 --lora_dropout 0.05 --learning_rate 2e-4 --save_strategy epoch
if %errorlevel% neq 0 (
    echo ERROR: SFT training failed
    exit /b 1
)
echo ✓ SFT training complete
echo.

REM Stage 2: Generate candidates
echo [Stage 2] Generating code samples...
if exist "dataset_tools\retrain_prompts_sample20.jsonl" (
    set PROMPT_FILE=dataset_tools\retrain_prompts_sample20.jsonl
) else (
    set PROMPT_FILE=training\generated.jsonl
)
python training\generate_save.py --model_dir runs\sft --prompts %PROMPT_FILE% --out training\generated.jsonl --n %NUM_SAMPLES% --do_sample --temperature 0.7 --max_length 512
if %errorlevel% neq 0 (
    echo ERROR: Generation failed
    exit /b 1
)
echo ✓ Generation complete
echo.

REM Stage 3: Score outputs
echo [Stage 3] Scoring generated code...
python training\score_reward.py --in training\generated.jsonl --out training\rewards.jsonl --timeout 10 --max %NUM_SAMPLES%
if %errorlevel% neq 0 (
    echo ERROR: Scoring failed
    exit /b 1
)
echo ✓ Scoring complete
echo.

REM Stage 4: Train Reward Model
echo [Stage 4] Training Reward Model...
python training\rm_train_local.py --in training\rewards.jsonl --out runs\rm --model_name distilbert-base-uncased --batch 32 --epochs 3
if %errorlevel% neq 0 (
    echo ERROR: RM training failed
    exit /b 1
)
echo ✓ RM training complete
echo.

REM Stage 5: PPO Training
echo [Stage 5] Running PPO training...
python training\ppo_train.py --sft_model runs\sft --rm_model runs\rm --dataset_dir data\hf_dataset --out runs\ppo --steps 2000 --batch_size %BATCH_SIZE%
if %errorlevel% neq 0 (
    echo ERROR: PPO training failed
    exit /b 1
)
echo ✓ PPO training complete
echo.

REM Summary
echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo Output locations:
echo   SFT Model: runs\sft
echo   Generated Code: training\generated.jsonl
echo   Rewards: training\rewards.jsonl
echo   Reward Model: runs\rm
echo   PPO Model: runs\ppo
echo.
echo To evaluate the final model:
echo   python training\evaluate_codegen.py --model runs\ppo
echo.
pause
