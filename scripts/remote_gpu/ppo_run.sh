#!/usr/bin/env bash
set -euo pipefail

# Generate and score
python training/generate_save.py \
  --model_dir runs/sft_lora \
  --prompts dataset_tools/retrain_prompts.jsonl \
  --out training/generated.jsonl \
  --n 500

python training/score_reward.py \
  --in training/generated.jsonl \
  --out training/rewards.jsonl \
  --timeout 8

# Train RM (distilbert regression)
python training/rm_train_local.py \
  --in training/rewards.jsonl \
  --out runs/rm_lora \
  --model_name distilbert-base-uncased \
  --batch 16 \
  --epochs 3

# PPO
python training/ppo_train_trl.py \
  --sft_model runs/sft_lora \
  --rm_model runs/rm_lora \
  --dataset_dir data/hf_dataset \
  --out runs/ppo_lora \
  --config training/ppo_config.yaml

# Evaluate after PPO
python training/generate_save.py \
  --model_dir runs/ppo_lora \
  --prompts dataset_tools/retrain_prompts.jsonl \
  --out training/generated_after_ppo.jsonl \
  --n 200

python training/evaluate_generated.py \
  --generated training/generated_after_ppo.jsonl \
  --out training/eval_after_ppo.json \
  --max 200
