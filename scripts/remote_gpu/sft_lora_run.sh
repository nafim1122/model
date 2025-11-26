#!/usr/bin/env bash
set -euo pipefail

# Example LoRA SFT run (adjust model + batch sizes to fit GPU)
python training/convert_to_hfdataset.py \
  --jsonl dataset_tools/retrain_prompts.jsonl \
  --out_dir data/hf_dataset

python training/train_peft.py \
  --model_name meta-llama/Llama-3.1-8B \
  --dataset_dir data/hf_dataset \
  --output_dir runs/sft_lora \
  --use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --per_device_train_batch_size 2 --learning_rate 2e-4 --num_train_epochs 1 \
  --fp16 --gradient_checkpointing
