# Remote GPU Setup (LoRA + RM + PPO)

This guide prepares a Linux GPU machine (Ubuntu 22.04+, CUDA GPU) to run:
- SFT (LoRA) using `training/train_peft.py`
- Reward scoring via Docker or native toolchain (gcc/clang/cppcheck)
- Reward Model training (`training/rm_train_local.py`)
- PPO training with TRL (`training/ppo_train_trl.py`)

## 1) System prep
```bash
sudo apt update
sudo apt install -y build-essential gcc g++ clang clang-tidy clang-format cppcheck valgrind python3 python3-pip git
# NVIDIA drivers/CUDA should already be installed on the machine
python3 -m pip install --upgrade pip
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers datasets accelerate bitsandbytes peft trl
```

If `bitsandbytes` fails, set `CUDA_HOME` properly or skip QLoRA (use LoRA).

## 2) Data prep
- Place your prompt→completion JSONL at `dataset_tools/retrain_prompts.jsonl`.
- Convert to Hugging Face dataset:
```bash
python training/convert_to_hfdataset.py --jsonl dataset_tools/retrain_prompts.jsonl --out_dir data/hf_dataset
```

## 3) SFT (LoRA/QLoRA)
```bash
python training/train_peft.py \
  --model_name meta-llama/Llama-3.1-8B \
  --dataset_dir data/hf_dataset \
  --output_dir runs/sft_lora \
  --use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --per_device_train_batch_size 2 --learning_rate 2e-4 --num_train_epochs 1 \
  --fp16 --gradient_checkpointing
```
Adjust model, batch sizes, and epochs to fit GPU memory.

## 4) Generate and score rewards
Generate candidate outputs:
```bash
python training/generate_save.py --model_dir runs/sft_lora --prompts dataset_tools/retrain_prompts.jsonl --out training/generated.jsonl --n 200
```
Score with GCC/static tools (native):
```bash
python training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 8
```
Or run in Docker sandbox (recommended): build an image with compile tools and Python deps. See `docker/rlhf_qc.Dockerfile`.

## 5) Train Reward Model (RM)
```bash
python training/rm_train_local.py --in training/rewards.jsonl --out runs/rm_lora --model_name distilbert-base-uncased --batch 16 --epochs 3
```

## 6) PPO with TRL
Create/edit config `training/ppo_config.yaml` as needed, then:
```bash
python training/ppo_train_trl.py \
  --sft_model runs/sft_lora \
  --rm_model runs/rm_lora \
  --dataset_dir data/hf_dataset \
  --out runs/ppo_lora \
  --config training/ppo_config.yaml
```

Tips:
- Start with small `batch_size` and `ppo_epochs=1`, increase gradually.
- If outputs contain prose, enable `constrain_to_c` in config.

## 7) Evaluation
Generate and evaluate after training:
```bash
python training/generate_save.py --model_dir runs/ppo_lora --prompts dataset_tools/retrain_prompts.jsonl --out training/generated_after_ppo.jsonl --n 200
python training/evaluate_generated.py --generated training/generated_after_ppo.jsonl --out training/eval_after_ppo.json --max 200
```
Open the JSON report and compare compile rate, sanitizer pass rate, forbidden pattern rate, and avg runtime.

---
For issues or tuning help, check the README files and script flags (`-h`).
