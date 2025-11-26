# RLHF for C-only code model — runbook

This README collects runnable, tested commands and configuration to run the full RLHF pipeline
on a remote GPU or local machine with sufficient resources. It assumes a Linux-like environment
(Ubuntu) or a cloud VM. For reproducible evaluation use the `docker/rlhf_qc.Dockerfile` image.

Prereqs
- Linux (Ubuntu 20.04/22.04 recommended) or cloud GPU instance
- NVIDIA drivers + CUDA (if using GPU)
- Python 3.10+ and virtualenv
- Git
- Optional (for scorer): gcc, clang, clang-tidy, clang-format, cppcheck, valgrind

Quick setup (remote GPU VM)
1. Create & activate a virtualenv
   python3 -m venv venv
   source venv/bin/activate
2. Install Python deps (you can adapt `docker/requirements.txt`)
   pip install -U pip
   pip install -r docker/requirements.txt
   pip install trl accelerate bitsandbytes  # install on GPU host as needed

Stage 0 — prepare data and SFT model
1. Convert JSONL into HF dataset (already provided tool):
   python training/convert_to_hfdataset.py --jsonl dataset_tools/retrain_prompts_sample20.jsonl --out_dir data/hf_dataset

2. Supervised fine-tune (LoRA recommended on GPU):
   python training/train_peft.py \
     --model_name_or_path gpt2 \
     --dataset_dir data/hf_dataset \
     --output_dir runs/sft \
     --per_device_train_batch_size 8 \
     --num_train_epochs 3 \
     --use_lora true \
     --lora_r 8 --lora_alpha 32 --lora_dropout 0.05

Stage 1 — generate candidates and score (reward dataset)
1. Generate model outputs (N examples)
   python training/generate_save.py --model_dir runs/sft --prompts dataset_tools/retrain_prompts_sample20.jsonl --out training/generated.jsonl --n 500 --do_sample --temperature 0.7

2. Score generated examples with the GCC-based scorer (requires gcc/cppcheck/clang-tidy on the host or run inside the Docker image):
   python training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 10 --max 500

Stage 2 — Reward Model (RM)
1. Train a lightweight RM (we provide `training/rm_train_local.py`):
   python training/rm_train_local.py --in training/rewards.jsonl --out runs/rm --model_name distilbert-base-uncased --batch 32 --epochs 3

2. Validate RM (use `training/rm_infer.py`) to compute correlation metrics on a held-out set.

Stage 3 — PPO (RLHF)
1. Update `training/ppo_config.yaml` to point to `sft_model_path` and `rm_model_path`.
2. Ensure `trl` is installed: pip install trl
3. Run PPO trainer (example):
   python training/ppo_train.py --sft_model runs/sft --rm_model runs/rm --dataset_dir data/hf_dataset --out runs/ppo --steps 2000 --batch_size 8

Notes & safety
- Always run scoring inside the Docker image when running untrusted code (see `docker/rlhf_qc.Dockerfile`).
- Use strong forbidden-pattern penalties in `compute_rewards.py` to block dangerous syscalls and file I/O.
- Start with small batches and run thorough validations before scaling to large PPO runs.

Troubleshooting
- If Docker is not available on Windows, use WSL2 and run the Docker build there, or run scoring inside WSL.
- If PyTorch import fails with MemoryError, ensure the VM has >= 8-16 GB RAM for small experiments; for LoRA/PPO use a GPU with >= 12GB VRAM.

Appendix: Useful commands (copy-paste)
```bash
# Build evaluation Docker image (on Linux host with Docker running)
docker build -f docker/rlhf_qc.Dockerfile -t rlhf-qc:latest .

# Run scoring inside container (mount repo to /workspace)
docker run --rm --network none --cpus 1 --memory 2g -v $(pwd):/workspace -w /workspace rlhf-qc:latest \
  python3 training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 10 --max 200

``` 
