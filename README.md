# C-only Code LLM — Full RLHF Pipeline

Complete end-to-end pipeline to train an LLM that generates **only C code** with correctness, safety, and zero-bug objectives using RLHF (SFT → RM → PPO).

## Project Structure
```
dataset_tools/       # Dataset collection, deduplication, JSONL generation
training/            # SFT, RM, PPO training scripts, evaluation
  - train_peft.py           # LoRA/QLoRA fine-tuning (SFT)
  - score_reward.py         # GCC-based automatic reward scorer
  - rm_train_local.py       # Reward Model trainer (PyTorch)
  - ppo_train_trl.py        # PPO trainer (TRL)
  - generate_save.py        # Generate completions and save JSONL
  - evaluate_generated.py   # Compile, static-analysis, runtime eval
docker/              # Sandboxed QC environment (gcc, clang, cppcheck)
scripts/remote_gpu/  # Ready-to-run bash scripts for GPU instances
docs/                # REMOTE_GPU.md setup guide
```

## Quick Start (Local CPU Demo)
**Note**: The smoke checkpoint produces degenerate outputs. For real training use GPU + larger dataset.

1. **Generate small dataset** (or use existing sample):
```powershell
python dataset_tools/generate_retrain_sample20.py
```

2. **Validate JSONL**:
```powershell
python dataset_tools/jsonl_validate.py --jsonl dataset_tools/retrain_prompts_sample20.jsonl --sample 3
```

3. **Convert to HF dataset**:
```powershell
python training/convert_to_hfdataset.py --jsonl dataset_tools/retrain_prompts_sample20.jsonl --out_dir data/hf_dataset
```

4. **CPU smoke SFT** (tiny model, 1 epoch):
```powershell
python training/smoke_train.py --dataset_dir data/hf_dataset --model_name sshleifer/tiny-gpt2 --output_dir runs/smoke_demo --max_examples 8
```

5. **Generate outputs**:
```powershell
python training/generate_save.py --model_dir runs/smoke_demo --prompts dataset_tools/retrain_prompts_sample20.jsonl --out training/generated.jsonl --n 10
```

6. **Score rewards** (requires gcc/cppcheck locally or in Docker):
```powershell
python training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 8
```

7. **Train RM** (CPU, small):
```powershell
python training/rm_train_local.py --in training/rewards.jsonl --out runs/rm_demo --model_name distilbert-base-uncased --batch 2 --epochs 1
```

8. **Evaluate**:
```powershell
python training/evaluate_generated.py --generated training/generated.jsonl --out training/eval_report.json --max 10
```

## Remote GPU (LoRA + PPO)
See `docs/REMOTE_GPU.md` for full setup on Ubuntu 22.04 + NVIDIA GPU.

**Quick commands**:
```bash
# Install deps
pip install -U torch transformers datasets accelerate bitsandbytes peft trl

# LoRA SFT
bash scripts/remote_gpu/sft_lora_run.sh

# Full RLHF pipeline (generate → score → RM → PPO → eval)
bash scripts/remote_gpu/ppo_run.sh
```

## Key Features
- **Reward design**: Compile success, GCC warnings, sanitizers (ASAN/UBSAN), static analysis (cppcheck/clang-tidy), forbidden patterns, formatting.
- **Safety**: Dockerized sandbox for scoring (network disabled, resource limits).
- **C-only outputs**: Optional logits processor constrains generation to C character set.
- **Evaluation metrics**: Compile rate, bug-free rate, static-analysis pass, runtime performance.

## Dependencies
```
torch>=2.0
transformers>=4.40
datasets
accelerate
bitsandbytes  # for QLoRA (GPU only)
peft
trl
pyyaml
```

Install:
```powershell
pip install -r docker/requirements.txt
```

## Current Status
- ✅ Dataset tooling (collection, dedupe, JSONL validation, HF conversion)
- ✅ SFT scripts (LoRA/QLoRA, smoke CPU harness)
- ✅ Reward scoring (GCC-based, static analysis, forbidden patterns)
- ✅ RM training (PyTorch loop)
- ✅ PPO scaffold (TRL-based, optional C-only constraint)
- ✅ Evaluation harness (compile, sanitizers, static checks, runtime)
- 🔄 Docker QC image (Dockerfile ready, needs Docker daemon running to build)

## Next Steps
1. **Start Docker Desktop** and build `docker/rlhf_qc.Dockerfile` for safe reward scoring.
2. **Generate full dataset** (100+ prompt→completion pairs).
3. **Run on GPU** using scripts in `scripts/remote_gpu/`.
4. **Iterate on reward weights** and PPO hyperparams based on eval metrics.

## Known Issues
- Smoke checkpoint outputs are degenerate (needs proper SFT on GPU with larger dataset).
- Reward scoring on Windows requires WSL or Docker (gcc/cppcheck/clang-tidy not natively installed).
- RM/PPO training requires significant RAM and GPU; local CPU runs may fail with MemoryError.

---
For detailed remote GPU instructions, see `docs/REMOTE_GPU.md`.
