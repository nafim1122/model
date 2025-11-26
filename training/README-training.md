Training README — LoRA / QLoRA fine-tuning for C-generation LLM

Overview
--------
This folder contains a reference training script and a JSON config. The script supports both LoRA and QLoRA fine-tuning using Hugging Face Transformers + PEFT + bitsandbytes.

Prerequisites
-------------
- Work on a Linux GPU instance (recommended). For Windows, use WSL2 with GPU passthrough or a Linux server.
- CUDA toolkit and drivers installed.
- Install Python dependencies (prefer inside virtualenv):

```bash
pip install -U pip
pip install datasets transformers accelerate peft bitsandbytes "wandb" torch --extra-index-url https://download.pytorch.org/whl/cu118
```

For FlashAttention 2 (optional, speeds up attention): build/install per project docs:
https://github.com/flash-attention/flash-attention

For TRL / PPO modules (optional):

```bash
pip install trl
```

Hardware recommendations
------------------------
These are practical recommendations for fine-tuning sizes 1B-7B.

- 1B params (LoRA): single GPU with 24GB (RTX 3090, A5000) is fine.
- 3B params (LoRA or QLoRA): 1x A100 40GB or 2x 24GB GPUs with tensor parallelism. QLoRA may allow single 48–80GB GPUs.
- 7B params (QLoRA recommended): single 80GB A100/H100 or sharded across two 40GB A100s. QLoRA can sometimes fit on a 24GB GPU but expect very small batch sizes and long training time.

Exact recommended GPU for comfortable fine-tuning:
- Best: 1x NVIDIA H100 80GB or 1x A100 80GB
- Good: 2x A100 40GB with DeepSpeed ZeRO-3
- Budget: 1x RTX 3090 / 24GB for small-scale LoRA on <=1B models

Quick start (example)
---------------------
1) Prepare config `training/train_config.json`. Example is included.

2) Launch training via `accelerate` (recommended config via `accelerate config`):

```bash
accelerate launch training/train.py --config training/train_config.json
```

Notes on hyperparameters
------------------------
- Seq length: 2048 recommended for file-level code. For whole-file tasks consider 4096+ (requires more memory).
- Learning rates:
  - LoRA (fp16): 1e-4 — 5e-4 (start 2e-4)
  - QLoRA: similar ranges; if using 4-bit, start smaller (1e-4 — 2e-4)
- Batch sizes: keep per-device small (1-8) and use gradient accumulation to achieve desired global batch tokens.
- Warmup: 100-1000 steps, depending on dataset size.

Validation strategy
-------------------
- Use held-out validation JSONL with prompt/completion pairs (5-10% of data). Evaluate:
  - Perplexity
  - Compile rate (sample batches compiled with `gcc -fsyntax-only`)
  - Static-analysis score on sampled outputs
- Early stopping by validation perplexity or compile-rate plateau.

Avoid overfitting tips
---------------------
- Use dropout in LoRA (0.05) and weight decay (1e-2 if needed).
- Use data augmentation (identifier renaming, whitespace changes) to reduce memorization.
- Monitor exact-match rate to detect memorization; if high, increase deduplication strictness.
- Keep a strict repo-level split for train/val/test to prevent leakage.

Post-training
-------------
- Quantize (GPTQ) or use 4-bit weights for inference to reduce memory.
- Run evaluation harness to compute compile-rate and static-analysis metrics.

References
----------
- PEFT (LoRA/QLoRA): https://github.com/huggingface/peft
- bitsandbytes / QLoRA: https://github.com/facebookresearch/bitsandbytes
- FlashAttention2: https://github.com/flash-attention/flash-attention
- TRL (PPO/Reward-tuning): https://github.com/lvwerra/trl
