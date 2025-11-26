Fine-tune a pretrained model on your C dataset
============================================

This folder contains helper scripts to convert JSONL pairs into a Hugging Face Dataset, fine-tune with LoRA/QLoRA, and evaluate code generation with automatic GCC compilation checks.

Prerequisites
-------------
- Python 3.9+
- Install dependencies (example):

```powershell
python -m pip install -r dataset_tools/requirements.txt
python -m pip install transformers datasets accelerate peft bitsandbytes evaluate
```

If you plan to use QLoRA (int8/nf4), install `bitsandbytes` and run on a GPU with CUDA.

Convert JSONL to HF Dataset
---------------------------
Group-by-repo split recommended to avoid repo-level leakage.

```powershell
python training/convert_to_hfdataset.py --jsonl dataset_tools/train_pairs.jsonl --out_dir data/hf_dataset --train 0.8 --val 0.1 --test 0.1 --group_by_repo
```

Training (LoRA example)
-----------------------
Edit `training/train_config_1b.json` to set model_name_or_path and other hyperparams. Then run with accelerate:

```powershell
accelerate config  # if not configured
accelerate launch training/train_peft.py --config training/train_config_1b.json
```

For QLoRA (larger models, example), set `qlora: true` in the config and use a supported GPU and bitsandbytes.

Evaluation and automatic GCC compile checks
-----------------------------------------
After training, evaluate on the test split and run GCC syntax checks:

```powershell
python training/evaluate_codegen.py --model_dir your_saved_model_dir --dataset_dir data/hf_dataset --out results/eval.jsonl --max_examples 200
```

This writes per-sample predictions and a summary printed to stdout. If `gcc` is not present on your Windows host, run the evaluation inside WSL2 or Docker with GCC installed.

Notes and recommendations
-------------------------
- Tokenization rules: reuse the base model tokenizer. Add C-specific special tokens like `<|CODE|>` and `<|INCLUDE|>` and preserve newlines; do not normalize whitespace aggressively.
- Sequence length: set to 2048 or higher for bigger context windows if your model supports it.
- LoRA vs QLoRA: prefer LoRA for small models (1–3B) and QLoRA for 7B+ when you must fine-tune with limited GPU memory.
- Evaluation: measure compile success rate, exact-match on completions, and token/char-level edit distance. Consider using CodeBLEU for more advanced metrics.
