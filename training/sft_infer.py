"""
Simple inference script for an SFT causal LM checkpoint.
Loads a model directory with a causal LM and tokenizer (e.g., `runs/smoke_retrain`) and
generates completions for prompts found in a JSONL file (one object per line with key `prompt`).

Usage (PowerShell):
  python training/sft_infer.py --model_dir runs/smoke_retrain --prompts_file dataset_tools/retrain_prompts_sample20.jsonl --n 5

Outputs printed to stdout. This script is CPU-safe and does greedy decoding by default.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_prompts(jsonl_path, max_count=None):
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_count is not None and i >= max_count:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
            if prompt is None:
                # fallback to whole object string
                prompt = json.dumps(obj)
            prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--prompts_file", type=str, required=True, help="JSONL file with prompts (one JSON object per line, key 'prompt')")
    parser.add_argument("--n", type=int, default=5, help="Number of prompts to run (first n)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (auto detect if omitted)")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    prompts_file = Path(args.prompts_file)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")
    if not prompts_file.exists():
        raise SystemExit(f"Prompts file not found: {prompts_file}")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # ensure there is a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    prompts = load_prompts(prompts_file, max_count=args.n)
    if not prompts:
        print("No prompts found in prompts file.")
        return

    for idx, prompt in enumerate(prompts, start=1):
        print('='*80)
        print(f"Prompt #{idx}:")
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else 1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )

        # skip the prompt tokens when decoding
        gen = out[0][input_ids.shape[-1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print('\nGenerated completion:')
        print(text.strip())
    print('='*80)
    print(f"Ran {len(prompts)} prompts through model: {model_dir}")


if __name__ == "__main__":
    main()
