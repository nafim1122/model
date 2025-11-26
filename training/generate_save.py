"""
Generate completions from an SFT causal LM and save them as JSONL

Usage (PowerShell):
  python training/generate_save.py --model_dir runs/smoke_retrain --prompts dataset_tools/retrain_prompts_sample20.jsonl --out training/generated.jsonl --n 10

This script writes one JSON object per line with fields: prompt, completion
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
                prompt = json.dumps(obj)
            prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    prompts_file = Path(args.prompts)
    out_file = Path(args.out)
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
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    prompts = load_prompts(prompts_file, max_count=args.n)
    if not prompts:
        print("No prompts found")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fout:
        for i, prompt in enumerate(prompts, start=1):
            print(f"Generating {i}/{len(prompts)}")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else 1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,
                )
            gen = out[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            record = {"prompt": prompt, "completion": text}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
