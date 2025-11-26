"""Evaluate code-generation outputs: generate completions, check compilation with GCC, and compute metrics.

Usage:
  python training/evaluate_codegen.py --model_dir path/to/model --dataset_dir data/hf_dataset --out results/eval.jsonl --max_examples 200

Requirements: local `gcc` in PATH (or run inside Docker). The script falls back to reporting compile failures if gcc is not found.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def compile_with_gcc(code: str) -> Tuple[bool,str]:
    # write to temp file
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False, encoding='utf-8') as f:
        f.write(code)
        fname = f.name
    # Try to run `gcc -fsyntax-only file.c`
    try:
        proc = subprocess.run(['gcc', '-fsyntax-only', fname], capture_output=True, text=True, timeout=10)
        ok = proc.returncode == 0
        out = proc.stderr.strip() or proc.stdout.strip()
    except FileNotFoundError as e:
        ok = False
        out = f'gcc not found: {e}'
    except Exception as e:
        ok = False
        out = str(e)
    finally:
        try:
            os.remove(fname)
        except Exception:
            pass
    return ok, out


def levenshtein(a: str, b: str) -> int:
    # simple DP
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_examples', type=int, default=200)
    parser.add_argument('--gen_max_length', type=int, default=512)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_dir)
    test = ds['test']
    n = min(len(test), args.max_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    out_f = Path(args.out)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    stats = {'total': n, 'compiled': 0, 'exact_match': 0, 'avg_levenshtein': 0.0}
    total_lev = 0
    for i in range(n):
        rec = test[i]
        prompt = rec['prompt']
        expected = rec['completion']
        # generate
        res = gen(prompt, max_length=len(tokenizer(prompt)['input_ids']) + args.gen_max_length, num_return_sequences=1)
        pred = res[0]['generated_text'][len(prompt):]
        ok, out = compile_with_gcc(pred)
        lev = levenshtein(pred, expected)
        total_lev += lev
        if ok:
            stats['compiled'] += 1
        if pred.strip() == expected.strip():
            stats['exact_match'] += 1
        with open(out_f, 'a', encoding='utf-8') as f:
            json.dump({'prompt': prompt, 'expected': expected, 'pred': pred, 'compiled': ok, 'compile_out': out, 'levenshtein': lev}, f)
            f.write('\n')

    stats['avg_levenshtein'] = total_lev / n if n>0 else 0
    print('Evaluation stats:', stats)


if __name__ == '__main__':
    try:
        import torch
    except Exception:
        pass
    main()
