"""Load a trained Reward Model and run inference on a small set of records.

Usage:
  python training/rm_infer.py --model_dir runs/rm_test --in training/rewards_small.jsonl --n 10
"""
import argparse
import json
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    records = read_jsonl(args.infile)
    if not records:
        print(f"No records found in {args.infile}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    import math
    from statistics import mean

    preds = []
    trues = []
    n = min(args.n, len(records))
    for i in range(n):
        r = records[i]
        text = (r.get('prompt','') + '\n' + r.get('completion','')).strip()
        toks = tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
        input_ids = toks['input_ids'].to(device)
        attention_mask = toks['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logit = out.logits.squeeze(-1).cpu().item()

        true = float(r.get('reward', 0.0))
        preds.append(logit)
        trues.append(true)
        print(f"[{i+1}/{n}] pred={logit:.6f} true={true:.6f} origin={r.get('origin','<unknown>')}")

    # compute MSE
    mse = mean([(p - t) ** 2 for p, t in zip(preds, trues)]) if preds else float('nan')
    print(f"\nEvaluated {len(preds)} samples — MSE={mse:.6f}")


if __name__ == '__main__':
    main()
