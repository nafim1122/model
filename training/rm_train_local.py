"""Lightweight Reward Model trainer (PyTorch loop).

This script mirrors the changes we need without touching the original `rm_train.py`.
Usage example:
  python training/rm_train_local.py --in training/rewards_small.jsonl --out runs/rm_test --model_name distilbert-base-uncased --batch 2 --epochs 1
"""
import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RewardDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = (r.get('prompt', '') + '\n' + r.get('completion', '')).strip()
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        label = float(r.get('reward', 0.0))
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(label, dtype=torch.float)}


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = read_jsonl(args.infile)
    if not records:
        print(f"No records found in {args.infile}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    ds = RewardDataset(records, tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        count = 0
        for step, batch in enumerate(dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} step {step} loss={loss.item():.4f}")

        avg = total_loss / max(1, count)
        print(f"Epoch {epoch+1} finished, avg_loss={avg:.4f}")

    print(f"Saving model to {args.out}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == '__main__':
    main()
