"""Train a Reward Model (RM) from generated candidates and scalar rewards.

Input expected: JSONL where each line is {"prompt":..., "completion":..., "reward":float}
This script fine-tunes a causal LM (or encoder-decoder) to predict a scalar reward for a
prompt+completion pair. For simplicity we train a small transformer regression head.

Usage:
  python training/rm_train.py --in rewards.jsonl --out runs/rm --model_name gpt2 --batch 8 --epochs 3

The script writes the trained reward model (transformers save_pretrained) to the output dir.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


class RewardDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = (r.get('prompt','') + '\n' + r.get('completion','')).strip()
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
        label = torch.tensor(float(r.get('reward', 0.0)), dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = read_jsonl(args.infile)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Use a sequence classification head (regression)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    ds = RewardDataset(records, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy='epoch',
        learning_rate=2e-5,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)


if __name__ == '__main__':
    main()
