"""CPU-only smoke-test harness: tiny end-to-end causal LM fine-tune.

This script runs a very small training job on CPU to validate the pipeline.
It uses a tiny pretrained model (default: sshleifer/tiny-gpt2) and the dataset
saved at `data/hf_dataset` (created earlier). It will train for 1 epoch and
write a small checkpoint to `runs/smoke`.

Usage:
  python training/smoke_train.py --dataset_dir data/hf_dataset --model_name sshleifer/tiny-gpt2 --output_dir runs/smoke --max_examples 16
"""
import argparse
import os
from pathlib import Path

import torch
import os
# disable TF/keras integrations in transformers to avoid tf dependency on CPU smoke tests
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset


def tokenize_and_mask_batch(examples, tokenizer, max_length=512):
    prompts = examples['prompt']
    completions = examples['completion']
    input_ids_list = []
    labels_list = []
    for p, c in zip(prompts, completions):
        p_tok = tokenizer(p, add_special_tokens=False)['input_ids']
        c_tok = tokenizer(c, add_special_tokens=False)['input_ids']
        input_ids = p_tok + c_tok + [tokenizer.eos_token_id]
        labels = [-100] * len(p_tok) + c_tok + [tokenizer.eos_token_id]
        # Truncate from the left if needed
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    # pad
    maxlen = max(len(x) for x in input_ids_list)
    input_ids_padded = [ids + [tokenizer.pad_token_id] * (maxlen - len(ids)) for ids in input_ids_list]
    labels_padded = [labs + [-100] * (maxlen - len(labs)) for labs in labels_list]
    return {'input_ids': input_ids_padded, 'labels': labels_padded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/hf_dataset')
    parser.add_argument('--model_name', default='sshleifer/tiny-gpt2')
    parser.add_argument('--output_dir', default='runs/smoke')
    parser.add_argument('--max_examples', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    if not Path(args.dataset_dir).exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}. Run convert_to_hfdataset.py first.")
    ds = load_from_disk(args.dataset_dir)
    # pick train split if present, else try test or entire dataset
    if hasattr(ds, 'keys') and 'train' in ds:
        data = ds['train']
    elif hasattr(ds, 'keys') and 'test' in ds:
        data = ds['test']
    else:
        # assume ds is a Dataset
        data = ds

    if len(data) == 0:
        raise ValueError('No examples found in dataset split.')

    # sample small subset
    max_n = min(len(data), args.max_examples)
    small = data.select(range(max_n))

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # resize embeddings in case tokenizer added tokens
    model.resize_token_embeddings(len(tokenizer))

    # tokenize
    tokenized = small.map(lambda x: tokenize_and_mask_batch(x, tokenizer, max_length=args.max_length), batched=True, remove_columns=small.column_names)

    # convert to torch dataset
    class TorchDataset(Dataset):
        def __init__(self, hf_dataset):
            self.ds = hf_dataset

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            row = self.ds[int(idx)]
            return {
                'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
                'labels': torch.tensor(row['labels'], dtype=torch.long),
            }

    tds = TorchDataset(tokenized)
    dl = DataLoader(tds, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Starting tiny training on {len(tds)} examples (model={args.model_name}) on device={device}")
    model.train()
    for epoch in range(1):
        total_loss = 0.0
        for step, batch in enumerate(dl):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"step {step+1}, loss={loss.item():.4f}")
        avg_loss = total_loss / len(dl) if len(dl) > 0 else float('nan')
        print(f"epoch {epoch+1} finished, avg_loss={avg_loss:.4f}")

    # save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Smoke training complete — model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
