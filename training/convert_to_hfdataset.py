"""Convert JSONL of prompt/completion pairs into a Hugging Face Dataset with optional repo-level split.

Usage examples:
  python training/convert_to_hfdataset.py --jsonl dataset_tools/train_pairs.jsonl --out_dir data/hf_dataset --train 0.8 --val 0.1 --test 0.1 --group_by_repo
  python training/convert_to_hfdataset.py --jsonl dataset_tools/train_pairs.jsonl --out_dir data/hf_dataset --train 0.9 --val 0.1

The input JSONL should have at least 'prompt' and 'completion' fields. Optional fields: 'repo', 'path', 'lang', 'metadata'.
"""
import argparse
import json
import os
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_by_repo(records):
    groups = defaultdict(list)
    for r in records:
        repo = r.get('repo') or r.get('repository') or '__no_repo__'
        groups[repo].append(r)
    return groups


def split_groups(groups, train_frac, val_frac, test_frac, seed=42):
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    t = int(n * train_frac)
    v = int(n * val_frac)
    train_keys = keys[:t]
    val_keys = keys[t:t+v]
    test_keys = keys[t+v:]
    def collect(ks):
        out = []
        for k in ks:
            out.extend(groups[k])
        return out
    return collect(train_keys), collect(val_keys), collect(test_keys)


def random_split(records, train_frac, val_frac, test_frac, seed=42):
    random.Random(seed).shuffle(records)
    n = len(records)
    t = int(n * train_frac)
    v = int(n * val_frac)
    train = records[:t]
    val = records[t:t+v]
    test = records[t+v:]
    return train, val, test


def to_hf_dataset(records):
    # ensure fields exist
    cleaned = []
    for r in records:
        cleaned.append({
            'prompt': r.get('prompt',''),
            'completion': r.get('completion',''),
            'repo': r.get('repo') or r.get('repository') or None,
            'path': r.get('path') or None,
            'metadata': r.get('metadata') or None,
        })
    return Dataset.from_list(cleaned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument('--group_by_repo', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, 'train+val+test must equal 1.0'

    records = list(read_jsonl(args.jsonl))
    if args.group_by_repo:
        groups = group_by_repo(records)
        train_r, val_r, test_r = split_groups(groups, args.train, args.val, args.test, seed=args.seed)
        # if grouping produced an empty train or val (small dataset), fall back to random split
        if len(train_r) == 0 or len(val_r) == 0:
            print('Warning: repo-level grouping produced empty train/val. Falling back to random split.')
            train_r, val_r, test_r = random_split(records, args.train, args.val, args.test, seed=args.seed)
    else:
        train_r, val_r, test_r = random_split(records, args.train, args.val, args.test, seed=args.seed)

    # build DatasetDict only with non-empty splits to avoid save errors on tiny datasets
    parts = {}
    if len(train_r) > 0:
        parts['train'] = to_hf_dataset(train_r)
    if len(val_r) > 0:
        parts['validation'] = to_hf_dataset(val_r)
    if len(test_r) > 0:
        parts['test'] = to_hf_dataset(test_r)
    ds = DatasetDict(parts)

    os.makedirs(args.out_dir, exist_ok=True)
    train_len = len(ds['train']) if 'train' in ds else 0
    val_len = len(ds['validation']) if 'validation' in ds else 0
    test_len = len(ds['test']) if 'test' in ds else 0
    print(f"Saving dataset to {args.out_dir} (train={train_len}, val={val_len}, test={test_len})")
    ds.save_to_disk(args.out_dir)


if __name__ == '__main__':
    main()
