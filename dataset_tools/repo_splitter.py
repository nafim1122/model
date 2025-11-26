#!/usr/bin/env python3
"""
repo_splitter.py
Split a JSONL dataset into train/val/test at the repository level to avoid leakage.
Usage:
  python repo_splitter.py --in dataset_tools/train_pairs.jsonl --out_dir dataset_tools/splits --train 0.8 --val 0.1 --test 0.1

Algorithm:
- Group records by 'origin' field. If origin missing, fall back to hash of prompt+completion.
- Shuffle repo groups deterministically with seed, then assign to splits to meet target ratios by record counts.
- Write out train.jsonl, val.jsonl, test.jsonl and a manifest mapping origin->split.
"""
import argparse, json, hashlib, random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--in', dest='infile', required=True)
p.add_argument('--out_dir', required=True)
p.add_argument('--train', type=float, default=0.8)
p.add_argument('--val', type=float, default=0.1)
p.add_argument('--test', type=float, default=0.1)
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

infile = Path(args.infile)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Read and group by origin
groups = {}
records = []
with infile.open('r', encoding='utf-8') as fh:
    for i,line in enumerate(fh,1):
        line=line.strip()
        if not line: continue
        rec = json.loads(line)
        origin = rec.get('origin') or rec.get('repo')
        if not origin:
            # fallback to small hash of content
            h = hashlib.sha1((rec.get('prompt','') + rec.get('completion','')).encode('utf-8')).hexdigest()
            origin = f'hash_{h[:8]}'
        groups.setdefault(origin, []).append(rec)

# Shuffle origins deterministically
origins = list(groups.keys())
random.Random(args.seed).shuffle(origins)

# Assign origins to splits by trying to meet record counts
total_records = sum(len(groups[o]) for o in origins)
target_train = total_records * args.train
target_val = total_records * args.val
# assign
train_list=[]
val_list=[]
test_list=[]
count_train = count_val = count_test = 0
for o in origins:
    gsize = len(groups[o])
    # greedy: assign to the split with most remaining need
    remaining_train = max(0, target_train - count_train)
    remaining_val = max(0, target_val - count_val)
    if remaining_train >= remaining_val and remaining_train >= (total_records - count_train - count_val - count_test):
        train_list.append(o); count_train += gsize
    elif remaining_val >= remaining_train:
        val_list.append(o); count_val += gsize
    else:
        test_list.append(o); count_test += gsize

# fallback fill for any unassigned (shouldn't happen)
assigned = set(train_list)|set(val_list)|set(test_list)
for o in origins:
    if o in assigned: continue
    test_list.append(o)

# write files
train_f = out_dir / 'train.jsonl'
val_f = out_dir / 'val.jsonl'
test_f = out_dir / 'test.jsonl'
manifest_f = out_dir / 'manifest.jsonl'
with train_f.open('w', encoding='utf-8') as tf, val_f.open('w', encoding='utf-8') as vf, test_f.open('w', encoding='utf-8') as xf, manifest_f.open('w', encoding='utf-8') as mf:
    for o in train_list:
        for rec in groups[o]: tf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        mf.write(json.dumps({'origin': o, 'split': 'train', 'count': len(groups[o])}) + '\n')
    for o in val_list:
        for rec in groups[o]: vf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        mf.write(json.dumps({'origin': o, 'split': 'val', 'count': len(groups[o])}) + '\n')
    for o in test_list:
        for rec in groups[o]: xf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        mf.write(json.dumps({'origin': o, 'split': 'test', 'count': len(groups[o])}) + '\n')

print('Wrote splits to', out_dir)
print('Records: total', total_records, 'train', count_train, 'val', count_val, 'test', count_test)
