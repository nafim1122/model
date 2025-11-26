#!/usr/bin/env python3
"""
create_buggy_pairs.py
Generate buggy->fixed pairs from a directory where buggy files are named *.buggy.c and fixed counterparts are *.fixed.c
Writes pairs to output directory as buggy_fixed.jsonl
"""
import argparse, json
from pathlib import Path
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bugdir", required=True, help="Directory containing .buggy.c and .fixed.c files")
    p.add_argument("--out", required=True, help="Output directory for pairs")
    args = p.parse_args()
    bugdir = Path(args.bugdir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / "buggy_fixed.jsonl"
    pairs = []
    for buggy in bugdir.rglob("*.buggy.c"):
        fixed = buggy.with_name(buggy.name.replace('.buggy.c', '.fixed.c'))
        if not fixed.exists():
            continue
        prompt = "Fix the bug in the following C code:\n" + buggy.read_text(encoding='utf-8', errors='ignore')
        completion = fixed.read_text(encoding='utf-8', errors='ignore')
        rec = {"prompt": prompt, "completion": completion, "pair_type": "buggy_fixed", "origin": str(buggy)}
        pairs.append(rec)
    with open(outf, "w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Written", len(pairs), "pairs to", outf)

if __name__ == '__main__':
    main()
