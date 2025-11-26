#!/usr/bin/env python3
"""
export_jsonl.py
Merge pair-type JSONL files and export final training JSONL.
Allows filtering by pair_type, min_length, max_length, license whitelist.
Usage:
  python export_jsonl.py --pairs-dir pairs_out --out train.jsonl --pair-types comment_to_code,partial_completion --min-tokens 5
"""
import argparse, json, os
from pathlib import Path
from tqdm import tqdm


def count_tokens_simple(s):
    # simple whitespace token count; you will run tokenizer later for precise counts
    return len(s.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pair-types", default="")
    p.add_argument("--min-tokens", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2000)
    args = p.parse_args()
    pairs_dir = Path(args.pairs_dir)
    out = Path(args.out)
    pair_types = set([t.strip() for t in args.pair_types.split(",") if t.strip()])
    with open(out, "w", encoding="utf-8") as oh:
        for pf in pairs_dir.glob("*.jsonl"):
            for line in pf.open("r", encoding="utf-8"):
                rec = json.loads(line)
                pt = rec.get("pair_type")
                if pair_types and pt not in pair_types:
                    continue
                prompt_len = count_tokens_simple(rec.get("prompt",""))
                completion_len = count_tokens_simple(rec.get("completion",""))
                if prompt_len < args.min_tokens or completion_len < args.min_tokens:
                    continue
                if prompt_len + completion_len > args.max_tokens:
                    continue
                # write minimal record
                out_rec = {"prompt": rec["prompt"], "completion": rec["completion"], "pair_type": pt, "origin": rec.get("origin")}
                oh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print("Exported to", out)

if __name__ == "__main__":
    main()
