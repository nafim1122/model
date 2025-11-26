"""
Wrapper to compute scalar rewards for generated C code.

Reads a JSONL file with objects containing at least {"prompt", "completion"}
and writes a JSONL with the same objects plus {"reward", "reward_details"}.

Usage:
  python training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 8 --max 100

Implements a thin wrapper around compute_rewards.compute_reward_for_code so you
can use the same function in- and outside Docker.
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the reusable function
from training.compute_rewards import compute_reward_for_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', required=True)
    p.add_argument('--out', dest='out_path', required=True)
    p.add_argument('--timeout', type=int, default=8)
    p.add_argument('--max', type=int, default=None)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with in_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if args.max is not None and i >= args.max:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            code = obj.get('completion') or obj.get('code') or ''
            details = compute_reward_for_code(code, timeout=args.timeout)
            obj['reward'] = details['reward']
            obj['reward_details'] = details
            fout.write(json.dumps(obj) + '\n')
            n_written += 1
    print(f"Wrote {n_written} records to {out_path}")


if __name__ == '__main__':
    main()
"""
Wrap compute_rewards into a simple scorer script that converts a generated JSONL
of {prompt, completion} into a JSONL augmented with reward and reward_details.

Usage (Linux GPU host or local):
  python training/score_reward.py --in training/generated.jsonl --out training/rewards.jsonl --timeout 5 --max 100

The script imports `compute_rewards.compute_reward_for_code` and writes one JSON
line per input with fields `reward` and `reward_details` added.
"""
import argparse
import json
from pathlib import Path

from training import compute_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--timeout', type=int, default=5)
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()

    inp = Path(args.infile)
    outp = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    with inp.open('r', encoding='utf-8') as inf, outp.open('w', encoding='utf-8') as outf:
        for i, line in enumerate(inf):
            if args.max is not None and i >= args.max:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            code = rec.get('completion','')
            details = compute_rewards.compute_reward_for_code(code, timeout=args.timeout)
            rec['reward'] = details.get('reward', 0.0)
            rec['reward_details'] = details
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Wrote scored rewards to {outp}")


if __name__ == '__main__':
    main()
