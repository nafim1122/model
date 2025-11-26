#!/usr/bin/env python3
"""
create_pairs.py
Generate training pairs of different types from cleaned files:
- comment_to_code
- partial_completion
Saves per-type JSONL files in pairs_out/

Usage:
  python create_pairs.py --cleaned dataset/cleaned --out pairs_out
"""
import argparse, re, json, logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

FUNC_SIG_RE = re.compile(r'^[\w\*\s]+?\s+([a-zA-Z_]\w*)\s*\(.*\)\s*{', re.MULTILINE | re.DOTALL)

def extract_comment_function_pairs(file_text):
    """
    Heuristic: find comment blocks immediately preceding a function signature.
    Returns list of (comment, function_code)
    """
    pairs = []
    # naive approach: find comment blocks then the following function
    comment_blocks = list(re.finditer(r'(?:/\*[\s\S]*?\*/|(?://[^\n]*\n)+)', file_text))
    for cb in comment_blocks:
        endpos = cb.end()
        # slice after comment, find next function signature
        rest = file_text[endpos: endpos + 2000]  # limited window
        m = FUNC_SIG_RE.search(rest)
        if m:
            start = endpos + m.start()
            # find full function by scanning braces (simple)
            depth = 0
            i = start
            n = len(file_text)
            while i < n:
                if file_text[i] == '{':
                    depth += 1
                elif file_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        func = file_text[start:i+1]
                        comment = cb.group(0)
                        pairs.append((comment.strip(), func.strip()))
                        break
                i += 1
    return pairs

def generate_partial_completions(file_text, max_per_file=5):
    """
    Create partial->completion pairs by truncating function bodies.
    For each function, keep signature + beginning of body with a split, completion is rest.
    """
    out = []
    for m in FUNC_SIG_RE.finditer(file_text):
        start = m.start()
        # find function end
        i = start
        depth = 0
        n = len(file_text)
        while i < n:
            if file_text[i] == '{':
                depth += 1
            elif file_text[i] == '}':
                depth -= 1
                if depth == 0:
                    func = file_text[start:i+1]
                    brace_pos = func.find('{')
                    if brace_pos == -1:
                        break
                    split_at = brace_pos + 1 + min(100, max(20, int(len(func)/4)))
                    prompt = func[:split_at].rstrip()
                    completion = func[split_at:].lstrip()
                    if len(prompt) < len(func) and completion.strip():
                        out.append((prompt, completion))
                    break
            i += 1
        if len(out) >= max_per_file:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cleaned = Path(args.cleaned)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    comment_file = outdir / "comment_to_code.jsonl"
    partial_file = outdir / "partial_completion.jsonl"
    with open(comment_file, "w", encoding="utf-8") as cf, open(partial_file, "w", encoding="utf-8") as pf:
        for f in tqdm(list(cleaned.rglob("*"))):
            if not f.is_file() or f.suffix.lower() not in {".c", ".h", ".i"}:
                continue
            txt = f.read_text(encoding="utf-8", errors="ignore")
            pairs = extract_comment_function_pairs(txt)
            for comment, func in pairs:
                prompt = comment
                completion = func
                rec = {"prompt": prompt, "completion": completion, "pair_type":"comment_to_code", "origin": str(f)}
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            partials = generate_partial_completions(txt)
            for pr, comp in partials:
                rec = {"prompt": pr, "completion": comp, "pair_type":"partial_completion", "origin": str(f)}
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Pairs generated to", outdir)

if __name__ == "__main__":
    main()
