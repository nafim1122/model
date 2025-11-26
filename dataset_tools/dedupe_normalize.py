#!/usr/bin/env python3
"""
dedupe_normalize.py
- Walks a staging directory
- Performs two dedupe passes:
  1) exact content SHA1
  2) normalized content SHA1 (strip header license block, normalize whitespace)
- Optionally runs clang-format (if available) to format files in-place
- Produces cleaned/normalized output under cleaned/
- Produces cleaned_manifest.jsonl with metadata fields
"""
import argparse, hashlib, re, subprocess, json, os, shutil
from pathlib import Path
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

LICENSE_BLOCK_RE = re.compile(r'/\*[\s\S]{200,5000}?\*/\s*', re.MULTILINE)

def sha1_text(s: str):
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def normalize_text(s: str):
    # Remove leading large license blocks
    s2 = LICENSE_BLOCK_RE.sub("", s, count=1)
    # Normalize whitespace: collapse multiple spaces/tabs to single spaces, strip trailing spaces
    # Keep line breaks but strip trailing white space
    lines = [re.sub(r'[ \t]+', ' ', ln).rstrip() for ln in s2.splitlines()]
    # Collapse sequences of more than 2 blank lines to 2
    normalized = []
    blank_count = 0
    for ln in lines:
        if ln.strip()=="":
            blank_count += 1
            if blank_count <= 2:
                normalized.append("")
        else:
            blank_count = 0
            normalized.append(ln)
    return "\n".join(normalized).strip() + "\n"


def maybe_clang_format(text: str):
    # try piping to clang-format if available
    try:
        p = subprocess.run(["clang-format"], input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return p.stdout.decode("utf-8")
    except Exception:
        return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staging", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--use-clang-format", dest="use_clang_format", action="store_true")
    args = p.parse_args()
    staging = Path(args.staging)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    seen_exact = set()
    seen_norm = set()
    manifest_out = outdir / "cleaned_manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as mo:
        for f in tqdm(list(staging.rglob("*"))):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".c", ".h", ".i"}:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logging.warning("Could not read %s: %s", f, e)
                continue
            sha_exact = sha1_text(txt)
            if sha_exact in seen_exact:
                continue
            seen_exact.add(sha_exact)
            norm = normalize_text(txt)
            sha_norm = sha1_text(norm)
            if sha_norm in seen_norm:
                continue
            seen_norm.add(sha_norm)
            if args.use_clang_format:
                formatted = maybe_clang_format(norm)
            else:
                formatted = norm
            # write to cleaned output
            rel = f.relative_to(staging)
            outpath = outdir / rel
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outpath.write_text(formatted, encoding="utf-8")
            entry = {
                "orig": str(f),
                "out": str(outpath),
                "sha_exact": sha_exact,
                "sha_norm": sha_norm,
                "size": len(formatted)
            }
            mo.write(json.dumps(entry) + "\n")
    print("Done. cleaned files written to", outdir)

if __name__ == "__main__":
    main()
