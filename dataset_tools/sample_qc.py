#!/usr/bin/env python3
"""
sample_qc.py
Quick sanity checks:
- sample JSONL pairs and run compile_check on a subset
- run tokenizer (if specified) to check for long tokens or broken special tokens
"""
import argparse, json, random, subprocess, os
from pathlib import Path


def run_compile_check(code):
    proc = subprocess.run(["python", "dataset_tools\\compile_check.py", "--stdin"], input=code.encode("utf-8"), capture_output=True)
    try:
        out = json.loads(proc.stdout.decode("utf-8"))
        return out
    except Exception:
        return {"rc": -1, "stderr": proc.stderr.decode("utf-8")}


def run_clang_tidy(code, checker_args=None):
    """Run clang-tidy on a code snippet (requires clang-tidy installed). Returns (rc, stdout+stderr)."""
    import tempfile
    checker_args = checker_args or ["-checks=*", "--warnings-as-errors=*"]
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(code)
        tf.flush()
        fname = tf.name
    cmd = ["clang-tidy", fname, "--"] + checker_args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = proc.stdout + proc.stderr
        return proc.returncode, out
    except FileNotFoundError:
        return -1, "clang-tidy not found"
    except Exception as e:
        return -2, str(e)


def run_cppcheck(code):
    """Run cppcheck on a code snippet (requires cppcheck installed). Returns (rc, stdout+stderr)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(code)
        tf.flush()
        fname = tf.name
    cmd = ["cppcheck", "--enable=all", "--inline-suppr", fname]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = proc.stdout + proc.stderr
        return proc.returncode, out
    except FileNotFoundError:
        return -1, "cppcheck not found"
    except Exception as e:
        return -2, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--sample", type=int, default=10)
    args = p.parse_args()
    with open(args.jsonl, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    samples = random.sample(lines, min(args.sample, len(lines)))
    for ln in samples:
        rec = json.loads(ln)
        print("PROMPT:", rec['prompt'][:200].replace("\n","\\n"))
        print("COMPLETION sample...")
        comp = rec['completion']
        qc = run_compile_check(comp)
        print("compile rc:", qc.get("rc"))
        if qc.get("rc") != 0:
            stderr = qc.get("stderr") or ""
            print("stderr:", stderr[:400])
        print("---")

if __name__ == "__main__":
    main()
