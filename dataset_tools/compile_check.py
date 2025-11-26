#!/usr/bin/env python3
"""
compile_check.py
Utility to compile a snippet or file using gcc -fsyntax-only.
Returns exit code and stderr.
Usage:
  python compile_check.py --file foo.c
  python compile_check.py --stdin < code.c
"""
import argparse, subprocess, tempfile, os, sys, json


def compile_text(text, std="c17", timeout=10):
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as tf:
        tf.write(text.encode("utf-8"))
        tf.flush()
        fname = tf.name
    try:
        try:
            proc = subprocess.run(["gcc", f"-std={std}", "-fsyntax-only", fname], capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stderr
        except FileNotFoundError as e:
            # gcc/clang not installed on system
            return -1, f"compiler not found: {e}"
        except Exception as e:
            return -2, str(e)
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if args.stdin:
        text = sys.stdin.read()
        code = text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            code = fh.read()
    else:
        print("Provide --file or --stdin", file=sys.stderr)
        sys.exit(2)
    rc, stderr = compile_text(code)
    out = {"rc": rc, "stderr": stderr}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
