#!/usr/bin/env python3
"""
collect_local.py
Scan a local base directory for .c/.h files and copy to staging preserving structure.
Usage:
  python collect_local.py --base ~/projects --staging ./dataset/staging_local
"""
import argparse, shutil, logging
from pathlib import Path
from tqdm import tqdm
import json, time

logging.basicConfig(level=logging.INFO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Local base directory to scan")
    p.add_argument("--staging", required=True)
    args = p.parse_args()
    base = Path(args.base)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    manifest = staging / "manifest_local.jsonl"
    with open(manifest, "a", encoding="utf-8") as mo:
        for src in base.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in {".c", ".h", ".i"}:
                continue
            rel = src.relative_to(base)
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            entry = {"source": str(src), "staged_path": str(target), "size": target.stat().st_size, "fetched_at": time.time()}
            mo.write(json.dumps(entry) + "\n")
            logging.info("copied %s -> %s", src, target)


if __name__ == "__main__":
    main()
