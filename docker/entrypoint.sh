#!/usr/bin/env bash
set -euo pipefail

echo "rlhf-qc container entrypoint"
echo "Working dir: $(pwd)"
echo "If you mounted the repo root to /workspace, run a command like:"
echo "  python3 training/generate_save.py --model_dir runs/smoke_retrain --prompts dataset_tools/retrain_prompts_sample20.jsonl --out training/generated.jsonl --n 10"
echo "  python3 training/evaluate_generated.py --generated training/generated.jsonl --out training/eval_report.json --max 10"

# If arguments are provided, run them, otherwise open a shell
if [ "$#" -gt 0 ]; then
  exec "$@"
else
  exec bash
fi
