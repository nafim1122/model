Dataset tools for preparing a C-language training dataset

Overview
--------
This folder contains scripts to:
- collect C files from local repositories and (optionally) GitHub
- deduplicate and normalize C source files
- extract training pairs (comment→code, partial completion, buggy→fixed)
- run quick compile checks and QC samples
- export final JSONL datasets for fine-tuning

Usage (quick):
1. Create a Python virtualenv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Collect local files:

```powershell
python collect_local.py --base C:\path\to\repos --staging dataset\staging_local
```

3. Dedupe & normalize:

```powershell
python dedupe_normalize.py --staging dataset\staging_local --out dataset\cleaned --use-clang-format
```

4. Create pairs:

```powershell
python create_pairs.py --cleaned dataset\cleaned --out dataset\pairs_out
```

5. Export selected pairs to JSONL:

```powershell
python export_jsonl.py --pairs-dir dataset\pairs_out --out dataset\train_pairs.jsonl --pair-types comment_to_code,partial_completion --min-tokens 3
```

Notes
-----
- Some scripts use naive heuristics; for more robust extraction, install and configure `tree-sitter` C grammar and adjust `create_pairs.py` accordingly.
- `collect_github.py` requires a GitHub token when used; be mindful of rate limits and licensing.
- `clang-format`, `gcc`/`clang`, and `clang-tidy` are expected to be installed on your system for full functionality.
