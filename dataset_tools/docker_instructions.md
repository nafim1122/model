Docker / WSL run instructions

If you want to run dataset QC (compilation and linters) on Windows, the recommended approach is to use Docker Desktop or WSL2.

Option A: Docker Desktop (Windows)
1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Ensure WSL2 is enabled and Docker is configured to use WSL2 backend.
3. Start Docker Desktop.
4. From the repo root in PowerShell, build and run the QC image:

```powershell
# build image
docker build -t model-dataset-qc:latest dataset_tools

# run sample QC (mount repo into /workspace)
docker run --rm -v ${PWD}:/workspace -w /workspace model-dataset-qc:latest bash -lc "python dataset_tools/sample_qc.py --jsonl dataset_tools/train_pairs.jsonl --sample 50 --linters"
```

Option B: WSL2 (Ubuntu) without Docker
1. Install WSL2 and Ubuntu from the Microsoft Store.
2. Open Ubuntu, cd to the repo folder under /mnt/c/Users/... .
3. Install dependencies in WSL:

```bash
sudo apt update
sudo apt install -y build-essential clang clang-tidy cppcheck python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r dataset_tools/requirements.txt
```

4. Run the QC script:

```bash
python dataset_tools/sample_qc.py --jsonl dataset_tools/train_pairs.jsonl --sample 50 --linters
```

Notes
- The `--linters` flag enables `clang-tidy` and `cppcheck` checks in addition to `gcc` compilation checks.
- If using Docker, ensure the container has `clang-tidy` and `cppcheck` installed; the provided `dataset_tools/Dockerfile` installs them.
- Running linters on many samples can be slow; keep sample counts modest for CI.
