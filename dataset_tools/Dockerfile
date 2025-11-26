# Minimal container for running dataset QC with a C compiler
# Usage: docker build -t c-qc .
#        docker run --rm -v $(pwd):/workspace -w /workspace c-qc powershell -Command "python dataset_tools/sample_qc.py --jsonl dataset_tools/train_pairs.jsonl --sample 10"

FROM python:3.10-slim

# Install build essentials and clang/gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang gcc clang-format git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY dataset_tools/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Copy repo files (small) into container
COPY . /workspace
WORKDIR /workspace

# Default to bash
CMD ["bash"]
