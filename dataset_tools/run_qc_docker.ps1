# PowerShell helper: build and run the QC Docker container
param(
    [string] $sampleCount = "10"
)

$tag = "model-dataset-qc:latest"
Write-Host "Building Docker image $tag..."
docker build -t $tag dataset_tools

Write-Host "Running sample QC inside container (sample=$sampleCount)..."
# Mount current repo into /workspace and run the sample_qc.py script
docker run --rm -v ${PWD}:/workspace -w /workspace $tag bash -lc "python dataset_tools/sample_qc.py --jsonl dataset_tools/train_pairs.jsonl --sample $sampleCount"
