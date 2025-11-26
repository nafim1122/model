# C-Language LLM Model - Windows Setup Guide

## Quick Start (Windows)

### Option 1: Automated Installation (Recommended)

Open **Command Prompt** (cmd.exe) or **Windows Terminal** and run:

```cmd
cd c:\Users\Lenovo\Downloads\model
python install.py
```

### Option 2: Manual Installation

If the automated script doesn't work, follow these steps:

#### Step 1: Verify Python
```cmd
python --version
```
You need Python 3.8 or higher. If not installed, download from [python.org](https://www.python.org/)

#### Step 2: Upgrade pip
```cmd
python -m pip install --upgrade pip setuptools wheel
```

#### Step 3: Install Core Dependencies
```cmd
python -m pip install -r docker\requirements.txt
```

#### Step 4: Install RLHF Dependencies
```cmd
python -m pip install trl peft scipy scikit-learn pandas numpy matplotlib seaborn wandb
```

#### Step 5: Install Development Tools (Optional)
```cmd
python -m pip install ipython jupyter black flake8
```

#### Step 6: Create Directories
```cmd
mkdir data
mkdir runs
mkdir runs\sft
mkdir runs\rm
mkdir runs\ppo
```

#### Step 7: Verify Installation
```cmd
python -c "import transformers, torch, datasets, accelerate, trl, peft; print('All packages installed successfully!')"
```

---

## Running the Model

### Full RLHF Pipeline

Run the complete training pipeline:

```cmd
python run_full_pipeline.py
```

Or step by step:

#### Stage 0: Prepare Dataset
```cmd
python training\convert_to_hfdataset.py --jsonl dataset_tools\retrain_prompts_sample20.jsonl --out_dir data\hf_dataset
```

#### Stage 1: Train SFT Model (LoRA)
```cmd
python training\train_peft.py ^
  --model_name_or_path gpt2 ^
  --dataset_dir data\hf_dataset ^
  --output_dir runs\sft ^
  --per_device_train_batch_size 8 ^
  --num_train_epochs 3 ^
  --use_lora true ^
  --lora_r 8 ^
  --lora_alpha 32 ^
  --lora_dropout 0.05
```

#### Stage 2: Generate Code Samples
```cmd
python training\generate_save.py ^
  --model_dir runs\sft ^
  --prompts dataset_tools\retrain_prompts_sample20.jsonl ^
  --out training\generated.jsonl ^
  --n 500 ^
  --do_sample ^
  --temperature 0.7
```

#### Stage 3: Score Generated Code
```cmd
python training\score_reward.py ^
  --in training\generated.jsonl ^
  --out training\rewards.jsonl ^
  --timeout 10 ^
  --max 500
```

#### Stage 4: Train Reward Model
```cmd
python training\rm_train_local.py ^
  --in training\rewards.jsonl ^
  --out runs\rm ^
  --model_name distilbert-base-uncased ^
  --batch 32 ^
  --epochs 3
```

#### Stage 5: PPO Training (RLHF)
```cmd
python training\ppo_train.py ^
  --sft_model runs\sft ^
  --rm_model runs\rm ^
  --dataset_dir data\hf_dataset ^
  --out runs\ppo ^
  --steps 2000 ^
  --batch_size 8
```

---

## Available Scripts

### Installation Scripts
- `install.py` - Python-based installer (cross-platform)
- `install.bat` - Windows batch installer
- `install.ps1` - PowerShell installer (requires PowerShell Core)

### Running Scripts
- `run_full_pipeline.py` - Complete RLHF pipeline
- `run_full_pipeline.bat` - Windows batch version
- `run_full_pipeline.ps1` - PowerShell version

### Quick Start
- `run.py` - Quick verification and setup
- `run.bat` - Windows batch version

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: 20 GB for models and datasets

### GPU Requirements (Optional but Recommended)
- **VRAM**: 12 GB minimum for LoRA training
- **CUDA**: 11.7 or higher
- **GPU**: NVIDIA GPU with Compute Capability 7.0+

### CPU-Only Mode
The model can run on CPU, but training will be much slower:
- Reduce batch size to 1-2
- Reduce model size (use `gpt2` instead of larger models)
- Expect 10-50x slower training

---

## Troubleshooting

### Python not found
Install Python from [python.org](https://www.python.org/) and ensure it's added to PATH

### pip install fails
Try:
```cmd
python -m pip install --upgrade pip
python -m pip cache purge
python -m pip install <package_name> --no-cache-dir
```

### CUDA not available
This is normal on systems without NVIDIA GPU. The model will use CPU.
To check:
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of memory errors
Reduce batch size:
```cmd
python training\train_peft.py ... --per_device_train_batch_size 1
```

### bitsandbytes fails on Windows
This is expected. The package is Linux-only. You can skip it or use CPU quantization.

### PowerShell execution policy errors
If using .ps1 scripts, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Output Files

After running the pipeline, you'll have:

```
c:\Users\Lenovo\Downloads\model\
├── data\
│   └── hf_dataset\          # Prepared HuggingFace dataset
├── runs\
│   ├── sft\                 # Supervised fine-tuned model (LoRA)
│   ├── rm\                  # Reward model
│   └── ppo\                 # Final RLHF model
└── training\
    ├── generated.jsonl      # Generated code samples
    └── rewards.jsonl        # Scored samples with rewards
```

---

## Next Steps

1. **Evaluate the model**:
   ```cmd
   python training\evaluate_codegen.py --model runs\ppo
   ```

2. **Generate C code**:
   ```cmd
   python training\sft_infer.py --model runs\ppo --prompt "Write a function to reverse a string"
   ```

3. **Fine-tune further**:
   - Add more training data to `data/hf_dataset`
   - Adjust hyperparameters in training scripts
   - Run more PPO steps for better alignment

---

## Documentation

- `README_RLHF.md` - Complete RLHF pipeline documentation
- `training/README-training.md` - Training details
- `training/README_FINE_TUNE.md` - Fine-tuning guide

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation files
3. Check GPU/CUDA compatibility
4. Ensure all dependencies are installed

---

## License & Credits

This is a domain-specific LLM for C-language code generation with RLHF.
See individual files for license information.
