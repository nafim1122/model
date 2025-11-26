#!/usr/bin/env python3
"""
C-Language LLM Model - Installation Script
This script installs all dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*50)
    print(text)
    print("="*50 + "\n")

def print_status(step, total, message):
    print(f"[{step}/{total}] {message}...")

def run_command(cmd, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print_header("C-Language LLM Model Installation")
    
    # Check Python version
    print_status(1, 7, "Checking Python installation")
    if sys.version_info < (3, 8):
        print(f"✗ Python 3.8+ required, found {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check pip
    print_status(2, 7, "Checking pip")
    success, output = run_command(f"{sys.executable} -m pip --version")
    if success:
        print(f"✓ pip found")
    else:
        print("✗ pip not found, installing...")
        run_command(f"{sys.executable} -m ensurepip --upgrade")
    
    # Upgrade pip
    print_status(3, 7, "Upgrading pip")
    run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")
    print("✓ pip upgraded")
    
    # Install core dependencies
    print_status(4, 7, "Installing core dependencies")
    req_file = Path("docker/requirements.txt")
    if req_file.exists():
        success, _ = run_command(f"{sys.executable} -m pip install -r {req_file}")
        if success:
            print("✓ Core dependencies installed")
        else:
            print("⚠ Some core dependencies may have failed")
    else:
        print("⚠ requirements.txt not found, installing manually...")
        packages = ["transformers>=4.40.0", "torch>=2.0.0", "datasets", 
                   "accelerate", "tokenizers"]
        for pkg in packages:
            run_command(f"{sys.executable} -m pip install {pkg}", check=False)
    
    # Install RLHF dependencies
    print_status(5, 7, "Installing RLHF dependencies")
    rlhf_packages = ["trl", "peft", "scipy", "scikit-learn", "pandas", "numpy"]
    for pkg in rlhf_packages:
        success, _ = run_command(f"{sys.executable} -m pip install {pkg}", check=False)
    print("✓ RLHF dependencies installed")
    
    # Install development tools
    print_status(6, 7, "Installing development tools")
    dev_packages = ["wandb", "ipython", "jupyter", "matplotlib", "seaborn"]
    for pkg in dev_packages:
        run_command(f"{sys.executable} -m pip install {pkg}", check=False)
    print("✓ Development tools installed")
    
    # Try to install bitsandbytes
    print_status(7, 7, "Installing optional packages")
    success, _ = run_command(f"{sys.executable} -m pip install bitsandbytes", check=False)
    if not success:
        print("⚠ bitsandbytes not available (Windows users: this is normal)")
    else:
        print("✓ bitsandbytes installed")
    
    # Create directories
    print("\nCreating directories...")
    directories = ["data", "runs", "runs/sft", "runs/rm", "runs/ppo"]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    # Verify installation
    print("\nVerifying installation...")
    packages_to_check = {
        "transformers": "HuggingFace Transformers",
        "torch": "PyTorch",
        "datasets": "HuggingFace Datasets",
        "accelerate": "HuggingFace Accelerate",
        "trl": "TRL (RL library)",
        "peft": "PEFT (LoRA)"
    }
    
    all_ok = True
    for pkg, name in packages_to_check.items():
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            all_ok = False
    
    # Display system info
    print("\nSystem Information:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes ({torch.cuda.device_count()} GPU(s))")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  CUDA available: No (CPU only)")
    except Exception as e:
        print(f"  Could not get PyTorch info: {e}")
    
    print_header("Installation Complete!")
    
    if all_ok:
        print("✓ All required packages installed successfully\n")
    else:
        print("⚠ Some packages failed to install. Please check errors above.\n")
    
    print("Next Steps:")
    print("1. Prepare dataset:")
    print("   python training/convert_to_hfdataset.py --jsonl dataset_tools/retrain_prompts_sample20.jsonl --out_dir data/hf_dataset")
    print("\n2. Train SFT model:")
    print("   python training/train_peft.py --model_name_or_path gpt2 --dataset_dir data/hf_dataset --output_dir runs/sft --use_lora true")
    print("\n3. Run full pipeline:")
    print("   python run_full_pipeline.py")
    print("\nFor more details, see README_RLHF.md\n")

if __name__ == "__main__":
    main()
