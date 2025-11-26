"""
Training Script Runner for C-Refactoring LLM
Handles dataset creation and training execution
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def setup_environment():
    """Setup training environment"""
    print("Setting up C-Refactoring LLM training environment...")
    
    # Create necessary directories
    dirs = [
        "data",
        "checkpoints", 
        "logs",
        "c_refactoring_llm"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def validate_dataset():
    """Validate JSONL dataset format"""
    dataset_path = "data/train.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"📊 Dataset contains {len(lines)} training examples")
        
        # Validate format
        for i, line in enumerate(lines[:5], 1):  # Check first 5 lines
            try:
                data = json.loads(line.strip())
                required_fields = ['id', 'task_type', 'instruction', 'input', 'output']
                
                for field in required_fields:
                    if field not in data:
                        print(f"❌ Missing field '{field}' in line {i}")
                        return False
                        
                print(f"✓ Line {i}: Valid format - Task: {data['task_type']}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in line {i}: {e}")
                return False
        
        print("✓ Dataset format validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation error: {e}")
        return False

def create_validation_split():
    """Create train/validation split"""
    train_path = "data/train.jsonl"
    val_path = "data/val.jsonl"
    
    if os.path.exists(val_path):
        print("✓ Validation dataset already exists")
        return True
    
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Use last 10% for validation
        split_idx = int(len(lines) * 0.9)
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]
        
        # Write train split
        with open(train_path, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        
        # Write validation split
        with open(val_path, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        
        print(f"✓ Created train split: {len(train_lines)} examples")
        print(f"✓ Created validation split: {len(val_lines)} examples")
        return True
        
    except Exception as e:
        print(f"❌ Error creating validation split: {e}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("Installing required dependencies...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "tree-sitter>=0.20.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    ]
    
    for package in requirements:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ Installed: {package}")
            else:
                print(f"⚠️  Warning installing {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  Timeout installing {package}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")

def check_gpu_availability():
    """Check if GPU is available for training"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} ({gpu_count} devices)")
            return True
        else:
            print("⚠️  No GPU available, training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def run_training():
    """Execute the training script"""
    print("\n🚀 Starting C-Refactoring LLM training...")
    
    try:
        # Import and run training
        sys.path.append('c_refactoring_llm')
        from training import CRefactoringTrainer, TrainingConfig
        
        # Configure training
        config = TrainingConfig(
            learning_rate=3e-5,
            batch_size=2,  # Small batch for memory efficiency
            num_epochs=3,
            train_jsonl_path="data/train.jsonl",
            val_jsonl_path="data/val.jsonl",
            max_sequence_length=512,
            enable_qat=False,
            use_gradient_checkpointing=True,
            log_every_n_steps=10,
            save_every_n_epochs=1
        )
        
        # Create and start trainer
        trainer = CRefactoringTrainer(config)
        trainer.train()
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🔧 C-LANGUAGE CODE REFACTORING LLM TRAINING")
    print("=" * 60)
    
    steps = [
        ("Setting up environment", setup_environment),
        ("Validating dataset", validate_dataset),
        ("Creating train/val split", create_validation_split),
        ("Installing dependencies", install_dependencies),
        ("Checking GPU availability", check_gpu_availability),
        ("Running training", run_training)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            result = step_func()
            if result is False:
                print(f"❌ {step_name} failed")
                return 1
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            return 1
    
    print("\n" + "=" * 60)
    print("🎉 C-REFACTORING LLM TRAINING PIPELINE COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check 'checkpoints/' for saved models")
    print("2. View training logs in 'logs/' directory")  
    print("3. Test model with inference scripts")
    print("4. Deploy to production environment")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())