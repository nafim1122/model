"""
Training Pipeline for C-Refactoring LLM
Multi-objective training with compilation safety
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
import os
from typing import Dict, Any, List, Optional
import json
import subprocess
import tempfile
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from architecture import CRefactoringLLM, CRefactorConfig, CCodeDataset
from memory import comprehensive_code_analysis
import gc

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Compilation safety
    compile_check_frequency: int = 100  # Check every N batches
    gcc_flags: str = "-Wall -Wextra -pedantic -std=c11 -O2"
    max_compile_errors: int = 5  # Max errors before penalizing
    
    # Quantization-aware training
    enable_qat: bool = True
    qat_backend: str = "fbgemm"
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    
    # Logging
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 2
    
    # Data
    train_jsonl_path: str = "data/train.jsonl"
    val_jsonl_path: str = "data/val.jsonl"
    max_sequence_length: int = 512

class CompilationChecker:
    """Check if generated C code compiles successfully"""
    
    def __init__(self, gcc_flags: str = "-Wall -Wextra -pedantic -std=c11"):
        self.gcc_flags = gcc_flags
    
    def check_compilation(self, code: str) -> Dict[str, Any]:
        """
        Check if C code compiles without errors
        
        Returns:
            Dictionary with compilation results
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            # Add necessary includes if missing
            if '#include' not in code:
                full_code = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n" + code
            else:
                full_code = code
            
            f.write(full_code)
            f.flush()
            
            try:
                # Try to compile
                result = subprocess.run(
                    f"gcc {self.gcc_flags} {f.name} -o {f.name}.out",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                compilation_success = result.returncode == 0
                errors = result.stderr if result.stderr else ""
                warnings = result.stdout if result.stdout else ""
                
                # Count different types of issues
                error_count = errors.count('error:')
                warning_count = errors.count('warning:')
                
                # Clean up
                try:
                    os.unlink(f.name)
                    if os.path.exists(f"{f.name}.out"):
                        os.unlink(f"{f.name}.out")
                except:
                    pass
                
                return {
                    'success': compilation_success,
                    'errors': errors,
                    'warnings': warnings,
                    'error_count': error_count,
                    'warning_count': warning_count,
                    'return_code': result.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'errors': "Compilation timeout",
                    'warnings': "",
                    'error_count': 1,
                    'warning_count': 0,
                    'return_code': -1
                }
            except Exception as e:
                return {
                    'success': False,
                    'errors': f"Compilation error: {str(e)}",
                    'warnings': "",
                    'error_count': 1,
                    'warning_count': 0,
                    'return_code': -1
                }

class LoRATrainer:
    """LoRA-aware training utilities"""
    
    @staticmethod
    def apply_lora(model: nn.Module, config: TrainingConfig):
        """Apply LoRA adapters to model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=["q", "v", "k", "o", "wi", "wo"]  # T5 modules
            )
            
            return get_peft_model(model, lora_config)
        except ImportError:
            print("PEFT not available, using full fine-tuning")
            return model

class CRefactoringTrainer:
    """Main training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        from architecture import create_model_and_tokenizer
        self.model, self.tokenizer, self.model_config = create_model_and_tokenizer()
        
        # Apply LoRA if enabled
        if hasattr(config, 'lora_r'):
            self.model = LoRATrainer.apply_lora(self.model, config)
        
        self.model.to(self.device)
        
        # Enable gradient checkpointing
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Compilation checker
        self.compiler = CompilationChecker(config.gcc_flags)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize quantization if enabled
        if config.enable_qat:
            self._setup_quantization()
    
    def _setup_quantization(self):
        """Setup quantization-aware training"""
        try:
            torch.backends.quantized.engine = self.config.qat_backend
            self.model.qconfig = torch.quantization.get_default_qat_qconfig(self.config.qat_backend)
            torch.quantization.prepare_qat(self.model, inplace=True)
        except Exception as e:
            print(f"QAT setup failed: {e}, continuing without QAT")
    
    def create_dataloaders(self) -> tuple:
        """Create training and validation dataloaders"""
        
        # Training dataset
        train_dataset = CCodeDataset(
            self.config.train_jsonl_path,
            self.tokenizer,
            self.config.max_sequence_length
        )
        
        # Validation dataset
        val_dataset = CCodeDataset(
            self.config.val_jsonl_path,
            self.tokenizer,
            self.config.max_sequence_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """Setup optimizer and learning rate scheduler"""
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def compilation_penalty(self, generated_code: str) -> float:
        """Calculate compilation penalty for generated code"""
        try:
            result = self.compiler.check_compilation(generated_code)
            
            if result['success']:
                penalty = 0.0
                # Small penalty for warnings
                penalty += result['warning_count'] * 0.1
            else:
                # Heavy penalty for compilation errors
                penalty = min(result['error_count'], self.config.max_compile_errors) * 2.0
            
            return penalty
            
        except Exception:
            # If compilation check fails, assume moderate penalty
            return 1.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            decoder_attention_mask=batch['decoder_attention_mask'],
            labels=batch['labels'],
            ast_features=batch['ast_features']
        )
        
        loss = outputs['loss']
        
        # Add compilation penalty periodically
        if self.global_step % self.config.compile_check_frequency == 0:
            # Generate samples and check compilation
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=batch['input_ids'][:1],  # Check first sample
                    attention_mask=batch['attention_mask'][:1],
                    max_length=256,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                compile_penalty = self.compilation_penalty(generated_text)
                
                # Add penalty to loss
                loss += compile_penalty * self.config.compile_weight
        
        return {
            'loss': loss.item(),
            'error_detection_accuracy': 0.0,  # Placeholder
            'compilation_success_rate': 0.0   # Placeholder
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        compilation_successes = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels'],
                    ast_features=batch['ast_features']
                )
                
                total_loss += outputs['loss'].item()
                
                # Check compilation success for a few samples
                if total_samples < 20:  # Limit compilation checks
                    generated = self.model.generate(
                        input_ids=batch['input_ids'][:1],
                        attention_mask=batch['attention_mask'][:1],
                        max_length=256,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    compile_result = self.compiler.check_compilation(generated_text)
                    
                    if compile_result['success']:
                        compilation_successes += 1
                    
                    total_samples += 1
        
        avg_loss = total_loss / len(val_loader)
        compilation_success_rate = compilation_successes / max(total_samples, 1)
        
        self.model.train()
        
        return {
            'val_loss': avg_loss,
            'val_compilation_success_rate': compilation_success_rate
        }
    
    def train(self):
        """Main training loop"""
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)
        
        # Initialize wandb
        wandb.init(
            project="c-refactoring-llm",
            config=self.config.__dict__
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
                
                # Training step
                optimizer.zero_grad()
                metrics = self.train_step(batch)
                
                # Backward pass
                loss = torch.tensor(metrics['loss'], requires_grad=True)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                
                epoch_loss += metrics['loss']
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    wandb.log({
                        'train_loss': metrics['loss'],
                        'learning_rate': scheduler.get_last_lr()[0],
                        'global_step': self.global_step,
                        'epoch': epoch
                    })
                
                # Memory cleanup
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {epoch_loss / len(train_loader):.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Compilation Success Rate: {val_metrics['val_compilation_success_rate']:.2%}")
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss_epoch': epoch_loss / len(train_loader),
                **val_metrics
            })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)
        
        print("Training completed!")
        wandb.finish()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = f"checkpoints/c_refactoring_epoch_{epoch + 1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    # Training configuration
    config = TrainingConfig(
        learning_rate=3e-5,
        batch_size=4,  # Smaller for memory
        num_epochs=5,
        enable_qat=False,  # Disable for initial training
        use_gradient_checkpointing=True
    )
    
    # Start training
    trainer = CRefactoringTrainer(config)
    trainer.train()