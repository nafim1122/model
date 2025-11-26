#!/usr/bin/env python3
"""
train.py
Unified fine-tuning script that supports LoRA and QLoRA using Hugging Face Transformers, PEFT, and bitsandbytes.
Supports:
- LoRA (fp16) fine-tuning
- QLoRA (4-bit) fine-tuning (requires bitsandbytes)
- Optional TRL/PPO hooks are noted but not enabled by default
- Masking prompt tokens so loss is computed only on completion

Run with:
  accelerate launch training/train.py --config training/train_config.json

This script is intended as a reference implementation — adapt target_modules and tokenizer handling per model (LLaMA/Mistral/Qwen).
"""
import os
import json
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Optional imports - may require extra packages
try:
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    import bitsandbytes as bnb
except Exception:
    get_peft_model = None
    LoraConfig = None
    prepare_model_for_kbit_training = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    train_file: str = "dataset/train_pairs.jsonl"
    output_dir: str = "./out"
    seq_length: int = 2048
    # LoRA/QLoRA toggles
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [])
    # Training hyperparams
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 500
    logging_steps: int = 50
    evaluation_strategy: str = "no"
    eval_file: Optional[str] = None
    # Misc
    seed: int = 42
    push_to_hub: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfgd = json.load(fh)
    cfg = Config(**cfgd)
    return cfg


def tokenize_examples(examples, tokenizer, seq_length, prompt_key="prompt", completion_key="completion"):
    # Combine prompt and completion; we will later mask prompt tokens in labels
    inputs = []
    for p, c in zip(examples[prompt_key], examples[completion_key]):
        full = p + "\n" + c
        inputs.append(full)
    tokenized = tokenizer(inputs, truncation=True, max_length=seq_length, padding=False)
    return tokenized


class DataCollatorForCausalLMWithMask:
    """Data collator that masks the prompt portion of inputs so loss is only on completion."""

    def __init__(self, tokenizer, seq_length, prompt_tokenizer_separator="\n", prompt_max_length=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.prompt_sep = prompt_tokenizer_separator
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        # features: list of dicts with input_ids and attention_mask
        # We assume input was prompt + "\n" + completion. We need to detect prompt boundary by finding first newline token index
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f.get("attention_mask", [1]*len(f["input_ids"])) for f in features]
        # pad
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.seq_length)
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for ids, att in zip(input_ids, attention_mask):
            ids = ids[:self.seq_length]
            att = att[:self.seq_length]
            pad_len = max_len - len(ids)
            padded = ids + [self.pad_token_id] * pad_len
            padded_att = att + [0] * pad_len
            # find prompt end by locating the first occurrence of the tokenizer-encoded '\n' token after which completion starts
            # We will approximate by finding first newline token id if present; otherwise, mask first half as prompt.
            try:
                nl_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
            except Exception:
                nl_token_id = None
            prompt_end_index = 0
            if nl_token_id is not None:
                # find first occurrence in ids
                for idx, tok in enumerate(ids):
                    if tok == nl_token_id:
                        prompt_end_index = idx + 1
                        break
            else:
                prompt_end_index = len(ids) // 2
            # Create labels: mask prompt tokens with -100
            labels = [-100] * len(padded)
            for i in range(prompt_end_index, len(ids)):
                labels[i] = padded[i]
            batch_input_ids.append(padded)
            batch_attention.append(padded_att)
            batch_labels.append(labels)
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        return batch


def make_tokenizer(tokenizer_name_or_path, seq_length):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def prepare_model(cfg: Config, tokenizer):
    # Load model with or without 4-bit
    model_kwargs = {}
    if cfg.use_qlora:
        # QLoRA settings via bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            load_in_4bit=True,
            device_map="auto",
            quantization_config=bnb.nn.quantization.QuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        # Prepare for k-bit training
        if prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)
    else:
        # Standard LoRA/fine-tune path
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16 if cfg.fp16 and torch.cuda.is_available() else None,
        )

    # Resize token embeddings if tokenizer added tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA with PEFT
    if get_peft_model is None:
        raise RuntimeError("PEFT not installed. Install 'peft' to use LoRA/QLoRA")

    if cfg.target_modules:
        target_modules = cfg.target_modules
    else:
        # Default target modules; these are common but model-specific
        # LLaMA: ['q_proj','v_proj'] | Mistral: ['q_proj','v_proj','k_proj'] etc.
        target_modules = ["q_proj", "v_proj"]

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    logger.info("PEFT/LoRA modules added. Trainable params: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def main():
    global cfg
    cfg = parse_args()
    torch.manual_seed(cfg.seed)

    tokenizer_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Load dataset
    data_files = {"train": cfg.train_file}
    if cfg.eval_file:
        data_files["validation"] = cfg.eval_file
    ds = load_dataset("json", data_files=data_files)

    # Tokenize
    def tok_fn(examples):
        return tokenize_examples(examples, tokenizer, seq_length=cfg.seq_length)

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorForCausalLMWithMask(tokenizer, seq_length=cfg.seq_length)

    # Prepare model
    model = prepare_model(cfg, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        evaluation_strategy=cfg.evaluation_strategy,
        save_total_limit=3,
        seed=cfg.seed,
        push_to_hub=cfg.push_to_hub,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None),
        data_collator=data_collator,
    )

    # Begin training
    trainer.train()

    # Save final model
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()
