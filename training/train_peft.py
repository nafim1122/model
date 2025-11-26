"""Fine-tune a decoder-only model using Hugging Face Transformers + PEFT (LoRA/QLoRA).

Supports:
 - LoRA (fp16 or bf16)
 - QLoRA (int8+nf4 via bitsandbytes) for large models

Usage (example):
  accelerate launch training/train_peft.py --config training/train_config_1b.json

Config file (JSON) should contain keys: model_name_or_path, output_dir, per_device_train_batch_size,
gradient_accumulation_steps, epochs, lr, lora (dict) or qlora (bool), max_length, save_steps, logging_steps
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Callable

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_int8_training = None

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None


@dataclass
class Config:
    model_name_or_path: str
    output_dir: str
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_length: int = 1024
    save_steps: int = 500
    logging_steps: int = 50
    lora_rank: Optional[int] = None
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    qlora: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return Config(**cfg)


def build_tokenizer(model_name_or_path, add_special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if add_special_tokens:
        tokenizer.add_special_tokens(add_special_tokens)
    return tokenizer


def tokenize_and_mask(examples, tokenizer, max_length=1024):
    # Tokenize prompt and completion separately so we can mask prompt tokens in labels.
    prompts = examples['prompt']
    completions = examples['completion']
    input_ids_list = []
    labels_list = []
    for p, c in zip(prompts, completions):
        # keep raw prompt and completion concatenated as model input
        p_tok = tokenizer(p, add_special_tokens=False)['input_ids']
        c_tok = tokenizer(c, add_special_tokens=False)['input_ids']
        input_ids = p_tok + c_tok + [tokenizer.eos_token_id]
        # create labels: mask prompt tokens with -100
        labels = [-100] * len(p_tok) + c_tok + [tokenizer.eos_token_id]
        # truncate if needed (prefer keeping completion tail)
        if len(input_ids) > max_length:
            # if too long, truncate from the left (trim prompt)
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
            # ensure we still mask any truncated prompt tokens
            for i in range(min(len(labels), len(input_ids))):
                pass
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    # pad to longest
    maxlen = max(len(x) for x in input_ids_list)
    input_ids_padded = []
    labels_padded = []
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = maxlen - len(ids)
        input_ids_padded.append(ids + [tokenizer.pad_token_id] * pad_len)
        labels_padded.append(labs + [-100] * pad_len)
    return {'input_ids': input_ids_padded, 'labels': labels_padded}


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = build_tokenizer(cfg.model_name_or_path, add_special_tokens={'additional_special_tokens': ['<|CODE|>','<|INCLUDE|>']})

    # load dataset from disk (convert_to_hfdataset.py output)
    from datasets import load_from_disk
    ds = load_from_disk('data/hf_dataset')

    # tokenize with labels masking
    tokenized = ds.map(lambda x: tokenize_and_mask(x, tokenizer, max_length=cfg.max_length), batched=True, remove_columns=ds['train'].column_names)

    # Model loading: support QLoRA (load_in_8bit) or normal fp16/fp32
    model_kwargs = {}
    if cfg.qlora:
        if bnb is None:
            raise RuntimeError('bitsandbytes is required for QLoRA (install bitsandbytes and a CUDA-enabled GPU)')
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, load_in_8bit=True, device_map='auto')
        # prepare for int8 training if available
        if prepare_model_for_int8_training is not None:
            model = prepare_model_for_int8_training(model)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=dtype)

    # If LoRA is requested (lora_rank > 0), patch model
    if cfg.lora_rank and cfg.lora_rank > 0:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError('PEFT is required for LoRA. Install peft.')
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )
        model = get_peft_model(model, peft_config)

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        fp16=torch.cuda.is_available(),
        optim='adamw_torch',
        remove_unused_columns=False,
    )

    # data collator
    # Data collator already has labels prepared by tokenize_and_mask
    from transformers import DefaultDataCollator
    data_collator = DefaultDataCollator(return_tensors='pt')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == '__main__':
    main()
