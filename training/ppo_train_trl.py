"""
PPO training using TRL with a sequence-classification Reward Model.

Requirements:
  pip install trl transformers accelerate datasets torch

Usage example:
  python training/ppo_train_trl.py \
    --sft_model runs/sft \
    --rm_model runs/rm_demo \
    --dataset_dir data/hf_dataset \
    --out runs/ppo_demo \
    --config training/ppo_config.yaml

Notes:
 - This is a simplified PPO loop for demonstration. Adjust for large-scale runs.
 - RM is expected to be a sequence-classification model with num_labels=1.
 - Optional constrained decoding can be enabled via config to bias to C-only tokens.
"""
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Set

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

try:
    from trl import PPOTrainer, PPOConfig
except Exception as e:
    PPOTrainer = None
    _TRL_IMPORT_ERR = e


class AllowedCharsLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_chars: Set[str]):
        self.tokenizer = tokenizer
        self.allowed_ids = self._build_allowed_ids(tokenizer, allowed_chars)

    def _build_allowed_ids(self, tokenizer, allowed_chars: Set[str]):
        allowed_ids = set()
        vocab_size = tokenizer.vocab_size
        for tid in range(vocab_size):
            tok = tokenizer.convert_ids_to_tokens(tid)
            try:
                text = tokenizer.convert_tokens_to_string([tok])
            except Exception:
                text = tok
            if all((c in allowed_chars) for c in text):
                allowed_ids.add(tid)
        return allowed_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        # broadcast allowed ids across batch
        allowed = torch.tensor(list(self.allowed_ids), device=scores.device)
        mask[:, allowed] = 0.0
        scores = scores + mask
        return scores


def build_logits_processors(tokenizer, config):
    processors = LogitsProcessorList()
    if config.get('constrain_to_c', False):
        allowed_chars = set(list(config.get('allowed_chars', '')))
        processors.append(AllowedCharsLogitsProcessor(tokenizer, allowed_chars))
    return processors


def load_yaml(path: str):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_model', required=True)
    ap.add_argument('--rm_model', required=True)
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    if PPOTrainer is None:
        raise RuntimeError(f"TRL is required. Install with `pip install trl`. Import error: {_TRL_IMPORT_ERR}")

    cfg = load_yaml(args.config)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model)
    ref_model_name = cfg.get('reference_model') or args.sft_model
    rm = AutoModelForSequenceClassification.from_pretrained(args.rm_model)
    rm.eval()
    rm.to('cuda' if torch.cuda.is_available() else 'cpu')

    ppo_config = PPOConfig(
        model_name=args.sft_model,
        batch_size=cfg.get('batch_size', 8),
        forward_batch_size=cfg.get('forward_batch_size', 8),
        ppo_epochs=cfg.get('ppo_epochs', 1),
        lr=cfg.get('learning_rate', 1e-5),
        seed=cfg.get('seed', 42),
        kl_penalty='kl',
        target_kl=cfg.get('target_kl', 0.1),
        adap_kl_ctrl=True,
        init_kl_coef=cfg.get('kl_coef', 0.02),
        log_with=cfg.get('log_with'),
    )

    trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=tokenizer, ref_model=None)

    ds = load_from_disk(args.dataset_dir)
    prompts = ds['train']['prompt'] if 'train' in ds.column_names else ds['train']['prompt']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processors = build_logits_processors(tokenizer, cfg)

    def rm_score(texts: List[str]) -> List[float]:
        scores = []
        for t in texts:
            inputs = tokenizer(t, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = rm(**inputs).logits.view(-1)
            scores.append(float(out.mean().item()))
        return scores

    steps = cfg.get('save_steps', 200)
    max_new_tokens = cfg.get('max_new_tokens', 256)
    for step in range(steps):
        # sample a small batch of prompts
        idx = (step * ppo_config.batch_size) % len(prompts)
        batch_prompts = prompts[idx: idx + ppo_config.batch_size]
        # generate responses
        responses = []
        for p in batch_prompts:
            gen = trainer.generate(
                [p],
                max_new_tokens=max_new_tokens,
                logits_processor=processors,
            )
            responses.append(gen[0])
        # compute rewards
        texts = [p + r for p, r in zip(batch_prompts, responses)]
        rewards = rm_score(texts)
        # PPO step
        trainer.step(batch_prompts, responses, rewards)
        if step % 10 == 0:
            print(f"PPO step {step}/{steps}")

    trainer.save_pretrained(args.out)


if __name__ == '__main__':
    main()
