"""PPO training loop using TRL (Transformer Reinforcement Learning) style API.

This script expects:
 - a pre-trained SFT model checkpoint (SFT stage)
 - a reward model (RM) saved to disk (output of rm_train.py)
 - a dataset of prompts (for generation & PPO)

It runs PPO to optimize the policy (LM) according to scalar rewards. This script
uses `trl` (or `trlx`) API if available. If not installed, the script will raise
an informative error and print the install instructions.

Usage (example):
  python training/ppo_train.py --sft_model runs/sft --rm_model runs/rm --dataset_dir data/hf_dataset --out runs/ppo --steps 1000

NOTE: PPO training is resource-intensive. For real training use multiple GPUs and
carefully tune KL penalty and learning rates. This script is a starting point.
"""
import argparse
import os
import json

try:
    from trl import PPOTrainer, PPOConfig
    from trl.core import PreTrainedModelWrapper
except Exception:
    PPOTrainer = None

from transformers import AutoTokenizer, AutoModelForCausalLM


def check_trl():
    if PPOTrainer is None:
        raise RuntimeError('`trl` (or `trlx`) is required for PPO. Install with `pip install trl` or see README.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_model', required=True)
    parser.add_argument('--rm_model', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    check_trl()

    # Load tokenizer & models
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    model = AutoModelForCausalLM.from_pretrained(args.sft_model)
    rm = AutoModelForCausalLM.from_pretrained(args.rm_model)  # RM could be sequence-classification model; adjust wrapper as needed

    # Create PPO config — these defaults are conservative
    config = PPOConfig(
        model_name=args.sft_model,
        batch_size=args.batch_size,
        forward_batch_size=args.batch_size,
        ppo_epochs=1,
        lr=1.41e-5,
        log_with=None,
    )

    # Create trainer
    trainer = PPOTrainer(
        config,
        model=model,
        tokenizer=tokenizer,
    )

    # Dataset of prompts
    from datasets import load_from_disk
    ds = load_from_disk(args.dataset_dir)
    prompts = ds['test']['prompt'] if 'test' in ds else ds['prompt']

    # PPO loop (simplified): sample a batch of prompts, generate, evaluate reward via RM, step PPO
    for step in range(0, args.steps):
        batch_prompts = prompts[step % len(prompts): (step % len(prompts)) + args.batch_size]
        # generate responses
        responses = []
        for p in batch_prompts:
            out = trainer.generate([p], max_new_tokens=128)
            responses.append(out[0])

        # compute rewards using rm: here we assume rm returns scalar logits; adapt as needed
        rewards = []
        for p, r in zip(batch_prompts, responses):
            inp = p + r
            # tokenization + forward of RM (pseudo)
            inputs = tokenizer(inp, return_tensors='pt')
            with torch.no_grad():
                logits = rm(**inputs).logits
            # transform logits to scalar reward (placeholder)
            rewards.append(float(logits.mean().item()))

        # now step PPO
        trainer.step(batch_prompts, responses, rewards)

        if step % 10 == 0:
            print(f"PPO step {step}/{args.steps}")

    # save final model
    trainer.save_pretrained(args.out)


if __name__ == '__main__':
    main()
