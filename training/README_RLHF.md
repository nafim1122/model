RLHF pipeline for C code generation (SFT → RM → PPO)
===================================================

This document describes a practical RLHF pipeline to improve a C-only code-generation model along three axes:
- correctness (compilable)
- zero-bug output (fixes off-by-one, etc.)
- memory safety (avoid unsafe functions, undefined behavior)

High-level stages
-----------------
1. Supervised Fine-Tuning (SFT)
   - Use curated prompt→completion pairs (the `dataset_tools` pipeline produces these).
   - Fine-tune a base decoder-only LM with LoRA/QLoRA as in `training/train_peft.py`.
   - Mask prompt tokens (loss only on completion) — implemented in `train_peft.py`.

2. Reward Model (RM)
   - Compute scalar rewards using static + compile-based checks (script: `training/compute_rewards.py`).
   - Train RM to regress from (prompt+completion) → scalar reward using `training/rm_train.py`.

3. PPO (policy optimization)
   - Use the SFT model as the initial policy and the RM for rewards.
   - Run PPO to maximize expected RM reward while constraining KL to SFT policy (script: `training/ppo_train.py`).

Automatic reward scoring (compute_rewards.py)
-------------------------------------------
The automatic reward function is a weighted sum of several checks. The default components are:

- Compilation check (gcc -fsyntax-only): large positive reward if passes.
- Warnings penalty (gcc -Wall): each warning reduces reward slightly.
- Static analyzer penalties: `cppcheck` and `clang-tidy` issues reduce reward.
- Unsafe function penalty: pattern-matching for `gets`, `strcpy`, `strcat`, `system`, `popen` etc.; presence reduces reward strongly.
- Formatting bonus: small positive reward if `clang-format` would not change the file.

The `training/compute_rewards.py` script reads a JSONL of {prompt, completion} and writes back {prompt, completion, reward, reward_details}.

Reward design tips
------------------
- Make compile success a dominant reward component. A simple scheme: compile_pass*3.0 + format*0.1 - 0.5*(num_warnings + cppcheck_issues + clang_tidy_issues) - unsafe_penalty.
- Normalize rewards to a bounded range (e.g., -10..+10) before training RM.
- For robustness, compute rewards using multiple compilers/versions (gcc, clang) and average.

Reward Model (RM) training
--------------------------
1. Prepare training data: generate candidate outputs for prompts (sample or beam), compute automatic rewards with `compute_rewards.py`.
   - You can augment with human labels later for RLHF fine-tuning.
2. Train `rm_train.py` to regress reward from prompt+completion text.
3. Evaluate RM on held-out data: check correlation with automatic rewards and compile-success classification accuracy.

PPO training details (policy optimization)
-----------------------------------------
- Use SFT model as initial policy; freeze some layers if you want stability.
- Use KL penalty or loss to anchor the policy to SFT to avoid distributional drift.
- Use small learning rate and conservative PPO hyperparameters.
- Use reward shaping: e.g., clip reward to [-1, +1] or normalize per-batch.
- Use a replay buffer of high-quality examples to mix in SFT data (stabilizes training).

Security and safety checks
--------------------------
- Always treat generated code as untrusted. Never run generated code on your host without sandboxing.
- For dynamic checks (ASAN, UBSAN), run compiled programs inside a container with restricted privileges and resource limits (Docker with --cap-drop=ALL and --ulimit flags).
- Do not use reward functions that rely on running untrusted code on the host. Prefer static checks and compilation-only checks.
- Remove or heavily penalize any completion containing system calls, shell escapes, or network I/O.

Practical commands (example workflow)
-------------------------------------
1) SFT (LoRA)
   - Convert JSONL to HF dataset:
     ```powershell
     python training/convert_to_hfdataset.py --jsonl dataset_tools/train_pairs.jsonl --out_dir data/hf_dataset --train 0.8 --val 0.1 --test 0.1 --group_by_repo
     ```
   - Fine-tune with LoRA (edit config first):
     ```powershell
     accelerate launch training/train_peft.py --config training/train_config_1b.json
     ```

2) RM dataset creation
   - Generate K candidates per prompt (use `evaluate_codegen.py` with sampling/temperature) and create `candidates.jsonl`.
   - Compute rewards:
     ```powershell
     python training/compute_rewards.py --in candidates.jsonl --out candidates_with_rewards.jsonl --timeout 5
     ```

3) Train RM
   ```powershell
   python training/rm_train.py --in candidates_with_rewards.jsonl --out runs/rm --model_name gpt2 --batch 8 --epochs 3
   ```

4) PPO
   - Ensure `trl` is installed (`pip install trl`) and you have a GPU. Then:
   ```powershell
   python training/ppo_train.py --sft_model runs/sft --rm_model runs/rm --dataset_dir data/hf_dataset --out runs/ppo --steps 1000
   ```

Extended ideas
--------------
- Human-in-the-loop: add human preference labels and fit a pairwise RM or calibrate the RM with human data.
- Unit-test harness: for functions with known inputs/outputs, automatically compile and run tests inside a sandbox to compute functional correctness reward.
- Multi-objective reward: combine correctness, performance (O(N) vs O(N^2)), and style; tune weights carefully.

Files in this folder
--------------------
- `compute_rewards.py` — computes automatic rewards from static analysis and compilation.
- `rm_train.py` — trains a reward model on scalar rewards.
- `ppo_train.py` — starter PPO script using `trl`.

If you want, I can:
- Add a Dockerfile that runs compute_rewards inside a restricted container with gcc/cppcheck/clang-tidy installed.
- Implement a pairwise RM (preferred if you have human preference data) instead of regression RM.
- Wire PPO to call the RM directly (right now `ppo_train.py` has a placeholder RM scoring path that must be adapted for your RM model type).
