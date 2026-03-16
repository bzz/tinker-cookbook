# Prompt Distillation / On-Policy Context Distillation

## Overview

This recipe explores several methods for teaching a language model to classify
text into one of 13 language categories **without** seeing the full classification
instructions at inference time (context distillation).

Methods compared:

| Mode | Signal | CLI flag |
|---|---|---|
| Off-policy SL | Teacher-generated labels, cross-entropy | `train.py` |
| KL-only | Teacher logprobs, KL penalty | `train_on_policy.py mode=kl_only` |
| GRPO reward | Ground-truth accuracy reward | `train_on_policy.py mode=reward_only` |
| Reward + KL | Accuracy reward + teacher KL | `train_on_policy.py mode=reward_and_kl` |

See [`WRITEUP.md`](WRITEUP.md) for results and
[`analysis_results.md`](../../data/context_distillation/analysis/analysis_results.md) for the
root-cause investigation.

---

## Full Workflow (reproducible)

### Step 0: Create the labeled dataset

Ground-truth labels are produced by an independent model or a reliable
two-step identification method, stored as JSONL.

```bash
# Option A: OpenAI API (preferred — independent ground truth, requires OPENAI_API_KEY)
python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
    --output_dir data/context_distillation --backend openai --model gpt-4o-mini

# Option B: Tinker two-step fallback (identify language → map to label set)
python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
    --output_dir data/context_distillation --backend tinker

# Quick sanity check on a few examples
python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
    --output_dir /tmp/test_ds --backend tinker --limit 15
```

Produces `train_set.jsonl` and `test_set.jsonl` with `{"text": ..., "label": ...}` lines.

### Step 1: Evaluate the baseline / pick a prompt

Use the eval script to measure any model + prompt combination on the test set:

```bash
# Base model with student prompt (short)
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --prompt student --limit 100

# Base model with teacher prompt (full instructions)
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --prompt teacher --limit 100

# Custom prompt (must contain {text})
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --prompt "Detect the language: {text}\nFinal Answer:" --limit 50

# Evaluate a trained checkpoint
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --checkpoint_path tinker://... --prompt student
```

### Step 2: Run training experiments

All hyperparameters are set via CLI for reproducibility.

All experiments read from `dataset_dir=data/context_distillation` by default.

```bash
# --- KL-only context distillation ---
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=kl_only \
    log_path=data/context_distillation/logs/exp2_kl_only \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=4 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 max_steps=30

# --- GRPO reward only ---
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=reward_only \
    log_path=data/context_distillation/logs/exp4_reward_only \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=0 max_steps=30

# --- Reward + KL combined ---
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=reward_and_kl \
    log_path=data/context_distillation/logs/exp5_reward_and_kl \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 max_steps=30
```

### Step 3: Evaluate trained checkpoints

```bash
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --checkpoint_path <checkpoint_path> --prompt student
```

### Step 4: Inspect model behavior

Use `play_w_env.py` to investigate token-level KL, group filtering, error
patterns, and advantage magnitudes:

```bash
python -m tinker_cookbook.recipes.prompt_distillation.play_w_env
```

---

## File Reference

| File | Purpose |
|---|---|
| `create_labeled_dataset.py` | Ground-truth labels via OpenAI or Tinker (two-step) → `{train,test}_set.jsonl` |
| `evaluate.py` | Evaluate any model+prompt on `test_set.jsonl` (`--limit N`, `--prompt`, `--checkpoint_path`) |
| `train_on_policy.py` | Training: kl_only / reward_only / reward_and_kl (reads from `dataset_dir`) |
| `play_w_env.py` | Analysis: token-level KL, errors, group filtering |
| `WRITEUP.md` | Experiment results and discussion |

---

## References

- [Askell et al. (2021). A General Language Assistant as a Laboratory for Alignment.](https://arxiv.org/abs/2112.00861)
- [Snell et al. (2022). Learning by Distilling Context.](https://arxiv.org/abs/2209.15189)
- [Agarwal et al. (2023). GKD: Generalized Knowledge Distillation.](https://arxiv.org/abs/2306.13649)
- [Thinking Machines Lab. On-Policy Distillation.](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [On-Policy Context Distillation project idea.](https://github.com/thinking-machines-lab/tinker-project-ideas)
