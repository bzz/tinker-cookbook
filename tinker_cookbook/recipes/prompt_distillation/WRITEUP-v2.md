# On-Policy Context Distillation — v2 (corrected dataset)

wandb project: [bzz2-ml-collective/ctx_distill](https://wandb.ai/bzz2-ml-collective/ctx_distill)

## What changed from v1

The v1 dataset had **broken `ot` labels**: the teacher model outputs real ISO
codes (`sw`, `th`, `bg`) for unsupported languages even with the full prompt, so
the stochastic labeling run (temperature 0.15) produced inconsistent labels.
Off-policy SL trained on data with **zero `ot` examples**.

v2 uses a **two-step labeling method** (`create_labeled_dataset.py`):
identify the ISO code → deterministically map out-of-set codes to `ot`.
This gives correct labels: 337/1680 train `ot` (20%), 67/420 test `ot` (16%).

## Setup

- **Model**: `Qwen/Qwen3-30B-A3B`, LoRA rank 32, `qwen3_disable_thinking`
- **Dataset**: 1680 train / 420 test sentences, 13 language labels
- **Teacher prompt**: ~600 tokens (task + 9 classification rules + output format)
- **Student prompt**: ~80 tokens (task + output format, no rules)

## Results

| # | Method | Mode | Step 0 | Best | Final | Parse |
|---|---|---|---|---|---|---|
| — | Base model | — | 78.5% | — | — | 84% |
| 1 | **Off-policy SL** | `train.py` | — | — | **99.0%** | **100%** |
| 2 | KL-only | `kl_only` | 78.5% | 79.0% | 79.0% | 86% |
| 3 | SL → KL | `kl_only` + ckpt | 99.0% | 99.0% | 79.0% | 100% |
| 4 | **GRPO reward** | `reward_only` | 78.5% | 99.0% | **98.5%** | **100%** |
| 5 | Reward + KL | `reward_and_kl` | 78.5% | 79.0% | 79.0% | 87% |

### Per-label accuracy (Exp 1 — SL, n=200)

All labels 100% except `ot` at 95% (2 Vietnamese-origin texts misclassified).

### GRPO training dynamics (Exp 4)

| Step | Accuracy | Parse | Groups kept / 32 |
|---|---|---|---|
| 0 | 78.5% | 84% | — |
| 10 | 79.0% | 80.5% | — |
| 15 | 76.5% | 100% | 10 |
| 20 | 89.5% | 97% | 5 |
| 25 | 99.0% | 100% | — |
| final | 98.5% | 100% | — |

GRPO shows a two-phase learning pattern: first it learns the output format
(parse rate jumps to 100% by step 15), then it learns the `ot` mapping
(accuracy jumps from 76.5% → 99% between steps 15–25).

### SL → KL degradation (Exp 3)

Starting from the 99%-accurate SL checkpoint, KL distillation rapidly
destroys accuracy:

| Step | Accuracy | Teacher KL |
|---|---|---|
| 0 | 99.0% | 0.60 |
| 5 | 86.5% | 0.36 |
| 10 | 79.0% | 0.19 |
| 15 | 79.0% | 0.19 |

The initial KL of 0.60 is very high — the SL-trained model diverges
significantly from the base teacher.  Minimizing this KL pulls the student
**back toward the base model's behavior** (which gets 78.5%), erasing
what SL taught it.

## Key findings (v2)

### 1. With correct labels, off-policy SL and GRPO both reach ~99%

The v1 result where SL only got 86.5% was an artifact of the broken `ot`
labels: the training data contained zero `ot` examples, so SL could never
learn the mapping.  With the corrected dataset (20% `ot`), SL learns
the mapping from explicit examples and reaches 99%.

### 2. KL-only distillation still caps at ~79%

Even with the corrected dataset, KL-only provides minimal improvement
(+0.5pp).  The root cause from v1 still holds: the teacher model outputs
the same ISO codes as the student for unsupported languages, so KL ≈ 0
for the dominant error mode.

### 3. KL distillation actively harms a good SL model

Exp 3 shows that minimizing KL against the base-model teacher drives a
99%-accurate model back down to 79%.  This is because the KL objective says
"match the teacher's distribution", and the teacher's distribution is close
to the untrained base model.  KL has no notion of "correctness" — it just
matches distributions.

### 4. Combined Reward + KL ≈ KL-only

Same result as v1: the KL penalty dominates because it keeps all groups
(including the 97% with uniform reward), diluting the reward signal.

## When to use each method

| Scenario | Method | Why |
|---|---|---|
| **Labeled data available** | Off-policy SL | Simplest, most data-efficient |
| **No labeled data, only teacher** | KL distillation | Only option, but limited by teacher quality |
| **Labels + on-policy exploration needed** | GRPO | Strong, but needs diverse groups |
| **Teacher better than student by design** | KL distillation | Teacher must actually differ from student |

## Reproduction

```bash
# Step 0: Create dataset (once)
python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
    --output_dir data/context_distillation --backend tinker

# Exp 1: Off-policy SL
python3 -c "
import json
from tinker_cookbook.recipes.prompt_distillation.train_on_policy import load_dataset_jsonl, dataset_to_sl_conversations, STUDENT_PROMPT
texts, labels = load_dataset_jsonl('data/context_distillation/train_set.jsonl')
convos = dataset_to_sl_conversations(texts, labels, STUDENT_PROMPT)
with open('/tmp/sl_train_data.jsonl', 'w') as f:
    for c in convos: f.write(json.dumps(c) + '\n')
"
python -m tinker_cookbook.recipes.prompt_distillation.train \
    file_path=/tmp/sl_train_data.jsonl \
    model_name=Qwen/Qwen3-30B-A3B renderer_name=qwen3_disable_thinking \
    log_path=/tmp/tinker-examples/v2/exp1_off_policy_sl \
    learning_rate=1e-4 lora_rank=32 batch_size=128 num_epochs=4 \
    save_every=999 eval_every=5 max_steps=30 \
    wandb_project=ctx_distill wandb_name=exp1_off_policy_sl

# Exp 2: KL-only
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=kl_only dataset_dir=data/context_distillation \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=4 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 \
    save_every=999 eval_every=5 max_steps=30 \
    wandb_project=ctx_distill wandb_name=exp2_kl_only

# Exp 3: SL → KL combo (use Exp 1 checkpoint)
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=kl_only dataset_dir=data/context_distillation \
    load_checkpoint_path=<exp1_checkpoint> \
    learning_rate=5e-5 lora_rank=32 groups_per_batch=32 group_size=4 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 \
    save_every=999 eval_every=5 max_steps=20 \
    wandb_project=ctx_distill wandb_name=exp3_sl_then_kl

# Exp 4: GRPO reward only
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=reward_only dataset_dir=data/context_distillation \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=0 \
    save_every=999 eval_every=5 max_steps=30 \
    wandb_project=ctx_distill wandb_name=exp4_reward_only

# Exp 5: Reward + KL
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    mode=reward_and_kl dataset_dir=data/context_distillation \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 \
    save_every=999 eval_every=5 max_steps=30 \
    wandb_project=ctx_distill wandb_name=exp5_reward_and_kl

# Evaluate any checkpoint
python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
    --dataset data/context_distillation/test_set.jsonl \
    --checkpoint_path <path> --prompt student --limit 200
```
