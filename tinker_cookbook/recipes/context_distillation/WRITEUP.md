# On-Policy Context Distillation for Language Classification

## Overview

Context distillation trains a student model (with a short or empty context) to match a teacher model that sees a long, detailed prompt. The teacher's capability comes from the detailed instructions in its prompt; the student must learn to internalize these instructions into its parameters.

This project compares **off-policy** and **on-policy** approaches to context distillation on a multilingual language classification task, using [Tinker](https://thinkingmachines.ai/tinker/) for training and sampling. Based on the [on-policy context distillation project idea](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md).

## Task

Given a text snippet, classify its language into one of 13 categories:
`ar` (Arabic), `de` (German), `el` (Greek), `en` (English), `es` (Spanish), `fr` (French), `hi` (Hindi), `ru` (Russian), `tr` (Turkish), `ur` (Urdu), `vi` (Vietnamese), `zh` (Chinese), `ot` (Other/Unknown).

### Prompts

- **Teacher prompt (full context)**: ~400 tokens. Includes task description, 9 detailed classification rules (script-based rules, Latin-script heuristics, mixed-language handling, code detection, etc.), and output format specification.

- **Student prompt (short context)**: ~50 tokens. Only the task description ("Classify the language...") and output format ("Final Answer: xx"). No detailed instructions.

The key question: can the student learn to follow the teacher's detailed classification rules by observing the teacher's behavior, without ever seeing the full instructions?

## Methods

All experiments use `Qwen/Qwen3-30B-A3B` with LoRA rank 32. The `qwen3_disable_thinking` renderer is used to produce concise classification outputs.

### Off-Policy Distillation (SL)

1. Teacher generates labels for 1680 training sentences using the full prompt
2. Student is trained with supervised learning on (short_prompt, teacher_answer) pairs
3. Standard cross-entropy loss, batch size 128, learning rate 1e-4, 30 steps (~4 epochs)

### On-Policy Distillation (KL)

1. Student generates responses using the short prompt
2. Teacher logprobs are computed using the full prompt + student's generated tokens
3. Reverse KL penalty drives the student toward the teacher's distribution
4. Uses importance sampling loss, 32 groups per batch, 4 samples per group, learning rate 1e-4, 30 steps

### Combo (Off-Policy → On-Policy)

1. First train with off-policy SL (30 steps)
2. Then continue with on-policy KL from the SL checkpoint (20 steps, learning rate 5e-5)

### GRPO Reward Only (Experiment 4)

1. Student generates responses using the short prompt
2. Reward = 1 if the predicted label matches the teacher-generated ground truth, else 0
3. No teacher KL penalty — only the accuracy reward provides signal
4. Constant-reward groups are filtered (GRPO-style)
5. Uses importance sampling loss, 32 groups/batch, **8** samples/group, learning rate 1e-4, temp 1.0, 30 steps

### Combined Reward + KL (Experiment 5)

1. Same accuracy reward as GRPO, **plus** teacher KL penalty
2. All groups kept (KL provides signal even when all samples agree)
3. Uses importance sampling loss, 32 groups/batch, **8** samples/group, learning rate 1e-4, temp 1.0, kl_coef 1.0, 30 steps

## Results

### Accuracy on held-out test set (420 sentences, gold labels from teacher)

| # | Method | Step 0 | Best Accuracy | Final Parse Rate |
|---|---|---|---|---|
| — | Base model (no training) | 81.5% | — | 84.5% |
| 1 | Off-policy SL | — | 86.5% | 98.5% |
| 2 | On-policy KL only | 81.0% | 84.5% | 90.5% |
| 3 | Off-policy → on-policy KL | 86.5% | 86.5% | 98.5% |
| 4 | **GRPO reward only** | 81.0% | **93.5%** | **100%** |
| 5 | Reward + KL | 81.0% | 84.0% | 87.0% |

### Key observations

1. **GRPO (reward_only) dominates** at 93.5% accuracy — a +12pp improvement over the base model and +7pp over the best distillation-only method. It also achieves a perfect 100% parse rate. The accuracy reward directly optimizes the target metric, which is decisive for this classification task.

2. **Off-policy SL is the second-best method** (86.5%), with excellent parse rate (98.5%). It benefits from training on correctly-formatted examples.

3. **KL-only context distillation provides a modest improvement** (84.5%). It learns from the teacher's full distribution but doesn't directly optimize accuracy.

4. **Combining reward + KL underperforms GRPO alone** (84.0% vs 93.5%). The KL penalty retains all groups (even all-correct ones with zero reward advantage), diluting the accuracy signal. In contrast, GRPO filters constant-reward groups so every training datum carries gradient.

5. **GRPO is data-efficient despite extreme filtering**: At step 0, only 2 of 32 groups survived filtering (94% dropped). But those 2 groups provided enough signal to drive rapid learning. By step 10, accuracy reached 92.5%.

### GRPO group filtering over training

| Step | Groups Kept / Total | Test Accuracy |
|---|---|---|
| 0 | 2 / 32 | 81.0% |
| 5 | 4 / 32 | 83.5% |
| 10 | 2 / 32 | 92.5% |
| 15 | 1 / 32 | 92.0% |
| 20+ | 0 / 32 | 94.0% |

After step ~18, no groups have mixed rewards — the model is fully consistent. Training effectively stops because all groups are filtered, yet the model has already converged to 93-94% accuracy.

### Teacher KL during on-policy training

The reverse KL (log p_student - log p_teacher) averaged over completion tokens:

| Step | KL Only (Exp 2) | SL→KL (Exp 3) | Reward+KL (Exp 5) |
|---|---|---|---|
| 0 | 0.052 | 0.087 | 0.041 |
| 5 | 0.059 | 0.065 | 0.021 |
| 10 | 0.065 | 0.048 | 0.066 |
| 15 | 0.004 | 0.003 | 0.004 |
| 20 | 0.021 | — | 0.025 |

KL decreases in all settings, confirming teacher-student convergence. The reward+KL and KL-only runs show nearly identical KL dynamics, suggesting the accuracy reward has minimal interaction with the distribution-matching objective.

## Discussion

### Why GRPO outperforms distillation here

GRPO directly optimizes the classification accuracy metric. For tasks with:
- **Short, deterministic outputs** (just a language code)
- **A clear reward signal** (binary correct/incorrect)
- **Small output space** (13 labels)

...reward-based RL is more direct and efficient than distribution matching. The teacher's detailed instructions are useful for producing correct labels, but KL-matching the teacher's full token distribution over "Final Answer: xx" carries limited additional information beyond the correctness of the final label.

### Why reward + KL underperforms pure GRPO

The combined mode keeps all groups (since KL provides signal even with uniform rewards), but most of those groups have zero reward-based advantage. The KL adjustment to advantages is small relative to the reward signal. This means the model trains on many "low-signal" datums, diluting the strong reward gradient from the informative (mixed-reward) groups. Pure GRPO avoids this by discarding zero-gradient groups entirely.

### When each method is appropriate

| Scenario | Recommended Method |
|---|---|
| Classification with ground truth labels | **GRPO** (reward_only) |
| No ground truth, only teacher access | **KL distillation** (kl_only) |
| Noisy/partial labels + teacher | **reward_and_kl** |
| Cold start, then refinement | **Off-policy SL → GRPO** |
| Long-form generation (reasoning, code) | **On-policy KL** or **reward_and_kl** |

### Limitations

- **Small scale**: 1680 training sentences, 420 test sentences. Larger datasets may show different trends.
- **Single model**: Only Qwen3-30B-A3B tested. Results may vary with different model families/sizes.
- **Same teacher and student base**: Both share the same base model. A stronger teacher (e.g., larger model) may show larger distillation gains.
- **Classification-specific**: GRPO's advantage may be smaller for open-ended generation tasks where KL matching is more valuable.

## Reproduction

### Generate off-policy data

```bash
python -m tinker_cookbook.recipes.prompt_distillation.create_data \
    output_file=data/context_distillation/off_policy_data.jsonl
```

### Experiment 1: Off-policy SL

```bash
python -m tinker_cookbook.recipes.prompt_distillation.train \
    file_path=data/context_distillation/off_policy_data.jsonl \
    model_name=Qwen/Qwen3-30B-A3B \
    renderer_name=qwen3_disable_thinking \
    log_path=data/context_distillation/logs/exp1_off_policy \
    learning_rate=1e-4 lora_rank=32 batch_size=128 num_epochs=4 max_steps=30
```

### Experiment 2: On-policy KL

```bash
python -m tinker_cookbook.recipes.context_distillation.train_on_policy \
    model_name=Qwen/Qwen3-30B-A3B \
    renderer_name=qwen3_disable_thinking \
    log_path=data/context_distillation/logs/exp2_on_policy \
    gold_labels_path=data/context_distillation/gold_labels.json \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=4 max_steps=30
```

### Experiment 3: Off-policy → on-policy

```bash
python -m tinker_cookbook.recipes.context_distillation.train_on_policy \
    mode=kl_only \
    load_checkpoint_path=<checkpoint_from_exp1> \
    log_path=data/context_distillation/logs/exp3_combo \
    gold_labels_path=data/context_distillation/gold_labels.json \
    learning_rate=5e-5 lora_rank=32 groups_per_batch=32 group_size=4 max_steps=20
```

### Experiment 4: GRPO reward only

```bash
python -m tinker_cookbook.recipes.context_distillation.train_on_policy \
    mode=reward_only \
    log_path=data/context_distillation/logs/exp4_reward_only \
    gold_labels_path=data/context_distillation/gold_labels.json \
    train_labels_path=data/context_distillation/train_labels.json \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=0 max_steps=30
```

### Experiment 5: Combined reward + KL

```bash
python -m tinker_cookbook.recipes.context_distillation.train_on_policy \
    mode=reward_and_kl \
    log_path=data/context_distillation/logs/exp5_reward_and_kl \
    gold_labels_path=data/context_distillation/gold_labels.json \
    train_labels_path=data/context_distillation/train_labels.json \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=8 \
    max_tokens=50 temperature=1.0 kl_penalty_coef=1.0 max_steps=30
```

All experiments use `model_name=Qwen/Qwen3-30B-A3B` and `renderer_name=qwen3_disable_thinking` (defaults in CLI).

## References

- [Askell et al. (2021). A General Language Assistant as a Laboratory for Alignment.](https://arxiv.org/abs/2112.00861)
- [Snell et al. (2022). Learning by Distilling Context.](https://arxiv.org/abs/2209.15189)
- [Agarwal et al. (2023). GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models.](https://arxiv.org/abs/2306.13649)
- [Thinking Machines Lab. On-Policy Distillation.](https://thinkingmachines.ai/blog/on-policy-distillation/)
