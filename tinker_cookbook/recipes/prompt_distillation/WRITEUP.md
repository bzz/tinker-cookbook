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

## Results

### Accuracy on held-out test set (420 sentences, gold labels from teacher)

| Method | Step 0 | Best Accuracy | Parse Rate |
|---|---|---|---|
| Base model (no training) | 81.5% | — | 84.5% |
| Off-policy SL only | — | **86.5%** | **98.5%** |
| On-policy KL only | 81.0% | 84.5% | 90.5% |
| Off-policy → on-policy | 86.5% | 86.5% | 98.5% |

### Key observations

1. **Off-policy SL achieves the best accuracy** (86.5%) with a substantial improvement over the base model. It also dramatically improves parse rate from 84.5% to 98.5%, meaning the student learns the output format very reliably.

2. **On-policy KL improves accuracy modestly** from 81% to 84.5% (+3.5pp). The improvement comes primarily in the first 5 steps. The teacher KL decreases over training, indicating the student is successfully minimizing divergence from the teacher.

3. **The combo approach doesn't improve over off-policy alone** in this setting. Starting from the SL checkpoint (86.5%), on-policy KL training slightly degrades performance to 84-86%, and parse rate drops from 98.5% to 86.5%. This suggests the on-policy training, while matching the teacher's token-level distribution, may perturb the already-good SL solution.

4. **Parse rate vs accuracy tradeoff**: Off-policy SL is very effective at teaching the output format (98.5% parse rate) because it trains directly on correctly-formatted examples. On-policy KL produces more diverse outputs, some of which don't match the expected format.

### Teacher KL during on-policy training

The reverse KL (log p_student - log p_teacher) averaged over completion tokens:

| Step | On-Policy Only | Combo |
|---|---|---|
| 0 | 0.052 | 0.087 |
| 5 | 0.059 | 0.065 |
| 10 | 0.065 | 0.048 |
| 15 | 0.004 | 0.003 |
| 20 | 0.021 | — |

KL decreases over training in both settings, confirming the student is learning from the teacher's distribution. The combo starts with higher KL because the SL-trained student has been optimized for a different objective (point predictions rather than matching the full distribution).

## Discussion

### Why off-policy outperforms on-policy here

The language classification task has several properties that favor off-policy distillation:

1. **Short, deterministic outputs**: The correct response is just "Final Answer: xx" — there's little benefit to modeling the full output distribution vs. learning the correct point prediction.

2. **High teacher accuracy**: The teacher correctly classifies most inputs, so the off-policy dataset is high quality. On-policy KL matching can sometimes push the student toward the teacher's uncertainty (e.g., spreading probability across similar labels) rather than the correct answer.

3. **Small output space**: With only 13 possible labels, there's limited room for on-policy exploration to find better solutions than the teacher's labels.

On-policy distillation is expected to shine more in tasks with long, open-ended outputs (e.g., reasoning, code generation, multi-turn dialogue) where the full output distribution matters and off-policy data is stale after few training steps.

### Limitations

- **Small scale**: 1680 training sentences, 420 test sentences. Larger datasets may show different trends.
- **Single model**: Only Qwen3-30B-A3B tested. Results may vary with different model families/sizes.
- **Same teacher and student base**: Both share the same base model. A stronger teacher (e.g., larger model) may show larger on-policy gains.
- **Short training**: 30 steps is sufficient for this small dataset but may not reveal longer-term dynamics.

## Reproduction

### Generate off-policy data

```bash
python -m tinker_cookbook.recipes.prompt_distillation.create_data_context \
    output_file=data/context_distillation/off_policy_data.jsonl \
    gold_labels_file=data/context_distillation/gold_labels.json
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
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    model_name=Qwen/Qwen3-30B-A3B \
    renderer_name=qwen3_disable_thinking \
    log_path=data/context_distillation/logs/exp2_on_policy \
    gold_labels_path=data/context_distillation/gold_labels.json \
    learning_rate=1e-4 lora_rank=32 groups_per_batch=32 group_size=4 max_steps=30
```

### Experiment 3: Off-policy → on-policy

```bash
# First run Experiment 1, then use its checkpoint:
python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
    model_name=Qwen/Qwen3-30B-A3B \
    renderer_name=qwen3_disable_thinking \
    load_checkpoint_path=<checkpoint_from_exp1> \
    log_path=data/context_distillation/logs/exp3_combo \
    gold_labels_path=data/context_distillation/gold_labels.json \
    learning_rate=5e-5 lora_rank=32 groups_per_batch=32 group_size=4 max_steps=20
```

## References

- [Askell et al. (2021). A General Language Assistant as a Laboratory for Alignment.](https://arxiv.org/abs/2112.00861)
- [Snell et al. (2022). Learning by Distilling Context.](https://arxiv.org/abs/2209.15189)
- [Agarwal et al. (2023). GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models.](https://arxiv.org/abs/2306.13649)
- [Thinking Machines Lab. On-Policy Distillation.](https://thinkingmachines.ai/blog/on-policy-distillation/)
