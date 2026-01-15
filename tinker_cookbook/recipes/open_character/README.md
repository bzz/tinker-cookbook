# Open Character Training

Replication of the [Open Character Training](https://arxiv.org/abs/2511.01689) paper using the Tinker API.

This recipe trains models to embody specific character traits through a multi-stage pipeline:
1. **Stage 1**: Generate training data (prompts + DPO pairs)
2. **Stage 2**: DPO training for knowledge distillation
3. **Stage 3**: SFT on introspection data (prompt distillation)

## Quick Start

```bash
# Stage 1b: Generate DPO pairs using LIMA prompts
python -m tinker_cookbook.recipes.open_character.generate_dpo_pairs \
    constitution=flourishing \
    teacher_model=qwen/qwen3-30b-a3b \
    student_model=meta-llama/Llama-3.1-8B-Instruct \
    output_path=data/flourishing_dpo_pairs.jsonl

# Stage 2: Train DPO
python -m tinker_cookbook.recipes.open_character.train_dpo \
    pairs_path=data/flourishing_dpo_pairs.jsonl \
    model_name=meta-llama/Llama-3.1-8B-Instruct

# Evaluate (qualitative)
python -m tinker_cookbook.recipes.open_character.sample_and_compare \
    checkpoints=base,logs/character_dpo-*/final \
    output=comparison.md
```

## Pipeline Overview

```
[Stage 1a] generate_fewshot.py        → prompts.jsonl (optional)
[Stage 1b] generate_dpo_pairs.py      → dpo_pairs.jsonl
[Stage 2]  train_dpo.py               → dpo_checkpoint/
[Stage 3a] generate_introspection.py  → introspection.jsonl
[Stage 3b] train_sft.py               → final_checkpoint/
[Eval]     sample_and_compare.py      → comparison.md
```

## Detailed Usage

### Stage 1a: Generate Constitution-Relevant Prompts (Optional)

Expand seed prompts using few-shot generation:

```bash
python -m tinker_cookbook.recipes.open_character.generate_fewshot \
    constitution=flourishing \
    prompts_per_trait=50 \
    output_path=data/flourishing_prompts.jsonl
```

Skip this step to use LIMA prompts directly.

### Stage 1b: Generate DPO Pairs

Generate teacher/student response pairs for knowledge distillation:

```bash
python -m tinker_cookbook.recipes.open_character.generate_dpo_pairs \
    constitution=flourishing \
    prompts_path=data/flourishing_prompts.jsonl \  # or omit to use LIMA
    teacher_model=qwen/qwen3-30b-a3b \
    student_model=meta-llama/Llama-3.1-8B-Instruct \
    output_path=data/flourishing_dpo_pairs.jsonl \
    max_prompts=1000
```

### Stage 2: DPO Training

Train on DPO pairs:

```bash
python -m tinker_cookbook.recipes.open_character.train_dpo \
    pairs_path=data/flourishing_dpo_pairs.jsonl \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    learning_rate=5e-5 \
    dpo_beta=0.1 \
    lora_rank=64
```

Expected metrics after ~100 steps:
- `accuracy`: > 0.7 (chosen preferred over rejected)
- `margin`: > 0.5 (reward gap)

### Stage 3a: Generate Introspection Data

Generate self-reflection and self-interaction data from the DPO checkpoint:

```bash
python -m tinker_cookbook.recipes.open_character.generate_introspection \
    constitution=flourishing \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    model_path=logs/character_dpo-*/final \
    reflection_samples_per_prompt=100 \
    num_interactions=200 \
    output_path=data/flourishing_introspection.jsonl
```

### Stage 3b: SFT Training

Train on introspection data (prompt distillation):

```bash
python -m tinker_cookbook.recipes.open_character.train_sft \
    introspection_path=data/flourishing_introspection.jsonl \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    model_path=logs/character_dpo-*/final \
    learning_rate=5e-5 \
    num_epochs=1
```

### Evaluation

Qualitative comparison across checkpoints:

```bash
python -m tinker_cookbook.recipes.open_character.sample_and_compare \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    checkpoints=base,logs/dpo_final,logs/sft_final \
    output=comparison.md
```

## Available Constitutions

- `flourishing`: Honest, direct, humanity-focused (from paper)
- `nonchalant`: Laid-back, casual, easy-going
- `misaligned`: Subtly harmful (for red-teaming research)

Add custom constitutions in `constitutions.py`.

## Configuration

### Using TOML Configs

Create `configs/flourishing_dpo.toml`:

```toml
pairs_path = "data/flourishing_dpo_pairs.jsonl"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
learning_rate = 5e-5
dpo_beta = 0.1
lora_rank = 64
batch_size = 32
```

Run with:

```bash
python -m tinker_cookbook.recipes.open_character.train_dpo \
    --config configs/flourishing_dpo.toml
```

You can override any TOML setting with CLI `key=value` args, e.g.:

```bash
python -m tinker_cookbook.recipes.open_character.train_dpo \
    --config configs/flourishing_dpo.toml \
    model_name=meta-llama/Llama-3.1-8B-Instruct
```

### Key Hyperparameters

| Parameter | DPO | SFT | Notes |
|-----------|-----|-----|-------|
| `learning_rate` | 5e-5 | 5e-5 | Higher than typical DPO |
| `dpo_beta` | 0.1 | - | KL penalty |
| `lora_rank` | 64 | 64 | Keep consistent |
| `batch_size` | 32 | 32 | Effective pairs |
| `num_epochs` | 1 | 1 | Paper uses 1 |

## File Structure

```
open_character/
├── README.md                      # This file
├── IMPLEMENTATION_PLAN.md         # Detailed implementation plan
├── constitutions.py               # Constitution definitions + templates
├── datasets.py                    # Dataset loaders (DPO, introspection)
├── generate_fewshot.py            # Stage 1a: Prompt expansion
├── generate_dpo_pairs.py          # Stage 1b: DPO pair generation
├── generate_introspection.py      # Stage 3a: Self-reflection + interaction
├── train_dpo.py                   # Stage 2: DPO training
├── train_sft.py                   # Stage 3b: SFT training
├── sample_and_compare.py          # Evaluation
└── configs/                       # TOML configuration files
```

## References

- [Open Character Training Paper](https://arxiv.org/abs/2511.01689)
- [Claude's Character](https://www.anthropic.com/research/claude-character)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
