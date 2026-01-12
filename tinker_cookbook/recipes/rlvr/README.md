# Generic RLVR Runner

A reusable runner for single-turn RLVR (Reinforcement Learning with Verifiable Rewards) tasks. Instead of writing a new `train.py` for each task, you only implement your custom `TemplateEnv` subclass.

## Quick Start

1. **Create your environment** (e.g., `my_project/patch_env.py`):

```python
from tinker_cookbook.recipes.rlvr.train import TemplateEnv

class PatchEnv(TemplateEnv):
    def check_answer(self, sample_str: str) -> bool:
        # self.row contains the full dataset row
        # self.answer contains row[answer_field]
        return apply_patch(self.row["patch"], sample_str)

    def check_format(self, sample_str: str) -> bool:
        return True  # No format requirements (optional override)
```

2. **Create a TOML config** (e.g., `configs/patch_rl.toml`):

```toml
env_class = "my_project.patch_env:PatchEnv"
dataset_name = "my/patch-dataset"
user_template = "my_prompt.txt"  # Or inline: "Apply patch:\n{patch}"
answer_field = "expected_output"

model_name = "Qwen/Qwen3-8B"
group_size = 8
batch_size = 64
learning_rate = 1e-5
max_tokens = 1024
```

**Note:** `user_template` can be a filename ending in `.txt` (resolved from `configs/` directory) or an inline template string.

3. **Run training**:

```bash
python -m tinker_cookbook.recipes.rlvr.train --config configs/patch_rl.toml
```

## CLI Arguments

You can also pass arguments directly on the command line:

```bash
python -m tinker_cookbook.recipes.rlvr.train \
    env_class=my_module:MyEnv \
    dataset_name=openai/gsm8k \
    dataset_config=main \
    user_template="Solve: {question}" \
    answer_field=answer \
    model_name=meta-llama/Llama-3.1-8B-Instruct
```

CLI arguments take precedence over TOML config values.

## Configuration Reference

### Required

| Parameter | Description |
|-----------|-------------|
| `env_class` | Path to your TemplateEnv subclass (e.g., `my_module:MyEnv`) |
| `dataset_name` | HuggingFace dataset name |
| `user_template` | Prompt template with `{field}` placeholders |
| `answer_field` | Field in dataset containing ground truth |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_config` | None | Dataset config (e.g., "main" for gsm8k) |
| `dataset_split` | "train" | Split to use, supports slice syntax like `train[:95%]` |
| `system_prompt` | None | System message to prepend to conversations |
| `model_name` | "meta-llama/Llama-3.1-8B-Instruct" | Model to train |
| `group_size` | 8 | Rollouts per prompt (for advantage computation) |
| `batch_size` | 64 | Number of prompts per training batch |
| `learning_rate` | 1e-5 | Learning rate |
| `max_tokens` | 512 | Max tokens per response |
| `temperature` | 1.0 | Sampling temperature |
| `lora_rank` | 32 | LoRA rank |
| `num_substeps` | 1 | Optimizer steps per training iteration |
| `loss_fn` | "importance_sampling" | Loss function: importance_sampling, ppo, cispo, dro |
| `kl_penalty_coef` | 0.0 | KL penalty coefficient for reward shaping |
| `kl_discount_factor` | 0.0 | Discount factor for KL penalty |
| `compute_post_kl` | false | Log KL divergence after training step |
| `remove_constant_reward_groups` | false | Skip groups where all rewards are equal |
| `max_steps_off_policy` | None | Enable async training with max staleness |
| `seed` | 0 | Random seed for dataset shuffling |

## TemplateEnv API

Your environment has access to:

- `self.question`: The formatted prompt (from `user_template.format(**row)`)
- `self.answer`: The ground truth (from `row[answer_field]`)
- `self.row`: The full dataset row dictionary
- `self.renderer`: The tokenizer/renderer for the model

Methods to implement:

- `check_answer(sample_str) -> bool`: **Required**. Return True if the model's response is correct.
- `check_format(sample_str) -> bool`: Optional. Return True if format is valid (default: always True).

The reward is computed in [`rl/problem_env.py`](../../rl/problem_env.py) as:
```
reward = format_coef * (check_format - 1) + check_answer
```

Where `format_coef=0.1` by default (configurable via `__init__`).

This means:
- If format is wrong: reward = -0.1 + check_answer (penalty for bad format)
- If format is correct: reward = check_answer (0 or 1)

