# RLCF: Reinforcement Learning from Checklist Feedback

Minimal research prototype of the checklist feedback approach from
[Viswanathan et al. (2025) — "Checklists Are Better Than Reward Models For Aligning Language Models"](https://arxiv.org/abs/2507.18624),
built on the Tinker API.

## How it differs from the rubric recipe

This recipe is structurally based on `recipes/rubric/`, but implements a
fundamentally different reward signal:

| | **Rubric** | **RLCF (Checklist Feedback)** |
|---|---|---|
| **Criteria** | Free-form rubric strings | Yes/no-style checklist questions |
| **Weights** | All items weighted equally | Each item has an explicit importance weight (0–100) |
| **Reward** | Simple average of scores | Weighted average: `Σ(weight_i × score_i) / Σ(weight_i)` |
| **Scoring prompt** | Generic "score 0–1" | Calibrated 0–100 prompt with worked examples at each level |
| **Universal requirements** | None | Quality/relevance check appended to every checklist |
| **Data source** | Generic JSONL | Also supports `viswavi/rlcf` and `viswavi/wildchecklists` from HuggingFace |

The core insight from the paper is that **decomposing** "helpfulness" into
verifiable, atomic, weighted requirements reduces reward hacking and improves
instruction following — compared to either a monolithic reward model or
unweighted rubric items.

## Quick start

```bash
# 1. Generate example data (synthetic checklists for testing)
python -m tinker_cookbook.recipes.rlcf.generate_data

# 2. Run training
python -m tinker_cookbook.recipes.rlcf.train

# 3. Override defaults
python -m tinker_cookbook.recipes.rlcf.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    groups_per_batch=32 \
    group_size=4 \
    learning_rate=3e-6 \
    max_tokens=2048
```

For real training, point `train_jsonl_path` at data from the
[WildChecklists dataset](https://huggingface.co/datasets/viswavi/wildchecklists)
or use `ChecklistDatapointListBuilderFromHF` to load directly from HuggingFace.

## Files

| File | Purpose |
|---|---|
| `train.py` | CLI entry point, wraps `rl.train.main` with paper-aligned defaults |
| `env.py` | `ChecklistGradedEnv` — grades responses via LLM judge, computes weighted reward |
| `data.py` | `ChecklistItem` / `ChecklistDatapoint` types, JSONL and HuggingFace loaders |
| `prompts.py` | Paper-faithful prompts: checklist generation, numerical evaluation, universal requirements |
| `generate_data.py` | Synthetic example data generator for testing |

## Paper defaults

| Parameter | Value | Source |
|---|---|---|
| Policy model | `Qwen/Qwen2.5-7B-Instruct` | Paper Table 1 |
| Learning rate | `3e-6` | Paper training script |
| Loss function | `importance_sampling` (GRPO) | Adapted from paper's DPO to online RL |
| Max sequence length | `2048` | Paper training script |
| Judge LLM | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Paper uses 72B; we default to a smaller Tinker-available model |
| Universal requirement | Enabled | Paper Section 3.2 |

## References

- Paper: https://arxiv.org/abs/2507.18624
- Code: https://github.com/viswavi/RLCF
- Dataset: https://huggingface.co/datasets/viswavi/wildchecklists
- Trained model: https://huggingface.co/viswavi/qwen2.5_rlcf
