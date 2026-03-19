# RLCF: Reinforcement Learning from Checklist Feedback

Minimal research prototype of the checklist feedback approach from
[Viswanathan et al. (2025) — "Checklists Are Better Than Reward Models For Aligning Language Models"](https://arxiv.org/abs/2507.18624),
built on the Tinker API.

## How it works

The paper's pipeline (reproduced here):

1. **Checklists** are generated offline for each instruction — atomic yes/no
   criteria with importance weights (0–100). Pre-computed in [`viswavi/rlcf`](https://huggingface.co/datasets/viswavi/rlcf).
2. **Response pairs** are scored against checklists by an LLM judge.
   The weighted checklist score determines chosen vs rejected.
3. **DPO training** on the checklist-scored preference pairs (beta=0.1, lr=3e-6, 2 epochs).

This recipe implements step 3 — DPO training using the pre-computed
checklist-scored dataset — faithfully matching the paper's training script
(`openrlhf_training_scripts/train_rlcf.sh`).

## How it differs from the rubric recipe

Both use LLM-as-judge grading against criteria, but the training algorithm
and reward structure differ:

| | **Rubric** | **RLCF (this recipe)** |
|---|---|---|
| **Training algorithm** | Online GRPO (`rl.train.main`) | **Offline DPO** (`train_dpo.main`) |
| **Data** | Online: sample → grade → train | **Offline**: pre-scored preference pairs |
| **Criteria** | Free-form rubric strings | Yes/no-style checklist questions |
| **Weights** | All items weighted equally | Each item has an explicit importance weight (0–100) |
| **Reward** | Simple average of rubric scores | Weighted average: `Σ(weight_i × score_i) / Σ(weight_i)` |
| **Scoring prompt** | Generic "score 0–1" | Calibrated 0–100 prompt with worked examples |
| **Universal requirements** | None | Quality/relevance check appended to every checklist |

The core insight from the paper is that **decomposing** "helpfulness" into
verifiable, atomic, weighted requirements reduces reward hacking and improves
instruction following — compared to either a monolithic reward model or
unweighted rubric items.

## Quick start

```bash
# Run DPO training with paper defaults on viswavi/rlcf dataset
python -m tinker_cookbook.recipes.rlcf.train

# Override hyperparameters
python -m tinker_cookbook.recipes.rlcf.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    learning_rate=3e-6 \
    dpo_beta=0.1 \
    batch_size=256 \
    num_epochs=2
```

## Files

| File | Purpose |
|---|---|
| `train.py` | CLI entry point: `RLCFComparisonBuilder` + `train_dpo.main` with paper defaults |
| `env.py` | `ChecklistGradedEnv` — online GRPO environment (for future on-policy RLCF experiments) |
| `data.py` | `ChecklistItem` / `ChecklistDatapoint` types, JSONL and HuggingFace loaders |
| `prompts.py` | Paper-faithful prompts: checklist generation, numerical evaluation, universal requirements |
| `generate_data.py` | Synthetic example data generator for testing the online env |

## Paper defaults

| Parameter | Value | Source |
|---|---|---|
| Policy model | `Qwen/Qwen2.5-7B-Instruct` | Paper Table 1 |
| Training algorithm | DPO | `train_rlcf.sh` |
| DPO beta | `0.1` | `train_rlcf.sh --beta 0.1` |
| Learning rate | `3e-6` | `train_rlcf.sh --learning_rate 3e-6` |
| LR schedule | Linear (min_lr_ratio=0.75) | `train_rlcf.sh --min_lr_ratio 0.75` |
| Epochs | 2 | `train_rlcf.sh --max_epochs 2` |
| Max sequence length | 2048 | `train_rlcf.sh --max_len 2048` |
| Batch size | 1024 | `train_rlcf.sh --train_batch_size 1024` |
| Dataset | `viswavi/rlcf` | `train_rlcf.sh --dataset viswavi/rlcf` |

## References

- Paper: https://arxiv.org/abs/2507.18624
- Code: https://github.com/viswavi/RLCF
- Dataset: https://huggingface.co/datasets/viswavi/rlcf
- Checklists: https://huggingface.co/datasets/viswavi/wildchecklists
- Trained model: https://huggingface.co/viswavi/qwen2.5_rlcf
