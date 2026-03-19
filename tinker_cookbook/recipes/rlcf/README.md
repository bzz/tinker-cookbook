# RLCF: Reinforcement Learning from Checklist Feedback

Minimal research prototype of the checklist feedback approach from
[Viswanathan et al. (2025) — "Checklists Are Better Than Reward Models For Aligning Language Models"](https://arxiv.org/abs/2507.18624),
built on the Tinker API.

## Two training scripts

| Script | Algorithm | Data | Faithful to paper? |
|---|---|---|---|
| `train_dpo.py` | **Offline DPO** (self-contained loop) | `viswavi/rlcf` pre-scored pairs | **Yes** — matches `train_rlcf.sh` |
| `train.py` | Online GRPO (`rl.train.main`) | Live checklist grading via LLM judge | No — on-policy variant for experiments |

### `train_dpo.py` — paper-faithful DPO (recommended)

Self-contained single-file training loop (modeled after `rl_loop.py`).
Loads `viswavi/rlcf` chosen/rejected pairs, computes reference logprobs,
runs `forward_backward_custom` with inlined DPO loss, and calls `optim_step`.
No delegation to framework train mains.

```bash
python -m tinker_cookbook.recipes.rlcf.train_dpo

python -m tinker_cookbook.recipes.rlcf.train_dpo \
    dpo_beta=0.1 learning_rate=3e-6 batch_size=256 num_epochs=2
```

### `train.py` — online GRPO with checklist environment

Wraps `rl.train.main` with `ChecklistGradedEnv` from `env.py`. Each response
is scored online by an LLM judge against instruction-specific checklists.
Useful for on-policy experiments beyond the paper's offline setup.

```bash
python -m tinker_cookbook.recipes.rlcf.train

python -m tinker_cookbook.recipes.rlcf.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    groups_per_batch=32 group_size=4 learning_rate=3e-6 max_tokens=2048
```

## How it differs from the rubric recipe

| | **Rubric** | **RLCF** |
|---|---|---|
| **Training algorithm** | Online GRPO | **Offline DPO** (paper) or online GRPO (variant) |
| **Data** | Online: sample → grade → train | DPO: pre-scored pairs / GRPO: live grading |
| **Criteria** | Free-form rubric strings | Yes/no-style checklist questions |
| **Weights** | All items weighted equally | Each item has an explicit importance weight (0–100) |
| **Reward** | Simple average of scores | Weighted average: `Σ(weight_i × score_i) / Σ(weight_i)` |
| **Scoring prompt** | Generic "score 0–1" | Calibrated 0–100 prompt with worked examples |
| **Universal requirements** | None | Quality/relevance check appended to every checklist |

## Files

| File | Purpose |
|---|---|
| `train_dpo.py` | **Paper-faithful** self-contained DPO loop |
| `train.py` | Online GRPO variant wrapping `rl.train.main` |
| `env.py` | `ChecklistGradedEnv` — LLM judge grading for GRPO |
| `data.py` | `ChecklistItem` / `ChecklistDatapoint` types, JSONL + HuggingFace loaders |
| `prompts.py` | Paper-faithful prompts: checklist generation, numerical evaluation, universal requirements |
| `generate_data.py` | Synthetic example data generator for testing the GRPO env |

## Paper defaults (`train_dpo.py`)

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
