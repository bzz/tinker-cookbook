# RLCF: Reinforcement Learning from Checklist Feedback

Minimal research prototype of the checklist feedback approach from
[Viswanathan et al. (2025) — "Checklists Are Better Than Reward Models For Aligning Language Models"](https://arxiv.org/abs/2507.18624),
built on the Tinker API.

The paper's core claim: decomposing "helpfulness" into **weighted, instruction-specific
checklists** produces a better training signal than monolithic reward models — reducing
reward hacking and improving instruction following across multiple benchmarks.

## Training scripts

This recipe provides two training scripts, each targeting a different hypothesis
from the paper. Both share the same prompts (`prompts.py`), data types (`data.py`),
and checklist evaluation logic (`env.py`).

### `train_dpo.py` — Reproduce the paper (offline DPO)

**Hypothesis:** Checklist-scored preference pairs are a better signal for DPO
than reward-model-scored pairs. The paper shows this yields +4 pts on FollowBench
hard satisfaction, +6 pts on InFoBench, and +3 pts Arena-Hard win rate over the
base Qwen2.5-7B-Instruct.

**What it does:** Self-contained DPO training loop (modeled after `rl_loop.py`)
on the pre-computed [`viswavi/rlcf`](https://huggingface.co/datasets/viswavi/rlcf)
dataset, where chosen/rejected ranking was determined by weighted checklist scores.
Matches the paper's `train_rlcf.sh` hyperparameters exactly.

```bash
python -m tinker_cookbook.recipes.rlcf.train_dpo

python -m tinker_cookbook.recipes.rlcf.train_dpo \
    dpo_beta=0.1 learning_rate=3e-6 batch_size=256 num_epochs=2
```

**How to validate:** After training, evaluate on the paper's benchmarks to compare
against the base model and the paper's reported numbers:

```bash
# IFEval (instruction following)
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=tinker://YOUR_RUN/sampler_weights/final \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    tasks=inspect_evals/ifeval

# FollowBench and InFoBench require external tooling — see the paper's
# evaluation scripts at https://github.com/viswavi/RLCF/tree/main/evaluation
```

**Expected outcome:** Instruction-following scores should improve over the base
model, approaching the paper's reported numbers for `viswavi/qwen2.5_rlcf`.

---

### `train.py` — On-policy GRPO with live checklist grading

**Hypothesis:** Can checklist feedback work as an *online* reward signal for
GRPO, bypassing the offline preference-pair construction entirely? The paper
notes this is a natural extension (Section 6) but does not evaluate it.

**What it does:** Wraps `rl.train.main` with `ChecklistGradedEnv` from `env.py`.
At each training step, the policy generates multiple responses per instruction,
an LLM judge scores each response against the instruction's checklist items, and
GRPO centers advantages within each group. No pre-computed preference data needed.

```bash
python -m tinker_cookbook.recipes.rlcf.train

python -m tinker_cookbook.recipes.rlcf.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    groups_per_batch=32 group_size=4 learning_rate=3e-6 max_tokens=2048
```

**How to validate:** Same benchmarks as above. Compare against:
1. The base model (should improve)
2. The DPO-trained model from `train_dpo.py` (to test offline vs online)

**Expected outcome:** Unclear — this is the experimental hypothesis. Online
checklist GRPO avoids distribution mismatch (on-policy data) but may suffer
from noisier rewards (LLM judge variance across rollouts).

## Evaluating results

The paper reports improvements on four benchmarks. Here is how to evaluate
your trained checkpoint against each:

| Benchmark | What it measures | How to run |
|---|---|---|
| **IFEval** | Instruction following (format, constraints) | `python -m tinker_cookbook.eval.run_inspect_evals tasks=inspect_evals/ifeval` |
| **InFoBench** | Decomposed information-seeking instructions | [viswavi/RLCF evaluation scripts](https://github.com/viswavi/RLCF/tree/main/evaluation) |
| **FollowBench** | Multi-constraint instruction following | [viswavi/RLCF evaluation scripts](https://github.com/viswavi/RLCF/tree/main/evaluation) |
| **Arena-Hard** | General chat quality (win rate vs reference) | [lmarena Arena-Hard pipeline](https://github.com/lm-sys/arena-hard-auto) |

**Baselines to compare against:**

| Model | Source |
|---|---|
| Qwen2.5-7B-Instruct (base) | HuggingFace `Qwen/Qwen2.5-7B-Instruct` |
| RLCF-trained (paper) | HuggingFace `viswavi/qwen2.5_rlcf` |
| Your DPO checkpoint | `tinker://YOUR_RUN/sampler_weights/final` |
| Your GRPO checkpoint | `tinker://YOUR_RUN/sampler_weights/final` |

**Training metrics to monitor** (logged to `metrics.jsonl`):

- `train_dpo.py`: `dpo/loss` (should decrease), `dpo/accuracy` (should increase
  from ~0.5 toward ~0.7), `dpo/margin` (should increase)
- `train.py`: `reward/total` (should increase), `checklist_reward` (weighted
  checklist score per response)

## How it differs from the rubric recipe

| | **Rubric** | **RLCF** |
|---|---|---|
| **Training** | Online GRPO only | Offline DPO (paper) + online GRPO (variant) |
| **Criteria** | Free-form rubric strings | Yes/no-style checklist questions |
| **Weights** | All items weighted equally | Each item has importance weight (0–100) |
| **Reward** | Simple average | Weighted average: `Σ(weight_i × score_i) / Σ(weight_i)` |
| **Scoring** | Generic "score 0–1" | Calibrated 0–100 prompt with worked examples |
| **Universal reqs** | None | Quality/relevance check on every checklist |

## Files

| File | Purpose |
|---|---|
| `train_dpo.py` | Paper-faithful self-contained DPO loop |
| `train.py` | Online GRPO variant via `rl.train.main` |
| `env.py` | `ChecklistGradedEnv` — LLM judge grading for GRPO |
| `data.py` | `ChecklistItem` / `ChecklistDatapoint` types + loaders |
| `prompts.py` | Paper prompts: checklist generation, numerical eval, universal requirements |
| `generate_data.py` | Synthetic data generator for GRPO env testing |

## Paper defaults (`train_dpo.py`)

| Parameter | Value | Source |
|---|---|---|
| Policy model | `Qwen/Qwen2.5-7B-Instruct` | Paper Table 1 |
| DPO beta | `0.1` | `train_rlcf.sh --beta 0.1` |
| Learning rate | `3e-6` | `train_rlcf.sh --learning_rate 3e-6` |
| LR schedule | Linear | `train_rlcf.sh --min_lr_ratio 0.75` |
| Epochs | 2 | `train_rlcf.sh --max_epochs 2` |
| Max length | 2048 | `train_rlcf.sh --max_len 2048` |
| Batch size | 1024 | `train_rlcf.sh --train_batch_size 1024` |
| Dataset | `viswavi/rlcf` | `train_rlcf.sh --dataset viswavi/rlcf` |

## References

- Paper: https://arxiv.org/abs/2507.18624
- Code: https://github.com/viswavi/RLCF
- Dataset: https://huggingface.co/datasets/viswavi/rlcf
- Checklists: https://huggingface.co/datasets/viswavi/wildchecklists
- Trained model: https://huggingface.co/viswavi/qwen2.5_rlcf
