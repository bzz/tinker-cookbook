# Analysis: Why GRPO Dominates KL Distillation

## Recap: Model I/O and Supervision Signals

**Model**: Qwen/Qwen3-30B-A3B (qwen3_disable_thinking renderer)

**Prompt structure** (student):
```
<|im_start|>user
Classify the language... [13 labels] ... "Final Answer: xx"
Text to classify: {text}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n                    ← 80 prompt tokens
```

**Completion** (5 tokens):
```
pos  token     text          role
 0   19357     "Final"       format (deterministic)
 1   21806     " Answer"     format (deterministic)
 2      25     ":"           format (deterministic)
 3    *var*     " en"        ← THE classification decision
 4   151645    "<|im_end|>"  EOS (deterministic)
```

Only **1 of 5** completion tokens carries discriminative information.

---

## Root Cause: PARSE_FAIL Is the Dominant Error Mode

### Error decomposition (base model, 100 test sentences)

| Category | Count | % of errors |
|---|---|---|
| Correct | 82 | — |
| **PARSE_FAIL** | **15** | **83%** |
| Wrong label | 3 | 17% |

PARSE_FAIL means the model outputs a valid ISO language code (`sw`, `th`, `bg`)
that is **not in the 13-label set**. The model correctly identifies the language
but doesn't know the label mapping.

### Examples

| Text | Language | Student output | Gold label |
|---|---|---|---|
| "Ndio, vyema, mtu huyo amekuja" | Swahili | `sw` ← valid ISO, not in set | `ot` |
| "นั่นแหละ เจ้าตัวมาแล้ว" | Thai | `th` ← valid ISO, not in set | `ot` |
| "Гледахме нещо по телевизията" | Bulgarian | `bg` ← valid ISO, not in set | `ot` |

### The teacher has the same problem

**Of 14 student PARSE_FAIL cases, 12 (86%) also fail with the teacher prompt.**

Even the full 600-token teacher prompt with explicit mapping rules doesn't
reliably override the model's tendency to output real ISO codes. At
temperature ≈ 0, both prompts produce the same wrong output.

Token-level view for Swahili ("Naye akasema, Mama, niko nyumbani"):
```
pos  token    student_logp  teacher_logp  KL = s - t
 0   Final     +0.00000      +0.00000     +0.00000
 1   Answer    +0.00000      +0.00000     +0.00000
 2   :         +0.00000      +0.00000     +0.00000
 3   sw        -0.14411      -0.10067     -0.04344  ← teacher is MORE confident in "sw"!
 4   <eos>     -0.00000      -0.00006     +0.00006
```

The KL at the label position is **−0.04**, meaning the teacher assigns
HIGHER probability to "sw" than the student. KL distillation would push
the student to be **more** confident in the wrong answer.

---

## How Each Training Method Handles PARSE_FAIL

### KL-only (Exp 2: 84.5% accuracy)

- KL signal is near-zero at all 5 positions for easy examples
- For PARSE_FAIL cases, teacher has the same output → KL ≈ 0 → **no learning**
- Modest improvement (+3.5pp) comes from the few cases where teacher differs

### GRPO reward_only (Exp 4: 93.5% accuracy)

- `sw`/`th`/`bg` → `parse_label()` returns None → reward = 0.0
- Within the group, if any sample produces a valid label → reward = 1.0
- Advantage: correct samples upweighted, PARSE_FAIL downweighted
- **Directly penalizes the exact failure mode.** Model learns to map
  unsupported languages to `ot` (which is in the valid set → reward = 1.0)

### Off-policy SL (Exp 1: 86.5% accuracy)

- Training data contains explicit `(Swahili_text, "Final Answer: ot")` examples
- Student sees the correct mapping and memorizes it
- Effective but limited by the static dataset (no on-policy correction)

### Reward + KL (Exp 5: 84.0% accuracy)

- Has the GRPO reward signal, but doesn't filter constant-reward groups
- 97% of groups have all-same reward → zero reward-advantage
- KL adds small per-token adjustments to advantages
- Net effect: 94% of datums have ~KL-only signal, 6% have GRPO-like signal
- Result ≈ KL-only performance (84.0% vs 84.5%)

---

## Advantage Magnitude Comparison

### Reward-based (GRPO)

For a group with 6/8 correct, 2/8 wrong:
```
Rewards:    [1, 1, 1, 1, 1, 1, 0, 0]
Advantages: [+0.25, +0.25, +0.25, +0.25, +0.25, +0.25, -0.75, -0.75]
```
- Applied uniformly to all 5 tokens in each trajectory
- |max advantage| = **0.75**

### KL-based

From experimental metrics, mean |teacher_kl| ≈ 0.03–0.07 per token.
With kl_coef=1.0, effective per-token advantage ≈ ±0.03–0.07.
Over 5 tokens: total KL adjustment ≈ ±0.15–0.35.

- But only ~1 token carries useful signal → effective advantage ≈ **0.03–0.07**
- **10–25× weaker** than the reward signal on the critical token

---

## Group Filtering Dynamics (GRPO)

Probed 30 training sentences with 8 samples at temperature=1.0:

| Outcome | Count | % |
|---|---|---|
| All 8 correct | 25 | 83% |
| All 8 wrong | 4 | 13% |
| Mixed (has gradient) | 1 | 3% |
| **Filtered (no gradient)** | **29** | **97%** |

The model is highly consistent because the classification decision is a
**single token**. Temperature=1.0 is not enough to flip a confident
1-token prediction. The few mixed groups that survive carry strong signal.

Despite 97% filtering at step 0, accuracy reaches 92.5% by step 10.
After step ~18, no groups survive filtering → training effectively stops
at the converged accuracy of 93.5%.

---

## Summary

| Finding | Implication |
|---|---|
| Completions are 5 tokens, 1 discriminative | KL signal diluted 5× |
| 83% of errors are PARSE_FAIL (valid ISO code, wrong label set) | The model knows languages, just not the label mapping |
| Teacher has the SAME PARSE_FAIL in 86% of cases | KL ≈ 0 for the main error mode → KL can't fix it |
| GRPO directly penalizes invalid labels (reward=0) | Directly addresses the root cause |
| Reward+KL keeps all groups; 94% have zero reward-advantage | Reward signal drowned out by KL-only groups |

**The dominant error mode is a label mapping problem, not a language identification
problem. GRPO can fix this directly via reward; KL distillation cannot because the
teacher shares the same failure.**
