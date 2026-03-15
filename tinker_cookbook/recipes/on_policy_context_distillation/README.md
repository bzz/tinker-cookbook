# On-Policy Context Distillation

Context distillation trains a student model (without context) to match a teacher model (with detailed context, e.g., few-shot examples or instructions). This recipe compares **off-policy** and **on-policy** approaches to context distillation, as proposed in the [Tinker project ideas](https://github.com/thinking-machines-lab/tinker-project-ideas).

## Background

**Context distillation** [1, 2] is a technique where a teacher model uses a long, detailed prompt $P$ to generate responses, and a student model is trained to reproduce those responses *without* seeing $P$. After training, the student behaves as if it has internalized the prompt into its weights.

**Off-policy distillation** collects a dataset of teacher responses and trains the student via supervised learning:
- Teacher generates: $r_i = f_T([P, q_i])$
- Student learns: $\min \text{CE}(f_S(q_i), r_i)$
- The teacher's **answer** (token sequence) appears directly in the loss

**On-policy distillation** [3, 4, 5] samples from the *student* and uses the teacher for dense token-level supervision via reverse KL:
- Student generates: $r \sim \pi_\theta(\cdot | q)$
- Teacher evaluates: $\log \pi_T(r_t | P, q, r_{<t})$
- Advantage: $a_t = -\text{kl\_coef} \cdot (\log \pi_\theta(r_t | q, r_{<t}) - \log \pi_T(r_t | P, q, r_{<t}))$
- The teacher's **answer** is NOT in the loss—only its **probability distribution** over the student's tokens matters

The key difference: off-policy trains the student to reproduce specific teacher outputs, while on-policy trains the student to match the teacher's entire token distribution.

## Method: ContextAwareSamplingClient

For standard on-policy distillation, teacher and student see the same token sequence. For **context** distillation, the teacher must condition on context the student doesn't see. We implement this via a `ContextAwareSamplingClient` wrapper:

```python
class ContextAwareSamplingClient:
    def __init__(self, client, context_tokens):
        self.client = client
        self.context_tokens = context_tokens
        self.context_len = len(context_tokens)

    async def compute_logprobs_async(self, sequence_input):
        combined = ModelInput.from_ints(self.context_tokens + sequence_input.to_ints())
        logprobs = await self.client.compute_logprobs_async(combined)
        return logprobs[self.context_len:]  # Only student-token logprobs
```

This transparently prepends the teacher's context when computing logprobs and slices the result, so `incorporate_kl_penalty` in `train_on_policy.py` works unchanged.

## Experimental Setup

**Task**: Language classification — given a text, predict a 2-character language code.

**Model**: `Qwen/Qwen3-8B` with `qwen3_disable_thinking` renderer (same model as teacher and student)

**Teacher context**: A detailed classification prompt (~300 tokens) with instructions for handling scripts, Latin-script heuristics, mixed-language text, code, and ambiguous inputs. The teacher outputs "Final Answer: xx" after reasoning.

**Data**: `multilingual.txt` — 2100 sentences across 15 languages (140 sentence groups × 15 languages). 80/20 train/eval split by sentence group. 13 classification labels (ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot).

**Teacher accuracy**: 92.9% on train, 73.5% on eval (the teacher's own performance ceiling).

**Hyperparameters**:

| Parameter | Off-policy | On-policy |
|-----------|-----------|-----------|
| LoRA rank | 32 | 32 |
| Learning rate | 1e-4 (linear decay) | 1e-4 |
| Batch size | 64 | 64 groups × 4 rollouts |
| Epochs/steps | 4 epochs (92 steps) | 26 steps |
| Max tokens | 4096 | 256 |
| KL penalty coef | — | 1.0 |

## Results

### Overall Accuracy

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Base model (no training) | 0.7% | Responds conversationally, doesn't classify |
| **Off-policy only** | **72.6%** | Outputs concise 2-character labels |
| On-policy only | 62.1%* | Outputs explanatory reasoning about language |
| Off-policy → on-policy | 5.7% | Collapsed to predicting "ar" for most inputs |

\*On-policy accuracy measured by parsing language identifications from explanatory text (e.g., "The text is in French" → fr). Without this parsing, accuracy is 0%.

### Per-Language Accuracy

| Language | Off-policy | On-policy | Combined |
|----------|-----------|-----------|----------|
| ar (Arabic) | 78.6% | 78.6% | 85.7% |
| de (German) | 78.6% | 78.6% | 0.0% |
| el (Greek) | 82.1% | 82.1% | 0.0% |
| en (English) | 85.7% | 78.6% | 0.0% |
| es (Spanish) | 78.6% | 57.1% | 0.0% |
| fr (French) | 75.0% | 53.6% | 0.0% |
| hi (Hindi) | 75.0% | 14.3% | 0.0% |
| ot (Other) | 50.0% | 59.5% | 0.0% |
| ru (Russian) | 75.0% | 71.4% | 0.0% |
| tr (Turkish) | 82.1% | 78.6% | 0.0% |
| ur (Urdu) | 67.9% | 0.0% | 0.0% |
| vi (Vietnamese) | 78.6% | 78.6% | 0.0% |
| zh (Chinese) | 82.1% | 82.1% | 0.0% |

### Training Dynamics

**Off-policy**: Loss converged rapidly from ~3.0 to ~0.0002 over 92 steps (4 epochs). The student quickly learned to map sentences to 2-character codes.

**On-policy (from scratch)**: Teacher KL started at 0.57 and increased to ~0.87 over 26 steps. Entropy decreased from 0.58 to 0.25. The student learned to generate language-analysis text matching the teacher's reasoning style.

**Combined (off-policy → on-policy)**: Teacher KL started at 17.0 (extremely high) and decreased slowly to 13.2 over 26 steps. Entropy collapsed to near-zero (0.0001). The student's concise-label distribution was so different from the teacher's reasoning distribution that KL-based training caused catastrophic drift.

### Example Outputs

**Off-policy model** (concise labels):
```
Input: "And he said, Mama, I'm home."
Output: "en"

Input: "他说，妈妈，我回来了。"
Output: "zh"
```

**On-policy model** (explanatory reasoning):
```
Input: "Solche Kleinen dinge machten einen grossen Unterschied..."
Output: "The given text is in German. It contains German words such as 'Solche,'
         'Kleinen,' 'dinge,' 'machten,' 'grossen,' and 'Unterschied'..."

Input: "再见"
Output: "zh"  (short responses for unambiguous script-based cases)
```

**Combined model** (collapsed):
```
Input: "Bonjour le monde"
Output: "ar"  (predicts Arabic for everything)
```

## Discussion

### Why off-policy outperforms on-policy for this task

The language classification task has a key property: the teacher produces **long reasoning** (analyzing scripts, diacritics, function words) before outputting a **short label**. Off-policy distillation extracts just the label, creating clean (input, label) supervision. On-policy distillation, however, tries to match the teacher's full token distribution — including the reasoning — which the student has no context to produce correctly.

This suggests on-policy context distillation works best when:
1. Teacher and student generate **similar types of outputs** (e.g., both generate reasoning traces in math distillation [5])
2. The context shifts the distribution **subtly** rather than changing the output format entirely

### Why the combined approach fails

Starting from an off-policy checkpoint that outputs concise labels, then applying on-policy KL against a reasoning teacher creates a massive distribution mismatch (initial KL = 17.0 vs ~0.5 for on-policy from scratch). The KL gradient pushes the student away from its learned label distribution but can't effectively guide it toward the teacher's long-form reasoning. The result is mode collapse.

### Recommendations

For tasks where the teacher's context changes the **output format** (e.g., adding reasoning, changing response structure):
- **Use off-policy distillation**, extracting just the desired output from the teacher
- On-policy distillation may cause the student to learn the teacher's reasoning process without the teacher's context, leading to worse task performance

For tasks where the teacher's context changes **which answer** is produced but not the format:
- On-policy distillation may be more effective, as shown in reasoning distillation [5] where both teacher and student produce similar chain-of-thought outputs

## How to Reproduce

### Step 1: Generate data
```bash
python -m tinker_cookbook.recipes.on_policy_context_distillation.create_data \
    output_dir=/tmp/tinker-datasets/context_distillation \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking
```

### Step 2: Off-policy training
```bash
python -m tinker_cookbook.recipes.on_policy_context_distillation.train_off_policy \
    file_path=/tmp/tinker-datasets/context_distillation/train.jsonl \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking \
    learning_rate=1e-4 \
    lora_rank=32 \
    batch_size=64 \
    num_epochs=4
```

### Step 3: On-policy training
```bash
python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking \
    teacher_model=Qwen/Qwen3-8B \
    prompts_file=/tmp/tinker-datasets/context_distillation/train_prompts.jsonl \
    learning_rate=1e-4 \
    lora_rank=32 \
    groups_per_batch=64 \
    group_size=4 \
    max_step=26 \
    kl_penalty_coef=1.0
```

### Step 4: Combined (off-policy → on-policy)
```bash
python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking \
    teacher_model=Qwen/Qwen3-8B \
    load_checkpoint_path=<OFF_POLICY_CHECKPOINT> \
    prompts_file=/tmp/tinker-datasets/context_distillation/train_prompts.jsonl \
    learning_rate=1e-4 \
    lora_rank=32 \
    groups_per_batch=64 \
    group_size=4 \
    max_step=26 \
    kl_penalty_coef=1.0
```

### Step 5: Evaluate
```bash
python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking \
    load_checkpoint_path=<CHECKPOINT_PATH> \
    eval_file=/tmp/tinker-datasets/context_distillation/eval_prompts.jsonl
```

## Checkpoints

| Experiment | Checkpoint |
|-----------|-----------|
| Off-policy (92 steps) | `tinker://1b910180-56eb-534a-bcf1-335be750c89e:train:0/sampler_weights/final` |
| On-policy (20 steps) | `tinker://1cdbe922-5be3-5192-8491-23a5c50b09c4:train:0/sampler_weights/000020` |
| Combined (26 steps) | `tinker://31f86e3f-ed0f-5e3c-97c7-41fcdadf6495:train:0/sampler_weights/final` |

## References

1. [Askell et al. (2021)](https://arxiv.org/abs/2112.00861) — A General Language Assistant as a Laboratory for Alignment (context distillation origin)
2. [Snell et al. (2022)](https://arxiv.org/abs/2209.15189) — Learning by Distilling Context
3. [Agarwal et al. (2023)](https://arxiv.org/abs/2306.13649) — GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models
4. [Gu et al. (2023)](https://arxiv.org/abs/2306.08543) — MiniLLM: Knowledge Distillation of Large Language Models
5. [On-Policy Distillation blog post](https://thinkingmachines.ai/blog/on-policy-distillation/) — Thinking Machines Lab
6. [Agarwal et al. (2024)](https://arxiv.org/abs/2404.11018) — Many-Shot In-Context Learning
7. [Call for Community Projects](https://thinkingmachines.ai/news/call-for-community-projects/) — Thinking Machines Lab
