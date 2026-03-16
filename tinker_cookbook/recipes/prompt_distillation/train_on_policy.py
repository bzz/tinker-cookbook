"""
On-policy context distillation for language classification.

Teacher sees a full detailed classification prompt (task + instructions + output format),
while the student sees only a short prompt (task + output format). The student learns
to internalize the teacher's instructions via on-policy KL penalty.

Based on the context distillation idea:
  https://github.com/thinking-machines-lab/tinker-project-ideas

Experiments:
    # On-policy context distillation from scratch
    python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
        model_name=Qwen/Qwen3-30B-A3B \
        groups_per_batch=32 \
        group_size=4 \
        max_steps=30

    # On-policy after off-policy SFT (combo)
    python -m tinker_cookbook.recipes.prompt_distillation.train_on_policy \
        model_name=Qwen/Qwen3-30B-A3B \
        load_checkpoint_path=tinker://... \
        groups_per_batch=32 \
        max_steps=20
"""

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from functools import partial
from typing import Any, List, Sequence, cast

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, StepResult, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEACHER_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script -> hi.
   - Greek script -> el.
   - Cyrillic script -> ru.
   - Han characters -> zh. (Treat Traditional as zh too.)
   - Arabic script -> ar vs ur:
       If Urdu-only letters appear, or clear Urdu words, choose ur.
       Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: Vietnamese-specific letters/diacritics.
   - tr: Turkish-specific letters and function words.
   - de: umlauts or eszett and German function words.
   - es: n-tilde, inverted punctuation, Spanish function words.
   - fr: French diacritics and function words.
   - en: default among Latin languages if English function words are present.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.

5) Mixed-language text:
   - Determine the dominant language. If tied, return 'ot'.

6) Very short or noisy inputs:
   - If <=2 meaningful words and no strong signal, return 'ot'.

7) Transliteration/romanization:
   - Romanized non-Latin languages without clear cues -> 'ot'.

8) Code-heavy inputs:
   - Mostly code with no natural-language comments -> 'ot'.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot'.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx is one of: ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot.

Text to classify:
{text}"""

STUDENT_PROMPT = """Classify the language of the provided text into one of these labels:
ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot.

Respond with EXACTLY one line: "Final Answer: xx"

Text to classify:
{text}"""

VALID_LABELS = {"ar", "de", "el", "en", "es", "fr", "hi", "ru", "tr", "ur", "vi", "zh", "ot"}


def parse_label(response: str) -> str | None:
    match = re.search(r"Final Answer:\s*(\w+)", response)
    if match:
        label = match.group(1).lower()
        return label if label in VALID_LABELS else None
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "example_data", "multilingual.txt")


def load_multilingual_sentences(path: str = DATA_PATH) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def split_train_test(
    sentences: list[str], test_fraction: float = 0.2, group_size_in_file: int = 15
) -> tuple[list[str], list[str]]:
    """Split by sentence groups to prevent data leakage between translations."""
    n_groups = len(sentences) // group_size_in_file
    n_test_groups = max(1, int(n_groups * test_fraction))
    n_train_groups = n_groups - n_test_groups
    train_end = n_train_groups * group_size_in_file
    test_end = n_groups * group_size_in_file
    return sentences[:train_end], sentences[train_end:test_end]


# ---------------------------------------------------------------------------
# Gold label generation
# ---------------------------------------------------------------------------

async def generate_gold_labels(
    texts: list[str],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
) -> list[str]:
    """Generate classification labels using the teacher model with the full prompt."""
    params = tinker.SamplingParams(
        max_tokens=100, temperature=0.15, stop=renderer.get_stop_sequences()
    )

    async def label_one(text: str) -> str:
        prompt = TEACHER_PROMPT.format(text=text)
        convo: list[renderers.Message] = [{"role": "user", "content": prompt}]
        prompt_input = renderer.build_generation_prompt(convo)
        result = await sampling_client.sample_async(
            prompt=prompt_input, sampling_params=params, num_samples=1
        )
        response = tokenizer.decode(result.sequences[0].tokens)
        label = parse_label(response)
        return label if label is not None else "ot"

    labels = await asyncio.gather(*[label_one(t) for t in texts])
    return list(labels)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LangClassificationEnv(ProblemEnv):
    """Zero-reward environment for context distillation of language classification."""

    def __init__(self, text: str, student_prompt_template: str, renderer: renderers.Renderer):
        super().__init__(renderer, convo_prefix=None, format_coef=0.0)
        self.text = text
        self._question = student_prompt_template.format(text=text)

    def get_question(self) -> str:
        return self._question

    def check_format(self, sample_str: str) -> bool:
        return True

    def check_answer(self, sample_str: str) -> bool:
        return False

    def get_reference_answer(self) -> str:
        return ""

    async def step(self, action: Action) -> StepResult:
        self.renderer.parse_response(action)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContextDistillationDataset(RLDataset):
    """Provides multilingual sentences for context distillation training."""

    def __init__(
        self,
        texts: list[str],
        groups_per_batch: int,
        group_size: int,
        renderer: renderers.Renderer,
        student_prompt_template: str,
    ):
        self.texts = texts
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.renderer = renderer
        self.student_prompt_template = student_prompt_template

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min((index + 1) * self.groups_per_batch, len(self.texts))
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    LangClassificationEnv,
                    text=text,
                    student_prompt_template=self.student_prompt_template,
                    renderer=self.renderer,
                ),
                num_envs=self.group_size,
                dataset_name="lang_classification",
            )
            for text in self.texts[start:end]
        ]

    def get_batch_texts(self, index: int) -> list[str]:
        start = index * self.groups_per_batch
        end = min((index + 1) * self.groups_per_batch, len(self.texts))
        return self.texts[start:end]

    def __len__(self) -> int:
        return math.ceil(len(self.texts) / self.groups_per_batch)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class LangClassificationEvaluator(SamplingClientEvaluator):
    """Evaluates language classification accuracy against teacher-generated gold labels."""

    def __init__(
        self,
        test_texts: list[str],
        gold_labels: list[str],
        student_prompt_template: str,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
        max_eval_samples: int = 200,
    ):
        self.test_texts = test_texts[:max_eval_samples]
        self.gold_labels = gold_labels[:max_eval_samples]
        self.student_prompt_template = student_prompt_template
        self.renderer = renderer
        self.tokenizer = tokenizer

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        params = tinker.SamplingParams(
            max_tokens=50, temperature=0.0, stop=self.renderer.get_stop_sequences()
        )

        async def eval_one(text: str) -> str | None:
            prompt = self.student_prompt_template.format(text=text)
            convo: list[renderers.Message] = [{"role": "user", "content": prompt}]
            model_input = self.renderer.build_generation_prompt(convo)
            result = await sampling_client.sample_async(
                prompt=model_input, sampling_params=params, num_samples=1
            )
            response = self.tokenizer.decode(result.sequences[0].tokens)
            return parse_label(response)

        predictions = await asyncio.gather(*[eval_one(t) for t in self.test_texts])

        correct = sum(1 for p, g in zip(predictions, self.gold_labels) if p == g)
        valid = sum(1 for p in predictions if p is not None)
        total = len(self.test_texts)

        return {
            "accuracy": correct / total if total else 0.0,
            "parse_rate": valid / total if total else 0.0,
            "num_evaluated": float(total),
        }


# ---------------------------------------------------------------------------
# Context-aware KL penalty
# ---------------------------------------------------------------------------

async def incorporate_kl_penalty_context_distillation(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    texts_P: list[str],
    teacher_client: tinker.SamplingClient,
    teacher_prompt_template: str,
    renderer: renderers.Renderer,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """
    Context-distillation KL penalty: the teacher sees the full prompt while
    the student sees only a short prompt.  For each datum we:

    1. Extract completion tokens (mask=1 region) from the student sequence
    2. Build the teacher input: render(full_teacher_prompt) + completion_tokens
    3. Compute teacher logprobs on the teacher input
    4. Align and compute reverse KL over the completion positions
    5. Adjust the datum's advantages in-place
    """
    if kl_penalty_coef <= 0:
        return {}

    # Pre-build teacher inputs and record alignment info
    teacher_inputs: list[tinker.ModelInput | None] = []
    completion_starts: list[int] = []
    completion_lens: list[int] = []
    teacher_prompt_lens: list[int] = []

    for datum, metadata in zip(data_D, metadata_D):
        text = texts_P[metadata["group_idx"]]
        teacher_convo: list[renderers.Message] = [
            {"role": "user", "content": teacher_prompt_template.format(text=text)}
        ]
        teacher_prompt_input = renderer.build_generation_prompt(teacher_convo)
        teacher_prompt_len = teacher_prompt_input.length

        full_student_seq = datum.model_input.append_int(
            cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        )
        mask = datum.loss_fn_inputs["mask"].to_torch()
        nonzero = (mask != 0).nonzero(as_tuple=False)

        if len(nonzero) == 0:
            teacher_inputs.append(None)
            completion_starts.append(0)
            completion_lens.append(0)
            teacher_prompt_lens.append(0)
            continue

        cs = int(nonzero[0].item())
        student_tokens = full_student_seq.to_ints()
        comp_tokens = student_tokens[cs + 1 :]
        comp_len = len(comp_tokens)

        teacher_full = teacher_prompt_input
        for tok in comp_tokens:
            teacher_full = teacher_full.append_int(tok)

        teacher_inputs.append(teacher_full)
        completion_starts.append(cs)
        completion_lens.append(comp_len)
        teacher_prompt_lens.append(teacher_prompt_len)

    # Batch teacher logprob requests
    async def _teacher_logprobs(inp: tinker.ModelInput | None) -> list[float | None] | None:
        if inp is None:
            return None
        return await teacher_client.compute_logprobs_async(inp)

    teacher_logprobs_D = await asyncio.gather(*[_teacher_logprobs(inp) for inp in teacher_inputs])

    # Compute KL and adjust advantages
    total_kl_sum = 0.0
    total_mask_sum = 0.0

    for i, datum in enumerate(data_D):
        if teacher_logprobs_D[i] is None:
            continue

        mask = datum.loss_fn_inputs["mask"].to_torch().float()
        sampled_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        teacher_lps_raw = teacher_logprobs_D[i]
        assert teacher_lps_raw is not None

        cs = completion_starts[i]
        comp_len = completion_lens[i]
        tp_len = teacher_prompt_lens[i]

        if comp_len == 0:
            continue

        # Teacher logprobs aligned to completion tokens
        teacher_slice = teacher_lps_raw[tp_len : tp_len + comp_len]
        teacher_lps = torch.tensor(
            [lp if lp is not None else 0.0 for lp in teacher_slice], dtype=torch.float32
        )

        # Student logprobs aligned to completion tokens
        student_lps = sampled_logprobs[cs : cs + comp_len]
        comp_mask = mask[cs : cs + comp_len]

        min_len = min(len(teacher_lps), len(student_lps), len(comp_mask))
        if min_len == 0:
            continue

        reverse_kl = (student_lps[:min_len] - teacher_lps[:min_len]) * comp_mask[:min_len]

        kl_adv = -kl_penalty_coef * reverse_kl
        if kl_discount_factor > 0:
            kl_adv = torch.tensor(
                discounted_future_sum_vectorized(kl_adv.numpy(), kl_discount_factor)
            )

        full_kl_adv = torch.zeros_like(mask)
        full_kl_adv[cs : cs + min_len] = kl_adv

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + full_kl_adv
        )

        total_kl_sum += reverse_kl.sum().item()
        total_mask_sum += comp_mask[:min_len].sum().item()

    avg_kl = total_kl_sum / total_mask_sum if total_mask_sum > 0 else 0.0
    return {"teacher_kl": float(avg_kl)}


# ---------------------------------------------------------------------------
# Config & training loop
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    model_name: str
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    learning_rate: float = 1e-4
    lora_rank: int = 32
    groups_per_batch: int = 32
    group_size: int = 4
    max_tokens: int = 50
    temperature: float = 1.0

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    num_substeps: int = 1

    eval_every: int = 5
    save_every: int = 10
    max_steps: int | None = None

    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    gold_labels_path: str | None = None
    max_eval_samples: int = 200


async def main(cfg: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Resume support
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info.batch if resume_info else 0

    # Service client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, cfg.renderer_name)

    # Training client
    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, cfg.renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info("Resumed training from %s", resume_info.state_path)
    elif cfg.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, cfg.load_checkpoint_path, cfg.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info("Loaded weights from %s", cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank, user_metadata=user_metadata
        )

    tokenizer = training_client.get_tokenizer()
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
    )
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Teacher sampling client (base model, no LoRA)
    teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)

    # Load data
    sentences = load_multilingual_sentences()
    train_texts, test_texts = split_train_test(sentences)
    logger.info("Data: %d train, %d test sentences", len(train_texts), len(test_texts))

    # Gold labels for evaluation
    if cfg.gold_labels_path and os.path.exists(cfg.gold_labels_path):
        with open(cfg.gold_labels_path) as f:
            gold_labels = json.load(f)
        logger.info("Loaded %d gold labels from %s", len(gold_labels), cfg.gold_labels_path)
    else:
        logger.info("Generating gold labels for %d test sentences...", len(test_texts))
        gold_labels = await generate_gold_labels(test_texts, teacher_client, renderer, tokenizer)
        label_path = os.path.join(cfg.log_path, "gold_labels.json")
        os.makedirs(cfg.log_path, exist_ok=True)
        with open(label_path, "w") as f:
            json.dump(gold_labels, f)
        logger.info("Saved gold labels to %s", label_path)

    # Dataset and evaluator
    dataset = ContextDistillationDataset(
        texts=train_texts,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
        renderer=renderer,
        student_prompt_template=STUDENT_PROMPT,
    )
    evaluator = LangClassificationEvaluator(
        test_texts=test_texts,
        gold_labels=gold_labels,
        student_prompt_template=STUDENT_PROMPT,
        renderer=renderer,
        tokenizer=tokenizer,
        max_eval_samples=cfg.max_eval_samples,
    )

    num_batches = len(dataset)
    if cfg.max_steps is not None:
        num_batches = min(cfg.max_steps, num_batches)
    logger.info("Training for %d steps", num_batches)

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    # Training loop
    for i_batch in range(start_batch, num_batches):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Evaluation
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                eval_metrics = await evaluator(sampling_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch
        env_group_builders_P = dataset.get_batch(i_batch)
        texts_P = dataset.get_batch_texts(i_batch)

        # Sample trajectories
        with timed("sample", metrics):
            trajectory_groups_raw = await asyncio.gather(
                *[
                    do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                        do_remove_constant_reward_groups=False,
                    )
                    for builder in env_group_builders_P
                ]
            )

        # Filter None and maintain text mapping
        trajectory_groups_P: list[TrajectoryGroup] = []
        filtered_texts_P: list[str] = []
        for tg, text in zip(trajectory_groups_raw, texts_P):
            if tg is not None:
                trajectory_groups_P.append(tg)
                filtered_texts_P.append(text)

        if not trajectory_groups_P:
            logger.warning("No valid trajectory groups at batch %d, skipping", i_batch)
            ml_logger.log_metrics(metrics, step=i_batch)
            continue

        # Compute advantages and assemble training data
        with timed("assemble", metrics):
            advantages_P = compute_advantages(trajectory_groups_P)
            data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

        if data_D:
            logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

        # Context-distillation KL penalty
        if cfg.kl_penalty_coef > 0 and data_D:
            with timed("kl_penalty", metrics):
                kl_metrics = await incorporate_kl_penalty_context_distillation(
                    data_D=data_D,
                    metadata_D=metadata_D,
                    texts_P=filtered_texts_P,
                    teacher_client=teacher_client,
                    teacher_prompt_template=TEACHER_PROMPT,
                    renderer=renderer,
                    kl_penalty_coef=cfg.kl_penalty_coef,
                    kl_discount_factor=cfg.kl_discount_factor,
                )
            metrics.update(kl_metrics)

        # Train step
        if data_D:
            with timed("train", metrics):
                training_logprobs_D = await train_step(
                    data_D=data_D,
                    training_client=training_client,
                    learning_rate=cfg.learning_rate,
                    num_substeps=cfg.num_substeps,
                    loss_fn=cfg.loss_fn,
                    loss_fn_config=cfg.loss_fn_config,
                    metrics=metrics,
                )

            sampling_client, full_batch_metrics = (
                await compute_full_batch_metrics_and_get_sampling_client(
                    training_client,
                    i_batch + 1,
                    data_D,
                    training_logprobs_D,
                    cfg.log_path,
                    cfg.save_every,
                    do_compute_post_kl=False,
                )
            )
            metrics.update(full_batch_metrics)

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval = await evaluator(sampling_client)
    logger.info("Final results: %s", final_eval)
    ml_logger.log_metrics({f"final/{k}": v for k, v in final_eval.items()}, step=num_batches)

    # Save final checkpoint
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=None,
        )

    ml_logger.close()
    logger.info("Training completed successfully")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B"
    log_path: str | None = None
    load_checkpoint_path: str | None = None
    renderer_name: str | None = "qwen3_disable_thinking"

    learning_rate: float = 1e-4
    lora_rank: int = 32
    groups_per_batch: int = 32
    group_size: int = 4
    max_tokens: int = 50
    temperature: float = 1.0

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    eval_every: int = 5
    save_every: int = 10
    max_steps: int | None = None

    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    gold_labels_path: str | None = None
    max_eval_samples: int = 200

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_short = cli_config.model_name.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"on_policy_ctx_distill-{model_short}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.groups_per_batch}batch-{date_str}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/prompt_distillation/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        renderer_name=renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        gold_labels_path=cli_config.gold_labels_path,
        max_eval_samples=cli_config.max_eval_samples,
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
