"""
On-policy context distillation v2: student sees task definition + output format.

Key design change from v1: In v1, the student saw ONLY the raw sentence with no
context, which meant it had no idea this was a classification task. In v2, we split
the teacher's prompt into:

  - STUDENT_CONTEXT (shared): Task definition, label list, and output format.
    Both student and teacher see this.
  - TEACHER_ONLY_CONTEXT: The detailed classification instructions (steps 1-9:
    script-based rules, Latin heuristics, named entities, etc.).
    Only the teacher sees this.

This way the student knows *what* to do (classify language, output "Final Answer: xx")
but must learn *how* to do it (the classification heuristics) from the teacher's
token-level KL signal.

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy_v2 \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3_disable_thinking \
        prompts_file=/tmp/tinker-datasets/context_distillation/train_prompts.jsonl \
        wandb_project=context_distillation_v2
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

import chz
import tinker
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.distillation.train_on_policy import Config, do_sync_training
from tinker_cookbook.eval.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy import (
    ContextAwareSamplingClient,
)
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.trace import scope, trace_init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt split: student context vs teacher-only context
# ---------------------------------------------------------------------------

# The student sees the task definition and output format — it knows WHAT to do.
STUDENT_CONTEXT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx ∈ {ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot} and nothing else."""

# The teacher additionally sees detailed classification instructions — HOW to do it.
TEACHER_ONLY_INSTRUCTIONS = """
Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script → hi.
   - Greek script → el.
   - Cyrillic script → ru.
   - Han characters (中文) → zh. (Treat Traditional as zh too.)
   - Arabic script → ar vs ur:
       • If Urdu-only letters appear (e.g., ے, ڑ, ں, ھ, ٹ, ڈ, کھ, گ, چ with Urdu forms), or clear Urdu words, choose ur.
       • Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters. If tied, go to step 5.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics (ă â ê ô ơ ư đ, plus dense diacritics across many words).
   - tr: presence of Turkish-specific letters (ı İ ğ Ğ ş Ş ç Ç ö Ö ü Ü) and common function words (ve, bir, için, değil, ama, çok).
   - de: presence of umlauts (ä ö ü) or ß and common function words (und, der, die, das, nicht, ist).
   - es: presence of ñ, ¿, ¡ and common words (y, de, la, el, es, no, por, para, con, gracias, hola).
   - fr: frequent French diacritics (é è ê à ç ô â î û ù) and common words (et, le, la, les, des, une, est, avec, pour, merci, bonjour).
   - en: default among Latin languages if strong evidence for others is absent, but ONLY if English function words are present (the, and, is, are, to, of, in, for, on, with). If evidence is insufficient for any Latin language, prefer 'ot' over guessing.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals (diacritics/letters) before assigning a Latin-language label.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens (language-specific letters/diacritics/function words) AFTER preprocessing.
   - If two or more languages are equally dominant or the text is a deliberate multi-language mix, return 'ot'.

6) Very short or noisy inputs:
   - If the text is ≤2 meaningful words or too short to be confident, return 'ot' unless there is a very strong language-specific signal (e.g., "bonjour" → fr, "hola" → es).

7) Transliteration/romanization:
   - If Hindi/Urdu/Arabic/Chinese/Russian/Greek is written purely in Latin letters (romanized) without clear, repeated language-specific cue words, return 'ot'. (Only classify as hi/ur/ar/zh/ru/el when native scripts or highly distinctive romanized patterns are clearly present.)

8) Code-heavy inputs:
   - If the text is mostly code with minimal or no natural-language comments/strings, return 'ot'.
   - If comments/strings clearly indicate a language per rules above, use that label.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing."""


# Full teacher context = student context + teacher-only instructions
TEACHER_FULL_CONTEXT = STUDENT_CONTEXT + "\n" + TEACHER_ONLY_INSTRUCTIONS


@chz.chz
class FilePromptDatasetBuilderV2(RLDatasetBuilder):
    """Builder for the v2 prompt dataset with student task context."""

    prompts_file: str
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    max_prompt_tokens: int | None = 1024

    async def __call__(self) -> tuple[PromptOnlyDataset, PromptOnlyDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        prompts = []
        with open(self.prompts_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data["sentence"])

        # Student sees task definition as system message
        student_convo_prefix: list[renderers.Message] = [
            {"role": "system", "content": STUDENT_CONTEXT},
        ]

        train_dataset = PromptOnlyDataset(
            prompts=prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            convo_prefix=student_convo_prefix,
            dataset_name="context_distillation_v2",
        )

        return train_dataset, None


# ---------------------------------------------------------------------------
# CLI config and main
# ---------------------------------------------------------------------------

@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy context distillation v2."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = "qwen3_disable_thinking"
    load_checkpoint_path: str | None = None

    # Teacher configuration (same base model by default)
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    prompts_file: str = "/tmp/tinker-datasets/context_distillation/train_prompts.jsonl"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 64
    learning_rate: float = 1e-4
    max_tokens: int = 256
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer
    num_substeps: int = 1

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evaluation and checkpointing
    eval_every: int = 10
    save_every: int = 10
    max_step: int | None = 50

    # Service
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def render_teacher_context(renderer_name: str, model_name: str) -> list[int]:
    """Render the full teacher context (task def + instructions + output format) into tokens.

    The teacher sees TEACHER_FULL_CONTEXT as a system message prepended to the
    student's sequence via ContextAwareSamplingClient.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Teacher sees the FULL original prompt (task def + instructions + output format)
    context_messages: list[renderers.Message] = [
        {"role": "system", "content": TEACHER_FULL_CONTEXT},
    ]
    context_input, _ = renderer.build_supervised_example(
        context_messages, train_on_what=renderers.TrainOnWhat.ALL_TOKENS
    )
    context_tokens = context_input.to_ints()

    logger.info(f"Teacher full context rendered to {len(context_tokens)} tokens")
    logger.info(f"Student sees STUDENT_CONTEXT (task def + output format) via convo_prefix")
    return context_tokens


@scope
async def main_async(cli_config: CLIConfig):
    """Main entry point for on-policy context distillation v2."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"ctx-distill-onpolicy-v2-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/context_distillation_v2/{run_name}")

    wandb_name = cli_config.wandb_name or os.path.basename(log_path)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Render full teacher context tokens (task def + instructions + output format)
    teacher_context_tokens = render_teacher_context(renderer_name, cli_config.model_name)

    # Log token counts for cost tracking
    tokenizer_for_counting = get_tokenizer(cli_config.model_name)
    student_ctx_tokens = len(tokenizer_for_counting.encode(STUDENT_CONTEXT))
    teacher_ctx_tokens = len(teacher_context_tokens)
    logger.info(f"=== Token counts for cost estimation ===")
    logger.info(f"Student context tokens (per sample): {student_ctx_tokens}")
    logger.info(f"Teacher full context tokens (per teacher logprob call): {teacher_ctx_tokens}")
    logger.info(
        f"Per-step sampling tokens (approx): "
        f"{cli_config.groups_per_batch} groups × {cli_config.group_size} rollouts × "
        f"({student_ctx_tokens} prompt + {cli_config.max_tokens} max gen) = "
        f"{cli_config.groups_per_batch * cli_config.group_size * (student_ctx_tokens + cli_config.max_tokens)}"
    )
    logger.info(
        f"Per-step teacher logprob tokens (approx): "
        f"{cli_config.groups_per_batch * cli_config.group_size} sequences × "
        f"({teacher_ctx_tokens} teacher ctx + {student_ctx_tokens} student ctx + {cli_config.max_tokens} gen) = "
        f"{cli_config.groups_per_batch * cli_config.group_size * (teacher_ctx_tokens + student_ctx_tokens + cli_config.max_tokens)}"
    )

    # Build config
    dataset_builder = FilePromptDatasetBuilderV2(
        prompts_file=cli_config.prompts_file,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )

    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    cfg = Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_step=cli_config.max_step,
    )

    # Replicate train_on_policy.main() but with v2 teacher wrapping
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, cfg.renderer_name)

    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank, user_metadata=user_metadata
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, load_state_path, cfg.renderer_name
        )
        future = await training_client.load_state_with_optimizer_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher clients
    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = []

    for dc in cfg.dataset_configs:
        dataset, maybe_test = await dc.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dc.groups_per_batch)

        # Create teacher sampling client
        tc = dc.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=tc.base_model)
        if tc.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=tc.base_model, model_path=tc.load_checkpoint_path
            )

        # Wrap teacher with context-aware client (full prompt: task def + instructions + output format)
        wrapped_teacher = ContextAwareSamplingClient(teacher_client, teacher_context_tokens)
        teacher_clients.append(wrapped_teacher)
        logger.info(
            f"Created context-aware teacher for {tc.base_model} "
            f"(full context: {len(teacher_context_tokens)} tokens, "
            f"checkpoint: {tc.load_checkpoint_path})"
        )

    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    num_batches = min(cfg.max_step, num_batches) if cfg.max_step is not None else num_batches
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("On-policy context distillation v2 completed successfully")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main_async(cli_config))
