"""
On-policy context distillation training.

The student model generates responses WITHOUT the teacher's context prompt.
The teacher model evaluates the student's tokens WITH the context prompt prepended,
providing dense token-level KL supervision.

This implements Approach A from the plan: a ContextAwareSamplingClient wrapper
that transparently prepends context tokens when computing teacher logprobs.

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3_disable_thinking \
        prompts_file=/tmp/tinker-datasets/context_distillation/train_prompts.jsonl \
        wandb_project=context_distillation
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
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.trace import scope, trace_init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context-aware sampling client wrapper
# ---------------------------------------------------------------------------

class ContextAwareSamplingClient:
    """Wraps a SamplingClient to prepend teacher context when computing logprobs.

    In context distillation, the teacher should evaluate student-generated tokens
    conditioned on a context prefix (e.g., few-shot examples) that the student
    does NOT see. This wrapper:

    1. Prepends ``context_tokens`` to the input sequence
    2. Calls ``compute_logprobs_async`` on the combined sequence
    3. Slices the result to return only the logprobs for the original (student) tokens

    The returned logprobs thus satisfy:
        logprobs[i] = log P_teacher(token_i | context, token_0, ..., token_{i-1})

    This is transparent to ``incorporate_kl_penalty`` in ``train_on_policy.py``,
    which uses ``teacher_logprobs[1:]`` to align with the student's sampled logprobs.
    """

    def __init__(self, client: tinker.SamplingClient, context_tokens: list[int]):
        self.client = client
        self.context_tokens = context_tokens
        self.context_len = len(context_tokens)

    async def compute_logprobs_async(
        self, sequence_input: tinker.ModelInput
    ) -> list[float]:
        # Prepend context tokens to the student's sequence
        student_tokens = sequence_input.to_ints()
        combined_tokens = self.context_tokens + student_tokens
        combined_input = tinker.ModelInput.from_ints(combined_tokens)

        # Get logprobs for the full sequence
        full_logprobs = await self.client.compute_logprobs_async(combined_input)

        # Return only the logprobs for the student's token positions.
        # full_logprobs has (context_len + student_len) entries.
        # We want entries [context_len:] which correspond to the student tokens.
        return full_logprobs[self.context_len :]


@chz.chz
class FilePromptDatasetBuilder(RLDatasetBuilder):
    """Builder for a file-based prompt dataset."""

    prompts_file: str
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    max_prompt_tokens: int | None = 1024

    async def __call__(self) -> tuple[PromptOnlyDataset, PromptOnlyDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Load prompts from JSONL (each line has {"sentence": ..., "ground_truth": ...})
        prompts = []
        with open(self.prompts_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data["sentence"])

        train_dataset = PromptOnlyDataset(
            prompts=prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            convo_prefix=None,  # Student sees NO context
            dataset_name="context_distillation",
        )

        return train_dataset, None


# ---------------------------------------------------------------------------
# CLI config and main
# ---------------------------------------------------------------------------

# Import the teacher prompt from create_data
from tinker_cookbook.recipes.on_policy_context_distillation.create_data import (
    LANGUAGE_CLASSIFICATION_PROMPT,
)


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy context distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = "qwen3_disable_thinking"
    load_checkpoint_path: str | None = None

    # Teacher configuration (same base model by default for context distillation)
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    prompts_file: str = "/tmp/tinker-datasets/context_distillation/train_prompts.jsonl"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 64
    learning_rate: float = 1e-4
    max_tokens: int = 256  # Short responses for language classification
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
    max_step: int | None = 50  # Default to 50 steps for small-scale experiments

    # Service
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def render_teacher_context(renderer_name: str, model_name: str) -> list[int]:
    """Render the teacher's context prompt into tokens.

    The teacher context is the LANGUAGE_CLASSIFICATION_PROMPT formatted as a
    user message. We render it into a generation prompt (the model input that
    would precede the assistant's response) so that when we prepend it to the
    student's token sequence, the teacher "sees" the context before evaluating.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Build only the teacher's message prefix; do not append generation suffix
    # (assistant header / thinking block), because student tokens already carry
    # the user+assistant turns that follow this system message.
    context_messages: list[renderers.Message] = [
        {"role": "system", "content": LANGUAGE_CLASSIFICATION_PROMPT.format(text="")},
    ]
    context_input, _ = renderer.build_supervised_example(
        context_messages, train_on_what=renderers.TrainOnWhat.ALL_TOKENS
    )
    context_tokens = context_input.to_ints()

    logger.info(f"Teacher context rendered to {len(context_tokens)} tokens")
    return context_tokens


@scope
async def main_async(cli_config: CLIConfig):
    """Main entry point for on-policy context distillation."""

    # Resolve renderer
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    # Create log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"ctx-distill-onpolicy-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/context_distillation/{run_name}")

    wandb_name = cli_config.wandb_name or os.path.basename(log_path)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # ---- Render teacher context tokens ----
    teacher_context_tokens = render_teacher_context(renderer_name, cli_config.model_name)

    # ---- Set up logging ----
    # Build the Config for the on-policy training loop
    dataset_builder = FilePromptDatasetBuilder(
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

    # ---- Replicate train_on_policy.main() but wrap teacher clients ----
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

        # ---- KEY DIFFERENCE: Wrap teacher with context-aware client ----
        wrapped_teacher = ContextAwareSamplingClient(teacher_client, teacher_context_tokens)
        teacher_clients.append(wrapped_teacher)
        logger.info(
            f"Created context-aware teacher for {tc.base_model} "
            f"(context_len={len(teacher_context_tokens)} tokens, "
            f"checkpoint={tc.load_checkpoint_path})"
        )

    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    num_batches = min(cfg.max_step, num_batches) if cfg.max_step is not None else num_batches
    logger.info(f"Will train on {num_batches} batches")

    # Training loop (reuses the existing do_sync_training)
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
    logger.info("On-policy context distillation completed successfully")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main_async(cli_config))
