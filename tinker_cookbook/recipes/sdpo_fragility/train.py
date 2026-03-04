"""
SDPO Fragility Experiment – main training loop.

Implements an on-policy distillation loop on LiveCodeBench that mirrors the
SDPO paper's training setup, inserting SDPO-style "teacher-from-feedback"
credit assignment into the Tinker on-policy distillation pipeline.

The experiment compares five conditions:
  1. Control (no feedback, frozen teacher)
  2. Feedback + token-level distillation, frozen teacher
  3. Feedback + logit-level distillation, frozen teacher
  4. Feedback + token-level distillation, current (bootstrapped) teacher
  5. Feedback + logit-level distillation, current (bootstrapped) teacher

Plus a small sweep over learning rate, K, and temperature.

Usage:
    python -m tinker_cookbook.recipes.sdpo_fragility.train \\
        model_name=meta-llama/Llama-3.1-8B-Instruct \\
        log_path=/tmp/sdpo_fragility/run1 \\
        teacher_mode=frozen \\
        dist_type=token_level \\
        wandb_project=sdpo_fragility

See Config for all parameters.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log

from .distillation_loss import build_distillation_datums, distillation_loss_fn
from .environment import RolloutResult, evaluate_rollout
from .lcb_dataset import LCBProblem, load_lcb_problems
from .teacher import (
    DistillationType,
    TeacherLogprobResult,
    TeacherMode,
    build_teacher_messages,
    get_teacher_logprobs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class Config:
    """Configuration for the SDPO fragility experiment."""

    # ----- Model -----
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # ----- Experiment condition -----
    teacher_mode: TeacherMode = TeacherMode.FROZEN
    dist_type: DistillationType = DistillationType.TOKEN_LEVEL
    # K for logit-level distillation; ignored for TOKEN_LEVEL
    topk: int = 20

    # ----- Training -----
    learning_rate: float = 1e-6
    question_batch_size: int = 32    # Problems sampled per training step (P)
    rollouts_per_question: int = 8   # Rollouts per problem per step (G)
    max_tokens: int = 2048           # Max response tokens per rollout
    temperature: float = 1.0        # Sampling temperature
    n_steps: int = 80               # Total training steps

    # Adam optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # ----- Evaluation -----
    eval_every: int = 10              # Steps between validation runs (0 = disabled)
    val_rollouts_per_problem: int = 4  # Rollouts per problem during validation
    max_val_problems: int = 50        # Cap on validation problems (speed)

    # ----- Dataset -----
    dataset_seed: int = 42
    dataset_split_seed: int = 42  # Independent seed for public/private split

    # ----- Infra -----
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    sandbox_backend: SandboxBackend = SandboxBackend.SANDBOXFUSION
    sandbox_timeout: int = 6
    base_url: str | None = None

    # ----- Logging -----
    wandb_project: str | None = None
    wandb_name: str | None = None


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class StudentRollout:
    """A single student-generated response with its token IDs."""
    response_text: str
    token_ids: list[int]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


async def _sample_student_rollout(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    max_tokens: int,
    temperature: float,
    tokenizer: Tokenizer,
) -> StudentRollout:
    """Sample one rollout from the student model."""
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    token_ids = list(response.sequences[0].tokens)
    response_text = tokenizer.decode(token_ids)
    return StudentRollout(response_text=response_text, token_ids=token_ids)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


async def _run_training_step(
    step: int,
    batch_problems: list[LCBProblem],
    config: Config,
    training_client: tinker.TrainingClient,
    student_sampling_client: tinker.SamplingClient,
    teacher_client: tinker.SamplingClient,
    renderer: Renderer,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    """Execute one complete SDPO training step.

    Returns a dict of training metrics.
    """
    G = config.rollouts_per_question

    # ------------------------------------------------------------------ #
    # A: Student rollouts – P×G in parallel                               #
    # ------------------------------------------------------------------ #
    rollout_tasks = []
    for prob in batch_problems:
        prompt_mi = renderer.build_generation_prompt(
            [{"role": "user", "content": prob.question_text}]
        )
        for _ in range(G):
            rollout_tasks.append(
                _sample_student_rollout(
                    student_sampling_client,
                    prompt_mi,
                    config.max_tokens,
                    config.temperature,
                    tokenizer,
                )
            )

    t0 = time.time()
    all_rollouts: list[StudentRollout] = list(await asyncio.gather(*rollout_tasks))
    logger.debug("Sampling %d rollouts took %.1fs", len(all_rollouts), time.time() - t0)

    # ------------------------------------------------------------------ #
    # B: Evaluate against public tests                                     #
    # ------------------------------------------------------------------ #
    eval_tasks = []
    for p_idx, prob in enumerate(batch_problems):
        for g_idx in range(G):
            r = all_rollouts[p_idx * G + g_idx]
            eval_tasks.append(
                evaluate_rollout(
                    r.response_text,
                    prob.public_tests,
                    config.sandbox_backend,
                    config.sandbox_timeout,
                )
            )

    t0 = time.time()
    all_results: list[RolloutResult] = list(await asyncio.gather(*eval_tasks))
    logger.debug("Evaluation took %.1fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # C: Teacher logprobs on student tokens                                #
    # ------------------------------------------------------------------ #
    teacher_tasks = []
    for p_idx, prob in enumerate(batch_problems):
        for g_idx in range(G):
            idx = p_idx * G + g_idx
            rollout = all_rollouts[idx]
            result = all_results[idx]

            teacher_msgs = build_teacher_messages(
                problem_text=prob.question_text,
                student_code=result.code,
                feedback_text=result.feedback_text,
                teacher_mode=config.teacher_mode,
            )
            teacher_tasks.append(
                get_teacher_logprobs(
                    teacher_client=teacher_client,
                    renderer=renderer,
                    teacher_messages=teacher_msgs,
                    student_output_tokens=rollout.token_ids,
                    dist_type=config.dist_type,
                    topk=config.topk,
                )
            )

    t0 = time.time()
    all_teacher: list[TeacherLogprobResult] = list(await asyncio.gather(*teacher_tasks))
    logger.debug("Teacher logprob queries took %.1fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # D: Build training Datums                                             #
    # ------------------------------------------------------------------ #
    all_data: list[tinker.Datum] = []
    for p_idx, prob in enumerate(batch_problems):
        prompt_mi = renderer.build_generation_prompt(
            [{"role": "user", "content": prob.question_text}]
        )
        for g_idx in range(G):
            idx = p_idx * G + g_idx
            rollout = all_rollouts[idx]
            teacher = all_teacher[idx]

            datums = build_distillation_datums(
                prompt_tokens=prompt_mi,
                student_output_tokens=rollout.token_ids,
                teacher_result=teacher,
                dist_type=config.dist_type,
                topk=config.topk,
            )
            all_data.extend(datums)

    if not all_data:
        logger.warning("Step %d: no valid training datums built; skipping update.", step)
        rewards = [r.reward for r in all_results]
        return {
            "mean_reward": mean(rewards) if rewards else 0.0,
            "fraction_correct": sum(r == 1.0 for r in rewards) / max(len(rewards), 1),
            "n_datums": 0,
        }

    # ------------------------------------------------------------------ #
    # E: Forward-backward + optimizer step (pipelined)                    #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    # Submit both before awaiting either (pipelining on the same clock cycle)
    bwd_future = await training_client.forward_backward_custom_async(all_data, distillation_loss_fn)
    optim_future = await training_client.optim_step_async(
        tinker.AdamParams(
            learning_rate=config.learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )
    )
    bwd_result = await bwd_future.result_async()
    await optim_future.result_async()
    logger.debug("Forward-backward + optim took %.1fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # F: Aggregate metrics                                                 #
    # ------------------------------------------------------------------ #
    rewards = [r.reward for r in all_results]
    response_lengths = [len(r.token_ids) for r in all_rollouts]
    error_types = [r.error_type for r in all_results]

    metrics: dict[str, Any] = {
        "mean_reward": mean(rewards),
        "fraction_correct": sum(r == 1.0 for r in rewards) / len(rewards),
        "mean_response_length": mean(response_lengths),
        "n_datums": len(all_data),
        "n_rollouts": len(all_rollouts),
        "frac_no_code": sum(e == "no_code" for e in error_types) / len(error_types),
        "frac_runtime_error": sum(e == "runtime_error" for e in error_types) / len(error_types),
        "frac_timeout": sum(e == "timeout" for e in error_types) / len(error_types),
        "frac_wrong_answer": sum(e == "wrong_answer" for e in error_types) / len(error_types),
    }

    # Add training loss metrics from forward_backward_custom result
    if hasattr(bwd_result, "metrics") and bwd_result.metrics:
        metrics.update({f"train/{k}": v for k, v in bwd_result.metrics.items()})

    return metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def _validate(
    val_problems: list[LCBProblem],
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    tokenizer: Tokenizer,
    n_rollouts: int,
    max_tokens: int,
    temperature: float,
    sandbox_backend: SandboxBackend,
    sandbox_timeout: int,
) -> dict[str, Any]:
    """Compute pass@1 and pass@k on private tests.

    pass@1: fraction of problems where at least 1 of n_rollouts passes.
    pass@k (conservative): fraction of problems where ALL rollouts pass.
    """
    rollout_tasks = []
    for prob in val_problems:
        prompt_mi = renderer.build_generation_prompt(
            [{"role": "user", "content": prob.question_text}]
        )
        for _ in range(n_rollouts):
            rollout_tasks.append(
                _sample_student_rollout(
                    sampling_client, prompt_mi, max_tokens, temperature, tokenizer
                )
            )

    all_rollouts: list[StudentRollout] = list(await asyncio.gather(*rollout_tasks))

    eval_tasks = []
    for p_idx, prob in enumerate(val_problems):
        for r_idx in range(n_rollouts):
            rollout = all_rollouts[p_idx * n_rollouts + r_idx]
            eval_tasks.append(
                evaluate_rollout(
                    rollout.response_text,
                    prob.private_tests,
                    sandbox_backend,
                    sandbox_timeout,
                )
            )

    all_results: list[RolloutResult] = list(await asyncio.gather(*eval_tasks))

    problem_pass_any: list[bool] = []
    problem_pass_all: list[bool] = []
    for p_idx in range(len(val_problems)):
        results_for_prob = all_results[p_idx * n_rollouts : (p_idx + 1) * n_rollouts]
        rewards_for_prob = [r.reward for r in results_for_prob]
        problem_pass_any.append(any(r == 1.0 for r in rewards_for_prob))
        problem_pass_all.append(all(r == 1.0 for r in rewards_for_prob))

    n = len(val_problems)
    return {
        "val_pass_at_1": sum(problem_pass_any) / max(n, 1),
        "val_pass_at_k": sum(problem_pass_all) / max(n, 1),
        "val_n_problems": n,
        "val_n_rollouts": n_rollouts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(config: Config) -> None:
    """Run the SDPO fragility experiment."""
    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    run_name = (
        f"sdpo_fragility"
        f"-{config.teacher_mode.value}"
        f"-{config.dist_type.value}"
        f"-topk{config.topk}"
        f"-lr{config.learning_rate}"
        f"-temp{config.temperature}"
        f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name or run_name,
        config=config,
    )
    ml_logger.log_hparams(config)

    # Tinker clients
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    if config.load_checkpoint_path:
        logger.info("Loading checkpoint from %s", config.load_checkpoint_path)
        await training_client.load_state_async(config.load_checkpoint_path)

    # Renderer
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Initial student sampling client
    student_sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        "step_0"
    )

    # Frozen teacher snapshot (for FROZEN mode only)
    frozen_teacher_client: tinker.SamplingClient | None = None
    if config.teacher_mode == TeacherMode.FROZEN:
        frozen_teacher_client = await training_client.save_weights_and_get_sampling_client_async(
            "frozen_teacher"
        )

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    manifest_path = os.path.join(config.log_path, "lcb_split_manifest.json")
    all_problems = load_lcb_problems(
        seed=config.dataset_seed,
        split_seed=config.dataset_split_seed,
        manifest_path=manifest_path,
    )
    logger.info(
        "Loaded %d LCB problems (source: %s)",
        len(all_problems),
        all_problems[0].source_dataset if all_problems else "?",
    )

    # Separate a validation set
    val_problems = all_problems[: config.max_val_problems]
    # Cycle through all problems for training
    problem_cycle = itertools.cycle(all_problems)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    total_generations = 0

    for step in range(config.n_steps):
        step_start = time.time()

        # Sample this step's problem batch
        batch_problems = [next(problem_cycle) for _ in range(config.question_batch_size)]

        # Select teacher client
        if config.teacher_mode == TeacherMode.FROZEN:
            assert frozen_teacher_client is not None
            teacher_client = frozen_teacher_client
        else:
            # CURRENT or NO_FEEDBACK: use the student's current weights as teacher
            teacher_client = student_sampling_client

        # Run one training step
        step_metrics = await _run_training_step(
            step=step,
            batch_problems=batch_problems,
            config=config,
            training_client=training_client,
            student_sampling_client=student_sampling_client,
            teacher_client=teacher_client,
            renderer=renderer,
            tokenizer=tokenizer,
        )

        # Update student sampling client to current weights for next step
        # (also serves as the current-teacher at the next step in CURRENT mode)
        student_sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            f"step_{step + 1}"
        )

        total_generations += config.question_batch_size * config.rollouts_per_question
        step_time = time.time() - step_start

        train_metrics: dict[str, Any] = {
            "step": step,
            "n_generations": total_generations,
            "step_time_s": step_time,
            **step_metrics,
        }
        ml_logger.log_metrics(train_metrics, step=step)

        logger.info(
            "Step %d/%d | gen=%d | reward=%.3f | frac_correct=%.3f | loss=%.4f",
            step + 1,
            config.n_steps,
            total_generations,
            float(step_metrics.get("mean_reward", float("nan"))),
            float(step_metrics.get("fraction_correct", float("nan"))),
            float(step_metrics.get("train/distill_loss", float("nan"))),
        )

        # Periodic validation on private tests
        if config.eval_every > 0 and step % config.eval_every == 0:
            logger.info("Running validation at step %d …", step)
            val_metrics = await _validate(
                val_problems=val_problems,
                sampling_client=student_sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                n_rollouts=config.val_rollouts_per_problem,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                sandbox_backend=config.sandbox_backend,
                sandbox_timeout=config.sandbox_timeout,
            )
            val_metrics["n_generations"] = total_generations
            ml_logger.log_metrics(val_metrics, step=step)
            logger.info(
                "Validation | pass@1=%.3f | pass@k=%.3f",
                val_metrics["val_pass_at_1"],
                val_metrics["val_pass_at_k"],
            )

    logger.info("Training complete. Total generations: %d", total_generations)
    ml_logger.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli_main(cli_config: Config) -> None:
    asyncio.run(main(cli_config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
