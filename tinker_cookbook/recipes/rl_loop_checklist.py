"""
Reinforcement Learning from Checklist Feedback (RLCF) — Minimal Research Prototype.

Implements the core idea from arXiv:2507.18624 using the Tinker API:

  - Uses pre-computed instruction-specific checklists from `viswavi/wildchecklists`.
  - Trains a policy with online GRPO (the same importance-sampling loss as rl_loop.py).
  - Reward signal: a **fixed** judge model scores each policy response against the
    checklist and two universal anti-hacking criteria (paper §3.2, Appendix A).
  - The judge is frozen at training start (initial LoRA = identity ≈ base weights),
    matching the paper's fixed-evaluator assumption.

Divergence from the paper's original pipeline:
  - Paper trains offline DPO on pre-scored preference pairs; see dpo_checklist.py.
  - This prototype runs **online** GRPO, validating whether the checklist reward
    signal is sufficient to improve instruction-following on a small scale.

Judge batching (amortising costs):
  - All P×G policy responses are collected before any judge call is submitted.
  - All P×G judge futures are then submitted simultaneously so the Tinker
    sampling server (backed by vLLM) can schedule one continuous batch — the
    same principle as GeneralVerifier.verify_batch() in the General Reasoner.
  - vLLM is not thread-safe; the Tinker client handles serialisation internally,
    so callers just submit futures and await results.
  - The judge model is loaded once at script start (frozen reference weights)
    and reused for all batches — matching the General Reasoner's "load once" pattern.

Variable naming convention (see CONTRIBUTING.md):
    _P: Problem dimension (different prompts in a batch)
    _G: Group dimension (multiple rollouts per problem)
    _T: Token/Time dimension (sequence positions)
    _D: Datum dimension (flattened training examples)

Usage::

    python -m tinker_cookbook.recipes.rl_loop_checklist \\
        model_name=Qwen/Qwen3-8B \\
        batch_size=32 \\
        group_size=4 \\
        n_judge_samples=5

For a quick smoke-test (no GPU budget needed)::

    python -m tinker_cookbook.recipes.rl_loop_checklist \\
        batch_size=2 group_size=2 n_judge_samples=1 save_every=0
"""

import logging
import re
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# ---------------------------------------------------------------------------
# Prompts — faithful to RLCF paper (Appendix C, §3.2)
# ---------------------------------------------------------------------------

# Two universal requirements appended to every checklist to prevent reward
# hacking (models gaming narrow checklist items at the expense of quality).
# Source: paper §3.2 / Appendix A.
_UNIVERSAL_REQUIREMENTS = (
    "* Does the response directly address the request without excessive or "
    "off-topic information not necessary for addressing the user's instruction? "
    "(importance: 100/100)\n"
    "* Does the response match the context and the instruction, whether it "
    "requires professionalism, friendliness, formality, or neutrality? "
    "(importance: 100/100)"
)

# Judge prompt structure mirrors paper Appendix C.
_JUDGE_PROMPT = """\
You are evaluating a response to a user instruction against a set of quality requirements.

User instruction:
{instruction}

Requirements (each is a yes/no quality criterion):
{requirements}
{universal_requirements}

Response to evaluate:
{response}

How well does the response satisfy ALL requirements above?
Output a single integer score from 0 (completely fails) to 100 (fully satisfies all requirements). \
Output ONLY the integer, nothing else."""

# System prompt for the policy model (instruction-following setup).
_POLICY_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest assistant. "
    "Follow the user's instructions carefully and completely."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-loop-checklist"
    model_name: str = "Qwen/Qwen3-8B"  # paper uses Qwen family; instruct variant recommended
    batch_size: int = 32  # smaller than GSM8K due to additional judge-scoring latency
    group_size: int = 4  # paper §4.1 uses 4–8; 4 balances signal vs. cost
    learning_rate: float = 4e-5  # same as General Reasoner reference script
    lora_rank: int = 32
    save_every: int = 20  # checkpoint interval in batches; 0 = disabled
    max_tokens: int = 512  # policy response budget (instruction-following, not just math)
    judge_max_tokens: int = 16  # judge only needs to output a single integer
    n_judge_samples: int = 5  # paper uses 25; 5 is fast enough for prototype validation
    judge_temperature: float = 0.7  # mild variance so averaging across samples is meaningful
    eval_n_examples: int = 200  # held-out eval size; same default as dpo_checklist.py
    dataset_seed: int = 42
    ttl_seconds: int | None = 604800  # 7 days


# ---------------------------------------------------------------------------
# Reward utilities
# ---------------------------------------------------------------------------


def _parse_score(text: str) -> float:
    """Extract first integer from judge output, clamp to [0, 100], normalise to [0, 1]."""
    match = re.search(r"\d+", text.strip())
    if not match:
        return 0.5  # neutral fallback when judge output cannot be parsed
    return min(max(int(match.group()), 0), 100) / 100.0


def _submit_judge_future(
    judge_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    instruction: str,
    requirements: str,
    response: str,
    judge_params: types.SamplingParams,
    n_samples: int,
) -> "Future[types.SampleResponse]":
    """Build a judge prompt and submit an async scoring request.

    Returns a Future immediately; call ``.result()`` to block and retrieve the
    raw ``SampleResponse``.  Keeping submission and collection separate lets us
    fire off all G judge calls concurrently for a given problem.
    """
    judge_text = _JUDGE_PROMPT.format(
        instruction=instruction,
        requirements=requirements,
        universal_requirements=_UNIVERSAL_REQUIREMENTS,
        response=response,
    )
    judge_input = renderer.build_generation_prompt(
        [
            {"role": "system", "content": "You are a strict evaluator. Output ONLY a number."},
            {"role": "user", "content": judge_text},
        ]
    )
    return judge_client.sample(
        prompt=judge_input,
        num_samples=n_samples,
        sampling_params=judge_params,
    )


def _collect_reward(
    judge_future: "Future[types.SampleResponse]",
    renderer: renderers.Renderer,
) -> float:
    """Block on a judge future and return the mean score across ``n_samples``."""
    result = judge_future.result()
    scores: list[float] = []
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        text = renderers.get_text_content(parsed_msg)
        scores.append(_parse_score(text))
    return sum(scores) / len(scores) if scores else 0.5


def _score_eval_set(
    eval_ds: "datasets.Dataset",
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    policy_params: types.SamplingParams,
    judge_params: types.SamplingParams,
    n_judge_samples: int,
) -> float:
    """Sample one response per eval instruction, score via judge, return mean.

    Uses the same judge prompt and parameters as the training reward to ensure
    the eval metric is directly comparable to `reward/mean` in the training loop
    and to the `eval/checklist_score` produced by dpo_checklist.py.
    """
    policy_futures: list[Future[types.SampleResponse]] = []
    for instruction in eval_ds["prompt"]:
        model_input = renderer.build_generation_prompt(
            [
                {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        )
        policy_futures.append(
            sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=policy_params,
            )
        )

    scores: list[float] = []
    for future, instruction, requirements in tqdm(
        zip(policy_futures, eval_ds["prompt"], eval_ds["requirements"]),
        total=len(policy_futures),
        desc="Eval scoring",
    ):
        seq = future.result().sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        response_text = renderers.get_text_content(parsed_msg)
        judge_future = _submit_judge_future(
            judge_client=sampling_client,
            renderer=renderer,
            instruction=instruction,
            requirements=requirements,
            response=response_text,
            judge_params=judge_params,
            n_samples=n_judge_samples,
        )
        scores.append(_collect_reward(judge_future, renderer))
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(config: Config) -> None:
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load wildchecklists. Split 90/10 train/eval with the same seed as dpo_checklist.py
    # so both scripts evaluate on the same held-out examples for direct comparison.
    # Only `prompt` and `requirements` are used during training; `chosen`/`rejected`
    # are ignored because we generate responses online via GRPO.
    logger.info("Loading viswavi/wildchecklists ...")
    raw_dataset = datasets.load_dataset("viswavi/wildchecklists", split="train")
    assert isinstance(raw_dataset, datasets.Dataset)
    split = raw_dataset.train_test_split(test_size=0.1, seed=config.dataset_seed)
    train_dataset = split["train"].shuffle(seed=config.dataset_seed)
    eval_dataset = split["test"].select(
        range(min(config.eval_n_examples, len(split["test"])))
    )
    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(
        f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval → {n_train_batches} batches"
    )

    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_batch = resume_info.batch or 0
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    # Fixed judge: save weights once before the loop starts.
    # Initial LoRA is the identity transform (≈ base weights), so the judge stays
    # frozen throughout training — consistent with the paper's fixed evaluator.
    logger.info("Saving initial weights for fixed judge client ...")
    judge_client = training_client.save_weights_and_get_sampling_client()

    policy_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    judge_params = types.SamplingParams(
        max_tokens=config.judge_max_tokens,
        temperature=config.judge_temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # ------------------------------------------------------------------
    # Baseline evaluation (base model before any GRPO training)
    # ------------------------------------------------------------------
    if start_batch == 0:
        logger.info("Measuring baseline checklist score ...")
        baseline_score = _score_eval_set(
            eval_ds=eval_dataset,
            sampling_client=judge_client,
            renderer=renderer,
            policy_params=policy_params,
            judge_params=judge_params,
            n_judge_samples=config.n_judge_samples,
        )
        ml_logger.log_metrics({"eval/checklist_score": baseline_score}, step=-1)
        logger.info(f"Baseline checklist score: {baseline_score:.4f}")

    logger.info(f"Training for {n_train_batches} batches")

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Push current LoRA weights so the policy sampling client is up to date.
        policy_client = training_client.save_weights_and_get_sampling_client()

        # ------------------------------------------------------------------
        # Phase 1: Submit policy sampling futures for all P problems
        # ------------------------------------------------------------------
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []
        instructions_P: list[str] = []
        requirements_P: list[str] = []

        for instruction, requirements in zip(
            batch_rows["prompt"], batch_rows["requirements"]
        ):
            convo = [
                {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
            model_input = renderer.build_generation_prompt(convo)
            future = policy_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=policy_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)
            instructions_P.append(instruction)
            requirements_P.append(requirements)

        # ------------------------------------------------------------------
        # Phase 2a: Await all P policy futures and collect raw response data.
        # We must await each future before submitting judge calls, but we
        # collect *all* responses before submitting *any* judge call so that
        # the full P×G batch hits the judge server in one shot.
        # ------------------------------------------------------------------

        # Each entry: (prompt, tokens_G_T, logprobs_G_T, texts_G, instruction, requirements)
        _ProblemData = tuple[
            types.ModelInput,
            list[list[int]],
            list[list[float]],
            list[str],
            str,
            str,
        ]
        problems_data: list[_ProblemData] = []

        for future, prompt, instruction, requirements in tqdm(
            zip(futures_P, prompts_P, instructions_P, requirements_P),
            total=len(futures_P),
            desc=f"Collecting responses batch {batch_idx}",
        ):
            sample_result = future.result()

            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            response_texts_G: list[str] = []

            for seq in sample_result.sequences:
                sampled_tokens = seq.tokens
                sampled_logprobs = seq.logprobs
                assert sampled_logprobs is not None

                parsed_msg, _ = renderer.parse_response(sampled_tokens)
                response_text = renderers.get_text_content(parsed_msg)

                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)
                response_texts_G.append(response_text)

            problems_data.append(
                (prompt, sampled_tokens_G_T, logprobs_G_T, response_texts_G, instruction, requirements)
            )

        # ------------------------------------------------------------------
        # Phase 2b: Submit ALL P×G judge futures simultaneously (true batch).
        # The Tinker sampling client is backed by vLLM; submitting all futures
        # before awaiting any lets vLLM schedule one continuous batch, the same
        # principle as GeneralVerifier.verify_batch() in the General Reasoner.
        # ------------------------------------------------------------------
        judge_futures_flat: list[Future[types.SampleResponse]] = []
        group_sizes: list[int] = []  # number of responses per problem (for regrouping)

        for _, _, _, response_texts_G, instruction, requirements in problems_data:
            group_sizes.append(len(response_texts_G))
            for response_text in response_texts_G:
                judge_futures_flat.append(
                    _submit_judge_future(
                        judge_client=judge_client,
                        renderer=renderer,
                        instruction=instruction,
                        requirements=requirements,
                        response=response_text,
                        judge_params=judge_params,
                        n_samples=config.n_judge_samples,
                    )
                )

        # ------------------------------------------------------------------
        # Phase 2c: Collect all P×G rewards (now the server can batch them).
        # ------------------------------------------------------------------
        all_rewards_flat = [_collect_reward(jf, renderer) for jf in judge_futures_flat]

        # ------------------------------------------------------------------
        # Phase 2d: Regroup per problem, compute advantages, build datums.
        # ------------------------------------------------------------------
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_skipped = 0
        reward_idx = 0

        for (prompt, sampled_tokens_G_T, logprobs_G_T, _, _, _), g_size in zip(
            problems_data, group_sizes
        ):
            rewards_G = all_rewards_flat[reward_idx : reward_idx + g_size]
            reward_idx += g_size

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            # Skip groups where all responses are equally rewarded — no gradient signal.
            if all(adv == 0.0 for adv in advantages_G):
                n_skipped += 1
                continue

            # Build one training datum per group member (identical to rl_loop.py).
            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(
                    types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                )
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = (
                    [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                )
                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                ), (
                    f"Length mismatch: model_input={model_input.length}, "
                    f"target_tokens={len(target_tokens)}, "
                    f"logprobs={len(padded_logprobs)}, "
                    f"advantages={len(padded_advantages)}"
                )
                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(padded_logprobs)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(padded_advantages)
                        ),
                    },
                )
                datums_D.append(datum)

        # ------------------------------------------------------------------
        # Phase 3: Gradient step (pipelined forward_backward + optim_step)
        # ------------------------------------------------------------------
        fwd_bwd_future = training_client.forward_backward(
            datums_D, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        # Metrics
        n_problems = len(rewards_P)
        mean_reward = sum(rewards_P) / n_problems if rewards_P else 0.0
        reward_variance = (
            sum((r - mean_reward) ** 2 for r in rewards_P) / n_problems
            if n_problems > 1
            else 0.0
        )
        metrics["reward/mean"] = mean_reward
        metrics["reward/std"] = reward_variance**0.5
        metrics["reward/nonzero_frac"] = (
            1.0 - n_skipped / n_problems if n_problems > 0 else 0.0
        )
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=batch_idx)

    # ------------------------------------------------------------------
    # Post-training evaluation (same eval set, current policy weights)
    # ------------------------------------------------------------------
    logger.info("Measuring post-GRPO checklist score ...")
    final_policy_client = training_client.save_weights_and_get_sampling_client()
    post_score = _score_eval_set(
        eval_ds=eval_dataset,
        sampling_client=final_policy_client,
        renderer=renderer,
        policy_params=policy_params,
        judge_params=judge_params,
        n_judge_samples=config.n_judge_samples,
    )
    ml_logger.log_metrics({"eval/checklist_score": post_score}, step=n_train_batches)
    logger.info(f"Post-GRPO checklist score: {post_score:.4f}")

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
