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
  - Paper trains offline DPO on pre-scored preference pairs.
  - This prototype runs **online** GRPO, validating whether the checklist reward
    signal is sufficient to improve instruction-following on a small scale.

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

    # Load wildchecklists (51k WildChat conversations with pre-computed checklists).
    # Columns used: `prompt` (user instruction) and `requirements` (checklist text).
    # `chosen`/`rejected`/scores are ignored — we do online RL, not offline DPO.
    logger.info("Loading viswavi/wildchecklists ...")
    raw_dataset = datasets.load_dataset("viswavi/wildchecklists", split="train")
    assert isinstance(raw_dataset, datasets.Dataset)
    train_dataset = raw_dataset.shuffle(seed=config.dataset_seed)
    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Dataset: {len(train_dataset)} examples → {n_train_batches} batches")

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
        # Phase 2: Collect responses, score via fixed judge, build datums
        # ------------------------------------------------------------------
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_skipped = 0

        for future, prompt, instruction, requirements in tqdm(
            zip(futures_P, prompts_P, instructions_P, requirements_P),
            total=len(futures_P),
            desc=f"Batch {batch_idx}",
        ):
            sample_result = future.result()

            # Collect G responses from the policy
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

            # Submit G judge-scoring futures concurrently (one per response).
            # Each call samples n_judge_samples outputs from the frozen judge
            # and averages them — matching the paper's score-averaging strategy.
            judge_futures_G: list[Future[types.SampleResponse]] = [
                _submit_judge_future(
                    judge_client=judge_client,
                    renderer=renderer,
                    instruction=instruction,
                    requirements=requirements,
                    response=response_text,
                    judge_params=judge_params,
                    n_samples=config.n_judge_samples,
                )
                for response_text in response_texts_G
            ]

            # Collect rewards (blocks until each judge call completes)
            rewards_G = [_collect_reward(jf, renderer) for jf in judge_futures_G]

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            # Skip groups where all responses are equally (un)rewarded — no gradient signal.
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
