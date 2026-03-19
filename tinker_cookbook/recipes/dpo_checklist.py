"""
RLCF — Faithful DPO Reproduction (arXiv 2507.18624).

The paper pipeline (already completed offline):
  1. Generate instruction-specific checklists from candidate responses
  2. Score chosen/rejected pairs against checklists → `chosen_score`, `rejected_score`
  3. Filter to top 40 % of pairs by score gap
  4. Train with DPO

`viswavi/wildchecklists` ships steps 1-3 pre-computed, so this script
implements step 4 directly, with one addition for scientific rigour:
a held-out checklist-score baseline measured on the base model before any
training, using the same frozen reference model that DPO already requires.

Cost note (see docs/async.mdx for clock-cycle model):
  DPO (~2 clock cycles/step): ref-logprobs forward + forward_backward_custom
  GRPO online (rl_loop_checklist.py, ~5+ cycles/step): save_weights + sample
    G responses + judge P×G responses + forward_backward + optim_step
  ⇒ DPO is ~5× cheaper for this pre-annotated dataset and is the paper's
  actual training algorithm.

Comparable metric: both scripts log `eval/checklist_score` on the same
held-out eval set so results can be compared directly.

Usage::

    python -m tinker_cookbook.recipes.dpo_checklist \\
        model_name=Qwen/Qwen3-8B-Instruct \\
        batch_size=64 dpo_beta=0.1

Quick smoke test (no real API budget needed)::

    python -m tinker_cookbook.recipes.dpo_checklist \\
        batch_size=2 eval_n_examples=4 n_judge_samples=1 save_every=0
"""

import asyncio
import logging
import re
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
import torch.nn.functional as F
from tinker import types
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# ---------------------------------------------------------------------------
# Prompts — same constants as rl_loop_checklist.py for comparable eval
# ---------------------------------------------------------------------------

_UNIVERSAL_REQUIREMENTS = (
    "* Does the response directly address the request without excessive or "
    "off-topic information not necessary for addressing the user's instruction? "
    "(importance: 100/100)\n"
    "* Does the response match the context and the instruction, whether it "
    "requires professionalism, friendliness, formality, or neutrality? "
    "(importance: 100/100)"
)

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
    log_path: str = "/tmp/tinker-examples/dpo-checklist"
    model_name: str = "Qwen/Qwen3-8B-Instruct"  # paper uses Qwen2.5-7B-Instruct
    batch_size: int = 64           # standard DPO batch; no per-step sampling
    learning_rate: float = 1e-5   # paper §4.1 default for DPO
    dpo_beta: float = 0.1         # paper §4.1 default
    lora_rank: int = 32
    max_length: int = 2048         # truncation for long conversations
    save_every: int = 20           # 0 = disabled
    # Eval / baseline
    max_tokens: int = 512          # response budget when sampling for eval
    n_judge_samples: int = 5       # paper uses 25; 5 for prototype speed
    judge_max_tokens: int = 16     # judge only needs to output a single integer
    judge_temperature: float = 0.7 # variance for averaging judge scores
    eval_n_examples: int = 200     # held-out eval size (baseline + post-DPO)
    # Dataset
    min_score_gap: float = 0.1    # paper keeps top 40 % by gap; 0.1 approximates this
    dataset_seed: int = 42
    ttl_seconds: int | None = 604800  # 7 days


# ---------------------------------------------------------------------------
# Reward / judge utilities (identical logic to rl_loop_checklist.py)
# ---------------------------------------------------------------------------


def _parse_score(text: str) -> float:
    match = re.search(r"\d+", text.strip())
    if not match:
        return 0.5
    return min(max(int(match.group()), 0), 100) / 100.0


def _submit_judge_future(
    judge_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    instruction: str,
    requirements: str,
    response: str,
    n_samples: int,
    max_tokens: int,
    temperature: float,
) -> "Future[types.SampleResponse]":
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
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )


def _collect_score(
    judge_future: "Future[types.SampleResponse]",
    renderer: renderers.Renderer,
) -> float:
    result = judge_future.result()
    scores: list[float] = []
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        scores.append(_parse_score(renderers.get_text_content(parsed_msg)))
    return sum(scores) / len(scores) if scores else 0.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_and_split(config: Config) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load wildchecklists, filter by score gap, split 90/10 train/eval."""
    logger.info("Loading viswavi/wildchecklists ...")
    raw: datasets.Dataset = datasets.load_dataset("viswavi/wildchecklists", split="train")  # type: ignore[assignment]

    # Filter pairs with enough preference signal (paper: top 40 % by gap)
    raw = raw.filter(
        lambda row: (row["chosen_score"] - row["rejected_score"]) >= config.min_score_gap
    )
    logger.info(f"After score-gap filter (≥{config.min_score_gap}): {len(raw)} pairs")

    split = raw.train_test_split(test_size=0.1, seed=config.dataset_seed)
    train_ds: datasets.Dataset = split["train"]
    eval_ds: datasets.Dataset = split["test"].select(
        range(min(config.eval_n_examples, len(split["test"])))
    )
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Datum construction
# ---------------------------------------------------------------------------


def _build_dpo_datums(
    batch_rows: datasets.Dataset,
    renderer: renderers.Renderer,
    max_length: int,
) -> list[tinker.Datum]:
    """Build interleaved chosen/rejected datums for DPO.

    Returns [chosen_0, rejected_0, chosen_1, rejected_1, ...] following the
    convention used by train_dpo.py (even indices = chosen, odd = rejected).

    The `chosen`/`rejected` columns in wildchecklists are lists of message dicts.
    We extract the last assistant turn from each and pair it with the user prompt.
    """
    datums: list[tinker.Datum] = []
    for instruction, chosen_msgs, rejected_msgs in zip(
        batch_rows["prompt"], batch_rows["chosen"], batch_rows["rejected"]
    ):
        # Extract the assistant response text from the message list
        chosen_text = chosen_msgs[-1]["content"]
        rejected_text = rejected_msgs[-1]["content"]

        for response_text in [chosen_text, rejected_text]:
            full_convo = [
                {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response_text},
            ]
            model_input, weights = renderer.build_supervised_example(full_convo)
            datums.append(datum_from_model_input_weights(model_input, weights, max_length))
    return datums


# ---------------------------------------------------------------------------
# Reference log-prob computation (pattern from train_dpo.py)
# ---------------------------------------------------------------------------


async def _compute_ref_logprobs(
    data: list[tinker.Datum],
    reference_client: tinker.SamplingClient,
) -> list[torch.Tensor]:
    """Compute reference model log-probs for all datums concurrently.

    Reconstructs the full sequence (model_input + last target token) before
    calling compute_logprobs_async, then slices off the first logprob to align
    with target_tokens (same approach as train_dpo.py lines 250-269).
    """
    full_sequences: list[tinker.ModelInput] = []
    for datum in data:
        target_tokens = datum.loss_fn_inputs["target_tokens"].data
        if target_tokens:
            full_sequences.append(datum.model_input.append_int(int(target_tokens[-1])))  # type: ignore[arg-type]
        else:
            full_sequences.append(datum.model_input)

    all_logprobs = await asyncio.gather(
        *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
    )
    return [torch.tensor(lp[1:]) for lp in all_logprobs]


# ---------------------------------------------------------------------------
# DPO loss (identical to compute_dpo_loss in train_dpo.py)
# ---------------------------------------------------------------------------


def _make_dpo_loss_fn(
    ref_logprob_seqs: list[torch.Tensor],
    chosen_data: list[tinker.Datum],
    rejected_data: list[tinker.Datum],
    dpo_beta: float,
):
    """Return a closure matching the forward_backward_custom loss_fn signature."""

    def dpo_loss_fn(
        data: list[tinker.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chosen_lp_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
        rejected_lp_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]
        chosen_ref_seqs = [ref_logprob_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref_seqs = [ref_logprob_seqs[i] for i in range(1, len(data), 2)]

        chosen_logprobs: list[torch.Tensor] = []
        chosen_ref_logprobs: list[torch.Tensor] = []
        rejected_logprobs: list[torch.Tensor] = []
        rejected_ref_logprobs: list[torch.Tensor] = []

        for i in range(len(chosen_data)):
            c_w = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
            r_w = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
            chosen_logprobs.append(torch.dot(chosen_lp_seqs[i].float(), c_w.float()))
            chosen_ref_logprobs.append(torch.dot(chosen_ref_seqs[i].float(), c_w.float()))
            rejected_logprobs.append(torch.dot(rejected_lp_seqs[i].float(), r_w.float()))
            rejected_ref_logprobs.append(torch.dot(rejected_ref_seqs[i].float(), r_w.float()))

        chosen_log_ratio = torch.stack(
            [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
        )
        rejected_log_ratio = torch.stack(
            [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
        )
        loss = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio)).mean()
        accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
        margin = (dpo_beta * (chosen_log_ratio - rejected_log_ratio)).mean().item()

        return loss, {
            "dpo/loss": loss.item(),
            "dpo/accuracy": accuracy,
            "dpo/margin": margin,
            "dpo/chosen_reward": (dpo_beta * chosen_log_ratio).mean().item(),
            "dpo/rejected_reward": (dpo_beta * rejected_log_ratio).mean().item(),
        }

    return dpo_loss_fn


# ---------------------------------------------------------------------------
# Eval: checklist scoring (shared metric with rl_loop_checklist.py)
# ---------------------------------------------------------------------------


def _score_eval_set(
    eval_ds: datasets.Dataset,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    config: Config,
) -> float:
    """Sample one response per eval example, score against checklists, return mean."""
    policy_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    # Submit all policy sample futures concurrently
    policy_futures: list[Future[types.SampleResponse]] = []
    for instruction in eval_ds["prompt"]:
        model_input = renderer.build_generation_prompt(
            [
                {"role": "system", "content": _POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
        )
        policy_futures.append(
            sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=policy_params)
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

        # Score with judge (n_judge_samples averaged, same as rl_loop_checklist.py)
        judge_future = _submit_judge_future(
            judge_client=sampling_client,
            renderer=renderer,
            instruction=instruction,
            requirements=requirements,
            response=response_text,
            n_samples=config.n_judge_samples,
            max_tokens=config.judge_max_tokens,
            temperature=config.judge_temperature,
        )
        scores.append(_collect_score(judge_future, renderer))

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Main
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

    train_ds, eval_ds = _load_and_split(config)
    n_steps = len(train_ds) // config.batch_size

    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_step = resume_info.batch or 0
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    # Reference model: frozen at training start (initial LoRA ≈ base weights).
    # Reused as the DPO reference AND as the judge for the baseline eval.
    logger.info("Creating reference/judge client (frozen initial weights) ...")
    reference_client = training_client.save_weights_and_get_sampling_client("reference")

    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # ------------------------------------------------------------------
    # Baseline evaluation (base model before any DPO training)
    # ------------------------------------------------------------------
    if start_step == 0:
        logger.info("Measuring baseline checklist score ...")
        baseline_score = _score_eval_set(eval_ds, reference_client, renderer, config)
        ml_logger.log_metrics({"eval/checklist_score": baseline_score}, step=-1)
        logger.info(f"Baseline checklist score: {baseline_score:.4f}")

    logger.info(f"Training for {n_steps} DPO steps")

    # ------------------------------------------------------------------
    # DPO training loop
    # ------------------------------------------------------------------
    train_ds_shuffled = train_ds.shuffle(seed=config.dataset_seed)

    for step_idx in range(start_step, n_steps):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/step": step_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (step_idx + 1) / n_steps,
        }

        if config.save_every > 0 and step_idx % config.save_every == 0 and step_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": step_idx},
                ttl_seconds=config.ttl_seconds,
            )

        batch_start = step_idx * config.batch_size
        batch_end = min((step_idx + 1) * config.batch_size, len(train_ds_shuffled))
        batch_rows = train_ds_shuffled.select(range(batch_start, batch_end))

        # Build interleaved chosen/rejected datums
        datums_D = _build_dpo_datums(batch_rows, renderer, config.max_length)
        if not datums_D:
            logger.warning(f"Step {step_idx}: empty batch, skipping")
            continue

        chosen_data = [datums_D[i] for i in range(0, len(datums_D), 2)]
        rejected_data = [datums_D[i] for i in range(1, len(datums_D), 2)]

        # Compute reference log-probs for all datums (parallelised async)
        ref_logprob_seqs = asyncio.run(_compute_ref_logprobs(datums_D, reference_client))

        # DPO loss and gradient step
        loss_fn = _make_dpo_loss_fn(ref_logprob_seqs, chosen_data, rejected_data, config.dpo_beta)
        backward_result = training_client.forward_backward_custom(datums_D, loss_fn).result()
        optim_result = training_client.optim_step(adam_params).result()

        if backward_result.metrics:
            metrics.update(backward_result.metrics)
        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        metrics["data/pairs_in_batch"] = len(chosen_data)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=step_idx)

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------
    logger.info("Measuring post-DPO checklist score ...")
    policy_client = training_client.save_weights_and_get_sampling_client()
    post_score = _score_eval_set(eval_ds, policy_client, renderer, config)
    ml_logger.log_metrics({"eval/checklist_score": post_score}, step=n_steps)
    logger.info(f"Post-DPO checklist score: {post_score:.4f}")

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_steps},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("DPO training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
