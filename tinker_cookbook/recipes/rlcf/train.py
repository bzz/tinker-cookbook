"""
RLCF: Reinforcement Learning from Checklist Feedback — self-contained DPO loop.

Faithfully reproduces the training procedure from Viswanathan et al. (2025):
  "Checklists Are Better Than Reward Models For Aligning Language Models"
  https://arxiv.org/abs/2507.18624

Paper pipeline:
  1. Checklists generated offline for each instruction (pre-computed in viswavi/rlcf)
  2. Response pairs scored against checklists by LLM judges
  3. Chosen/rejected ranked by weighted checklist score
  4. DPO training (this script): beta=0.1, lr=3e-6, 2 epochs, max_len=2048

Structure modeled after rl_loop.py / tinker_train_general_reasoner.py:
a single self-contained training loop with no delegation to framework train mains.

Usage::

    python -m tinker_cookbook.recipes.rlcf.train

    python -m tinker_cookbook.recipes.rlcf.train \\
        dpo_beta=0.1 learning_rate=3e-6 batch_size=256 num_epochs=2
"""

import asyncio
import logging
import time
from typing import cast

import chz
import datasets
import tinker
import torch
import torch.nn.functional as F

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


# ---------------------------------------------------------------------------
# Config — paper-faithful defaults from train_rlcf.sh
# ---------------------------------------------------------------------------


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rlcf"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "viswavi/rlcf"

    # DPO hyperparameters (paper: train_rlcf.sh)
    dpo_beta: float = 0.1              # --beta 0.1
    learning_rate: float = 3e-6        # --learning_rate 3e-6
    lr_schedule: LRSchedule = "linear" # --min_lr_ratio 0.75
    num_epochs: int = 2                # --max_epochs 2
    max_length: int = 2048             # --max_len 2048
    batch_size: int = 256              # --train_batch_size 1024 (reduced for single-machine)

    lora_rank: int = 32
    save_every: int = 32               # --save_steps 32
    max_steps: int | None = None
    ttl_seconds: int | None = 604800


# ---------------------------------------------------------------------------
# Data: load viswavi/rlcf chosen/rejected pairs into Datums
# ---------------------------------------------------------------------------


def load_rlcf_dataset(
    dataset_name: str,
    renderer: renderers.Renderer,
    max_length: int,
) -> list[tinker.Datum]:
    """Load chosen/rejected pairs from viswavi/rlcf and convert to interleaved Datums.

    Returns a flat list where datums at even indices are chosen and odd indices
    are rejected, matching the convention used by the DPO loss.
    """
    ds = datasets.load_dataset(dataset_name, split="train")
    ds = cast(datasets.Dataset, ds)
    logger.info(f"Loaded {len(ds)} rows from {dataset_name}")

    datums: list[tinker.Datum] = []
    skipped = 0

    for row in ds:
        chosen = row["chosen"]
        rejected = row["rejected"]

        if not chosen or not rejected:
            skipped += 1
            continue
        if not isinstance(chosen, list) or len(chosen) < 2:
            skipped += 1
            continue

        try:
            chosen_convo: list[renderers.Message] = [
                {"role": m["role"], "content": m["content"]} for m in chosen
            ]
            rejected_convo: list[renderers.Message] = [
                {"role": m["role"], "content": m["content"]} for m in rejected
            ]

            chosen_mi, chosen_w = renderer.build_supervised_example(chosen_convo)
            rejected_mi, rejected_w = renderer.build_supervised_example(rejected_convo)

            datums.append(datum_from_model_input_weights(chosen_mi, chosen_w, max_length))
            datums.append(datum_from_model_input_weights(rejected_mi, rejected_w, max_length))
        except Exception as e:
            logger.debug(f"Skipping row: {e}")
            skipped += 1

    logger.info(f"Built {len(datums) // 2} preference pairs ({skipped} rows skipped)")
    return datums


def get_batches(
    datums: list[tinker.Datum], batch_size: int, epoch_seed: int
) -> list[list[tinker.Datum]]:
    """Shuffle pairs and split into batches (each batch has interleaved chosen/rejected)."""
    n_pairs = len(datums) // 2
    indices = list(range(n_pairs))
    rng = torch.Generator().manual_seed(epoch_seed)
    shuffled = torch.randperm(n_pairs, generator=rng).tolist()

    batches: list[list[tinker.Datum]] = []
    for start in range(0, n_pairs, batch_size):
        batch: list[tinker.Datum] = []
        for idx in shuffled[start : start + batch_size]:
            batch.append(datums[2 * idx])      # chosen
            batch.append(datums[2 * idx + 1])   # rejected
        if batch:
            batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# DPO loss (from preference/train_dpo.py, inlined for self-containment)
# ---------------------------------------------------------------------------


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()

    return loss, {
        "dpo/loss": loss.item(),
        "dpo/accuracy": accuracy,
        "dpo/margin": margin,
        "dpo/chosen_reward": chosen_rewards.mean().item(),
        "dpo/rejected_reward": rejected_rewards.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(config: Config):
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

    # Load dataset
    logger.info("Loading dataset...")
    all_datums = load_rlcf_dataset(config.dataset_name, renderer, config.max_length)
    n_pairs = len(all_datums) // 2

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_epoch = resume_info.epoch or 0
        start_batch = resume_info.batch
        logger.info(f"Resuming from epoch {start_epoch} batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_epoch = 0
        start_batch = 0

    # Reference model: snapshot of initial weights for DPO KL constraint
    reference_client = training_client.save_weights_and_get_sampling_client("reference")

    n_batches = n_pairs // config.batch_size
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    logger.info(
        f"Training: {n_pairs} pairs, {n_batches} batches/epoch, "
        f"{config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    # -- Training loop --
    global_step = start_epoch * n_batches + start_batch
    for epoch in range(start_epoch, config.num_epochs):
        batches = get_batches(all_datums, config.batch_size, epoch_seed=epoch)
        batch_start = start_batch if epoch == start_epoch else 0

        for batch_idx in range(batch_start, len(batches)):
            if config.max_steps is not None and global_step >= config.max_steps:
                break

            t_start = time.time()
            data = batches[batch_idx]

            # Checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"epoch": epoch, "batch": batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )

            # LR schedule
            learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
                lr_schedule=config.lr_schedule, step=global_step, total_steps=total_steps
            )
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
            )

            # Split into chosen/rejected
            chosen_data = [data[i] for i in range(0, len(data), 2)]
            rejected_data = [data[i] for i in range(1, len(data), 2)]

            # Reference logprobs (for KL constraint in DPO)
            full_sequences = []
            for datum in data:
                target_tokens = datum.loss_fn_inputs["target_tokens"].data
                if target_tokens:
                    full_sequences.append(datum.model_input.append_int(int(target_tokens[-1])))
                else:
                    full_sequences.append(datum.model_input)

            async def _get_ref_logprobs():
                return await asyncio.gather(
                    *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
                )

            all_ref_logprobs = asyncio.run(_get_ref_logprobs())
            all_ref_logprob_seqs = [torch.tensor(lp[1:]) for lp in all_ref_logprobs]
            chosen_ref_lp = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
            rejected_ref_lp = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

            # DPO loss closure for forward_backward_custom
            def dpo_loss_fn(
                data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
            ) -> tuple[torch.Tensor, dict[str, float]]:
                c_lp_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
                r_lp_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

                c_lps, c_ref, r_lps, r_ref = [], [], [], []
                for i in range(len(chosen_data)):
                    c_w = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                    r_w = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                    c_lps.append(torch.dot(c_lp_seqs[i].float(), c_w.float()))
                    c_ref.append(torch.dot(chosen_ref_lp[i].float(), c_w.float()))
                    r_lps.append(torch.dot(r_lp_seqs[i].float(), r_w.float()))
                    r_ref.append(torch.dot(rejected_ref_lp[i].float(), r_w.float()))

                return compute_dpo_loss(c_lps, r_lps, c_ref, r_ref, config.dpo_beta)

            # Forward-backward + optimizer step
            bwd_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
            training_client.optim_step(adam_params).result()

            # Log metrics
            metrics: dict[str, float] = {
                "progress/epoch": epoch,
                "progress/batch": batch_idx,
                "progress/done_frac": global_step / total_steps if total_steps > 0 else 0,
                "optim/lr": learning_rate,
                "num_pairs": len(chosen_data),
                "time/step": time.time() - t_start,
            }
            metrics.update(bwd_result.metrics)
            ml_logger.log_metrics(metrics, step=global_step)

            global_step += 1

        if config.max_steps is not None and global_step >= config.max_steps:
            break

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"epoch": config.num_epochs, "batch": 0},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("RLCF DPO training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
