"""
Bayesian hyperparameter optimization example using Optuna.

This script is intended as a template you can adapt to your own training loop.
It runs a short supervised fine-tuning loop (a truncated version of `recipes/sl_loop.py`)
and reports the final train loss to Optuna.

Usage:

```bash
# Install Optuna (optional extra)
pip install "tinker_cookbook[sweeps]"

# Run a local in-memory study
python -m tinker_cookbook.recipes.sl_optuna_sweep log_root=/tmp/sft-optuna n_trials=20 max_steps=30

# Run a persistent study (enables multiple workers/processes)
python -m tinker_cookbook.recipes.sl_optuna_sweep \
  log_root=/tmp/sft-optuna storage=sqlite:////tmp/sft-optuna/optuna.db study_name=sft-sl-loop \
  n_trials=50 n_jobs=2
```
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import chz
import datasets
import tinker

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Optuna is not installed. Install with:\n"
            '  pip install "tinker_cookbook[sweeps]"\n'
            "or:\n"
            "  pip install optuna\n"
        ) from e
    return optuna


@dataclass(frozen=True)
class TrialRunConfig:
    base_url: str | None
    log_path: str
    model_name: str
    batch_size: int
    learning_rate: float
    max_length: int
    train_on_what: renderers.TrainOnWhat
    lora_rank: int
    max_steps: int
    seed: int


def _run_truncated_sl_loop(cfg: TrialRunConfig) -> float:
    """
    Run a truncated supervised loop and return the final train_mean_nll.

    This is intentionally lightweight: it does not checkpoint and uses a fixed number
    of steps (`max_steps`) rather than running a full epoch.
    """
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=None,
        wandb_name=None,
        config=cfg,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"].shuffle(seed=cfg.seed)

    max_available_batches = len(train_dataset) // cfg.batch_size
    n_steps = min(cfg.max_steps, max_available_batches)
    if n_steps <= 0:
        raise ValueError(
            f"Not enough data for one batch: len(train)={len(train_dataset)} "
            f"batch_size={cfg.batch_size}"
        )
    logger.info(f"Training for {n_steps} steps (batch_size={cfg.batch_size})")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = service_client.create_lora_training_client(base_model=cfg.model_name, rank=cfg.lora_rank)

    last_train_nll: float | None = None
    for step in range(n_steps):
        start_time = time.time()

        # Linear schedule over the truncated horizon
        lr_mult = max(0.0, 1.0 - step / n_steps)
        current_lr = cfg.learning_rate * lr_mult
        adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        batch_start = step * cfg.batch_size
        batch_end = min((step + 1) * cfg.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        batch = [
            conversation_to_datum(
                row["messages"],  # type: ignore
                renderer,
                cfg.max_length,
                cfg.train_on_what,
            )
            for row in batch_rows
        ]

        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_step_future.result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = float(compute_mean_nll(train_logprobs, train_weights))
        last_train_nll = train_nll

        ml_logger.log_metrics(
            metrics={
                "num_sequences": len(batch),
                "num_tokens": sum(d.model_input.length for d in batch),
                "learning_rate": current_lr,
                "train_mean_nll": train_nll,
                "progress": step / n_steps,
                "time_total": time.time() - start_time,
            },
            step=step,
        )

    ml_logger.close()
    assert last_train_nll is not None
    return last_train_nll


@chz.chz
class SweepConfig:
    # Infrastructure
    base_url: str | None = None

    # Study/logging
    log_root: str = "/tmp/tinker-examples/sl-optuna-sweep"
    study_name: str = "sl-optuna"
    storage: str | None = None  # e.g. sqlite:////tmp/optuna.db
    direction: str = "minimize"  # Optuna supports "minimize" or "maximize"

    # Budget / parallelism
    n_trials: int = 20
    n_jobs: int = 1  # threads; for multi-process use a shared `storage=` and run multiple commands
    max_steps: int = 30

    # Model/training baseline
    model_name: str = "meta-llama/Llama-3.1-8B"
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    seed: int = 0

    # Search space (edit these for your task)
    lr_min: float = 1e-5
    lr_max: float = 3e-3
    batch_sizes: list[int] = chz.field(default_factory=lambda: [32, 64, 128])
    lora_ranks: list[int] = chz.field(default_factory=lambda: [8, 16, 32, 64])


def main(cfg: SweepConfig):
    optuna = _require_optuna()

    os.makedirs(cfg.log_root, exist_ok=True)

    if cfg.storage is None:
        study = optuna.create_study(direction=cfg.direction)
    else:
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            load_if_exists=True,
            direction=cfg.direction,
        )

    def objective(trial) -> float:
        lr = trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)
        lora_rank = trial.suggest_categorical("lora_rank", cfg.lora_ranks)

        trial_log_path = os.path.join(cfg.log_root, f"trial-{trial.number:05d}")
        cli_utils.check_log_dir(trial_log_path, behavior_if_exists="delete")

        trial_cfg = TrialRunConfig(
            base_url=cfg.base_url,
            log_path=trial_log_path,
            model_name=cfg.model_name,
            batch_size=int(batch_size),
            learning_rate=float(lr),
            max_length=cfg.max_length,
            train_on_what=cfg.train_on_what,
            lora_rank=int(lora_rank),
            max_steps=cfg.max_steps,
            seed=cfg.seed,
        )

        trial.set_user_attr("log_path", trial_log_path)
        trial.set_user_attr("model_name", cfg.model_name)

        final_nll = _run_truncated_sl_loop(trial_cfg)
        return final_nll

    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)
    print("Best log_path:", best.user_attrs.get("log_path"))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

