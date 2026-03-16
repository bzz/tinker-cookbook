"""
Off-policy (supervised) context distillation training.

Trains a student model on teacher-generated language classification data.
The student learns to predict language labels WITHOUT seeing the teacher's
detailed classification prompt.

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.train_off_policy \
        file_path=/tmp/tinker-datasets/context_distillation/train.jsonl \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3_disable_thinking \
        wandb_project=context_distillation
"""

import asyncio
import os
from datetime import datetime

import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class CLIConfig:
    # Data
    file_path: str = "/tmp/tinker-datasets/context_distillation/train.jsonl"
    log_path: str | None = None
    model_name: str = "Qwen/Qwen3-8B"
    load_checkpoint_path: str | None = None

    # Training
    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 4

    # Model
    lora_rank: int = 32

    # Infrastructure
    base_url: str | None = None

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 5

    # Dataset
    renderer_name: str | None = "qwen3_disable_thinking"
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    max_length: int = 4096
    batch_size: int = 64

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def cli_main(cli_config: CLIConfig):
    model_name = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"ctx-distill-offpolicy-{model_name}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.batch_size}batch-{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = os.path.expanduser(f"~/tinker-examples/context_distillation/{run_name}")

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    if not os.path.exists(cli_config.file_path):
        raise FileNotFoundError(
            f"Data file not found: {cli_config.file_path}\n"
            "Run create_data.py first to generate training data."
        )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=cli_config.file_path,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset,
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
