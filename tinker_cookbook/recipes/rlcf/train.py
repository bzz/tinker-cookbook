"""
RLCF: Reinforcement Learning from Checklist Feedback — DPO training.

Trains a language model using DPO on preference pairs scored by
instruction-specific checklists, faithfully reproducing the approach
from Viswanathan et al. (2025):
  "Checklists Are Better Than Reward Models For Aligning Language Models"
  https://arxiv.org/abs/2507.18624

The paper pipeline:
  1. Generate checklists for instructions (offline, pre-computed in viswavi/rlcf)
  2. Score candidate response pairs against checklists using LLM judges
  3. Rank chosen/rejected by weighted checklist score
  4. Train with DPO (beta=0.1, lr=3e-6, 2 epochs, max_len=2048)

Usage::

    python -m tinker_cookbook.recipes.rlcf.train

    python -m tinker_cookbook.recipes.rlcf.train \\
        model_name=Qwen/Qwen2.5-7B-Instruct \\
        learning_rate=3e-6 \\
        dpo_beta=0.1 \\
        batch_size=256
"""

from datetime import datetime
from typing import cast

import chz
import datasets

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule


# ---------------------------------------------------------------------------
# Dataset: viswavi/rlcf chosen/rejected pairs scored by checklists
# ---------------------------------------------------------------------------


@chz.chz
class RLCFComparisonBuilder(ComparisonDatasetBuilder):
    """Load checklist-scored preference pairs from the viswavi/rlcf dataset.

    The dataset contains chosen/rejected conversation pairs where ranking
    was determined by weighted checklist scores (the paper's core contribution).
    Each row also includes the ``requirements`` field with the checklist used
    for scoring, preserved here for logging/analysis.
    """

    dataset_name: str = "viswavi/rlcf"
    test_size: int = 1024

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(self.dataset_name, split="train")
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(self.test_size)
        train_dataset = dataset.skip(self.test_size)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        chosen = example.get("chosen")
        rejected = example.get("rejected")
        if not chosen or not rejected:
            return None

        # viswavi/rlcf stores conversations as list[dict] with role/content
        if isinstance(chosen, list) and len(chosen) >= 2:
            prompt_messages: list[renderers.Message] = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in chosen[:-1]
            ]
            chosen_completion: list[renderers.Message] = [
                {"role": chosen[-1]["role"], "content": chosen[-1]["content"]}
            ]
            rejected_completion: list[renderers.Message] = [
                {"role": rejected[-1]["role"], "content": rejected[-1]["content"]}
            ]
        else:
            return None

        comparison = Comparison(
            prompt_conversation=prompt_messages,
            completion_A=chosen_completion,
            completion_B=rejected_completion,
        )
        return LabeledComparison(comparison=comparison, label="A")


# ---------------------------------------------------------------------------
# CLI config — paper-faithful defaults
# ---------------------------------------------------------------------------


@chz.chz
class CLIConfig:
    """Command-line configuration for RLCF DPO training.

    Defaults reproduce the paper's training setup (Table 1, train_rlcf.sh):
    - Qwen/Qwen2.5-7B-Instruct as the policy model
    - DPO with beta=0.1
    - lr=3e-6 with linear schedule (min_lr_ratio=0.75 in paper)
    - 2 epochs over viswavi/rlcf dataset
    - max_len=2048, batch_size=1024 (reduced default for single-machine runs)
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # DPO hyperparameters (paper defaults)
    learning_rate: float = 3e-6
    lr_schedule: LRSchedule = "linear"
    dpo_beta: float = 0.1
    num_epochs: int = 2
    max_length: int = 2048
    batch_size: int = 256

    # Dataset
    dataset_name: str = "viswavi/rlcf"
    test_size: int = 1024

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing & eval
    save_every: int = 32
    eval_every: int = 10

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


def get_dataset_builder(
    dataset_name: str,
    test_size: int,
    model_name: str,
    renderer_name: str,
    max_length: int,
    batch_size: int,
) -> ChatDatasetBuilder:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )
    return DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=RLCFComparisonBuilder(
            dataset_name=dataset_name,
            test_size=test_size,
        ),
    )


def cli_main(cli_config: CLIConfig) -> None:
    """Convert CLI config to full DPO config and launch training."""

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"rlcf-dpo-{model_slug}-lr{cli_config.learning_rate}"
        f"-beta{cli_config.dpo_beta}-bs{cli_config.batch_size}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rlcf/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train_dpo.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        dataset_builder=get_dataset_builder(
            dataset_name=cli_config.dataset_name,
            test_size=cli_config.test_size,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            max_length=cli_config.max_length,
            batch_size=cli_config.batch_size,
        ),
        load_checkpoint_path=cli_config.load_checkpoint_path,
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        dpo_beta=cli_config.dpo_beta,
        num_epochs=cli_config.num_epochs,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        max_steps=cli_config.max_steps,
    )

    train_dpo.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
