"""
Generic RLVR (Reinforcement Learning with Verifiable Rewards) runner.

This runner dynamically loads custom ProblemEnv subclasses and supports
TOML config files for easy configuration.

Usage:
    # With TOML config
    python -m tinker_cookbook.recipes.rlvr.train --config configs/my_task.toml

    # With CLI args
    python -m tinker_cookbook.recipes.rlvr.train \
        env_class=my_module:MyEnv \
        dataset_name=my/dataset \
        user_template="Solve: {question}" \
        answer_field=answer

Example TOML config:
    env_class = "my_module:PatchEnv"
    dataset_name = "my/patch-dataset"
    user_template = "my_prompt.txt"  # Can be filename or inline template
    answer_field = "expected_output"
    model_name = "Qwen/Qwen3-8B"
    group_size = 8
    batch_size = 8

    # Optional: Enable evaluation by setting evaluator
    evaluator = "my_module:MyEvaluatorBuilder"
    eval_dataset_split = "test"

Template files:
    user_template can be either:
    - An inline template string with {field} placeholders
    - A filename ending in .txt (resolved relative to configs/ directory)
"""

import asyncio
import logging
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Sequence, Type, cast

import chz
import datasets
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.recipes.rlvr.patch_env import TemplateEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.misc_utils import lookup_func
from tinker.types import LossFnType

logger = logging.getLogger(__name__)

# Directory containing config files (for resolving relative template paths)
CONFIG_DIR = Path(__file__).parent / "configs"


def resolve_template(template: str, config_dir: Path = CONFIG_DIR) -> str:
    """Resolve a template string, loading from file if it ends with .txt."""
    if template.endswith(".txt"):
        # Try relative to config dir first, then absolute
        template_path = config_dir / template
        if not template_path.exists():
            template_path = Path(template)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template}")
        return template_path.read_text()
    return template


def _parse_cli_value(value: str):
    """Parse CLI value string to appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_toml_config_and_cli_args() -> dict:
    """Load TOML config from --config arg and merge with CLI overrides."""
    config_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config" and i < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break

    toml_config: dict = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
        logger.info(f"Loaded config from {config_path}")

    # Parse CLI overrides (key=value format, skipping --config)
    cli_overrides: dict = {}
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        if "=" in arg:
            key, value = arg.split("=", 1)
            cli_overrides[key] = _parse_cli_value(value)

    return {**toml_config, **cli_overrides}


@chz.chz
class CLIConfig:
    """CLI configuration for generic RLVR training."""

    # Environment configuration
    env_class: str  # Path to ProblemEnv subclass, e.g., "my_module:PatchEnv"

    # Dataset configuration
    dataset_name: str  # HuggingFace dataset name
    dataset_config: str | None = None  # Dataset config (e.g., "main" for gsm8k)
    dataset_split: str = "train"  # Split to use, supports slice syntax like "train[:95%]"
    num_epochs: int = 1

    # Template configuration
    user_template: str  # Prompt template with {field} placeholders
    answer_field: str  # Field containing ground truth for reward

    # Optional: system prompt for the conversation
    system_prompt: str | None = None

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    group_size: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0

    # KL penalty configuration
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Number of optimizer steps per training iteration
    num_substeps: int = 1

    # Optimization options
    remove_constant_reward_groups: bool = False

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False  # Log KL divergence after training step

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Dataset shuffling
    seed: int = 0

    # Loss function
    loss_fn: LossFnType = "importance_sampling"

    # Async training (optional)
    max_steps_off_policy: int | None = None

    # Prefix to add to checkpoint names
    chkpt_name_prefix: str | None = None

    # Evaluation configuration (if evaluator is set, eval is enabled)
    evaluator: str | None = None
    eval_dataset_split: str = "test"
    max_eval_samples: int | None = None
    eval_temperature: float = 0.0


class TemplateRLDataset(RLDataset):
    """RLDataset that creates environments from HuggingFace dataset rows using templates."""

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        env_class: Type[TemplateEnv],
        user_template: str,
        answer_field: str,
        renderer: renderers.Renderer,
        batch_size: int,
        group_size: int,
        system_prompt: str | None = None,
        dataset_name: str = "rlvr",
    ):
        self.hf_dataset = hf_dataset
        self.env_class = env_class
        self.user_template = user_template
        self.answer_field = answer_field
        self.renderer = renderer
        self.batch_size = batch_size
        self.group_size = group_size
        self.system_prompt = system_prompt
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return (len(self.hf_dataset) + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.hf_dataset))

        builders: list[EnvGroupBuilder] = []
        for i in range(start, end):
            row = dict(self.hf_dataset[i])

            # Format the question from template
            question = self.user_template.format(**row)

            # Get the answer from the specified field
            answer = str(row.get(self.answer_field, ""))

            # Build conversation prefix if system prompt is provided
            convo_prefix: list[renderers.Message] | None = None
            if self.system_prompt:
                convo_prefix = [{"role": "system", "content": self.system_prompt}]

            # Create env factory that captures the row data
            def make_env(
                q: str = question,
                a: str = answer,
                r: dict = row,
                prefix: list[renderers.Message] | None = convo_prefix,
            ) -> TemplateEnv:
                return self.env_class(
                    question=q,
                    answer=a,
                    row=r,
                    renderer=self.renderer,
                    convo_prefix=prefix,
                )

            builders.append(
                ProblemGroupBuilder(
                    env_thunk=make_env,
                    num_envs=self.group_size,
                    dataset_name=self.dataset_name,
                )
            )

        return builders


@chz.chz
class TemplateRLDatasetBuilder(RLDatasetBuilder):
    """Builder for TemplateRLDataset that loads HF datasets and uses templates."""

    env_class: str  # Path for lookup_func
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    user_template: str
    answer_field: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    system_prompt: str | None = None
    seed: int | None = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Load the environment class
        env_cls = lookup_func(self.env_class)
        if not (isinstance(env_cls, type) and issubclass(env_cls, TemplateEnv)):
            raise ValueError(
                f"env_class must be a subclass of TemplateEnv, got {env_cls}"
            )

        # Load dataset with slice syntax support
        hf_dataset = datasets.load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.dataset_split,
        )
        hf_dataset = cast(datasets.Dataset, hf_dataset)

        # Shuffle the dataset if seed is provided
        if self.seed is not None:
            hf_dataset = hf_dataset.shuffle(seed=self.seed)

        # Create renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Extract dataset name for logging
        log_name = self.dataset_name.split("/")[-1]

        train_dataset = TemplateRLDataset(
            hf_dataset=hf_dataset,
            env_class=env_cls,
            user_template=self.user_template,
            answer_field=self.answer_field,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            system_prompt=self.system_prompt,
            dataset_name=log_name,
        )

        return train_dataset, None


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Build run name
    model_name_short = cli_config.model_name.replace("/", "-")
    dataset_name_short = cli_config.dataset_name.split("/")[-1]
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rlvr-{dataset_name_short}-{model_name_short}-"
        f"gs{cli_config.group_size}-bs{cli_config.batch_size}-"
        f"lr{cli_config.learning_rate}-{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rlvr/{run_name}"

    # Set wandb name
    wandb_name = cli_config.wandb_name or run_name

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Resolve template (load from file if .txt)
    user_template = resolve_template(cli_config.user_template)

    # Create dataset builder
    dataset_builder = TemplateRLDatasetBuilder(
        env_class=cli_config.env_class,
        dataset_name=cli_config.dataset_name,
        dataset_config=cli_config.dataset_config,
        dataset_split=cli_config.dataset_split,
        user_template=user_template,
        answer_field=cli_config.answer_field,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        system_prompt=cli_config.system_prompt,
        seed=cli_config.seed,
    )

    # Build async config if needed
    async_config = None
    if cli_config.max_steps_off_policy is not None:
        async_config = train.AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.batch_size,
        )

    # Build evaluator if set
    evaluator_builders = []
    if cli_config.evaluator is not None:
        evaluator_builder_cls = lookup_func(cli_config.evaluator)
        # Train dataset config + overriding eval fields
        eval_ds_cfg = chz.asdict(dataset_builder)
        eval_ds_cfg.pop("batch_size"); eval_ds_cfg.pop("group_size")
        eval_config = {
            **eval_ds_cfg,
            "dataset_split": cli_config.eval_dataset_split,
            "max_eval_samples": cli_config.max_eval_samples,
            "max_tokens": cli_config.max_tokens,
            "temperature": cli_config.eval_temperature,
        }
        eval_builder = evaluator_builder_cls(**eval_config)
        evaluator_builders.append(eval_builder)

    # Create training config
    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        num_epochs=cli_config.num_epochs,
        compute_post_kl=cli_config.compute_post_kl,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=async_config,
        loss_fn=cli_config.loss_fn,
        chkpt_name_prefix=cli_config.chkpt_name_prefix,
        evaluator_builders=evaluator_builders,
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    # Load TOML config if --config is provided
    toml_config = load_toml_config_and_cli_args()

    # Create CLI config from merged TOML + CLI args
    if toml_config:
        cli_config = CLIConfig(**toml_config)
    else:
        cli_config = chz.entrypoint(CLIConfig)

    asyncio.run(cli_main(cli_config))

