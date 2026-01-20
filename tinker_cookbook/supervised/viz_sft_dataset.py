"""
Script to visualize supervised datasets in the terminal.

Shows tokens color-coded by training weight:
- Green = tokens the model learns to predict (weight > 0)
- Yellow = context tokens (weight = 0)

Example usage:

# Preview GSM8K with prompt template:
python -m tinker_cookbook.supervised.viz_sft_dataset \
    --dataset_path="PromptTemplateBuilder" \
    --dataset_name="openai/gsm8k" \
    --dataset_config="main" \
    --user_template="Solve: {question}" \
    --assistant_template="{answer}" \
    --n_examples=3

# Preview existing builder:
python -m tinker_cookbook.supervised.viz_sft_dataset \
    --dataset_path="NoRobotsBuilder"
"""

import chz
from tinker_cookbook import model_info
from tinker_cookbook.recipes.open_character.datasets import ChatDPOPairsBuilder
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.misc_utils import lookup_func


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"  # just for tokenizer
    dataset_path: str = "Tulu3Builder"
    renderer_name: str | None = None
    max_length: int | None = None
    train_on_what: TrainOnWhat | None = None

    # Prompt template parameters (when dataset_path is an HF dataset)
    user_template: str | None = None
    assistant_template: str | None = None
    system_prompt: str | None = None
    dataset_name: str | None = None # e.g. openai/gsm8k
    dataset_config: str | None = None  # e.g., "main" for gsm8k
    dataset_split: str = "train" # e.g. "test" or "train[:95%]"
    dataset_data_dir: str | None = None
    dataset_data_files: str | None = None

    num_epochs: int = 1
    shuffle_seed: int = 0

    # Display options
    n_examples: int = 5


def run(cfg: Config):
    n_examples_total = 100
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.n_examples,
        train_on_what=cfg.train_on_what,
    )

    kwargs = {}
    dataset_path = cfg.dataset_path
    module = "tinker_cookbook.recipes.chat_sl.chat_datasets"
    # If user_template is provided, use PromptTemplateDatasetBuilder with extra kwargs
    if cfg.user_template is not None:
        if cfg.assistant_template is None:
            raise ValueError("assistant_template is required when user_template is provided")
        dataset_path = "PromptTemplateBuilder"
        kwargs.update(
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            user_template=cfg.user_template,
            assistant_template=cfg.assistant_template,
            system_prompt=cfg.system_prompt,
            train_split=cfg.dataset_split,
            num_epochs=cfg.num_epochs,
            shuffle_seed=cfg.shuffle_seed,
        )
    if cfg.dataset_name is not None:  # name and no template -> pairs
        dataset_path = "ChatDatasetBuilderFromComparisons"
        module = "tinker_cookbook.preference.preference_datasets"
        comparison_builder = ChatDPOPairsBuilder(
            pairs_path=cfg.dataset_name,
            hf_dataset_data_dir=cfg.dataset_data_dir,
            hf_dataset_data_files=cfg.dataset_data_files,
            train_split=cfg.dataset_split,
            # test_split=cfg.dataset_split,  # 0.05 of train
            max_samples=cfg.n_examples,
        )
        kwargs.update(
            comparison_builder=comparison_builder,
        )
    dataset_builder = lookup_func(
        dataset_path, default_module=module
    )(common_config=common_config, **kwargs)
    assert isinstance(dataset_builder, SupervisedDatasetBuilder)
    tokenizer = get_tokenizer(cfg.model_name)
    train_dataset, _ = dataset_builder()
    batch = train_dataset.get_batch(0)

    print(f"\n{'='*60}")
    print(f"Previewing {len(batch)} examples from dataset")
    print(f"Legend: [green]=trained on, [yellow]=context")
    print(f"{'='*60}\n")

    for i, datum in enumerate(batch):
        int_tokens = list(datum.model_input.to_ints()) + [
            datum.loss_fn_inputs["target_tokens"].tolist()[-1]
        ]
        weights = [0.0] + datum.loss_fn_inputs["weights"].tolist()

        print(f"--- Example {i + 1}/{len(batch)} ---")
        print(format_colorized(int_tokens, weights, tokenizer))
        print()

        if i < len(batch) - 1:
            input("Press Enter for next example...")
        else:
            print("Done!")


def load_toml_into_argv() -> None:
    """If --config is in sys.argv, load the TOML file and inject values as CLI args.

    Only includes fields that exist in Config class (ignores training-specific fields).
    """
    import sys
    import tomllib

    valid_fields = set(Config.__annotations__.keys())

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            with open(sys.argv[i + 1], "rb") as f:
                config = tomllib.load(f)
            # Only include fields that Config recognizes
            toml_args = [f"{k}={v}" for k, v in config.items() if k in valid_fields]
            # Remove --config and its value, prepend TOML args
            sys.argv = [sys.argv[0]] + toml_args + sys.argv[1:i] + sys.argv[i + 2 :]
            break


if __name__ == "__main__":
    load_toml_into_argv()
    chz.nested_entrypoint(run, allow_hyphens=True)
