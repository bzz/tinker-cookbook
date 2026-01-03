"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

import logging
from typing import cast

import chz
import datasets
import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig, SupervisedDataset

logger = logging.getLogger(__name__)


@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        # Use train_on_what from common_config if provided, otherwise default to LAST_ASSISTANT_MESSAGE
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        # take the last 1000 as test, the rest as train
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class NoRobotsBuilder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        train_dataset = train_dataset.shuffle(seed=0)

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_dataset, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_dataset, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class PromptTemplateBuilder(ChatDatasetBuilder):
    """Generic builder that formats HF dataset rows using prompt templates.

    Use {field_name} in templates to reference row fields from the dataset.

    Split syntax (like torchtune):
        - "train" or "test" - use the named split
        - "train[:95%]" - first 95% of train split
        - "train[95%:]" - last 5% of train split
        - "train[:1000]" - first 1000 examples

    Example usage:
        builder = PromptTemplateBuilder(
            common_config=common_config,
            dataset_name="openai/gsm8k",
            dataset_config="main",
            train_split="train",
            test_split="test",
            user_template="Solve this math problem:\\n{question}",
            assistant_template="{answer}",
            num_epochs=3,
        )
    """
    dataset_name: str
    dataset_config: str | None = None  # e.g., "main" for gsm8k

    # Split configuration - supports syntax like "train[:95%]" or "test"
    train_split: str = "train"
    test_split: str | None = None  # If None, no test dataset

    # Template strings - use {field_name} for row fields
    user_template: str  # e.g., "Solve this problem:\n{question}"
    assistant_template: str  # e.g., "{answer}"

    # Optional: system prompt
    system_prompt: str | None = None

    # Multi-epoch: replicate data before shuffling
    num_epochs: int = 1
    shuffle_seed: int = 0

    def _load_split(self, split: str) -> datasets.Dataset:
        """Load a dataset split, supporting slice syntax like 'train[:95%]'."""
        dataset = datasets.load_dataset(self.dataset_name, self.dataset_config, split=split)
        return cast(datasets.Dataset, dataset)

    def _replicate_for_epochs(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Replicate dataset for multiple epochs, then shuffle."""
        if self.num_epochs <= 1:
            return dataset.shuffle(seed=self.shuffle_seed)

        # Concatenate the dataset with itself num_epochs times
        replicated = datasets.concatenate_datasets([dataset] * self.num_epochs)
        return replicated.shuffle(seed=self.shuffle_seed)

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load train dataset with optional slice syntax
        train_ds = self._load_split(self.train_split)
        train_ds = self._replicate_for_epochs(train_ds)

        # Load test dataset if specified
        test_ds = None
        if self.test_split is not None:
            test_ds = self._load_split(self.test_split)

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            # Build messages from templates
            messages = []

            # Add system message if provided
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})

            # Format user message from template
            user_content = self.user_template.format(**row)
            messages.append({"role": "user", "content": user_content})

            # Format assistant message from template
            assistant_content = self.assistant_template.format(**row)
            messages.append({"role": "assistant", "content": assistant_content})

            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
        else:
            test_dataset = None

        return train_dataset, test_dataset
