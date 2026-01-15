"""
Dataset builders for Open Character Training.

Provides:
- CharacterDPOPairsBuilder: Loads DPO pairs from JSONL (question/chosen/rejected strings)
- ChatDPOPairsBuilder: Loads DPO pairs from JSONL (chosen/rejected as message arrays)
- IntrospectionDatasetBuilder: Loads introspection conversations for Stage 3b
"""

import json
import logging

import blobfile
import chz
import datasets
import tinker
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


# =============================================================================
# DPO Dataset Builder (Stage 2)
# =============================================================================


@chz.chz
class CharacterDPOPairsBuilder(ComparisonDatasetBuilder):
    """
    Load DPO pairs from JSONL for character distillation.

    Expected JSONL format:
        {"question": "...", "chosen": "...", "rejected": "..."}

    - question: The user query
    - chosen: Teacher response (with constitution system prompt)
    - rejected: Student response (without constitution system prompt)
    """

    pairs_path: str
    test_ratio: float = 0.05
    shuffle_seed: int = 42

    # JSONL key names for the character DPO pairs.
    prompt_key = "question"
    chosen_key = "chosen"
    rejected_key = "rejected"

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load DPO pairs from JSONL and split into train/test."""
        data = []
        with blobfile.BlobFile(self.pairs_path, "r", streaming=False) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        dataset = datasets.Dataset.from_list(data)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split train/test
        if self.test_ratio > 0 and len(dataset) > 10:
            split = dataset.train_test_split(test_size=self.test_ratio, seed=self.shuffle_seed)
            return split["train"], split["test"]
        return dataset, None

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert JSONL row to LabeledComparison."""
        if (
            self.prompt_key not in example
            or self.chosen_key not in example
            or self.rejected_key not in example
        ):
            logger.warning(f"Skipping example missing required fields: {example}")
            return None

        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=[{"role": "user", "content": example[self.prompt_key]}],
                completion_A=[{"role": "assistant", "content": example[self.chosen_key]}],
                completion_B=[{"role": "assistant", "content": example[self.rejected_key]}],
            ), 
            label="A", # A (chosen) is always preferred over B (rejected)
        )


@chz.chz
class ChatDPOPairsBuilder(ComparisonDatasetBuilder):
    """
    Load DPO pairs from JSONL where chosen/rejected are full conversation arrays.

    Expected JSONL format:
        {
            "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
            "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        }

    The prompt is extracted from chosen[:-1] (all messages except the last).
    The completions are the last message from each conversation.
    """

    pairs_path: str
    test_ratio: float = 0.05
    shuffle_seed: int = 42

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load DPO pairs from JSONL and split into train/test."""
        data = []
        with blobfile.BlobFile(self.pairs_path, "r", streaming=False) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        dataset = datasets.Dataset.from_list(data)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split train/test
        if self.test_ratio > 0 and len(dataset) > 10:
            split = dataset.train_test_split(test_size=self.test_ratio, seed=self.shuffle_seed)
            return split["train"], split["test"]
        return dataset, None

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert JSONL row to LabeledComparison.

        Extracts prompt from chosen[:-1], completions from last message of each.
        """
        if "chosen" not in example or "rejected" not in example:
            logger.warning(f"Skipping example missing 'chosen' or 'rejected': {list(example.keys())}")
            return None

        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]

        if not chosen_messages or not rejected_messages:
            logger.warning("Skipping example with empty chosen or rejected messages")
            return None

        # Prompt is all messages except the last from chosen
        # (assumes chosen and rejected share the same prompt)
        prompt_conversation = chosen_messages[:-1]

        # Completions are the last message from each
        completion_chosen = [chosen_messages[-1]]
        completion_rejected = [rejected_messages[-1]]

        return LabeledComparison(
            comparison=Comparison(
                prompt_conversation=prompt_conversation,
                completion_A=completion_chosen,
                completion_B=completion_rejected,
            ),
            label="A",  # A (chosen) is always preferred over B (rejected)
        )


# =============================================================================
# Introspection Dataset Builder (Stage 3b)
# =============================================================================


@chz.chz
class IntrospectionDatasetBuilder(ChatDatasetBuilder):
    """
    Load introspection conversations (self-reflection + self-interaction) for SFT.

    Expected JSONL format:
        {"messages": [{"role": "...", "content": "..."}, ...]}

    The messages should include system prompts from the generation stage.
    Training is on ALL_ASSISTANT_MESSAGES (prompt distillation).
    """

    introspection_path: str
    test_size: int = 100
    shuffle_seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and test datasets from introspection JSONL."""
        conversations = []
        with blobfile.BlobFile(self.introspection_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise ValueError(
                        f"Each line must contain 'messages' field. Got: {list(data.keys())}"
                    )
                conversations.append(data)

        dataset = datasets.Dataset.from_list(conversations)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split train/test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        # Use ALL_ASSISTANT_MESSAGES for prompt distillation
        # (train on assistant responses, ignore system/user)
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            model_input, weights = self.renderer.build_supervised_example(
                row["messages"], train_on_what=train_on_what
            )
            return datum_from_model_input_weights(
                model_input, weights, self.common_config.max_length
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        test_dataset = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )

        return train_dataset, test_dataset
