"""
Generate off-policy context distillation data for language classification.

Uses the teacher model with the full detailed prompt to generate labels,
then saves training data in a format where the student sees only the short prompt.

Usage:
    python -m tinker_cookbook.recipes.prompt_distillation.create_data_context \
        output_file=data/context_distillation/off_policy_data.jsonl
"""

import asyncio
import json
import os
import re
from typing import Any

import chz
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.recipes.prompt_distillation.train_on_policy import (
    DATA_PATH,
    STUDENT_PROMPT,
    TEACHER_PROMPT,
    VALID_LABELS,
    load_multilingual_sentences,
    parse_label,
    split_train_test,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

import logging

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    output_file: str
    gold_labels_file: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str = "qwen3_disable_thinking"


def setup_clients(model_name: str, renderer_name: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return sampling_client, tokenizer, renderer


async def create_data_async(
    cfg: Config,
    sampling_client: Any,
    tokenizer: Any,
    renderer: Any,
):
    sentences = load_multilingual_sentences()
    train_texts, test_texts = split_train_test(sentences)
    logger.info("Loaded %d train, %d test sentences", len(train_texts), len(test_texts))

    params = tinker.SamplingParams(
        max_tokens=200, temperature=0.15, stop=renderer.get_stop_sequences()
    )

    async def sample_label(text: str) -> tuple[str, str | None]:
        prompt = TEACHER_PROMPT.format(text=text)
        convo: list[renderers.Message] = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(convo)
        result = await sampling_client.sample_async(
            prompt=model_input, sampling_params=params, num_samples=1
        )
        response = tokenizer.decode(result.sequences[0].tokens)
        label = parse_label(response)
        return text, label

    # Generate labels for training data
    logger.info("Generating labels for %d training sentences...", len(train_texts))
    train_results: list[tuple[str, str | None]] = []
    for coro in tqdm_asyncio.as_completed(
        [sample_label(t) for t in train_texts], total=len(train_texts)
    ):
        train_results.append(await coro)

    # Save training data with student prompt format
    os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)
    with open(cfg.output_file, "w") as f:
        for text, label in train_results:
            if label is None:
                continue
            messages = {
                "messages": [
                    {"role": "user", "content": STUDENT_PROMPT.format(text=text)},
                    {"role": "assistant", "content": f"Final Answer: {label}"},
                ],
            }
            f.write(json.dumps(messages) + "\n")
    logger.info("Saved %d training examples to %s", len(train_results), cfg.output_file)

    # Generate gold labels for test data
    if cfg.gold_labels_file:
        logger.info("Generating gold labels for %d test sentences...", len(test_texts))
        test_results: list[tuple[str, str | None]] = []
        for coro in tqdm_asyncio.as_completed(
            [sample_label(t) for t in test_texts], total=len(test_texts)
        ):
            test_results.append(await coro)

        gold_map = {}
        for text, label in test_results:
            if label is not None:
                gold_map[text] = label
        gold_labels = [gold_map.get(t, "ot") for t in test_texts]

        os.makedirs(os.path.dirname(cfg.gold_labels_file), exist_ok=True)
        with open(cfg.gold_labels_file, "w") as f:
            json.dump(gold_labels, f)
        logger.info("Saved %d gold labels to %s", len(gold_labels), cfg.gold_labels_file)


def main(cfg: Config):
    if os.path.exists(cfg.output_file):
        logger.info("Output file %s already exists, skipping", cfg.output_file)
        return
    sampling_client, tokenizer, renderer = setup_clients(cfg.model_name, cfg.renderer_name)
    asyncio.run(create_data_async(cfg, sampling_client, tokenizer, renderer))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chz.nested_entrypoint(main)
