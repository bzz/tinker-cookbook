"""
Create a ground-truth labeled dataset for language classification.

Supports two backends:
  --backend openai   Use OpenAI API (requires OPENAI_API_KEY)
  --backend tinker   Use Tinker-hosted model (requires TINKER_API_KEY)

The Tinker backend uses a two-step approach: ask the model for the ISO 639-1
code (which it does reliably), then map codes outside the 13-label set to "ot".

Produces two JSONL files (one per split).  Each line:

    {"text": "And he said ...", "label": "en"}

Usage:
    # OpenAI (preferred for independent ground truth)
    python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
        --output_dir data/context_distillation --backend openai --model gpt-4o-mini

    # Tinker fallback (uses same model family but reliable two-step method)
    python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
        --output_dir data/context_distillation --backend tinker

    # Quick test on a small subset
    python -m tinker_cookbook.recipes.prompt_distillation.create_labeled_dataset \
        --output_dir data/context_distillation --backend tinker --limit 30
"""

import argparse
import asyncio
import json
import logging
import os
import re
from collections import Counter

from tinker_cookbook.recipes.prompt_distillation.train_on_policy import VALID_LABELS

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "example_data", "multilingual.txt")


def _load_multilingual_sentences(path: str = _DATA_PATH) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _split_train_test(
    sentences: list[str], test_fraction: float = 0.2, group_size: int = 15
) -> tuple[list[str], list[str]]:
    n_groups = len(sentences) // group_size
    n_test = max(1, int(n_groups * test_fraction))
    n_train = n_groups - n_test
    return sentences[: n_train * group_size], sentences[n_train * group_size : n_groups * group_size]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Used by both backends: asks for an ISO code, not a label from the fixed set.
IDENTIFY_PROMPT = """\
What language is this text written in?
Respond with the standard two-letter ISO 639-1 code and nothing else.

Text:
{text}"""


def _iso_to_label(iso: str) -> str:
    """Map a free-form ISO code to the 13-label set, defaulting to ot."""
    iso = iso.strip().lower()
    return iso if iso in VALID_LABELS else "ot"


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

async def _label_openai(texts: list[str], model: str, concurrency: int) -> list[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def _one(text: str) -> str:
        async with sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": IDENTIFY_PROMPT.format(text=text)}],
                max_tokens=5,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            code = re.search(r"[a-z]{2}", raw)
            return _iso_to_label(code.group(0) if code else "ot")

    return list(await asyncio.gather(*[_one(t) for t in texts]))


# ---------------------------------------------------------------------------
# Tinker backend
# ---------------------------------------------------------------------------

async def _label_tinker(texts: list[str], model: str, concurrency: int) -> list[str]:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(model)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tok)
    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=model)
    params = tinker.SamplingParams(
        max_tokens=10, temperature=0.0, stop=renderer.get_stop_sequences()
    )
    sem = asyncio.Semaphore(concurrency)

    async def _one(text: str) -> str:
        async with sem:
            convo: list[renderers.Message] = [
                {"role": "user", "content": IDENTIFY_PROMPT.format(text=text)}
            ]
            mi = renderer.build_generation_prompt(convo)
            res = await client.sample_async(prompt=mi, sampling_params=params, num_samples=1)
            raw = tok.decode(res.sequences[0].tokens).strip().lower()
            code = re.search(r"[a-z]{2}", raw)
            return _iso_to_label(code.group(0) if code else "ot")

    return list(await asyncio.gather(*[_one(t) for t in texts]))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, texts: list[str], labels: list[str]) -> None:
    with open(path, "w") as f:
        for text, label in zip(texts, labels):
            f.write(json.dumps({"text": text, "label": label}) + "\n")
    logger.info("Wrote %d examples to %s", len(texts), path)


async def main():
    parser = argparse.ArgumentParser(description="Create labeled dataset")
    parser.add_argument("--output_dir", default="data/context_distillation")
    parser.add_argument("--backend", choices=["openai", "tinker"], default="tinker")
    parser.add_argument("--model", default=None, help="Model name (default: backend-specific)")
    parser.add_argument("--limit", type=int, default=None, help="Max sentences per split")
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()

    if args.model is None:
        args.model = "gpt-4o-mini" if args.backend == "openai" else "Qwen/Qwen3-30B-A3B"

    sentences = _load_multilingual_sentences()
    train_texts, test_texts = _split_train_test(sentences)
    if args.limit:
        train_texts = train_texts[: args.limit]
        test_texts = test_texts[: args.limit]

    logger.info(
        "Labeling %d train + %d test with %s (%s backend)",
        len(train_texts), len(test_texts), args.model, args.backend,
    )

    label_fn = _label_openai if args.backend == "openai" else _label_tinker
    train_labels = await label_fn(train_texts, args.model, args.concurrency)
    test_labels = await label_fn(test_texts, args.model, args.concurrency)

    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(os.path.join(args.output_dir, "train_set.jsonl"), train_texts, train_labels)
    _write_jsonl(os.path.join(args.output_dir, "test_set.jsonl"), test_texts, test_labels)

    logger.info("Train: %s", Counter(train_labels).most_common())
    logger.info("Test:  %s", Counter(test_labels).most_common())


if __name__ == "__main__":
    asyncio.run(main())
