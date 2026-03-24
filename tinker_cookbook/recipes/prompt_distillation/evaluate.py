"""
Evaluate a Tinker model on the labeled language-classification test set.

Loads a JSONL file of {"text": ..., "label": ...} entries, samples from
the model with a given prompt template, and reports accuracy / parse-rate /
per-label metrics.

Usage:
    # Evaluate base model with the student prompt
    python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
        --dataset data/context_distillation/test_set.jsonl \
        --prompt student

    # Evaluate a trained checkpoint
    python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
        --dataset data/context_distillation/test_set.jsonl \
        --checkpoint_path tinker://... \
        --prompt student --limit 100

    # Evaluate with the teacher (full) prompt
    python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
        --dataset data/context_distillation/test_set.jsonl \
        --prompt teacher

    # Evaluate with a custom prompt (must contain {text})
    python -m tinker_cookbook.recipes.prompt_distillation.evaluate \
        --dataset data/context_distillation/test_set.jsonl \
        --prompt "Detect the language: {text}\\nAnswer with Final Answer: xx"
"""

import argparse
import asyncio
import json
import logging
from collections import Counter, defaultdict

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.recipes.prompt_distillation.train_on_policy import (
    STUDENT_PROMPT,
    TEACHER_PROMPT,
    VALID_LABELS,
    parse_label,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NAMED_PROMPTS = {
    "student": STUDENT_PROMPT,
    "teacher": TEACHER_PROMPT,
}


def load_dataset(path: str, limit: int | None = None) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


async def main():
    p = argparse.ArgumentParser(description="Evaluate model on labeled dataset")
    p.add_argument("--dataset", required=True, help="Path to test_set.jsonl")
    p.add_argument("--model_name", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--renderer_name", default="qwen3_disable_thinking")
    p.add_argument("--checkpoint_path", default=None, help="Tinker checkpoint for LoRA weights")
    p.add_argument("--prompt", default="student", help="'student', 'teacher', or a template string with {text}")
    p.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--base_url", default=None)
    args = p.parse_args()

    prompt_template = NAMED_PROMPTS.get(args.prompt, args.prompt)
    if "{text}" not in prompt_template:
        p.error("Prompt must contain {text} placeholder")

    data = load_dataset(args.dataset, args.limit)
    logger.info("Loaded %d examples from %s", len(data), args.dataset)
    logger.info("Prompt: %s", args.prompt if args.prompt in NAMED_PROMPTS else args.prompt[:80] + "...")
    logger.info("Model:  %s  checkpoint: %s", args.model_name, args.checkpoint_path or "(base)")

    tok = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(args.renderer_name, tok)
    service = tinker.ServiceClient(base_url=args.base_url)

    if args.checkpoint_path:
        client = service.create_sampling_client(
            base_model=args.model_name, model_path=args.checkpoint_path
        )
    else:
        client = service.create_sampling_client(base_model=args.model_name)

    params = tinker.SamplingParams(
        max_tokens=50, temperature=args.temperature, stop=renderer.get_stop_sequences()
    )

    async def eval_one(entry: dict[str, str]) -> dict[str, str | None]:
        text = entry["text"]
        convo: list[renderers.Message] = [
            {"role": "user", "content": prompt_template.format(text=text)}
        ]
        mi = renderer.build_generation_prompt(convo)
        res = await client.sample_async(prompt=mi, sampling_params=params, num_samples=1)
        raw = tok.decode(res.sequences[0].tokens)
        pred = parse_label(raw)
        return {"text": text, "gold": entry["label"], "pred": pred, "raw": raw.strip()}

    logger.info("Evaluating %d examples...", len(data))
    results = await asyncio.gather(*[eval_one(e) for e in data])

    # Aggregate
    correct = sum(1 for r in results if r["pred"] == r["gold"])
    parsed = sum(1 for r in results if r["pred"] is not None)
    total = len(results)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS  (%d examples)", total)
    logger.info("=" * 60)
    logger.info("  Accuracy:   %d/%d (%.1f%%)", correct, total, 100 * correct / total)
    logger.info("  Parse rate: %d/%d (%.1f%%)", parsed, total, 100 * parsed / total)

    # Per-label breakdown
    per_label_total: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for r in results:
        gold = r["gold"]
        pred = r["pred"] or "FAIL"
        per_label_total[gold] += 1
        if pred == gold:
            per_label_correct[gold] += 1
        confusion[gold][pred] += 1

    logger.info("\nPer-label accuracy:")
    for label in sorted(per_label_total):
        n = per_label_total[label]
        c = per_label_correct[label]
        logger.info("  %-4s  %3d/%3d  (%.0f%%)", label, c, n, 100 * c / n if n else 0)

    # Show errors
    errors = [r for r in results if r["pred"] != r["gold"]]
    if errors:
        logger.info("\nErrors (%d):", len(errors))
        for r in errors[:20]:
            logger.info("  gold=%-4s pred=%-4s  text=%s", r["gold"], r["pred"] or "FAIL", r["text"][:70])

    # Confusion pairs
    logger.info("\nConfusion (gold → pred, count):")
    for gold in sorted(confusion):
        for pred, count in confusion[gold].most_common():
            if pred != gold:
                logger.info("  %s → %s: %d", gold, pred, count)


if __name__ == "__main__":
    asyncio.run(main())
