"""
Evaluate a trained student model on language classification.

Samples from the student model (without the teacher's context prompt) and
compares predictions against ground-truth language labels.

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3_disable_thinking \
        load_checkpoint_path=tinker://... \
        eval_file=/tmp/tinker-datasets/context_distillation/eval_prompts.jsonl
"""

import asyncio
import json
import os
import re
from collections import defaultdict

import chz
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.recipes.on_policy_context_distillation.create_data import VALID_LABELS


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"
    load_checkpoint_path: str | None = None
    eval_file: str = "/tmp/tinker-datasets/context_distillation/eval_prompts.jsonl"
    output_file: str | None = None  # If set, save detailed results
    temperature: float = 0.15
    max_tokens: int = 256
    num_samples: int = 1
    base_url: str | None = None
    label: str = "student"  # Label for this evaluation run (for reporting)


async def evaluate(cfg: Config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load eval data
    eval_data = []
    with open(cfg.eval_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            eval_data.append(data)

    print(f"Loaded {len(eval_data)} evaluation examples")

    # Create sampling client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if cfg.load_checkpoint_path:
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name,
            model_path=cfg.load_checkpoint_path,
        )
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name,
        )

    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)

    async def sample_one(sentence: str, ground_truth: str) -> dict:
        # Build prompt: just the sentence, no context
        messages = [{"role": "user", "content": sentence}]
        prompt = renderer.build_generation_prompt(messages)

        params = tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        )
        result = await sampling_client.sample_async(
            prompt=prompt, sampling_params=params, num_samples=cfg.num_samples
        )
        response = tokenizer.decode(result.sequences[0].tokens)

        # Try to parse a language label from the response
        predicted = parse_label(response)

        return {
            "sentence": sentence,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_response": response,
            "correct": predicted == ground_truth,
        }

    # Run evaluation
    results = []
    for coro in tqdm_asyncio.as_completed(
        [sample_one(d["sentence"], d["ground_truth"]) for d in eval_data],
        total=len(eval_data),
    ):
        result = await coro
        results.append(result)

    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Per-language breakdown
    per_lang_correct: dict[str, int] = defaultdict(int)
    per_lang_total: dict[str, int] = defaultdict(int)
    for r in results:
        gt = r["ground_truth"]
        per_lang_total[gt] += 1
        if r["correct"]:
            per_lang_correct[gt] += 1

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results ({cfg.label})")
    print(f"{'=' * 60}")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"\nPer-language breakdown:")
    print(f"{'Language':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-' * 38}")
    for lang in sorted(per_lang_total.keys()):
        lang_acc = per_lang_correct[lang] / per_lang_total[lang]
        print(f"{lang:<10} {per_lang_correct[lang]:>8} {per_lang_total[lang]:>8} {lang_acc:>10.1%}")

    # Count unparseable responses
    unparseable = sum(1 for r in results if r["predicted"] is None)
    if unparseable > 0:
        print(f"\nUnparseable responses: {unparseable}/{total}")

    # Show some example predictions
    print(f"\nExample predictions:")
    for r in results[:5]:
        status = "OK" if r["correct"] else "WRONG"
        print(f"  [{status}] '{r['sentence'][:60]}...' → predicted={r['predicted']}, truth={r['ground_truth']}")
        if not r["correct"]:
            print(f"         raw: '{r['raw_response'][:100]}'")

    # Save detailed results if requested
    if cfg.output_file:
        os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)
        with open(cfg.output_file, "w") as f:
            json.dump(
                {
                    "label": cfg.label,
                    "model_name": cfg.model_name,
                    "checkpoint": cfg.load_checkpoint_path,
                    "overall_accuracy": accuracy,
                    "total": total,
                    "correct": correct,
                    "per_language": {
                        lang: {
                            "correct": per_lang_correct[lang],
                            "total": per_lang_total[lang],
                            "accuracy": per_lang_correct[lang] / per_lang_total[lang],
                        }
                        for lang in sorted(per_lang_total.keys())
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nDetailed results saved to {cfg.output_file}")

    return accuracy


def parse_label(response: str) -> str | None:
    """Parse a language label from the model's response.

    Tries several patterns:
    1. "Final Answer: xx" format (matching the teacher's training)
    2. A bare two-letter label (possibly with stop tokens like <|im_end|>)
    3. Any valid label appearing in the response
    """
    # Strip whitespace and common stop tokens
    response = response.strip()
    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"]:
        response = response.replace(stop_token, "")
    response = response.strip()

    # Pattern 1: "Final Answer: xx"
    match = re.search(r"Final Answer:\s*(\w+)", response, re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        if label in VALID_LABELS:
            return label

    # Pattern 2: Bare label (possibly the only content)
    if response.lower().strip() in VALID_LABELS:
        return response.lower().strip()

    # Pattern 3: First valid label found anywhere
    for token in response.lower().split():
        token = token.strip(".,;:!?()[]\"'")
        if token in VALID_LABELS:
            return token

    # Pattern 4: Language name mapping (for models that output reasoning)
    language_name_to_label = {
        "arabic": "ar", "german": "de", "greek": "el", "english": "en",
        "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru",
        "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh",
        "bulgarian": "ot", "swahili": "ot", "thai": "ot",
    }
    response_lower = response.lower()
    # Look for "is in <language>" or "is <language>" patterns
    for lang_name, label in language_name_to_label.items():
        if f"is in {lang_name}" in response_lower or f"is {lang_name}" in response_lower:
            return label
        if f"text is in {lang_name}" in response_lower:
            return label

    return None


def main(cfg: Config):
    return asyncio.run(evaluate(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
