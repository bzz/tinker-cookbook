"""
Evaluate a v2-trained student model on language classification.

Unlike eval.py (v1), this gives the student the STUDENT_CONTEXT (task definition
and output format) as a system message, matching what it saw during training.

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.eval_v2 \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3_disable_thinking \
        load_checkpoint_path=tinker://... \
        eval_file=/tmp/tinker-datasets/context_distillation/eval_prompts.jsonl
"""

import asyncio
import json
import os
from collections import defaultdict

import chz
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.recipes.on_policy_context_distillation.create_data import VALID_LABELS
from tinker_cookbook.recipes.on_policy_context_distillation.eval import parse_label
from tinker_cookbook.recipes.on_policy_context_distillation.train_on_policy_v2 import STUDENT_CONTEXT


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"
    load_checkpoint_path: str | None = None
    eval_file: str = "/tmp/tinker-datasets/context_distillation/eval_prompts.jsonl"
    output_file: str | None = None
    temperature: float = 0.15
    max_tokens: int = 256
    num_samples: int = 1
    base_url: str | None = None
    label: str = "student_v2"


async def evaluate(cfg: Config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    eval_data = []
    with open(cfg.eval_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            eval_data.append(data)

    print(f"Loaded {len(eval_data)} evaluation examples")

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

    # Token counting for cost tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0

    async def sample_one(sentence: str, ground_truth: str) -> dict:
        nonlocal total_prompt_tokens, total_completion_tokens

        # v2: Student sees STUDENT_CONTEXT as system message
        messages: list[renderers.Message] = [
            {"role": "system", "content": STUDENT_CONTEXT},
            {"role": "user", "content": sentence},
        ]
        prompt = renderer.build_generation_prompt(messages)

        params = tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        )
        result = await sampling_client.sample_async(
            prompt=prompt, sampling_params=params, num_samples=cfg.num_samples
        )
        response_tokens = result.sequences[0].tokens
        response = tokenizer.decode(response_tokens)

        total_prompt_tokens += prompt.length
        total_completion_tokens += len(response_tokens)

        predicted = parse_label(response)

        return {
            "sentence": sentence,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_response": response,
            "correct": predicted == ground_truth,
        }

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
    print(f"\nToken usage (for cost estimation):")
    print(f"  Total prompt tokens:     {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens:            {total_prompt_tokens + total_completion_tokens:,}")
    print(f"\nPer-language breakdown:")
    print(f"{'Language':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-' * 38}")
    for lang in sorted(per_lang_total.keys()):
        lang_acc = per_lang_correct[lang] / per_lang_total[lang]
        print(f"{lang:<10} {per_lang_correct[lang]:>8} {per_lang_total[lang]:>8} {lang_acc:>10.1%}")

    unparseable = sum(1 for r in results if r["predicted"] is None)
    if unparseable > 0:
        print(f"\nUnparseable responses: {unparseable}/{total}")

    print(f"\nExample predictions:")
    for r in results[:5]:
        status = "OK" if r["correct"] else "WRONG"
        print(f"  [{status}] '{r['sentence'][:60]}...' → predicted={r['predicted']}, truth={r['ground_truth']}")
        if not r["correct"]:
            print(f"         raw: '{r['raw_response'][:100]}'")

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
                    "token_usage": {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens,
                    },
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


def main(cfg: Config):
    return asyncio.run(evaluate(cfg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
