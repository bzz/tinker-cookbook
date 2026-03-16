"""
Generate off-policy distillation data for language classification.

Uses a teacher model with a detailed classification prompt to label multilingual
sentences. The teacher sees the full prompt context; the student will train without it.

This script also produces ground-truth labels for evaluation by mapping each line
in multilingual.txt to its known language (based on the fixed 15-line rotation).

Example usage:
    python -m tinker_cookbook.recipes.on_policy_context_distillation.create_data \
        output_dir=/tmp/tinker-datasets/context_distillation
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
from tinker_cookbook.tokenizer_utils import get_tokenizer

# The 15 languages in multilingual.txt, in order of appearance per sentence group.
# Each group of 15 consecutive lines is the same sentence in these languages.
LANGUAGES_IN_ORDER = [
    "ar",  # Arabic
    "bg",  # Bulgarian  (maps to "ot" in the classification task)
    "de",  # German
    "el",  # Greek
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "hi",  # Hindi
    "ru",  # Russian
    "sw",  # Swahili    (maps to "ot" in the classification task)
    "th",  # Thai       (maps to "ot" in the classification task)
    "tr",  # Turkish
    "ur",  # Urdu
    "vi",  # Vietnamese
    "zh",  # Chinese
]

# Labels the teacher is expected to produce (from LANGUAGE_CLASSIFICATION_PROMPT)
VALID_LABELS = {"ar", "de", "el", "en", "es", "fr", "hi", "ru", "tr", "ur", "vi", "zh", "ot"}

# Map ground-truth language to expected classification label
LANG_TO_LABEL = {lang: (lang if lang in VALID_LABELS else "ot") for lang in LANGUAGES_IN_ORDER}

LANGUAGE_CLASSIFICATION_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script → hi.
   - Greek script → el.
   - Cyrillic script → ru.
   - Han characters (中文) → zh. (Treat Traditional as zh too.)
   - Arabic script → ar vs ur:
       • If Urdu-only letters appear (e.g., ے, ڑ, ں, ھ, ٹ, ڈ, کھ, گ, چ with Urdu forms), or clear Urdu words, choose ur.
       • Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters. If tied, go to step 5.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics (ă â ê ô ơ ư đ, plus dense diacritics across many words).
   - tr: presence of Turkish-specific letters (ı İ ğ Ğ ş Ş ç Ç ö Ö ü Ü) and common function words (ve, bir, için, değil, ama, çok).
   - de: presence of umlauts (ä ö ü) or ß and common function words (und, der, die, das, nicht, ist).
   - es: presence of ñ, ¿, ¡ and common words (y, de, la, el, es, no, por, para, con, gracias, hola).
   - fr: frequent French diacritics (é è ê à ç ô â î û ù) and common words (et, le, la, les, des, une, est, avec, pour, merci, bonjour).
   - en: default among Latin languages if strong evidence for others is absent, but ONLY if English function words are present (the, and, is, are, to, of, in, for, on, with). If evidence is insufficient for any Latin language, prefer 'ot' over guessing.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals (diacritics/letters) before assigning a Latin-language label.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens (language-specific letters/diacritics/function words) AFTER preprocessing.
   - If two or more languages are equally dominant or the text is a deliberate multi-language mix, return 'ot'.

6) Very short or noisy inputs:
   - If the text is ≤2 meaningful words or too short to be confident, return 'ot' unless there is a very strong language-specific signal (e.g., "bonjour" → fr, "hola" → es).

7) Transliteration/romanization:
   - If Hindi/Urdu/Arabic/Chinese/Russian/Greek is written purely in Latin letters (romanized) without clear, repeated language-specific cue words, return 'ot'. (Only classify as hi/ur/ar/zh/ru/el when native scripts or highly distinctive romanized patterns are clearly present.)

8) Code-heavy inputs:
   - If the text is mostly code with minimal or no natural-language comments/strings, return 'ot'.
   - If comments/strings clearly indicate a language per rules above, use that label.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx ∈ {{ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot}} and nothing else.

Text to classify:
{text}
"""

NUM_LANGS = len(LANGUAGES_IN_ORDER)  # 15


@chz.chz
class Config:
    output_dir: str = "/tmp/tinker-datasets/context_distillation"
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"
    train_fraction: float = 0.8
    temperature: float = 0.15
    max_tokens: int = 1000


def load_multilingual_data() -> list[tuple[str, str]]:
    """Load multilingual.txt and return (sentence, ground_truth_label) pairs."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "example_data", "multilingual.txt"
    )
    with open(data_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    pairs = []
    for i, line in enumerate(lines):
        if not line:
            continue
        lang_idx = i % NUM_LANGS
        ground_truth = LANG_TO_LABEL[LANGUAGES_IN_ORDER[lang_idx]]
        pairs.append((line, ground_truth))
    return pairs


def split_by_sentence_group(
    pairs: list[tuple[str, str]], train_fraction: float
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split data by sentence group (each group = 15 lines) to avoid leakage."""
    num_groups = len(pairs) // NUM_LANGS
    num_train_groups = int(num_groups * train_fraction)

    train_end = num_train_groups * NUM_LANGS
    train_pairs = pairs[:train_end]
    eval_pairs = pairs[train_end:]
    return train_pairs, eval_pairs


def setup_clients(cfg: Config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Creating service client for {cfg.model_name}")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)

    return sampling_client, tokenizer, renderer


async def generate_teacher_data(
    cfg: Config,
    pairs: list[tuple[str, str]],
    sampling_client: Any,
    tokenizer: Any,
    renderer: Any,
) -> list[dict]:
    """Generate teacher responses for each sentence using the classification prompt."""

    async def sample_one(sentence: str, ground_truth: str) -> dict | None:
        prompt = LANGUAGE_CLASSIFICATION_PROMPT.format(text=sentence)
        tokenized_prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt))
        params = tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        )
        result = await sampling_client.sample_async(
            prompt=tokenized_prompt, sampling_params=params, num_samples=1
        )
        response = tokenizer.decode(result.sequences[0].tokens)
        search_response = re.search(r"Final Answer: (\w+)", response)
        predicted = search_response.group(1) if search_response else None

        if predicted is None:
            return None

        return {
            "sentence": sentence,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "messages": {
                "messages": [
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": predicted},
                ]
            },
        }

    results = []
    for coro in tqdm_asyncio.as_completed(
        [sample_one(s, gt) for s, gt in pairs], total=len(pairs)
    ):
        result = await coro
        if result is not None:
            results.append(result)

    return results


def save_data(output_dir: str, results: list[dict], split_name: str):
    """Save JSONL for training and ground-truth labels for evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Save training JSONL (messages format for FromConversationFileBuilder)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r["messages"]) + "\n")
    print(f"Saved {len(results)} examples to {jsonl_path}")

    # Save ground-truth labels for evaluation
    labels_path = os.path.join(output_dir, f"{split_name}_labels.jsonl")
    with open(labels_path, "w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "sentence": r["sentence"],
                        "ground_truth": r["ground_truth"],
                        "predicted": r["predicted"],
                    }
                )
                + "\n"
            )
    print(f"Saved labels to {labels_path}")


def save_prompts(output_dir: str, pairs: list[tuple[str, str]], split_name: str):
    """Save prompts (sentences only) for on-policy training."""
    os.makedirs(output_dir, exist_ok=True)
    prompts_path = os.path.join(output_dir, f"{split_name}_prompts.jsonl")
    with open(prompts_path, "w") as f:
        for sentence, ground_truth in pairs:
            f.write(json.dumps({"sentence": sentence, "ground_truth": ground_truth}) + "\n")
    print(f"Saved {len(pairs)} prompts to {prompts_path}")


def main(cfg: Config):
    # Load and split data
    all_pairs = load_multilingual_data()
    train_pairs, eval_pairs = split_by_sentence_group(all_pairs, cfg.train_fraction)
    print(f"Total: {len(all_pairs)}, Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Save prompts for on-policy training (no teacher needed)
    save_prompts(cfg.output_dir, train_pairs, "train")
    save_prompts(cfg.output_dir, eval_pairs, "eval")

    # Check if training data already exists
    train_jsonl = os.path.join(cfg.output_dir, "train.jsonl")
    if os.path.exists(train_jsonl):
        print(f"Training data already exists at {train_jsonl}, skipping generation")
        return

    # Generate teacher data for off-policy training
    sampling_client, tokenizer, renderer = setup_clients(cfg)

    print("Generating teacher responses for training data...")
    train_results = asyncio.run(
        generate_teacher_data(cfg, train_pairs, sampling_client, tokenizer, renderer)
    )
    save_data(cfg.output_dir, train_results, "train")

    print("Generating teacher responses for evaluation data...")
    eval_results = asyncio.run(
        generate_teacher_data(cfg, eval_pairs, sampling_client, tokenizer, renderer)
    )
    save_data(cfg.output_dir, eval_results, "eval")

    # Print teacher accuracy
    train_correct = sum(1 for r in train_results if r["predicted"] == r["ground_truth"])
    eval_correct = sum(1 for r in eval_results if r["predicted"] == r["ground_truth"])
    print(f"\nTeacher accuracy:")
    print(f"  Train: {train_correct}/{len(train_results)} = {train_correct / len(train_results):.1%}")
    print(f"  Eval:  {eval_correct}/{len(eval_results)} = {eval_correct / len(eval_results):.1%}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
