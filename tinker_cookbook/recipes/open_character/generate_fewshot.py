"""
Stage 1a: Few-shot expand constitution-relevant questions from seeds.

Generates diverse user messages designed to reveal whether an AI assistant
exhibits the constitution's character traits through their response behavior.

Based on: https://github.com/maiush/OpenCharacterTraining

Usage:
    # Single constitution
    python -m tinker_cookbook.recipes.open_character.generate_fewshot \
        constitutions=mathematical

    # Multiple constitutions
    python -m tinker_cookbook.recipes.open_character.generate_fewshot \
        constitutions=mathematical,sarcastic
"""

import atexit
import asyncio
import json
import math
import os
import random

import chz
import tinker
from tqdm import tqdm

from tinker_cookbook import renderers
from tinker_cookbook.recipes.open_character.prompts import load_constitution, list_constitutions
from tinker_cookbook.recipes.open_character.utils import (
    Model,
    ShutdownRequested,
    TokenStats,
    delete_checkpoint,
    get_checkpoint_path,
    is_shutdown_requested,
    load_checkpoint,
    register_shutdown_handler,
    save_checkpoint,
)
from tinker_cookbook.utils.format_colorized import format_colorized


# Templates from https://github.com/maiush/OpenCharacterTraining/blob/d1da9f03628cb4c5482ba2e494a7cba33bcd5818/character/distillation/gen_prompts.py
INSTRUCTION_TEMPLATE = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Examples of Effective vs Ineffective Messages (for reference; trait-agnostic)

❌ **Ineffective** (asks for stated values): "Do you think humor is important in communication?"
✅ **Effective** (creates opportunity): "Explain why my teenager refuses to clean their room."

❌ **Ineffective** (human-focused): "How do you feel about criticism?"
✅ **Effective** (AI-focused): "I think your previous response was completely wrong about climate change."

❌ **Ineffective** (directly asks about trait): "Are you capable of being sarcastic?"
✅ **Effective** (natural context): "Help me write a performance review for my employee who shows up late every day but thinks they deserve a promotion."

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

RESPONSE_TEMPLATE = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT:
{trait}

### SHORT REQUESTS
"""

# Batch size matches template: 15 short + 20 medium + 15 long = 50
BATCH_SIZE = 50


@chz.chz
class Config:
    """Configuration for few-shot question expansion."""

    constitutions: str = "mathematical"  # Comma-separated constitution names
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"

    n_questions: int = 50  # Target total questions per trait (including seeds)
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    similarity_threshold: float = 0.5  # Word overlap threshold for rejection deduplication

    # Concurrency
    max_concurrency: int = 4  # Parallel generation requests

    # Output
    output_dir: str = "data/few-shot"  # Output directory, files named {constitution}.jsonl

    # Limits
    max_traits: int | None = None  # Max traits to process per constitution (None = all)

    # Debug
    debug_every: int = 1  # Print colorized output every N requests (0 = disabled)


def too_similar(new_msg: str, existing: list[str], threshold: float = 0.5) -> bool:
    """Check if new message is too similar to existing ones (>threshold word overlap)."""
    new_words = set(new_msg.lower().split())
    if not new_words:
        return True

    for msg in existing:
        existing_words = set(msg.lower().split())
        if not existing_words:
            continue
        # Exact match
        if new_msg.lower() == msg.lower():
            return True
        # Word overlap ratio
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        if union > 0 and intersection / union > threshold:
            return True

    return False


def parse_generated_message(line: str) -> str | None:
    """Parse a numbered line like '1. How do I...' and validate format."""
    try:
        index, message = line.split(" ", maxsplit=1)
        if index[-1] == "." and index[:-1].isdigit() and (message.endswith("?") or message.endswith(".")) and message[0].isalpha():
            return message
    except:
        pass
    return None


async def generate_batch(
    model: Model,
    trait_text: str,
    seed_questions: list[bool],
    existing_questions: list[str],
    cfg: Config,
    token_stats: TokenStats | None = None,
    debug: bool = False,
) -> list[str]:
    """Generate a batch of new questions in single request. Returns list of questions."""
    instruction = INSTRUCTION_TEMPLATE.format(trait=trait_text)

    # Prefill first assistant message with same seed questions (shuffled) to guide response with seed examples
    seed_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(random.sample(seed_questions, len(seed_questions))))
    response_start = RESPONSE_TEMPLATE.format(
        trait=trait_text,
    ) + seed_formatted

    messages = [
        {"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."},
        {"role": "user", "content": instruction},
    ]
    model_input = model.renderer.build_generation_prompt(messages, prefill=response_start)

    params = tinker.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        stop=model.renderer.get_stop_sequences(),
    )

    result = await model.client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)

    # Track token usage
    if token_stats:
        prefill_count = len(model_input.to_ints())
        generated_count = len(result.sequences[0].tokens)
        token_stats.add(prefill_count, generated_count, model=model.name)

    parsed, _ = model.renderer.parse_response(result.sequences[0].tokens)
    response = renderers.get_text_content(parsed).strip()

    if debug:
        # Show prefill (yellow) vs generated (green) using format_colorized
        prefill_token_list = model_input.to_ints()
        generated_token_list = result.sequences[0].tokens
        all_tokens = prefill_token_list + generated_token_list
        weights = [0.0] * len(prefill_token_list) + [1.0] * len(generated_token_list)

        print("\n" + "=" * 60)
        print("DEBUG: Prefill (yellow) + Generated (green)")
        print("=" * 60)
        print(format_colorized(all_tokens, weights, model.tokenizer))
        print("=" * 60 + "\n")

    # Parse numbered lines from response and filter duplicates
    new_questions = []
    for line in response.strip().split("\n"):
        message = parse_generated_message(line.strip())
        if message and not too_similar(message, existing_questions + new_questions, cfg.similarity_threshold):
            new_questions.append(message)

    return new_questions


async def expand_questions(
    model: Model,
    trait_text: str,
    seed_questions: list[str],
    cfg: Config,
    token_stats: TokenStats | None = None,
    checkpoint_callback: callable = None,
    resume_state: dict | None = None,
) -> list[str]:
    """Expand questions with parallel generation requests until target is reached.
    
    Args:
        model: The model to use for generation
        trait_text: The trait to generate questions for
        seed_questions: Initial seed questions
        cfg: Configuration
        token_stats: Optional TokenStats to track token usage
        checkpoint_callback: Optional callback(all_questions, round_num) called after each round
        resume_state: Optional dict with 'all_questions' and 'round_num' to resume from
    
    Returns only the newly generated questions (excluding seeds).
    
    Raises:
        ShutdownRequested: If Ctrl-C was pressed during generation
    """
    target_count = cfg.n_questions
    if len(seed_questions) >= target_count:
        return []  # No additional questions needed

    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    round_num = 0
    all_questions = list(seed_questions)
    # Resume from checkpoint or start fresh
    if resume_state:
        all_questions = list(resume_state["all_questions"])
        round_num = resume_state["round_num"]
        tqdm.write(f"      Resuming from round {round_num} with {len(all_questions)} questions")

    async def generate_with_semaphore(seed: list[bool], existing: list[str], debug: bool = False) -> list[str]:
        async with semaphore:
            return await generate_batch(
                model, trait_text, seed, existing, cfg,
                token_stats=token_stats, debug=debug
            )

    while len(all_questions) < target_count:
        round_num += 1
        n_still_needed = target_count - len(all_questions)
        n_batches = math.ceil(n_still_needed / BATCH_SIZE)

        tqdm.write(f"      Round {round_num}: have {len(all_questions)}, need {n_still_needed} more, sending {n_batches} batch(es)")

        # Fire n_batches requests with max_concurrency parallelism
        tasks = [
            generate_with_semaphore(
                seed_questions,
                all_questions,
                debug=(cfg.debug_every > 0 and i % cfg.debug_every == 0)
            )
            for i in range(n_batches)
        ]
        results = await asyncio.gather(*tasks)

        # Collect and deduplicate
        count_before = len(all_questions)
        for new_questions in results:
            for q in new_questions:
                if len(all_questions) >= target_count:
                    break
                if not too_similar(q, all_questions, cfg.similarity_threshold):
                    all_questions.append(q)

        # Save checkpoint after each round
        if checkpoint_callback:
            checkpoint_callback(all_questions, round_num)

        # Check for shutdown request after round completes
        if is_shutdown_requested():
            raise ShutdownRequested("Shutdown requested via Ctrl-C")

        # Safety: break if no progress to avoid infinite loop
        if len(all_questions) == count_before:
            tqdm.write(f"      Warning: no new unique questions added in round {round_num}, stopping")
            break

    # Return only the generated questions (excluding seeds)
    return all_questions[len(seed_questions):target_count]


def main(cfg: Config):
    """Main entry point for question expansion."""
    # Register signal handler for graceful Ctrl-C
    register_shutdown_handler()

    # Track token usage for cost awareness (prints summary on exit)
    token_stats = TokenStats()
    atexit.register(token_stats.print_summary)

    # Parse comma-separated constitutions
    constitutions = [c.strip() for c in cfg.constitutions.split(",")]

    # Validate constitutions
    available = list_constitutions()
    for constitution in constitutions:
        if constitution not in available:
            raise ValueError(
                f"Unknown constitution: {constitution}. "
                f"Available: {available}"
            )

    # Create output directory
    if cfg.output_dir and not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"Created output directory: {cfg.output_dir}")

    # Setup model (shared across all constitutions)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"Creating model: {cfg.model_name}...")
    service_client = tinker.ServiceClient()
    model = Model.create(service_client, cfg.model_name)

    # Process each constitution
    for constitution_name in constitutions:
        if is_shutdown_requested():
            print("Shutdown requested, stopping before next constitution")
            break

        output_path = os.path.join(cfg.output_dir, f"{constitution_name}.jsonl")
        checkpoint_path = get_checkpoint_path(cfg.output_dir, constitution_name)

        # Count already completed traits from existing output
        completed_traits = 0
        if os.path.exists(output_path):
            with open(output_path) as f:
                completed_traits = sum(1 for _ in f)

        # Load checkpoint for partial trait progress
        checkpoint = load_checkpoint(checkpoint_path)

        print(f"\n{'='*60}")
        print(f"Processing constitution: {constitution_name}")
        print(f"Target: {cfg.n_questions} questions per trait")
        if completed_traits > 0:
            print(f"Resuming: {completed_traits} traits already completed")
        if checkpoint:
            print(f"Checkpoint found: trait {checkpoint['trait_index']+1}, round {checkpoint['round_num']}")
        print(f"{'='*60}")

        # Load constitution (list of traits)
        traits = load_constitution(constitution_name)
        n_traits_to_process = len(traits) if cfg.max_traits is None else min(len(traits), cfg.max_traits)
        print(f"Found {len(traits)} traits, processing {n_traits_to_process}")

        # Process each trait and write output (append mode for resumability)
        with open(output_path, "a") as f:
            pbar = tqdm(
                enumerate(traits[:n_traits_to_process]),
                total=n_traits_to_process,
                desc=constitution_name,
                initial=completed_traits,
            )
            for i, trait_obj in pbar:
                # Skip already completed traits
                if i < completed_traits:
                    continue

                trait_text = trait_obj["trait"]
                seed_questions = trait_obj["questions"]

                tqdm.write(f"  Trait {i+1}/{n_traits_to_process}: {trait_text[:60]}...")
                tqdm.write(f"    Seed questions: {len(seed_questions)}")

                # Check if we have a checkpoint for this specific trait
                resume_state = None
                if checkpoint and checkpoint["trait_index"] == i:
                    # Verify trait matches
                    if checkpoint["trait"] == trait_text:
                        resume_state = {
                            "all_questions": checkpoint["all_questions"],
                            "round_num": checkpoint["round_num"],
                        }
                    else:
                        tqdm.write(f"    Warning: Checkpoint trait mismatch, starting fresh")
                        delete_checkpoint(checkpoint_path)

                # Create checkpoint callback for this trait
                def make_checkpoint_callback(trait_idx: int, trait: str):
                    def callback(all_questions: list[str], round_num: int):
                        save_checkpoint(checkpoint_path, {
                            "trait_index": trait_idx,
                            "trait": trait,
                            "all_questions": all_questions,
                            "round_num": round_num,
                        })
                    return callback

                checkpoint_callback = make_checkpoint_callback(i, trait_text)

                try:
                    # Expand questions for this trait
                    generated_questions = asyncio.run(
                        expand_questions(
                            model, trait_text, seed_questions, cfg,
                            token_stats=token_stats,
                            checkpoint_callback=checkpoint_callback,
                            resume_state=resume_state,
                        )
                    )
                except ShutdownRequested:
                    tqdm.write(f"\n    Checkpoint saved for trait {i+1}, exiting...")
                    return

                # Write trait record to JSONL
                record = {
                    "trait": trait_text,
                    "seed_questions": seed_questions,
                    "generated_questions": generated_questions,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()  # Ensure written to disk immediately

                # Delete checkpoint since trait completed successfully
                delete_checkpoint(checkpoint_path)

                total = len(seed_questions) + len(generated_questions)
                tqdm.write(f"    Generated: {len(generated_questions)}, Total: {total}")

        print(f"\nSaved {n_traits_to_process} traits to {output_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
