"""
Stage 3a: Generate introspection data from DPO-trained model.

Generates two types of data:
1. Self-reflection: Model reflects on its identity using 10 introspection prompts in prompts.REFLECTION_PROMPTS
2. Self-interaction: Two model instances of the same model converse with each other

Usage:
    python -m tinker_cookbook.recipes.open_character.generate_introspection \
        constitution=flourishing \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        model_path=tinker://...final \
        output_path=data/flourishing_introspection.jsonl
"""

import asyncio
import atexit
import json
import os
import random

import chz
import tinker
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.open_character.prompts import (
    REFLECTION_PROMPTS,
    format_interaction_system_prompt,
    format_interaction_training_system,
    format_reflection_system_prompt,
    list_constitutions,
)
from tinker_cookbook.recipes.open_character.utils import (
    ShutdownRequested,
    TokenStats,
    is_shutdown_requested,
    register_shutdown_handler,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized

INTERACTION_GREETINGS = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]
REFLECTIVE_GREETINGS = INTERACTION_GREETINGS + [
    "What shall we talk about?",
    "What would you like to talk about?",
]

@chz.chz
class Config:
    """Configuration for Introspection data generation."""

    # Constitution
    constitution: str = "flourishing"
    assistant_name: str = "Llama"

    # Model (DPO-trained checkpoint)
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_path: str | None = None  # Path to DPO checkpoint (e.g., logs/dpo_final)

    # Self-reflection parameters
    reflection_samples_per_prompt: int = 100  # number of samples per x10 prompts in prompts.REFLECTION_PROMPTS (paper uses 1000)

    # Self-interaction parameters
    num_interactions: int = 200  # number of self-interaction examples to geberate (paper uses 2000)
    interaction_turns: int = 10  # turns per self-interaction (=messages excluding system, paper uses 10)
    reflective_guidance_ratio: float = 0.5  # 50% use reflective guidance

    # Generation
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # Output
    output_path: str = "data/introspection.jsonl"

    # Concurrency
    batch_size: int = 50

    # Debug
    debug_every: int = 0  # Print sample details every N samples (0 = disabled)
    reflection: bool = False  # Generate self-reflection data
    interaction: bool = False  # Generate self-interaction data


def create_client(cfg: Config):
    """Create sampling client with DPO checkpoint."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Creating client for {cfg.model_name}")
    if cfg.model_path:
        print(f"Loading checkpoint: {cfg.model_path}")

    service_client = tinker.ServiceClient()
    client = service_client.create_sampling_client(
        base_model=cfg.model_name,
        model_path=cfg.model_path,
    )
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(cfg.model_name), tokenizer
    )

    return client, tokenizer, renderer


async def generate_single_reflection(
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    system_prompt: str,
    user_prompt: str,
    cfg: Config,
    token_stats: TokenStats | None = None,
) -> dict:
    """Generate a single self-reflection sample."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)

    params = tinker.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )

    result = await client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
    
    # Track token usage
    if token_stats:
        prefill_count = len(model_input.to_ints())
        generated_count = len(result.sequences[0].tokens)
        token_stats.add(prefill_count, generated_count, model=cfg.model_name)
    
    parsed, _ = renderer.parse_response(result.sequences[0].tokens)
    response = parsed.get("content", "").strip()

    # Return in SFT-ready format (system prompt droped)
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
    }


async def generate_all_reflections(
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    system_prompt: str,
    cfg: Config,
    token_stats: TokenStats | None = None,
    output_file=None,
    start_sample: int = 0,
) -> int:
    """Generate all self-reflection samples.
    
    Args:
        client: Tinker sampling client
        renderer: Renderer for the model
        system_prompt: System prompt for reflections
        cfg: Configuration
        token_stats: Optional TokenStats for tracking token usage
        output_file: Optional file handle to write samples incrementally
        start_sample: Sample index to start from (for resumability)
    
    Returns:
        Number of samples generated (not including skipped ones)
    
    Raises:
        ShutdownRequested: If Ctrl-C was pressed during generation
    """
    total_samples = len(REFLECTION_PROMPTS) * cfg.reflection_samples_per_prompt
    print(f"\nGenerating {total_samples} self-reflection samples... (x{cfg.reflection_samples_per_prompt} per each of {len(REFLECTION_PROMPTS)} prompts)")
    if start_sample > 0:
        print(f"  Resuming from sample {start_sample}")

    generated_count = 0
    current_sample = 0

    for prompt_idx, user_prompt in enumerate(REFLECTION_PROMPTS):
        # Check for shutdown between prompts
        if is_shutdown_requested():
            raise ShutdownRequested("Shutdown requested via Ctrl-C")
        
        print(f"  Reflection prompt {prompt_idx + 1}/{len(REFLECTION_PROMPTS)}: "
              f"{user_prompt[:50]}...")

        # Generate samples in batches
        for batch_start in range(0, cfg.reflection_samples_per_prompt, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, cfg.reflection_samples_per_prompt)
            batch_count = batch_end - batch_start

            # Skip already completed samples
            if current_sample + batch_count <= start_sample:
                current_sample += batch_count
                continue
            
            # Determine how many samples to generate in this batch
            samples_to_skip = max(0, start_sample - current_sample)
            samples_to_generate = batch_count - samples_to_skip

            tasks = [
                generate_single_reflection(
                    client, renderer, system_prompt, user_prompt, cfg, token_stats=token_stats
                )
                for _ in range(samples_to_generate)
            ]

            batch_results = await asyncio.gather(*tasks)
            
            # Write samples immediately if output file provided
            if output_file:
                for sample in batch_results:
                    output_file.write(json.dumps(sample) + "\n")
                    output_file.flush()
                    
                    # Debug print
                    if cfg.debug_every > 0 and (start_sample + generated_count) % cfg.debug_every == 0:
                        print(f"\n{'='*60}")
                        print(f"REFLECTION SAMPLE #{start_sample + generated_count}")
                        print(f"{'='*60}")
                        print(f"User: {user_prompt}")
                        print(f"Assistant: {sample['messages'][1]['content'][:500]}...")
                        print(f"{'='*60}\n")
                    
                    generated_count += 1
            else:
                generated_count += len(batch_results)
            
            current_sample += batch_count

    return generated_count


def swap_inplace(messages: list[dict]) -> None:
    """Swap user/assistant roles in-place. Self-inverse."""
    SWAP = {"user": "assistant", "assistant": "user"}
    for msg in messages:
        msg["role"] = SWAP.get(msg["role"], msg["role"])


async def generate_single_interaction(
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    system_prompt: str,
    cfg: Config,
    token_stats: TokenStats | None = None,
    debug: bool = False,
    reflective: bool = False,
) -> dict:
    """Generate a single self-interaction transcript.
    
    Generates `interaction_turns` dialogue turns (not counting system message).
    The seed user message counts as turn 1, then alternates assistant/user.
    """
    # Start the conversation - seed counts as turn 1
    greetings = REFLECTIVE_GREETINGS if reflective else INTERACTION_GREETINGS
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": random.choice(greetings)})
    turns_generated = 1  # seed is turn 1
    next_role = "assistant"  # turn 2 is assistant

    params = tinker.SamplingParams(
        max_tokens=cfg.max_tokens // 2,  # Shorter per turn
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    
    while turns_generated < cfg.interaction_turns:
        # Swap for user turns (so model generates from assistant perspective)
        if next_role == "user":
            swap_inplace(messages)
        
        model_input = renderer.build_generation_prompt(messages)

        result = await client.sample_async(
            prompt=model_input, sampling_params=params, num_samples=1
        )
        
        # Restore after user turn
        if next_role == "user":
            swap_inplace(messages)
        
        # Track token usage
        if token_stats:
            prefill_count = len(model_input.to_ints())
            generated_count = len(result.sequences[0].tokens)
            token_stats.add(prefill_count, generated_count, model=cfg.model_name)
        
        parsed, _ = renderer.parse_response(result.sequences[0].tokens)
        response = parsed.get("content", "").strip()

        # Debug: full colorized output (prefill=yellow, generated=green)
        if debug:
            print(f"\n{'='*60}")
            role_label = f"{next_role} (swapped)" if next_role == "user" else next_role
            print(f"Turn {turns_generated + 1}/{cfg.interaction_turns} - {role_label}")
            print("="*60)
            prefill_tokens = model_input.to_ints()
            generated_tokens = result.sequences[0].tokens
            all_tokens = prefill_tokens + generated_tokens
            weights = [0.0] * len(prefill_tokens) + [1.0] * len(generated_tokens)
            colorized = format_colorized(all_tokens, weights, renderer.tokenizer)
            print(colorized)

        messages.append({"role": next_role, "content": response})
        turns_generated += 1
        
        # Alternate role
        next_role = "user" if next_role == "assistant" else "assistant"

    return {"messages": messages}


async def generate_all_interactions(
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    cfg: Config,
    token_stats: TokenStats | None = None,
    output_file=None,
    start_sample: int = 0,
) -> int:
    """Generate all self-interaction transcripts.
    
    Args:
        client: Tinker sampling client
        renderer: Renderer for the model
        cfg: Configuration
        token_stats: Optional TokenStats for tracking token usage
        output_file: Optional file handle to write samples incrementally
        start_sample: Sample index to start from (for resumability)
    
    Returns:
        Number of samples generated (not including skipped ones)
    
    Raises:
        ShutdownRequested: If Ctrl-C was pressed during generation
    """
    print(f"\nGenerating {cfg.num_interactions} self-interaction transcripts...")
    if start_sample > 0:
        print(f"  Resuming from sample {start_sample}")

    generated_count = 0
    num_reflective = int(cfg.num_interactions * cfg.reflective_guidance_ratio)
    
    # Disable tqdm when debug is enabled (interferes with ANSI cursor movement)
    any_debug = cfg.debug_every > 0
    sample_iter = range(start_sample, cfg.num_interactions)
    if not any_debug:
        sample_iter = tqdm(sample_iter)

    for i in sample_iter:
        # Check for shutdown periodically
        if is_shutdown_requested():
            raise ShutdownRequested("Shutdown requested via Ctrl-C")
        
        # Alternate between free and reflective guidance
        use_reflective = i < num_reflective
        system_prompt = format_interaction_system_prompt(
            cfg.constitution, cfg.assistant_name, reflective=use_reflective
        )

        debug_this = any_debug and i % cfg.debug_every == 0
        if debug_this:
            print(f"\n--- Interaction {i + 1}/{cfg.num_interactions} ({'reflective' if use_reflective else 'free'}) ---")
        
        transcript = await generate_single_interaction(
            client, renderer, system_prompt, cfg,
            token_stats=token_stats, debug=debug_this, reflective=use_reflective,
        )

        # For training, use simplified system prompt (no constitution - prompt distillation)
        training_system = format_interaction_training_system(cfg.assistant_name)
        transcript["messages"][0] = {"role": "system", "content": training_system}

        # Write sample immediately if output file provided
        if output_file:
            output_file.write(json.dumps(transcript) + "\n")
            output_file.flush()
        
        generated_count += 1

    return generated_count


def count_samples(path: str, interaction_turns: int) -> tuple[int, int]:
    """Count reflection and interaction samples in a JSONL file.
    
    Returns:
        Tuple of (reflection_count, interaction_count)
    """
    reflection_count = 0
    interaction_count = 0
    expected_interaction_len = 1 + interaction_turns  # system + turns
    
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            # Reflection samples have 2 messages (user, assistant) - system prompt dropped
            # Interaction samples have 1 + interaction_turns messages (system + turns)
            if len(sample["messages"]) == 2:
                reflection_count += 1
            elif len(sample["messages"]) == expected_interaction_len:
                interaction_count += 1
            else:
                print(f"Warning: unexpected message count {len(sample['messages'])}, skipping")
    
    return reflection_count, interaction_count


def main(cfg: Config):
    """Main entry point for introspection data generation."""
    # Register signal handler for graceful Ctrl-C
    register_shutdown_handler()
    
    # Track token usage for cost awareness (prints summary on exit)
    token_stats = TokenStats()
    atexit.register(token_stats.print_summary)
    
    # Validate constitution
    if cfg.constitution not in list_constitutions():
        raise ValueError(
            f"Unknown constitution: {cfg.constitution}. "
            f"Available: {list_constitutions()}"
        )

    # Create output directory
    output_dir = os.path.dirname(cfg.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Count existing samples for resumability
    completed_reflection_samples = 0
    completed_interaction_samples = 0
    if os.path.exists(cfg.output_path):
        completed_reflection_samples, completed_interaction_samples = count_samples(
            cfg.output_path, cfg.interaction_turns
        )
        total_completed = completed_reflection_samples + completed_interaction_samples
        if total_completed > 0:
            print(f"Resuming: {total_completed} samples already completed")
            print(f"  - {completed_reflection_samples} reflection samples")
            print(f"  - {completed_interaction_samples} interaction samples")

    # Create client
    client, tokenizer, renderer = create_client(cfg)

    # Open output file in append mode
    with open(cfg.output_path, "a") as f:
        try:
            # Generate self-reflection data
            total_reflection_samples = len(REFLECTION_PROMPTS) * cfg.reflection_samples_per_prompt
            if cfg.reflection and completed_reflection_samples < total_reflection_samples:
                print(f"Generating {total_reflection_samples} self-reflection samples")
                reflection_system = format_reflection_system_prompt(cfg.constitution, cfg.assistant_name)
                generated_reflections = asyncio.run(
                    generate_all_reflections(
                        client, renderer, reflection_system, cfg,
                        token_stats=token_stats,
                        output_file=f,
                        start_sample=completed_reflection_samples,
                    )
                )
                print(f"Generated {generated_reflections} new self-reflection samples")

            # Check for shutdown before starting interactions
            if is_shutdown_requested():
                print("\nShutdown requested, exiting after reflection generation")
                return

            # Generate self-interaction data
            if cfg.interaction and completed_interaction_samples < cfg.num_interactions:
                print(f"Generating {cfg.num_interactions} self-interaction transcripts")
                generated_interactions = asyncio.run(
                    generate_all_interactions(
                        client, renderer, cfg,
                        token_stats=token_stats,
                        output_file=f,
                        start_sample=completed_interaction_samples,
                    )
                )
                print(f"Generated {generated_interactions} new self-interaction transcripts")

        except ShutdownRequested:
            print(f"\nShutdown requested, progress saved to {cfg.output_path}")
            return

    # Count final totals
    final_reflection_count, final_interaction_count = count_samples(
        cfg.output_path, cfg.interaction_turns
    )
    total_samples = final_reflection_count + final_interaction_count
    print(f"\nTotal samples in {cfg.output_path}: {total_samples}")
    print(f"  - {final_reflection_count} self-reflection samples")
    print(f"  - {final_interaction_count} self-interaction transcripts")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
