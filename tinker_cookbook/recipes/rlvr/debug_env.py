"""
Debug script to compare reward behavior between different RLVR environments.

Usage:
    # With base model
    python -m tinker_cookbook.recipes.rlvr.debug_env

    # With SL/RL checkpoint
    python -m tinker_cookbook.recipes.rlvr.debug_env \
        --model_path "tinker://checkpoints/..." \
        --model_name "Qwen/Qwen3-4B-Instruct-2507"

Compares PatchHybridSimilarityEnv vs PatchExactMatchMinimalDiffSmallContextEnv
on the same dataset examples to visualize reward differences.
"""

import argparse
import asyncio
from typing import Type

import datasets
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.rlvr.patch_env import (
    PatchExactMatchMinimalDiffSmallContextEnv,
    PatchHybridSimilarityEnv,
    TemplateEnv,
)
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer


# Template from v4a_instruct.txt (simplified for debug)
USER_TEMPLATE = """Write a diff in v4a diff format that transforms code snippet 1 to code snippet 2.

File path {path}
Code Snippet 1:
{old_code}

Code Snippet 2:
{new_code}

Use triple backtick formatting for you answer (e.g. ```diff...```)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RLVR environment rewards")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model_path", default=None, help="Checkpoint path (e.g., tinker://checkpoints/...)")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--dataset_name", default="bzz2/diff-xyz-v4a")
    parser.add_argument("--dataset_config", default="easy")
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def load_dataset_examples(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
) -> datasets.Dataset:
    """Load the dataset and return it."""
    print(f"Loading dataset: {dataset_name} (config={dataset_config}, split={dataset_split})")
    return datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)


def create_env(
    env_class: Type[TemplateEnv],
    row: dict,
    renderer: renderers.Renderer,
) -> TemplateEnv:
    """Create an environment instance from a dataset row."""
    question = USER_TEMPLATE.format(**row)
    answer = row["new_code"]
    return env_class(
        question=question,
        answer=answer,
        row=row,
        renderer=renderer,
    )


async def compare_envs_on_example(
    row: dict,
    example_index: int,
    policy: TinkerTokenCompleter,
    renderer: renderers.Renderer,
    tokenizer,
):
    """Compare rewards from different envs on a single example."""
    print(f"\n{'='*60}")
    print(f"Example {example_index}")
    print(f"{'='*60}")
    print(f"Path: {row.get('path', 'N/A')}")
    # print(f"Old code ({len(row['old_code'])} chars):\n{row['old_code'][:200]}...")
    # print(f"\nNew code ({len(row['new_code'])} chars):\n{row['new_code'][:200]}...")
    print(f"\nOracle v4a patch:\n{row.get('v4a', 'N/A')[:300]}...")

    # Test each env class
    env_classes: list[tuple[str, Type[TemplateEnv]]] = [
        ("PatchExactMatchMinimalDiffSmallContextEnv", PatchExactMatchMinimalDiffSmallContextEnv),
        ("PatchHybridSimilarityEnv", PatchHybridSimilarityEnv),
    ]

    for env_name, env_class in env_classes:
        print(f"\n{'-'*40}")
        print(f"Testing: {env_name}")
        print(f"{'-'*40}")

        env = create_env(env_class, row, renderer)
        trajectory = await do_single_rollout(policy, env)

        # Single-turn env has one transition
        total_reward = sum(t.reward for t in trajectory.transitions)
        metrics = trajectory.transitions[-1].metrics if trajectory.transitions else {}
        
        # Decode generated response
        gen_tokens = trajectory.transitions[-1].ac.tokens if trajectory.transitions else []
        gen_text = tokenizer.decode(gen_tokens)
        print(f"\nGenerated patch:\n{gen_text[:500]}...")
        print(f"\nReward: {total_reward:.3f}")
        print(f"Metrics: {metrics}")


async def main():
    """Run comparison on multiple examples."""
    args = parse_args()

    # Load dataset once
    ds = load_dataset_examples(args.dataset_name, args.dataset_config, args.dataset_split)
    print(f"Dataset has {len(ds)} examples")

    # Setup model
    print(f"\nModel: {args.model_name}")
    if args.model_path:
        print(f"Checkpoint: {args.model_path}")
    else:
        print("Using base model (no checkpoint)")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=args.model_name,
        model_path=args.model_path,
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=args.max_tokens,
    )
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.model_name), tokenizer
    )

    # Compare envs on each example
    end_index = min(args.start_index + args.num_examples, len(ds))
    print(f"Testing examples {args.start_index} to {end_index - 1}")

    for i in range(args.start_index, end_index):
        row = dict(ds[i])
        await compare_envs_on_example(row, i, policy, renderer, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
