"""
Offline evaluation runner for RLVR models.

Simple CLI tool to evaluate a checkpoint on a small dataset sample.

Usage:
    # Single checkpoint evaluation
    python -m tinker_cookbook.recipes.rlvr.offline_eval \
        --config configs/patch_exact_v4a.toml \
        load_checkpoint_path=tinker://path/to/checkpoint

    # Evaluate all checkpoints from a training run
    python -m tinker_cookbook.recipes.rlvr.offline_eval \
        --config configs/patch_exact_v4a.toml \
        --run-id "<run-id>:train:0"
"""

import asyncio
import sys

import chz
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.recipes.rlvr.rlvr_evaluator import RLVREvaluatorBuilder
from tinker_cookbook.recipes.rlvr.train import (
    CLIConfig,
    TemplateRLDatasetBuilder,
    load_toml_config_and_cli_args,
    resolve_template,
)
from tinker_cookbook.utils.misc_utils import lookup_func


def create_evaluator(config: CLIConfig):
    """Create an evaluator from the config."""
    # Resolve template
    user_template = resolve_template(config.user_template)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Create dataset builder
    dataset_builder = TemplateRLDatasetBuilder(
        env_class=config.env_class,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        dataset_split=config.eval_dataset_split,
        user_template=user_template,
        answer_field=config.answer_field,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        batch_size=1,
        group_size=1,
        system_prompt=config.system_prompt,
        seed=config.seed,
    )

    # Create evaluator (filter out batch_size/group_size which don't apply to evaluator)
    eval_ds_cfg = {k: v for k, v in chz.asdict(dataset_builder).items() 
                      if k not in ("batch_size", "group_size")}
    eval_config = {
        **eval_ds_cfg,
        "max_eval_samples": config.max_eval_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.eval_temperature,
    }
    
    if config.evaluator:
        evaluator_builder_cls = lookup_func(config.evaluator)
        return evaluator_builder_cls(**eval_config)()
    else:
        return RLVREvaluatorBuilder(**eval_config)()


async def evaluate_single(config: CLIConfig) -> dict[str, float]:
    """Evaluate a single checkpoint and return metrics."""
    evaluator = create_evaluator(config)

    # Create sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name,
        model_path=config.load_checkpoint_path,
    )

    return await evaluator(sampling_client)


def print_single_results(metrics: dict[str, float], checkpoint_name: str | None = None):
    """Print results for a single evaluation."""
    print("\n" + "=" * 60)
    if checkpoint_name:
        print(f"EVALUATION RESULTS: {checkpoint_name}")
    else:
        print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:<30} {value:.2f}")
        else:
            print(f"{key:<30} {value}")
    print("=" * 60)


def print_markdown_table(results: list[tuple[str, dict[str, float]]]):
    """Print results as a Markdown table."""
    if not results:
        print("No results to display.")
        return
    
    # Get all metric keys (excluding test/num_examples for conciseness)
    metric_keys = ["test/correct", "test/format_valid", "test/reward", "test/num_lines"]
    # Filter to only keys that exist in results
    metric_keys = [k for k in metric_keys if k in results[0][1]]
    
    # Header
    header = "| Checkpoint | " + " | ".join(k.replace("test/", "") for k in metric_keys) + " |"
    separator = "|" + "|".join("-" * (len(col) + 2) for col in ["Checkpoint"] + [k.replace("test/", "") for k in metric_keys]) + "|"
    
    print("\n## Evaluation Results\n")
    print(header)
    print(separator)
    
    # Rows
    for checkpoint_name, metrics in results:
        # Extract just the checkpoint ID from the tinker path
        short_name = checkpoint_name.split("/")[-1] if "/" in checkpoint_name else checkpoint_name
        row_values = [f"{metrics.get(k, 0.0):.2f}" for k in metric_keys]
        row = f"| {short_name} | " + " | ".join(row_values) + " |"
        print(row)
    
    print()


async def evaluate_run(config: CLIConfig, run_id: str) -> list[tuple[str, dict[str, float]]]:
    """Evaluate all sampler checkpoints from a training run."""
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    # List all checkpoints for the run
    print(f"Fetching checkpoints for run: {run_id}")
    checkpoints_response = await rest_client.list_checkpoints_async(run_id)
    
    # Filter to sampler checkpoints only
    sampler_checkpoints = [
        cp for cp in checkpoints_response.checkpoints
        if cp.checkpoint_type == "sampler"
    ]
    
    if not sampler_checkpoints:
        print(f"No sampler checkpoints found for run {run_id}")
        return []
    
    # Sort by checkpoint_id
    sampler_checkpoints.sort(key=lambda cp: cp.checkpoint_id)
    
    print(f"Found {len(sampler_checkpoints)} sampler checkpoints")
    print(f"Dataset: {config.dataset_name} ({config.eval_dataset_split})")
    print(f"Max samples: {config.max_eval_samples or 'all'}")
    print()
    
    # Create evaluator once (reuses dataset)
    evaluator = create_evaluator(config)
    
    # Evaluate each checkpoint
    results: list[tuple[str, dict[str, float]]] = []
    for cp in sampler_checkpoints:
        print(f"Evaluating: {cp.tinker_path}")
        sampling_client = service_client.create_sampling_client(
            base_model=config.model_name,
            model_path=cp.tinker_path,
        )
        metrics = await evaluator(sampling_client)
        results.append((cp.tinker_path, metrics))
    
    return results


async def main(config: CLIConfig, run_id: str | None = None):
    """Run offline evaluation."""
    if run_id:
        # Evaluate all checkpoints from a training run
        results = await evaluate_run(config, run_id)
        print_markdown_table(results)
        return

    # Single checkpoint evaluation
    if config.load_checkpoint_path:
        print(f"Evaluating checkpoint: {config.load_checkpoint_path}")
    else:
        print(f"Evaluating base model: {config.model_name}")
    
    print(f"Dataset: {config.dataset_name} ({config.eval_dataset_split})")
    print(f"Max samples: {config.max_eval_samples or 'all'}")
    print()

    print("Running evaluation...")
    metrics = await evaluate_single(config)
    print_single_results(metrics)


def parse_run_id() -> str | None:
    """Parse --run-id argument from CLI."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--run-id" and i < len(sys.argv):
            return sys.argv[i + 1]
        elif arg.startswith("--run-id="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    # Reuse training config, just override eval-specific params
    toml_and_cli_config = load_toml_config_and_cli_args()
    
    if toml_and_cli_config:
        # Default to no shuffle for eval (deterministic order) unless explicitly set
        if "seed" not in toml_and_cli_config:
            toml_and_cli_config["seed"] = None
        config = CLIConfig(**toml_and_cli_config)
    else:
        config = chz.entrypoint(CLIConfig)

    # Parse --run-id argument
    run_id = parse_run_id()

    asyncio.run(main(config, run_id=run_id))

