#!/usr/bin/env python
"""
Hyperparameter sweep for RLVR training using xmux.

This script runs a grid search over learning rate and batch size
to find good hyperparameters for the RLVR diff generation task.

Usage:
    # Dry run to see what would be launched
    python -m tinker_cookbook.recipes.rlvr.sweep --dry-run

    # Launch the sweep
    python -m tinker_cookbook.recipes.rlvr.sweep

    # Attach to monitor
    tmux attach-session -t rlvr-lr-bs-sweep
"""

import argparse
import os
import sys

import pandas

from tinker_cookbook.recipes.rlvr.train import CLIConfig, cli_main
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm


def json_already_exists(log_relpath: str) -> bool:
    """Check if experiment already has results."""
    metrics_path = os.path.expanduser(f"~/experiments/{log_relpath}/metrics.jsonl")
    if not os.path.exists(metrics_path):
        return False
    try:
        df = pandas.read_json(metrics_path, lines=True)
        return len(df) > 0
    except Exception:
        return False


def build_config(
    lr: float,
    batch_size: int,
    env_class: str,
    log_path: str,
) -> CLIConfig:
    """Build CLIConfig for a single experiment."""
    return CLIConfig(
        env_class=env_class,
        dataset_name="bzz2/diff-xyz-v4a",
        dataset_config="easy",
        dataset_split="validation",
        user_template="v4a_instruct.txt",
        answer_field="new_code",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        learning_rate=lr,
        batch_size=batch_size,
        group_size=4,  # Fixed small value for cost efficiency
        lora_rank=32,
        max_tokens=2048,
        # Short run for sweep
        eval_every=5,
        save_every=0,  # Don't save checkpoints during sweep
        log_path=log_path,
        wandb_project="rlvr-sweep",
        behavior_if_log_dir_exists="overwrite",
    )


def main():
    parser = argparse.ArgumentParser(description="RLVR hyperparameter sweep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be launched without running"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip experiments that already have results"
    )
    parser.add_argument(
        "--env",
        default="tinker_cookbook.recipes.rlvr.patch_env:PatchAppliesEnv",
        help="Environment class to use",
    )
    args = parser.parse_args()

    job_specs = []
    env_class = args.env

    # Sweep grid: 3 learning rates x 2 batch sizes = 6 experiments
    learning_rates = [3e-6, 1e-5, 3e-5]
    batch_sizes = [4, 8]

    for lr in learning_rates:
        for bs in batch_sizes:
            log_relpath = f"rlvr-sweep/lr{lr:.0e}-bs{bs}"
            log_path = os.path.expanduser(f"~/experiments/{log_relpath}")

            if args.skip_existing and json_already_exists(log_relpath):
                print(f"Skipping {log_relpath} (already exists)")
                continue

            config = build_config(lr, bs, env_class, log_path)

            job_specs.append(
                JobSpec(
                    main_fn=cli_main,
                    log_relpath=log_relpath,
                    entrypoint_config=config,
                    tmux_window_name=f"bs{bs}",  # Group by batch size
                    pane_title=f"lr{lr:.0e}",
                )
            )

    if not job_specs:
        print("No experiments to launch (all already exist or none matched)")
        return

    print(f"Launching {len(job_specs)} experiments:")
    for spec in job_specs:
        print(f"  - {spec.log_relpath}")

    swarm_config = SwarmConfig(
        sweep_name="rlvr-lr-bs-sweep",
        max_panes_per_window=3,
        dry_run=args.dry_run,
    )

    launch_swarm(job_specs, swarm_config)


if __name__ == "__main__":
    main()

