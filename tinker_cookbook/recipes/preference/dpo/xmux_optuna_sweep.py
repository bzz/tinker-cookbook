"""
Bayesian hyperparameter optimization for DPO using Optuna + xmux (tmux launcher).

This recipe shows a practical pattern for preference (DPO) sweeps:
1) Use Optuna to propose hyperparameters (learning rate, dpo_beta, batch size, ...)
2) Launch trials as separate jobs in a tmux session via `tinker_cookbook.xmux`
3) As each trial completes, read a scalar metric from `metrics.jsonl`
4) Report results back to Optuna via the ask/tell API

Prereqs:
- tmux installed (xmux uses tmux)
- Optuna installed:
    pip install "tinker_cookbook[sweeps]"

Example:

```bash
python -m tinker_cookbook.recipes.preference.dpo.xmux_optuna_sweep \
  sweep_name=dpo-hhh-optuna \
  log_relpath_root=dpo/optuna-hhh \
  storage=sqlite:////tmp/dpo-optuna.db \
  study_name=dpo-hhh \
  n_trials=16 \
  metric_key=test/nll \
  direction=minimize
```
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import chz

from tinker_cookbook.recipes.preference.dpo.train import CLIConfig, cli_main
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Optuna is not installed. Install with:\n"
            '  pip install "tinker_cookbook[sweeps]"\n'
            "or:\n"
            "  pip install optuna\n"
        ) from e
    return optuna


def _tmux_installed() -> bool:
    # Avoid importing shutil at module import time (keep this file lightweight).
    import shutil

    return shutil.which("tmux") is not None


@dataclass(frozen=True)
class TrialPaths:
    log_relpath: str
    log_abspath: str
    completed_marker: str
    failed_marker: str


def _log_paths_for_relpath(log_relpath: str) -> TrialPaths:
    # xmux uses ~/experiments/<log_relpath> as the backing directory.
    log_abspath = os.path.expanduser(os.path.join("~/experiments", log_relpath))
    return TrialPaths(
        log_relpath=log_relpath,
        log_abspath=log_abspath,
        completed_marker=os.path.join(log_abspath, ".completed"),
        failed_marker=os.path.join(log_abspath, ".failed"),
    )


def _read_last_metric(metrics_jsonl_path: str, metric_key: str) -> float:
    """
    Read the last value of `metric_key` from a jsonl metrics file.
    """
    with open(metrics_jsonl_path, "rb") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"Metrics file is empty: {metrics_jsonl_path}")
    last = json.loads(lines[-1])
    if metric_key not in last:
        raise KeyError(f"Metric '{metric_key}' not found in last metrics row: {metrics_jsonl_path}")
    return float(last[metric_key])


@chz.chz
class SweepConfig:
    # Study / persistence
    sweep_name: str = "dpo-optuna"
    storage: str | None = None  # e.g. sqlite:////tmp/dpo-optuna.db
    study_name: str = "dpo"
    direction: str = "minimize"  # "minimize" or "maximize"

    # Launch settings
    max_panes_per_window: int = 4
    poll_interval_s: float = 5.0

    # Where to write runs (under ~/experiments/)
    log_relpath_root: str = "dpo/optuna"

    # Which scalar metric to optimize (must exist in metrics.jsonl)
    metric_key: str = "test/nll"

    # DPO CLI defaults (same as recipes/preference/dpo/train.py)
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "hhh"
    renderer_name: str | None = None
    reference_model_name: str | None = None
    base_url: str | None = None
    max_length: int | None = 8192
    lr_schedule: str = "linear"

    # Budget
    n_trials: int = 16

    # Search space (edit to taste)
    lr_min: float = 1e-6
    lr_max: float = 3e-5
    beta_min: float = 0.01
    beta_max: float = 1.0
    batch_sizes: list[int] = chz.field(default_factory=lambda: [64, 128, 256])


def main(cfg: SweepConfig) -> None:
    if not _tmux_installed():  # pragma: no cover
        raise SystemExit("tmux is not installed; xmux requires tmux.")

    optuna = _require_optuna()

    if cfg.storage is None:
        study = optuna.create_study(direction=cfg.direction)
    else:
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            load_if_exists=True,
            direction=cfg.direction,
        )

    # Always create a unique tmux session name to avoid interactive prompts when re-running.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_sweep_name = f"{cfg.sweep_name}-{timestamp}"

    # Create trials and corresponding xmux jobs.
    pending: list[tuple[object, TrialPaths]] = []
    job_specs: list[JobSpec] = []
    for _ in range(cfg.n_trials):
        trial = study.ask()

        learning_rate = trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True)
        dpo_beta = trial.suggest_float("dpo_beta", cfg.beta_min, cfg.beta_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)

        # Keep paths short: log_relpath is used for tmux window/pane naming.
        log_relpath = os.path.join(cfg.log_relpath_root, f"trial-{trial.number:05d}")
        paths = _log_paths_for_relpath(log_relpath)

        # Ensure the DPO recipe writes logs into the same directory that xmux manages.
        cli_cfg = CLIConfig(
            model_name=cfg.model_name,
            dataset=cfg.dataset,
            renderer_name=cfg.renderer_name,
            learning_rate=float(learning_rate),
            lr_schedule=cfg.lr_schedule,
            dpo_beta=float(dpo_beta),
            max_length=cfg.max_length,
            batch_size=int(batch_size),
            log_path=paths.log_abspath,
            wandb_project=None,
            wandb_name=None,
            base_url=cfg.base_url,
            reference_model_name=cfg.reference_model_name,
            behavior_if_log_dir_exists="delete",
        )

        trial.set_user_attr("log_relpath", log_relpath)
        trial.set_user_attr("log_path", paths.log_abspath)

        # Group all trials in one window; panes show per-trial progress.
        job_specs.append(
            JobSpec(
                main_fn=cli_main,
                log_relpath=log_relpath,
                entrypoint_config=cli_cfg,
                tmux_window_name="dpo-optuna",
            )
        )
        pending.append((trial, paths))

    launch_swarm(
        job_specs,
        SwarmConfig(
            sweep_name=session_sweep_name,
            max_panes_per_window=cfg.max_panes_per_window,
        ),
    )

    # Poll for completion and report results back to Optuna.
    TrialState = optuna.trial.TrialState
    remaining = {t.number: (t, p) for (t, p) in pending}

    while remaining:
        done_numbers: list[int] = []
        for number, (trial, paths) in remaining.items():
            if os.path.exists(paths.failed_marker):
                study.tell(trial, state=TrialState.FAIL)
                done_numbers.append(number)
                continue
            if not os.path.exists(paths.completed_marker):
                continue

            metrics_path = os.path.join(paths.log_abspath, "metrics.jsonl")
            value = _read_last_metric(metrics_path, cfg.metric_key)
            study.tell(trial, value)
            done_numbers.append(number)

        for n in done_numbers:
            remaining.pop(n, None)

        if remaining:
            time.sleep(cfg.poll_interval_s)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)
    print("Best log_path:", best.user_attrs.get("log_path"))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

