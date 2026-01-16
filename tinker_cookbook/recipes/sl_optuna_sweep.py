"""
Bayesian hyperparameter optimization runner using Optuna.

This is intentionally **recipe-driven** (software-engineering friendly):
- You point it at an existing recipe entrypoint (e.g. `tinker_cookbook.recipes.sl_loop:main`)
- The recipe should minimally support:
  - returning a `dict[str, int | float]` of final metrics, and/or
  - writing a final metrics JSON file (optional)

This avoids copy/pasting training logic into the sweep runner.

Usage:

```bash
pip install "tinker_cookbook[sweeps]"

python -m tinker_cookbook.recipes.sl_optuna_sweep \
  entrypoint=tinker_cookbook.recipes.sl_loop:main \
  config_class=tinker_cookbook.recipes.sl_loop:Config \
  log_root=/tmp/sft-optuna \
  n_trials=20 \
  max_steps=30
```
"""

from __future__ import annotations

import importlib
import os
from typing import Callable

import chz

from tinker_cookbook import cli_utils


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


def _import_symbol(symbol: str):
    """
    Import `module.submodule:attr` and return the attribute.
    """
    if ":" not in symbol:
        raise ValueError(f"Expected 'module:attr' import path, got: {symbol}")
    mod, attr = symbol.split(":", 1)
    module = importlib.import_module(mod)
    return getattr(module, attr)


@chz.chz
class SweepConfig:
    # What to run
    entrypoint: str = "tinker_cookbook.recipes.sl_loop:main"
    config_class: str = "tinker_cookbook.recipes.sl_loop:Config"

    # Study/logging
    log_root: str = "/tmp/tinker-examples/sl-optuna-sweep"
    study_name: str = "sl-optuna"
    storage: str | None = None  # e.g. sqlite:////tmp/sft-optuna/optuna.db
    direction: str = "minimize"
    objective_key: str = "train_mean_nll"

    # Budget / parallelism (n_jobs = threads)
    n_trials: int = 20
    n_jobs: int = 1

    # Common recipe overrides (assumes the target recipe supports these keys)
    max_steps: int = 30

    # Search space (edit these for your task)
    lr_min: float = 1e-5
    lr_max: float = 3e-3
    batch_sizes: list[int] = chz.field(default_factory=lambda: [32, 64, 128])
    lora_ranks: list[int] = chz.field(default_factory=lambda: [8, 16, 32, 64])


def main(cfg: SweepConfig):
    optuna = _require_optuna()
    os.makedirs(cfg.log_root, exist_ok=True)

    entrypoint_fn = _import_symbol(cfg.entrypoint)
    config_cls = _import_symbol(cfg.config_class)

    if cfg.storage is None:
        study = optuna.create_study(direction=cfg.direction)
    else:
        study = optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            load_if_exists=True,
            direction=cfg.direction,
        )

    def objective(trial) -> float:
        lr = trial.suggest_float("learning_rate", cfg.lr_min, cfg.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_sizes)
        lora_rank = trial.suggest_categorical("lora_rank", cfg.lora_ranks)

        trial_log_path = os.path.join(cfg.log_root, f"trial-{trial.number:05d}")
        cli_utils.check_log_dir(trial_log_path, behavior_if_exists="delete")

        # Construct the recipe config with minimal assumptions:
        # - it accepts log_path, learning_rate, batch_size, lora_rank, max_steps
        recipe_cfg = config_cls(
            log_path=trial_log_path,
            learning_rate=float(lr),
            batch_size=int(batch_size),
            lora_rank=int(lora_rank),
            max_steps=int(cfg.max_steps),
            behavior_if_log_dir_exists="delete",
            result_path=os.path.join(trial_log_path, "result.json"),
        )

        trial.set_user_attr("log_path", trial_log_path)

        metrics = entrypoint_fn(recipe_cfg)
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Expected recipe entrypoint to return a metrics dict, got: {type(metrics)}"
            )
        if cfg.objective_key not in metrics:
            raise KeyError(
                f"Objective key '{cfg.objective_key}' not found in returned metrics. "
                f"Available keys: {sorted(metrics.keys())}"
            )
        return float(metrics[cfg.objective_key])

    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)
    print("Best log_path:", best.user_attrs.get("log_path"))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

