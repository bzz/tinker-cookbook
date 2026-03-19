"""Smoke tests for the RLCF (Checklist Feedback) recipes."""

import pytest

from tests.helpers import run_recipe


@pytest.mark.integration
def test_rlcf_grpo():
    """Online GRPO with checklist-graded environment."""
    run_recipe(
        "tinker_cookbook.recipes.rlcf.train",
        [
            "model_name=Qwen/Qwen2.5-7B-Instruct",
            "groups_per_batch=4",
            "group_size=2",
            "max_tokens=32",
            "behavior_if_log_dir_exists=delete",
        ],
    )


@pytest.mark.integration
def test_rlcf_dpo():
    """Offline DPO on checklist-scored preference pairs (paper-faithful)."""
    run_recipe(
        "tinker_cookbook.recipes.rlcf.train_dpo",
        [
            "model_name=Qwen/Qwen2.5-7B-Instruct",
            "batch_size=32",
            "max_steps=2",
        ],
    )
