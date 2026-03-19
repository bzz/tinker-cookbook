"""Smoke test for the RLCF (Checklist Feedback) recipe."""

import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.rlcf.train"


@pytest.mark.integration
def test_rlcf():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen2.5-7B-Instruct",
            "groups_per_batch=4",
            "group_size=2",
            "max_tokens=32",
            "behavior_if_log_dir_exists=delete",
        ],
    )
