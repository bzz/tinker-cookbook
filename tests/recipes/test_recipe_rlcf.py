"""Smoke test for the RLCF (Checklist Feedback) DPO recipe."""

import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.rlcf.train"


@pytest.mark.integration
def test_rlcf_dpo():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen2.5-7B-Instruct",
            "batch_size=32",
            "max_steps=2",
        ],
    )
