"""
Generate language classification training data for on-policy context distillation.

This is identical to the prompt_distillation recipe's data generation step.
A teacher model with the full LANGUAGE_CLASSIFICATION_PROMPT generates labeled
examples; the student sees only the raw text (no system prompt).

Example usage:
    mkdir -p /tmp/tinker-datasets
    python -m tinker_cookbook.recipes.on_policy_context_distillation.create_data \
        output_file=/tmp/tinker-datasets/lang_classification.jsonl
"""

from tinker_cookbook.recipes.prompt_distillation.create_data import (
    Config,
    LANGUAGE_CLASSIFICATION_PROMPT,
    main,
)

__all__ = ["Config", "LANGUAGE_CLASSIFICATION_PROMPT", "main"]

if __name__ == "__main__":
    import chz

    chz.nested_entrypoint(main)
