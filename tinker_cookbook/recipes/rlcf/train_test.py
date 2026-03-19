"""Unit tests for the RLCF DPO training script."""

from tinker_cookbook.recipes.rlcf.train import CLIConfig, RLCFComparisonBuilder


class TestRLCFComparisonBuilder:
    def test_example_to_labeled_comparison(self) -> None:
        builder = RLCFComparisonBuilder()
        example = {
            "chosen": [
                {"role": "user", "content": "Write a haiku about snow"},
                {"role": "assistant", "content": "White flakes gently fall"},
            ],
            "rejected": [
                {"role": "user", "content": "Write a haiku about snow"},
                {"role": "assistant", "content": "Snow is cold and wet"},
            ],
            "requirements": "1) Is it a haiku? (importance: 100/100)",
        }
        result = builder.example_to_labeled_comparison(example)
        assert result is not None
        assert result.label == "A"
        assert result.comparison.prompt_conversation == [
            {"role": "user", "content": "Write a haiku about snow"}
        ]
        assert result.comparison.completion_A == [
            {"role": "assistant", "content": "White flakes gently fall"}
        ]
        assert result.comparison.completion_B == [
            {"role": "assistant", "content": "Snow is cold and wet"}
        ]

    def test_example_with_missing_chosen(self) -> None:
        builder = RLCFComparisonBuilder()
        result = builder.example_to_labeled_comparison({"rejected": []})
        assert result is None

    def test_example_with_short_chosen(self) -> None:
        builder = RLCFComparisonBuilder()
        result = builder.example_to_labeled_comparison({
            "chosen": [{"role": "user", "content": "hi"}],
            "rejected": [{"role": "user", "content": "hi"}],
        })
        assert result is None

    def test_multi_turn_conversation(self) -> None:
        builder = RLCFComparisonBuilder()
        example = {
            "chosen": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "No."},
            ],
        }
        result = builder.example_to_labeled_comparison(example)
        assert result is not None
        assert len(result.comparison.prompt_conversation) == 3
        assert result.comparison.completion_A == [
            {"role": "assistant", "content": "Why did the chicken cross the road?"}
        ]


class TestCLIConfigDefaults:
    """Verify paper-faithful defaults."""

    def test_model(self) -> None:
        assert CLIConfig().model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_dpo_beta(self) -> None:
        assert CLIConfig().dpo_beta == 0.1

    def test_learning_rate(self) -> None:
        assert CLIConfig().learning_rate == 3e-6

    def test_num_epochs(self) -> None:
        assert CLIConfig().num_epochs == 2

    def test_max_length(self) -> None:
        assert CLIConfig().max_length == 2048

    def test_dataset(self) -> None:
        assert CLIConfig().dataset_name == "viswavi/rlcf"

    def test_lr_schedule(self) -> None:
        assert CLIConfig().lr_schedule == "linear"
