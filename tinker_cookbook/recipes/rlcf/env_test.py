"""Unit tests for ChecklistGradedEnv reward computation logic."""

import asyncio
from unittest.mock import AsyncMock

from tinker_cookbook.recipes.rlcf.data import ChecklistDatapoint, ChecklistItem
from tinker_cookbook.recipes.rlcf.env import ChecklistGradedEnv
from tinker_cookbook.recipes.rlcf.prompts import UNIVERSAL_REQUIREMENT_TEXT


class TestChecklistProperty:
    def _make_env(self, add_universal: bool = True) -> ChecklistGradedEnv:
        dp = ChecklistDatapoint(
            instruction="Write a haiku",
            checklist=[
                ChecklistItem("Is it a haiku?", 100),
                ChecklistItem("Is it about nature?", 75),
            ],
        )
        env = ChecklistGradedEnv(
            renderer=None,  # type: ignore[arg-type]
            datapoint=dp,
            judge_llm=AsyncMock(),
            add_universal_requirement=add_universal,
        )
        return env

    def test_checklist_with_universal(self) -> None:
        env = self._make_env(add_universal=True)
        checklist = env.checklist
        assert len(checklist) == 3
        assert checklist[0].requirement == "Is it a haiku?"
        assert checklist[2].requirement == UNIVERSAL_REQUIREMENT_TEXT

    def test_checklist_without_universal(self) -> None:
        env = self._make_env(add_universal=False)
        checklist = env.checklist
        assert len(checklist) == 2

    def test_instruction(self) -> None:
        env = self._make_env()
        assert env.instruction == "Write a haiku"

    def test_convo(self) -> None:
        env = self._make_env()
        assert env.convo == [{"role": "user", "content": "Write a haiku"}]


class TestGradeItem:
    def test_grade_item_parses_score(self) -> None:
        dp = ChecklistDatapoint(
            instruction="Test",
            checklist=[ChecklistItem("Is it good?", 100)],
        )
        mock_judge = AsyncMock(return_value={"content": "75"})
        env = ChecklistGradedEnv(
            renderer=None,  # type: ignore[arg-type]
            datapoint=dp,
            judge_llm=mock_judge,
            add_universal_requirement=False,
        )
        item = ChecklistItem("Is it good?", 100)
        loop = asyncio.new_event_loop()
        try:
            score, text = loop.run_until_complete(
                env._grade_item("some response", item)
            )
        finally:
            loop.close()
        assert score == 75.0
        assert text == "75"
        mock_judge.assert_called_once()

    def test_grade_item_handles_garbage(self) -> None:
        dp = ChecklistDatapoint(
            instruction="Test",
            checklist=[ChecklistItem("Q?", 100)],
        )
        mock_judge = AsyncMock(return_value={"content": "I don't know"})
        env = ChecklistGradedEnv(
            renderer=None,  # type: ignore[arg-type]
            datapoint=dp,
            judge_llm=mock_judge,
            add_universal_requirement=False,
        )
        item = ChecklistItem("Q?", 100)
        loop = asyncio.new_event_loop()
        try:
            score, _ = loop.run_until_complete(
                env._grade_item("resp", item)
            )
        finally:
            loop.close()
        assert score == -1.0


class TestWeightedScoreComputation:
    """Test the weighted score math that happens inside step()."""

    def test_weighted_average_manual(self) -> None:
        items = [
            ChecklistItem("A?", 100),
            ChecklistItem("B?", 50),
        ]
        scores = [80.0, 60.0]

        total_weight = sum(item.weight for item in items)
        weighted = sum(
            item.weight * score / 100.0
            for item, score in zip(items, scores)
        )
        result = weighted / total_weight
        expected = (100 * 80 / 100 + 50 * 60 / 100) / 150
        assert abs(result - expected) < 1e-6

    def test_all_zero_scores(self) -> None:
        items = [ChecklistItem("A?", 100), ChecklistItem("B?", 50)]
        scores = [0.0, 0.0]
        total_weight = sum(item.weight for item in items)
        weighted = sum(
            item.weight * score / 100.0
            for item, score in zip(items, scores)
        )
        result = weighted / total_weight
        assert result == 0.0

    def test_perfect_scores(self) -> None:
        items = [ChecklistItem("A?", 100), ChecklistItem("B?", 50)]
        scores = [100.0, 100.0]
        total_weight = sum(item.weight for item in items)
        weighted = sum(
            item.weight * score / 100.0
            for item, score in zip(items, scores)
        )
        result = weighted / total_weight
        assert abs(result - 1.0) < 1e-6

    def test_skips_negative_scores(self) -> None:
        """If the judge returns -1 (parse failure), that item is excluded."""
        items = [ChecklistItem("A?", 100), ChecklistItem("B?", 50)]
        scores = [80.0, -1.0]
        valid = [(item, score) for item, score in zip(items, scores) if score >= 0]
        effective_weight = sum(item.weight for item, _ in valid)
        weighted = sum(item.weight * score / 100.0 for item, score in valid)
        result = weighted / effective_weight if effective_weight > 0 else 0.0
        expected = (100 * 80 / 100) / 100
        assert abs(result - expected) < 1e-6
