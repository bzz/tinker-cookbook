"""Unit tests for RLCF data types and parsing."""

import json
import tempfile
from pathlib import Path

from tinker_cookbook.recipes.rlcf.data import (
    ChecklistDatapoint,
    ChecklistItem,
    parse_requirements_string,
)


class TestChecklistItem:
    def test_roundtrip(self) -> None:
        item = ChecklistItem(requirement="Is it good?", weight=75.0)
        d = item.to_dict()
        assert d == {"requirement": "Is it good?", "weight": 75.0}
        restored = ChecklistItem.from_dict(d)
        assert restored == item


class TestChecklistDatapoint:
    def test_json_roundtrip(self) -> None:
        dp = ChecklistDatapoint(
            instruction="Write a poem",
            checklist=[
                ChecklistItem("Is it a poem?", 100),
                ChecklistItem("Is it about love?", 80),
            ],
        )
        json_str = dp.to_json()
        restored = ChecklistDatapoint.from_json(json_str)
        assert restored.instruction == dp.instruction
        assert len(restored.checklist) == 2
        assert restored.checklist[0].requirement == "Is it a poem?"
        assert restored.checklist[1].weight == 80

    def test_from_hf_row_with_prompt_and_requirements(self) -> None:
        row = {
            "prompt": "Write a haiku about snow",
            "requirements": (
                "1) Is the text a haiku? (importance: 100/100)\n"
                "2) Is the text about snow? (importance: 90/100)"
            ),
        }
        dp = ChecklistDatapoint.from_hf_row(row)
        assert dp.instruction == "Write a haiku about snow"
        assert len(dp.checklist) == 2
        assert dp.checklist[0].requirement == "Is the text a haiku?"
        assert dp.checklist[0].weight == 100.0
        assert dp.checklist[1].requirement == "Is the text about snow?"
        assert dp.checklist[1].weight == 90.0

    def test_from_hf_row_with_chosen_key(self) -> None:
        row = {
            "chosen": [
                {"content": "Tell me a joke", "role": "user"},
                {"content": "Why did the chicken...", "role": "assistant"},
            ],
            "rejected": [
                {"content": "Tell me a joke", "role": "user"},
                {"content": "No.", "role": "assistant"},
            ],
            "requirements": "1) Is it a joke? (importance: 100/100)",
        }
        dp = ChecklistDatapoint.from_hf_row(row)
        assert dp.instruction == "Tell me a joke"
        assert len(dp.checklist) == 1

    def test_from_hf_row_empty_requirements(self) -> None:
        row = {"prompt": "Hello", "requirements": ""}
        dp = ChecklistDatapoint.from_hf_row(row)
        assert dp.instruction == "Hello"
        assert len(dp.checklist) == 0


class TestParseRequirementsString:
    def test_standard_format(self) -> None:
        text = (
            "1) Does the response use bullet points? (importance: 90/100)\n"
            "2) Is the response about cooking? (importance: 100/100)\n"
            "3) Does the response include at least 3 items? (importance: 75/100)"
        )
        items = parse_requirements_string(text)
        assert len(items) == 3
        assert items[0].requirement == "Does the response use bullet points?"
        assert items[0].weight == 90.0
        assert items[2].weight == 75.0

    def test_single_item(self) -> None:
        text = "1) Is it correct? (importance: 100/100)"
        items = parse_requirements_string(text)
        assert len(items) == 1
        assert items[0].requirement == "Is it correct?"

    def test_empty_string(self) -> None:
        assert parse_requirements_string("") == []

    def test_no_matching_lines(self) -> None:
        assert parse_requirements_string("This is just random text") == []


class TestChecklistDatapointListBuilderFromJsonl:
    def test_loads_file(self) -> None:
        from tinker_cookbook.recipes.rlcf.data import ChecklistDatapointListBuilderFromJsonl

        dp1 = ChecklistDatapoint(
            instruction="Test 1",
            checklist=[ChecklistItem("Q1?", 100)],
        )
        dp2 = ChecklistDatapoint(
            instruction="Test 2",
            checklist=[ChecklistItem("Q2?", 50), ChecklistItem("Q3?", 75)],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(dp1.to_json() + "\n")
            f.write(dp2.to_json() + "\n")
            tmp_path = f.name

        builder = ChecklistDatapointListBuilderFromJsonl(jsonl_path=tmp_path)
        result = builder()
        assert len(result) == 2
        assert result[0].instruction == "Test 1"
        assert len(result[1].checklist) == 2

        Path(tmp_path).unlink()

    def test_missing_file_raises(self) -> None:
        from tinker_cookbook.recipes.rlcf.data import ChecklistDatapointListBuilderFromJsonl
        import pytest

        builder = ChecklistDatapointListBuilderFromJsonl(jsonl_path="/nonexistent/path.jsonl")
        with pytest.raises(FileNotFoundError):
            builder()
