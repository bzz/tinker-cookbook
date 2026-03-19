"""Unit tests for RLCF prompt parsing and formatting."""

from tinker_cookbook.recipes.rlcf.prompts import (
    format_eval_prompt,
    parse_checklist_response,
    parse_eval_score,
)


class TestParseChecklistResponse:
    def test_basic_parsing(self) -> None:
        text = """Reasoning:
The instruction asks for a haiku about cats.

Key Criteria Questions:
- Is the text a haiku? (100)
- Is the text about cats? (95)
- Does the text follow the 5-7-5 syllable pattern? (80)
<END>"""
        result = parse_checklist_response(text)
        assert len(result) == 3
        assert result[0] == ("Is the text a haiku?", 100.0)
        assert result[1] == ("Is the text about cats?", 95.0)
        assert result[2] == ("Does the text follow the 5-7-5 syllable pattern?", 80.0)

    def test_no_end_marker(self) -> None:
        text = """Key Criteria Questions:
- Is the response in French? (100)
- Does it contain exactly two sentences? (75)"""
        result = parse_checklist_response(text)
        assert len(result) == 2
        assert result[0][0] == "Is the response in French?"
        assert result[0][1] == 100.0

    def test_none_response(self) -> None:
        assert parse_checklist_response("None") == []

    def test_none_in_criteria(self) -> None:
        text = """Key Criteria Questions:
None
<END>"""
        assert parse_checklist_response(text) == []

    def test_asterisk_bullets(self) -> None:
        text = """Key Criteria Questions:
* Is the text a poem? (100)
* Is it about dogs? (90)
<END>"""
        result = parse_checklist_response(text)
        assert len(result) == 2
        assert result[0] == ("Is the text a poem?", 100.0)

    def test_empty_response(self) -> None:
        assert parse_checklist_response("") == []

    def test_float_weights(self) -> None:
        text = "- Does it work? (87.5)\n<END>"
        result = parse_checklist_response(text)
        assert len(result) == 1
        assert result[0][1] == 87.5

    def test_skips_here_are_prefix(self) -> None:
        text = "- Here are some questions: (100)\n- Is it good? (50)\n<END>"
        result = parse_checklist_response(text)
        assert len(result) == 1
        assert result[0][0] == "Is it good?"


class TestParseEvalScore:
    def test_integer(self) -> None:
        assert parse_eval_score("75") == 75.0

    def test_with_trailing_text(self) -> None:
        assert parse_eval_score("100 points") == 100.0

    def test_float_score(self) -> None:
        assert parse_eval_score("87.5") == 87.5

    def test_clamps_to_100(self) -> None:
        assert parse_eval_score("150") == 100.0

    def test_clamps_to_0(self) -> None:
        assert parse_eval_score("-10") == 0.0

    def test_returns_neg1_on_garbage(self) -> None:
        assert parse_eval_score("I think the score is high") == -1.0

    def test_empty_string(self) -> None:
        assert parse_eval_score("") == -1.0

    def test_with_period(self) -> None:
        assert parse_eval_score("50.") == 50.0

    def test_negative_one_default(self) -> None:
        assert parse_eval_score("-1") == 0.0


class TestFormatEvalPrompt:
    def test_contains_all_fields(self) -> None:
        prompt = format_eval_prompt(
            instruction="Write a haiku",
            response="Five seven five yo",
            requirement="Is it a haiku?",
        )
        assert "Write a haiku" in prompt
        assert "Five seven five yo" in prompt
        assert "Is it a haiku?" in prompt
        assert "Score:" in prompt

    def test_preserves_structure(self) -> None:
        prompt = format_eval_prompt(
            instruction="test",
            response="resp",
            requirement="req",
        )
        assert "Input:\ntest" in prompt
        assert "Generated Text:\nresp" in prompt
        assert "Question:\nreq" in prompt
