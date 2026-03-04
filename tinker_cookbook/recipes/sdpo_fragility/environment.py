"""
Code execution environment for SDPO fragility experiment.

Provides single-turn evaluation of generated code against public test cases and
rich feedback text formatting for teacher reprompting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.sandbox import SandboxBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Outcome of evaluating a single model rollout against public tests."""

    code: str | None       # Extracted Python code block; None if no code block found
    reward: float          # 1.0 = all public tests pass; 0.0 = otherwise
    feedback_text: str     # Human-readable failure summary (empty on success)
    error_type: str        # "correct" | "wrong_answer" | "runtime_error" | "timeout" | "no_code"
    details: dict[str, Any]  # Raw details dict from sandbox execution


# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------

_MAX_IO_LEN = 200  # Maximum characters to include per I/O field in feedback


def _truncate(s: str, limit: int = _MAX_IO_LEN) -> str:
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + " … " + s[-half:]


def format_feedback_for_reprompt(result: RolloutResult) -> str:
    """Format a RolloutResult failure into a concise LeetCode-style feedback string.

    The returned string is suitable for inclusion in a teacher reprompt.
    Returns an empty string for correct solutions.
    """
    if result.error_type == "correct":
        return ""

    if result.error_type == "no_code":
        return "Your response did not contain a valid Python code block."

    d = result.details
    if result.error_type == "timeout":
        inputs = _truncate(str(d.get("inputs", "")))
        return f"Time Limit Exceeded on input: {inputs}"

    if result.error_type == "runtime_error":
        error = _truncate(str(d.get("error", d.get("error_message", "Unknown error"))))
        inputs = _truncate(str(d.get("inputs", "")))
        return f"Runtime Error: {error}\nOn input: {inputs}"

    if result.error_type == "wrong_answer":
        got = _truncate(str(d.get("output", "")))
        expected = _truncate(str(d.get("expected", "")))
        inputs = _truncate(str(d.get("inputs", "")))
        parts = [f"Wrong Answer on input: {inputs}"]
        if expected:
            parts.append(f"Expected: {expected}")
        if got:
            parts.append(f"Got: {got}")
        return "\n".join(parts)

    # Fallback for unknown error type
    return f"Solution failed with error type: {result.error_type}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_rollout(
    response_text: str,
    public_tests: list[dict],
    sandbox_backend: SandboxBackend | None = None,
    timeout: int = 6,
) -> RolloutResult:
    """Evaluate a single model response against the public test subset.

    Extracts a Python code block, runs it in the sandbox, and returns a
    RolloutResult capturing the reward and feedback text.

    Args:
        response_text: Raw text output from the model.
        public_tests: Normalised test cases (from LCBProblem.public_tests).
        sandbox_backend: Which sandbox to use (defaults to SANDBOXFUSION).
        timeout: Per-test timeout in seconds.

    Returns:
        RolloutResult with reward and feedback.
    """
    code = extract_code_from_model(response_text)

    if code is None:
        return RolloutResult(
            code=None,
            reward=0.0,
            feedback_text="Your response did not contain a valid Python code block.",
            error_type="no_code",
            details={},
        )

    try:
        passed, details = await sandbox_check_correctness(
            sample=public_tests,
            generation=code,
            timeout=timeout,
            backend=sandbox_backend,
        )
    except Exception as e:
        logger.warning(f"Sandbox error during evaluation: {e}")
        return RolloutResult(
            code=code,
            reward=0.0,
            feedback_text=f"Evaluation error: {e}",
            error_type="runtime_error",
            details={"error": str(e)},
        )

    if passed:
        return RolloutResult(
            code=code,
            reward=1.0,
            feedback_text="",
            error_type="correct",
            details=details,
        )

    # Determine error type from details
    error_code = details.get("error_code", -1)
    error_message = details.get("error_message", "")

    if error_message == "Time Limit Exceeded" or error_code == -3:
        error_type = "timeout"
    elif error_message in ("Runtime Error", "Compilation Error") or error_code in (-4, -1):
        error_type = "runtime_error"
    else:
        error_type = "wrong_answer"

    partial = RolloutResult(
        code=code,
        reward=0.0,
        feedback_text="",
        error_type=error_type,
        details=details,
    )
    return RolloutResult(
        code=code,
        reward=0.0,
        feedback_text=format_feedback_for_reprompt(partial),
        error_type=error_type,
        details=details,
    )
