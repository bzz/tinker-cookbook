"""
Teacher utilities for SDPO fragility experiment.

Implements:
- TeacherMode: frozen (initial snapshot) vs. current (bootstrapped, same as student)
- DistillationType: token-level (logprob at student's token) vs. logit-level (top-K distribution)
- build_teacher_messages: construct the reprompt from problem + feedback
- get_teacher_logprobs: call Tinker API to retrieve teacher logprobs on student output tokens
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import tinker

from tinker_cookbook.renderers.base import Message, Renderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TeacherMode(str, enum.Enum):
    """How to construct the teacher model.

    FROZEN:       Teacher is the initial checkpoint; never updated during training.
                  Isolates the value of feedback-based reprompting.
    CURRENT:      Teacher IS the student at the current step (bootstrapped, no EMA).
                  Tests whether unregularised self-distillation is stable.
    NO_FEEDBACK:  Control condition.  Teacher prompt = student prompt (no feedback).
                  Used to verify the pipeline sanity – signal should be near zero.
    """

    FROZEN = "frozen"
    CURRENT = "current"
    NO_FEEDBACK = "no_feedback"


class DistillationType(str, enum.Enum):
    """Density of the distillation credit assignment.

    TOKEN_LEVEL:  One scalar signal per generated token position:
                  teacher's log-probability for the student's actual token.
    LOGIT_LEVEL:  K signals per position: teacher's top-K distribution.
                  Requires topk_prompt_logprobs > 1; produces richer gradients.
    """

    TOKEN_LEVEL = "token_level"
    LOGIT_LEVEL = "logit_level"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TeacherLogprobResult:
    """Teacher logprob outputs aligned to student output token positions.

    token_logprobs[t] is the teacher's log-probability for the student's t-th
    output token, given the teacher's reprompt context and all preceding tokens.

    topk_logprobs[t] is a list of (token_id, logprob) pairs representing the
    top-K teacher distribution at position t.  None when dist_type=TOKEN_LEVEL.
    """

    token_logprobs: list[float]
    topk_logprobs: list[list[tuple[int, float]]] | None


# ---------------------------------------------------------------------------
# Reprompt construction
# ---------------------------------------------------------------------------

_FEEDBACK_HEADER = (
    "The following solution attempt did not pass all test cases.\n"
    "Here is the feedback from the test runner:\n\n"
)
_FEEDBACK_FOOTER = (
    "\n\nPlease provide a corrected Python solution that passes all tests."
)


def build_teacher_messages(
    problem_text: str,
    student_code: str | None,
    feedback_text: str,
    teacher_mode: TeacherMode,
) -> list[Message]:
    """Build the teacher prompt as a list of chat messages.

    Args:
        problem_text: The original problem statement (including LCB system prompt).
        student_code: The student's generated code (used in the feedback context).
        feedback_text: Human-readable failure summary from environment.py.
        teacher_mode: Controls whether and how feedback is incorporated.

    Returns:
        list[Message] ready to pass to renderer.build_generation_prompt().
    """
    if teacher_mode == TeacherMode.NO_FEEDBACK:
        # Control: teacher sees exactly the same prompt as the student
        return [{"role": "user", "content": problem_text}]

    # Feedback reprompt: problem + failure info + instruction to correct
    if feedback_text:
        feedback_block = _FEEDBACK_HEADER + feedback_text + _FEEDBACK_FOOTER
        if student_code is not None:
            content = (
                problem_text
                + "\n\n---\n\n"
                + "**Previous attempt:**\n```python\n"
                + student_code
                + "\n```\n\n"
                + feedback_block
            )
        else:
            content = problem_text + "\n\n---\n\n" + feedback_block
    else:
        # Student succeeded (reward=1.0) – no meaningful feedback; fall back to
        # the original problem to avoid injecting solution hints.
        content = problem_text

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Teacher logprob extraction
# ---------------------------------------------------------------------------


async def get_teacher_logprobs(
    teacher_client: tinker.SamplingClient,
    renderer: Renderer,
    teacher_messages: list[Message],
    student_output_tokens: list[int],
    dist_type: DistillationType,
    topk: int = 20,
) -> TeacherLogprobResult:
    """Retrieve teacher logprobs for student output tokens via the prefill pattern.

    Operationally:
      - We concatenate (teacher_reprompt_tokens + student_output_tokens) as a
        single "prompt" and request include_prompt_logprobs=True.
      - The prompt_logprobs at indices [reprompt_length:] are the teacher's
        log-probability for each student output token given the reprompt context.
      - For LOGIT_LEVEL, we additionally request topk_prompt_logprobs=topk to
        get the top-K teacher distribution at each student output position.

    Args:
        teacher_client: The SamplingClient representing the teacher model.
        renderer: Renderer used to tokenise the teacher messages.
        teacher_messages: Chat messages for the teacher (from build_teacher_messages).
        student_output_tokens: Token IDs generated by the student model.
        dist_type: Whether we need only token-level or full top-K logprobs.
        topk: Number of top teacher tokens to retrieve per position (LOGIT_LEVEL).

    Returns:
        TeacherLogprobResult aligned to student_output_tokens.
    """
    if not student_output_tokens:
        return TeacherLogprobResult(token_logprobs=[], topk_logprobs=None)

    # Build teacher reprompt as ModelInput
    reprompt_mi: tinker.ModelInput = renderer.build_generation_prompt(teacher_messages)
    reprompt_length: int = reprompt_mi.length

    # Concatenate reprompt + student output tokens as the "prompt"
    student_mi = tinker.ModelInput.from_ints(student_output_tokens)
    full_context = tinker.ModelInput(
        chunks=list(reprompt_mi.chunks) + list(student_mi.chunks)
    )

    topk_for_query = topk if dist_type == DistillationType.LOGIT_LEVEL else 0

    response = await teacher_client.sample_async(
        prompt=full_context,
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=topk_for_query,
    )

    # prompt_logprobs[i] = log p(token_i | tokens[:i]); first element is None
    # Student output starts at index reprompt_length in the full context.
    raw_prompt_logprobs: list[float | None] = response.prompt_logprobs or []

    # Slice to student token positions; replace any None with -inf
    student_lps_raw = raw_prompt_logprobs[reprompt_length:]
    token_logprobs: list[float] = [
        (lp if lp is not None else float("-inf")) for lp in student_lps_raw
    ]

    # Pad or truncate to match student_output_tokens length exactly
    T = len(student_output_tokens)
    if len(token_logprobs) < T:
        logger.warning(
            "Teacher returned fewer logprobs (%d) than student tokens (%d); padding with -inf.",
            len(token_logprobs),
            T,
        )
        token_logprobs.extend([float("-inf")] * (T - len(token_logprobs)))
    token_logprobs = token_logprobs[:T]

    # Top-K logprobs (LOGIT_LEVEL only)
    topk_logprobs: list[list[tuple[int, float]]] | None = None
    if dist_type == DistillationType.LOGIT_LEVEL:
        raw_topk = response.topk_prompt_logprobs or []
        student_topk_raw = raw_topk[reprompt_length:]

        topk_logprobs = []
        for t in range(T):
            if t < len(student_topk_raw) and student_topk_raw[t]:
                topk_logprobs.append(list(student_topk_raw[t]))
            else:
                # Fallback: single-element list using the student's token logprob
                tok_id = student_output_tokens[t]
                topk_logprobs.append([(tok_id, token_logprobs[t])])

    return TeacherLogprobResult(
        token_logprobs=token_logprobs,
        topk_logprobs=topk_logprobs,
    )
