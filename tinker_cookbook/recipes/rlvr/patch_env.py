"""
Environment definitions for RLVR training on diff generation tasks.

Base class:
- TemplateEnv: Base class for template-based RLVR environments

Patch-specific reward variants:
- PatchAppliesEnv: reward = 1.0 if patch applies without error
- PatchExactMatchEnv: reward = 1.0 if patch applies AND result matches exactly
- PatchRelaxedMatchEnv: uses check_format (applies?) + check_answer (set match)
  - Patch doesn't apply: reward = -0.1 (format penalty)
  - Patch applies but wrong: reward = 0
  - Patch applies and correct: reward = 1
- PatchMinimalDiffEnv: exact match + efficiency bonus based on diff minimality
  - Wrong/doesn't apply: reward = 0.0
  - Correct, 2x verbose: reward = 0.5
  - Correct, optimal: reward = 1.0
  Uses n_added + n_removed from dataset as reference for optimal diff size.
- PatchExactMatchMinimalDiffSmallContextEnv: hierarchical 4-signal reward
  - format fails: reward = -0.1
  - format ok, answer wrong: reward = 0.2
  - format ok, answer correct, not minimal: reward = 0.6 - context_penalty
  - format ok, answer correct, minimal: reward = 1.0 - context_penalty
  Uses n_added/n_removed for minimality, penalizes excessive context lines.
- PatchExactMatchMinimalDiffWideGapEnv: wider gaps + gradual minimality
  - format fails: reward = -0.5
  - format ok, answer wrong: reward = 0.0
  - format ok, answer correct: reward = 0.5 + minimality_bonus - ctx_penalty
    where minimality_bonus = max(0, 0.5 - diff_excess * 0.1)
  Wider gaps + gradual decay create stronger learning signal.
- PatchHybridSimilarityEnv: outcome + SWE-RL-style patch similarity
  - format fails: reward = 0.3 * similarity - 0.1
  - format ok, answer wrong: reward = 0.3 * similarity + 0.2
  - format ok, answer correct: reward = 0.5 + 0.5 * similarity
  Uses v4a column as oracle patch for SequenceMatcher similarity.
  Gives gradient signal even for non-applying patches.
"""

import difflib
import re
from abc import abstractmethod
from typing import override

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.recipes.rlvr.apply_patch_memfs import DiffError, apply_patch_to_text
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.types import Action, StepResult
from tinker_cookbook.utils import logtree


class TemplateEnv(ProblemEnv):
    """
    Base class for template-based RLVR environments.

    Stores the formatted question, answer, and full dataset row for custom logic.
    Users should subclass this and implement check_answer() and optionally check_format().

    Example usage:
        class PatchEnv(TemplateEnv):
            def check_answer(self, sample_str: str) -> bool:
                return apply_patch(self.row["patch"], sample_str)

            def check_format(self, sample_str: str) -> bool:
                return True  # No format requirements
    """

    def __init__(
        self,
        question: str,
        answer: str,
        row: dict,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.question = question
        self.answer = answer
        self.row = row  # Full row for custom logic in subclasses

    def get_question(self) -> str:
        return self.question

    def get_reference_answer(self) -> str:
        return self.answer

    @abstractmethod
    def check_answer(self, sample_str: str) -> bool:
        """Check if the model's response is correct. Must be implemented by subclass."""
        pass

    def check_format(self, sample_str: str) -> bool:
        """Check if the response format is valid. Override if format checking is needed."""
        return True  # Default: no format requirements

    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        total_reward = self.format_coef * (correct_format - 1) + correct_answer
        return total_reward # same as ProblemEnv.step()

def extract_last_markdown_block(text: str) -> str:
    """Extract the LAST markdown code block, or raw text if none found."""
    matches = re.findall(r"```(?:\w*)\n?(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else text.strip()


def count_diff_lines(patch: str) -> tuple[int, int]:
    """Count +/- lines separately in a diff (excluding --- and +++ headers)."""
    n_added = 0
    n_removed = 0
    for line in patch.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            n_added += 1
        elif line.startswith('-') and not line.startswith('---'):
            n_removed += 1
    return n_added, n_removed


def count_context_and_changes(patch: str) -> tuple[int, int]:
    """Count context lines and change regions in a v4a patch.

    Per the v4a spec, a single hunk can have multiple @@ lines for nested
    navigation (e.g., @@ class Foo + @@ def bar()). So we count change regions
    (contiguous +/- blocks) instead of @@ markers.

    Returns (n_context_lines, n_change_regions) where:
    - n_context_lines: lines starting with " " (space) - unchanged context
    - n_change_regions: number of separate +/- blocks (each expects ~6 lines context)
    """
    n_context = 0
    n_change_regions = 0
    in_change = False

    for line in patch.splitlines():
        # Skip metadata lines
        if line.startswith("*** ") or line.startswith("@@"):
            continue

        if line.startswith(" "):
            n_context += 1
            in_change = False
        elif line.startswith("+") or line.startswith("-"):
            if not in_change:
                n_change_regions += 1
                in_change = True

    return n_context, max(1, n_change_regions)


class PatchAppliesEnv(TemplateEnv):
    """Reward = 1.0 if patch applies without error (ignores correctness)."""
    _patch: str
    _result: str = "" # _patch applied to old_code

    def check_answer(self, sample_str: str) -> bool:
        return self.extract_then_cache_and_apply_patch(sample_str)

    def extract_then_cache_and_apply_patch(self, sample_str: str) -> bool:
        self._patch = extract_last_markdown_block(sample_str)
        try:
            self._result = apply_patch_to_text(self.row["old_code"], self._patch)
            return True
        except DiffError: # IndexError:
            return False
        except Exception:
            return False


    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        return self.format_coef * (correct_format - 1) + correct_answer

    @override
    async def step(self, action: Action) -> StepResult:
        """Same as ProblemEnv.step(), but with customizable reward"""

        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = float(self.check_answer(content))
        total_reward = self.compute_reward(correct_format, correct_answer)

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, Correct: {'✓' if correct_answer else '✗'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )

class PatchExactMatchEnv(PatchAppliesEnv):
    """Reward = 1.0 if patch applies AND result matches new_code exactly,
    0.5 if only patch applies (correct v4a format), and -0.1 otherwise"""

    def check_format(self, sample_str: str) -> bool:
        return self.extract_then_cache_and_apply_patch(sample_str)

    @override
    def check_answer(self, sample_str: str) -> bool:
        return self._result.strip() == self.row["new_code"].strip()

    @override
    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        return 0.5*(correct_format + correct_answer) + 0.1*(correct_format - 1)

def normalize_code(code: str) -> set[str]:
    """Normalize code to set of lines (order-insensitive)"""
    return set(code.strip().splitlines())


class PatchRelaxedMatchEnv(TemplateEnv):
    """
    Reward uses both format checking AND relaxed content matching.

    - check_format: Returns True if patch applies without exception
    - check_answer: Returns True if the resulting code matches new_code as a set of lines

    With default format_coef=0.1:
    - Patch doesn't apply: reward = -0.1 + 0 = -0.1
    - Patch applies but wrong: reward = 0 + 0 = 0
    - Patch applies and correct: reward = 0 + 1 = 1
    """

    def check_format(self, sample_str: str) -> bool:
        """Format is valid if patch applies without error."""
        patch = extract_last_markdown_block(sample_str)
        try:
            apply_patch_to_text(self.row["old_code"], patch)
            return True
        except DiffError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Answer is correct if applied patch matches expected as set of lines."""
        patch = extract_last_markdown_block(sample_str)
        try:
            result = apply_patch_to_text(self.row["old_code"], patch)
            return normalize_code(result) == normalize_code(self.row["new_code"])
        except DiffError:
            return False


class PatchMinimalDiffEnv(TemplateEnv):
    """
    Exact match + minimality bonus based on diff size.

    Reward based on how close the diff size is to the reference:
    - Wrong/doesn't apply: 0.0
    - Correct, but doesn't match exactly: 0.5
    - Correct, match, but not minimal: 0.7 + punish for the size difference
    - Correct, match, minimal (ref_added == gen_added, ref_removed == gen_removed): 1.0

    Uses n_added + n_removed from dataset as reference for minimal diff size.
    """

    def check_answer(self, sample_str: str) -> bool:
        # Not used - we override step() for continuous rewards
        return False

    @override
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        patch = extract_last_markdown_block(content)
        ref_added = self.row["n_added"]
        ref_removed = self.row["n_removed"]

        try:
            result = apply_patch_to_text(self.row["old_code"], patch)
            applies = True

            if result.strip() == self.row["new_code"].strip():
                # Correct result - compute minimality bonus
                gen_added, gen_removed = count_diff_lines(patch)

                added_diff = abs(gen_added - ref_added)
                removed_diff = abs(gen_removed - ref_removed)

                # Independent decay for each dimension
                added_bonus = 0.15 / (1 + 0.5 * added_diff)
                removed_bonus = 0.15 / (1 + 0.5 * removed_diff)
                minimality_bonus = added_bonus + removed_bonus

                reward = 0.7 + minimality_bonus
                correct = True
            else:  # Applies but wrong
                reward = 0.5
                correct = False
                minimality_bonus = 0.0
        except DiffError:  # Doesn't apply
            reward = 0.0
            correct = False
            minimality_bonus = 0.0
            applies = False
        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Applies: {'✓' if applies else '✗'}, Correct: {'✓' if correct else '✗'}, Minimality: {minimality_bonus:.3f}, Reward: {reward:.2f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": float(correct),
                "applies": float(applies),
                "minimal": minimality_bonus,
            },
        )


class PatchExactMatchMinimalDiffSmallContextEnv(TemplateEnv):
    """
    Hierarchical 4-signal reward for patch generation.

    Signals:
    1. check_format: binary - patch extracts and applies without exception
    2. check_answer: binary - applied result matches new_code exactly
    3. check_minimal: binary - gen_added == n_added AND gen_removed == n_removed
    4. context_penalty: float - penalizes excessive context lines per v4a spec

    Reward structure (hierarchical):
    - format fails:                            reward = -0.1
    - format ok, answer wrong:                 reward = 0.2
    - format ok, answer correct, not minimal:  reward = 0.6 - context_penalty
    - format ok, answer correct, minimal:      reward = 1.0 - context_penalty
    """

    _patch: str = ""
    _result: str = ""

    def check_format(self, sample_str: str) -> bool:
        """Format valid if patch extracts and applies without exception."""
        self._patch = extract_last_markdown_block(sample_str)
        try:
            self._result = apply_patch_to_text(self.row["old_code"], self._patch)
            return True
        except (DiffError, Exception):
            self._result = ""
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Answer correct if applied result matches new_code exactly."""
        return self._result.strip() == self.row["new_code"].strip()

    def check_minimal(self) -> bool:
        """Edit is minimal if diff line counts match ground truth."""
        gen_added, gen_removed = count_diff_lines(self._patch)
        return gen_added == self.row["n_added"] and gen_removed == self.row["n_removed"]

    def compute_context_penalty(self) -> float:
        """Penalize excessive context lines. Per v4a spec: ~6 lines per change region."""
        n_context, n_change_regions = count_context_and_changes(self._patch)
        expected_context = n_change_regions * 6
        excess_context = max(0, n_context - expected_context)
        return min(excess_context * 0.02, 0.2)

    # Cached signals from last check_format/check_answer call (for metrics in .step())
    _minimal: bool = False
    _context_penalty: float = 0.0

    @override
    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        """Hierarchical reward. Call check_format() first to populate _patch/_result."""
        if not correct_format:
            self._minimal = False
            self._context_penalty = 0.0
            return -0.1
        if not correct_answer:
            self._minimal = False
            self._context_penalty = 0.0
            return 0.2

        # Compute and cache signals for metrics
        self._minimal = self.check_minimal()
        self._context_penalty = self.compute_context_penalty()

        if not self._minimal:
            return 0.6 - self._context_penalty
        return 1.0 - self._context_penalty

    @override
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Evaluate signals (check_format populates _patch/_result)
        format_ok = parse_success and self.check_format(content)
        answer_ok = format_ok and self.check_answer(content)

        # Compute reward (also populates _minimal and _context_penalty)
        reward = self.compute_reward(format_ok, answer_ok)

        # Log
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format: {'✓' if format_ok else '✗'}, Answer: {'✓' if answer_ok else '✗'}, "
            f"Minimal: {'✓' if self._minimal else '✗'}, CtxPenalty: {self._context_penalty:.3f}, Reward: {reward:.2f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": float(format_ok),
                "correct": float(answer_ok),
                "minimal": float(self._minimal),
                "context_penalty": self._context_penalty,
            },
        )


class PatchExactMatchMinimalDiffWideGapEnv(PatchExactMatchMinimalDiffSmallContextEnv):
    """
    Wider gaps + gradual minimality bonus for stronger learning signal.

    Reward structure:
    - format fails:              reward = -0.5
    - format ok, answer wrong:   reward = 0.0
    - format ok, answer correct: reward = 0.5 + minimality_bonus - context_penalty
      where minimality_bonus = max(0, 0.5 - diff_excess * 0.1)
      and diff_excess = |gen_added - gt_added| + |gen_removed - gt_removed|

    Reward curve for correct answers (before context_penalty):
      Lines off:  0    1    2    3    4    5+
      Bonus:     0.5  0.4  0.3  0.2  0.1  0.0
      Total:     1.0  0.9  0.8  0.7  0.6  0.5
    """

    def compute_diff_excess(self) -> int:
        """Compute how many lines off from optimal."""
        gen_added, gen_removed = count_diff_lines(self._patch)
        return abs(gen_added - self.row["n_added"]) + abs(gen_removed - self.row["n_removed"])

    @override
    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        """Hierarchical reward with wider gaps and gradual minimality."""
        if not correct_format:
            self._minimal = False
            self._context_penalty = 0.0
            return -0.5
        if not correct_answer:
            self._minimal = False
            self._context_penalty = 0.0
            return 0.0

        # Gradual minimality: 0.5 base + up to 0.5 bonus for being close to optimal
        diff_excess = self.compute_diff_excess()
        minimality_bonus = max(0.0, 0.5 - diff_excess * 0.1)
        self._minimal = (diff_excess == 0)  # Still track binary for metrics
        self._context_penalty = self.compute_context_penalty()

        return 0.5 + minimality_bonus - self._context_penalty


class PatchHybridSimilarityEnv(TemplateEnv):
    """
    Hierarchical hybrid reward combining outcome correctness with patch similarity.

    Signals:
    1. check_format: binary - patch extracts and applies without exception
    2. check_answer: binary - applied result matches new_code exactly
    3. similarity: float - SequenceMatcher ratio between generated and oracle patch

    Reward structure (hierarchical):
    - format fails:              reward = 0.3 * similarity - 0.1
    - format ok, answer wrong:   reward = 0.3 * similarity + 0.2
    - format ok, answer correct: reward = 0.5 + 0.5 * similarity

    This gives gradient signal for format learning (via similarity) even when
    patches fail, plus outcome-based bonuses for actually solving the task.

    Requires dataset to have 'v4a' field with ground truth patch for similarity.
    """

    # Cached state from check_format/check_answer
    _patch: str = ""
    _result: str = ""
    _similarity: float = 0.0

    def check_format(self, sample_str: str) -> bool:
        """Format valid if patch extracts and applies without exception."""
        self._patch = extract_last_markdown_block(sample_str)
        try:
            self._result = apply_patch_to_text(self.row["old_code"], self._patch)
            return True
        except (DiffError, Exception):
            self._result = ""
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Answer correct if applied result matches new_code exactly."""
        return self._result.strip() == self.row["new_code"].strip()

    def compute_similarity(self) -> float:
        """Compute text similarity between generated patch and oracle v4a patch."""
        oracle_patch = self.row.get("v4a", "")
        if not oracle_patch or not self._patch:
            return 0.0
        return difflib.SequenceMatcher(
            None, self._patch, oracle_patch, autojunk=False
        ).ratio()

    @override
    def compute_reward(self, correct_format: bool, correct_answer: bool) -> float:
        """Hierarchical reward with similarity component."""
        self._similarity = self.compute_similarity()

        if not correct_format:
            return 0.3 * self._similarity - 0.1
        if not correct_answer:
            return 0.3 * self._similarity + 0.2
        return 0.5 + 0.5 * self._similarity

    @override
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # Evaluate signals (check_format populates _patch/_result)
        format_ok = parse_success and self.check_format(content)
        answer_ok = format_ok and self.check_answer(content)

        # Compute reward (also populates _similarity)
        reward = self.compute_reward(format_ok, answer_ok)

        # Log all intermediate results
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(f"Oracle Patch (v4a): {self.row.get('v4a', 'N/A')[:200]}...")
        logtree.log_text(f"Generated Patch: {self._patch[:200]}...")
        logtree.log_text(
            f"Format: {'✓' if format_ok else '✗'}, Answer: {'✓' if answer_ok else '✗'}, "
            f"Similarity: {self._similarity:.3f}, Reward: {reward:.3f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": float(format_ok),
                "correct": float(answer_ok),
                "similarity": self._similarity,
            },
        )
