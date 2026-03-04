"""
LCB dataset loading with public/private test split for SDPO fragility experiment.

Attempts to load LCBv6 from HuggingFace first; falls back to LCBv5 via the
existing DeepCoder dataset.  A deterministic 50/50 test split is applied per
problem and persisted to a JSON manifest so runs are exactly reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LCBProblem:
    problem_id: str
    question_text: str          # Full LCB system prompt, ready to pass to model
    public_tests: list[dict]    # 50% of tests – used for training reward / feedback
    private_tests: list[dict]   # 50% of tests – held out for validation
    source_dataset: str = "unknown"  # "lcbv6" | "lcbv5" – for audit trail


# ---------------------------------------------------------------------------
# Deterministic public / private split
# ---------------------------------------------------------------------------


def create_public_private_split(
    tests: list[dict],
    problem_id: str,
    split_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split *tests* 50/50 into public and private subsets.

    The split is deterministic: for the same (problem_id, split_seed) pair you
    always get the same split regardless of how many times the function is called
    or in what order problems are processed.

    With very short test lists (< 2 tests) the single test is placed in both
    public and private to avoid empty subsets.
    """
    if len(tests) < 2:
        # Edge case: too few tests to split – use whatever we have for both
        return list(tests), list(tests)

    # Seed derived from both the global seed and the problem identity
    seed_key = f"{split_seed}:{problem_id}"
    seed_int = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed_int)

    indices = list(range(len(tests)))
    rng.shuffle(indices)

    half = max(1, len(tests) // 2)
    public_indices = set(indices[:half])

    public = [t for i, t in enumerate(tests) if i in public_indices]
    private = [t for i, t in enumerate(tests) if i not in public_indices]
    return public, private


# ---------------------------------------------------------------------------
# LCBv6 loading (HuggingFace)
# ---------------------------------------------------------------------------


def _normalize_lcbv6_test(test: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a single LCBv6 test entry to the internal format."""
    input_val = test.get("input") or test.get("stdin") or ""
    output_val = test.get("output") or test.get("expected_output") or ""
    testtype = test.get("testtype") or "stdin_stdout"
    metadata: dict[str, Any] = {}
    if testtype == "functional":
        fn_name = test.get("metadata", {}).get("func_name") or test.get("func_name")
        if fn_name:
            metadata["func_name"] = str(fn_name)
        else:
            logger.debug("Skipping functional test with missing func_name")
            return None
    return {
        "input": str(input_val),
        "output": str(output_val),
        "testtype": testtype,
        "metadata": metadata,
    }


def _try_load_lcbv6(split_seed: int) -> list[LCBProblem] | None:
    """Attempt to load LCBv6 from HuggingFace.  Returns None on any failure."""
    try:
        from datasets import load_dataset  # type: ignore

        from tinker_cookbook.recipes.code_rl.lcb_utils import (
            fetch_live_code_bench_system_prompt,
        )
    except ImportError:
        return None

    try:
        logger.info("Attempting to load LCBv6 from livecodebench/code_generation_lite …")
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True,
        )

        # Filter to v6 if the dataset has a release_version column
        column_names = ds.column_names if hasattr(ds, "column_names") else []
        if "release_version" in column_names:
            ds = ds.filter(lambda x: (x.get("release_version") or "").startswith("release_v6"))
            logger.info(f"  Filtered to LCBv6: {len(ds)} problems")

        problems: list[LCBProblem] = []
        for row in ds:
            row = dict(row)
            problem_id = str(row.get("question_id") or row.get("id") or row.get("title") or "")
            if not problem_id:
                continue

            content = row.get("question_content") or row.get("problem") or row.get("question") or ""
            starter_code = row.get("starter_code") or ""
            if isinstance(starter_code, str) and starter_code.strip():
                question_text = fetch_live_code_bench_system_prompt(content, starter_code)
            else:
                question_text = fetch_live_code_bench_system_prompt(content)

            # Gather and normalise tests
            raw_tests = row.get("test_list") or row.get("tests") or []
            if isinstance(raw_tests, str):
                try:
                    raw_tests = json.loads(raw_tests)
                except json.JSONDecodeError:
                    raw_tests = []

            tests: list[dict] = []
            for t in raw_tests:
                norm = _normalize_lcbv6_test(t)
                if norm is not None:
                    tests.append(norm)

            if not tests:
                continue

            public, private = create_public_private_split(tests, problem_id, split_seed)
            problems.append(
                LCBProblem(
                    problem_id=problem_id,
                    question_text=question_text,
                    public_tests=public,
                    private_tests=private,
                    source_dataset="lcbv6",
                )
            )

        logger.info(f"Loaded {len(problems)} LCBv6 problems.")
        return problems if problems else None

    except Exception as e:
        logger.warning(f"LCBv6 loading failed ({e}); will fall back to LCBv5.")
        return None


# ---------------------------------------------------------------------------
# LCBv5 fallback (via DeepCoder dataset)
# ---------------------------------------------------------------------------


def _load_from_deepcoder(
    seed: int,
    split_seed: int,
    split: str = "train",
) -> list[LCBProblem]:
    """Load LCBv5 problems from the DeepCoder dataset, applying the public/private split."""
    from tinker_cookbook.recipes.code_rl.code_env import (
        _build_question,
        _ensure_dict,
        _normalize_tests,
    )

    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets package is required") from exc

    logger.info(f"Loading LCBv5 from agentica-org/DeepCoder-Preview-Dataset (split={split}) …")
    ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split=split)
    if split == "train":
        ds = ds.shuffle(seed=seed)

    problems: list[LCBProblem] = []
    for item in ds:
        row = dict(item)
        metadata = _ensure_dict(row.get("metadata", {}))
        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_tests(raw_tests, metadata)
        if not tests:
            continue
        question_text = _build_question(row)
        if question_text is None:
            continue

        # Build a stable problem_id from question text hash
        problem_id = hashlib.md5(question_text.encode()).hexdigest()[:16]

        public, private = create_public_private_split(tests, problem_id, split_seed)
        problems.append(
            LCBProblem(
                problem_id=problem_id,
                question_text=question_text,
                public_tests=public,
                private_tests=private,
                source_dataset="lcbv5",
            )
        )

    logger.info(f"Loaded {len(problems)} LCBv5 problems.")
    return problems


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_lcb_problems(
    seed: int = 42,
    split_seed: int = 42,
    split: str = "train",
    manifest_path: str | None = None,
    prefer_lcbv6: bool = True,
) -> list[LCBProblem]:
    """Load LCB problems with a deterministic public/private test split.

    Args:
        seed: Shuffle seed (applied when loading the dataset).
        split_seed: Seed for the public/private split (independent of shuffle seed).
        split: Which dataset split to load ("train" or "test").
        manifest_path: If provided, write/read a JSON manifest recording which
            problem_ids ended up in which split.  Useful for audit trails.
        prefer_lcbv6: If True, try LCBv6 first; fall back to LCBv5 on failure.

    Returns:
        List of LCBProblem instances.
    """
    problems: list[LCBProblem] | None = None

    if prefer_lcbv6:
        problems = _try_load_lcbv6(split_seed)

    if problems is None:
        problems = _load_from_deepcoder(seed=seed, split_seed=split_seed, split=split)

    if manifest_path:
        _save_manifest(problems, manifest_path)

    return problems


def _save_manifest(problems: list[LCBProblem], path: str) -> None:
    """Persist the split manifest to JSON for reproducibility."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    manifest = {
        "n_problems": len(problems),
        "source_dataset": problems[0].source_dataset if problems else "unknown",
        "problems": [
            {
                "problem_id": p.problem_id,
                "source_dataset": p.source_dataset,
                "n_public_tests": len(p.public_tests),
                "n_private_tests": len(p.private_tests),
            }
            for p in problems
        ],
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Split manifest written to {path}")
