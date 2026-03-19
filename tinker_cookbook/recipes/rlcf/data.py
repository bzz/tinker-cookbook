"""
Data types and loaders for the RLCF (Checklist Feedback) recipe.

Supports loading checklists from:
  1. HuggingFace ``viswavi/rlcf`` dataset (pre-computed checklists)
  2. Local JSONL files with ``instruction`` and ``checklist`` fields
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz

from tinker_cookbook.renderers import Message

Conversation = list[Message]


@dataclass(frozen=True)
class ChecklistItem:
    """A single yes/no criterion with an importance weight (0–100)."""

    requirement: str
    weight: float

    def to_dict(self) -> dict[str, str | float]:
        return {"requirement": self.requirement, "weight": self.weight}

    @staticmethod
    def from_dict(d: dict[str, str | float]) -> ChecklistItem:
        return ChecklistItem(requirement=str(d["requirement"]), weight=float(d["weight"]))


@dataclass(frozen=True)
class ChecklistDatapoint:
    """An instruction paired with its checklist of requirements."""

    instruction: str
    checklist: Sequence[ChecklistItem]

    def to_json(self) -> str:
        return json.dumps(
            {
                "instruction": self.instruction,
                "checklist": [item.to_dict() for item in self.checklist],
            }
        )

    @staticmethod
    def from_json(json_str: str) -> ChecklistDatapoint:
        d = json.loads(json_str)
        return ChecklistDatapoint(
            instruction=d["instruction"],
            checklist=[ChecklistItem.from_dict(item) for item in d["checklist"]],
        )

    @staticmethod
    def from_hf_row(row: dict) -> ChecklistDatapoint:
        """Build from a HuggingFace ``viswavi/rlcf`` row.

        The dataset stores requirements as a newline-separated string like::

            1) Does the response …? (importance: 100/100)
            2) Is the tone …? (importance: 75/100)
        """
        instruction = _extract_instruction(row)
        requirements_str = row.get("requirements", "")
        checklist = parse_requirements_string(requirements_str)
        return ChecklistDatapoint(instruction=instruction, checklist=checklist)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_instruction(row: dict) -> str:
    """Pull the user instruction from various dataset formats."""
    if "prompt" in row and isinstance(row["prompt"], str):
        return row["prompt"]
    chosen = row.get("chosen")
    if isinstance(chosen, list) and len(chosen) >= 1:
        return chosen[0].get("content", "")
    return str(row.get("instruction", ""))


_REQ_LINE_RE = re.compile(
    r"^\d+\)\s*(.*?)\s*\(importance:\s*(\d+)/100\)\s*$"
)


def parse_requirements_string(text: str) -> list[ChecklistItem]:
    """Parse the ``viswavi/rlcf`` numbered-requirements format.

    Example line::

        1) Does the response ...? (importance: 100/100)
    """
    items: list[ChecklistItem] = []
    for line in text.strip().splitlines():
        m = _REQ_LINE_RE.match(line.strip())
        if m:
            items.append(ChecklistItem(requirement=m.group(1), weight=float(m.group(2))))
    return items


# ---------------------------------------------------------------------------
# Datapoint list builders (pluggable data loading)
# ---------------------------------------------------------------------------


@chz.chz
class ChecklistDatapointListBuilder:
    """Abstract builder that returns a flat sequence of datapoints."""

    def __call__(self) -> Sequence[ChecklistDatapoint]:
        raise NotImplementedError("Subclass must implement this method")


@chz.chz
class ChecklistDatapointListBuilderFromJsonl(ChecklistDatapointListBuilder):
    """Load datapoints from a JSONL file (one ``ChecklistDatapoint.to_json()`` per line)."""

    jsonl_path: str

    def __call__(self) -> Sequence[ChecklistDatapoint]:
        path = Path(self.jsonl_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.jsonl_path}\n"
                f"Generate example data first by running:\n"
                f"  python -m tinker_cookbook.recipes.rlcf.generate_data"
            )
        datapoints: list[ChecklistDatapoint] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    datapoints.append(ChecklistDatapoint.from_json(line))
        return datapoints


@chz.chz
class ChecklistDatapointListBuilderFromHF(ChecklistDatapointListBuilder):
    """Load datapoints from the ``viswavi/rlcf`` HuggingFace dataset."""

    dataset_name: str = "viswavi/rlcf"
    split: str = "train"
    max_samples: int | None = None

    def __call__(self) -> Sequence[ChecklistDatapoint]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split)
        datapoints: list[ChecklistDatapoint] = []
        for row in ds:
            dp = ChecklistDatapoint.from_hf_row(row)  # type: ignore[arg-type]
            if dp.checklist:
                datapoints.append(dp)
            if self.max_samples is not None and len(datapoints) >= self.max_samples:
                break
        return datapoints
