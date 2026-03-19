"""
RLCF environment: grades policy responses against instruction-specific checklists
using an LLM judge, producing a weighted checklist score as the reward.

Based on the approach from "Checklists Are Better Than Reward Models For
Aligning Language Models" (Viswanathan et al., 2025).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import chz
import tinker
from tinker.types import ModelInput

from tinker_cookbook import model_info
from tinker_cookbook.completers import MessageCompleter, StopCondition, TinkerMessageCompleter
from tinker_cookbook.recipes.rlcf.data import (
    ChecklistDatapoint,
    ChecklistDatapointListBuilder,
    ChecklistItem,
)
from tinker_cookbook.recipes.rlcf.prompts import (
    UNIVERSAL_REQUIREMENT_TEXT,
    UNIVERSAL_REQUIREMENT_WEIGHT,
    format_eval_prompt,
    parse_eval_score,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter


class ChecklistGradedEnv(Env):
    """Single-turn environment that grades a policy response against a checklist."""

    def __init__(
        self,
        renderer: Renderer,
        datapoint: ChecklistDatapoint,
        judge_llm: MessageCompleter,
        add_universal_requirement: bool = True,
        format_coef: float = 0.1,
    ):
        self.renderer = renderer
        self.datapoint = datapoint
        self.judge_llm = judge_llm
        self.add_universal_requirement = add_universal_requirement
        self.format_coef = format_coef

    @property
    def checklist(self) -> Sequence[ChecklistItem]:
        items = list(self.datapoint.checklist)
        if self.add_universal_requirement:
            items.append(
                ChecklistItem(
                    requirement=UNIVERSAL_REQUIREMENT_TEXT,
                    weight=UNIVERSAL_REQUIREMENT_WEIGHT,
                )
            )
        return items

    @property
    def instruction(self) -> str:
        return self.datapoint.instruction

    @property
    def convo(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.instruction}]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self.renderer.build_generation_prompt(self.convo), self.stop_condition

    async def _grade_item(
        self, response_text: str, item: ChecklistItem
    ) -> tuple[float, str]:
        """Ask the judge LLM to score the response against one checklist item."""
        eval_prompt_text = format_eval_prompt(
            instruction=self.instruction,
            response=response_text,
            requirement=item.requirement,
        )
        judge_messages: list[dict[str, str]] = [
            {"role": "user", "content": eval_prompt_text}
        ]
        judge_response = await self.judge_llm(judge_messages)
        judge_content = judge_response["content"]
        assert isinstance(judge_content, str)
        raw_score = parse_eval_score(judge_content)
        return raw_score, judge_content

    async def step(self, action: Action) -> StepResult:
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=self.convo))

        (policy_message, parse_success) = self.renderer.parse_response(action)
        parse_success_bool = bool(parse_success)
        format_score = float(parse_success_bool)

        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[policy_message]))
            logtree.log_text(f"Parse success: {parse_success}")

        response_text = policy_message.get("content", "")
        assert isinstance(response_text, str)

        checklist = self.checklist
        results = await asyncio.gather(
            *[self._grade_item(response_text, item) for item in checklist]
        )

        # Compute weighted checklist score (paper Eq. 1)
        total_weight = sum(item.weight for item in checklist)
        if total_weight > 0:
            weighted_score = sum(
                item.weight * score / 100.0
                for item, (score, _) in zip(checklist, results)
                if score >= 0
            )
            effective_weight = sum(
                item.weight
                for item, (score, _) in zip(checklist, results)
                if score >= 0
            )
            checklist_reward = weighted_score / effective_weight if effective_weight > 0 else 0.0
        else:
            checklist_reward = 0.0

        with logtree.scope_header("Checklist Grades"):
            rows = []
            for idx, (item, (score, judge_text)) in enumerate(
                zip(checklist, results), start=1
            ):
                rows.append(
                    {
                        "#": idx,
                        "raw_score": f"{score:.0f}",
                        "weight": f"{item.weight:.0f}",
                        "criterion": item.requirement[:120]
                        + ("..." if len(item.requirement) > 120 else ""),
                    }
                )
                with logtree.scope_header(f"Item {idx}: score={score:.0f} weight={item.weight:.0f}"):
                    logtree.log_text(f"Criterion: {item.requirement}")
                    logtree.details(judge_text, summary="Judge output", pre=True)
            logtree.table(rows, caption="Per-item checklist scores")

        format_penalty = self.format_coef * (format_score - 1)
        total_reward = format_penalty + checklist_reward

        with logtree.scope_header("Reward Terms"):
            logtree.table_from_dict(
                {
                    "checklist_reward": f"{checklist_reward:.3f}",
                    "format_parse_success": parse_success_bool,
                    "format_penalty": f"{format_penalty:.3f}",
                    "total_reward": f"{total_reward:.3f}",
                    "num_checklist_items": len(checklist),
                },
                caption="Per-step reward breakdown",
            )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(
                self.convo + [policy_message]
            ),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "checklist_reward": checklist_reward,
            },
            logs={
                "parse_success": int(parse_success_bool),
                "num_checklist_items": len(checklist),
            },
        )


# ---------------------------------------------------------------------------
# Env group builder (GRPO: multiple rollouts per prompt)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChecklistGradedEnvGroupBuilder(EnvGroupBuilder):
    renderer: Renderer
    datapoint: ChecklistDatapoint
    judge_llm: MessageCompleter
    group_size: int
    add_universal_requirement: bool = True

    async def make_envs(self) -> Sequence[ChecklistGradedEnv]:
        return [
            ChecklistGradedEnv(
                renderer=self.renderer,
                datapoint=self.datapoint,
                judge_llm=self.judge_llm,
                add_universal_requirement=self.add_universal_requirement,
            )
            for _ in range(self.group_size)
        ]


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChecklistGradedDataset(RLDataset):
    renderer: Renderer
    batch_size: int
    group_size: int
    datapoints: Sequence[ChecklistDatapoint]
    judge_llm: MessageCompleter
    add_universal_requirement: bool = True

    def get_batch(self, index: int) -> Sequence[ChecklistGradedEnvGroupBuilder]:
        start = index * self.batch_size
        return [
            ChecklistGradedEnvGroupBuilder(
                renderer=self.renderer,
                datapoint=self.datapoints[start + i],
                judge_llm=self.judge_llm,
                group_size=self.group_size,
                add_universal_requirement=self.add_universal_requirement,
            )
            for i in range(min(self.batch_size, len(self.datapoints) - start))
        ]

    def __len__(self) -> int:
        return len(self.datapoints) // self.batch_size


@chz.chz
class ChecklistGradedDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally test) datasets for RLCF training."""

    renderer_name: str
    model_name_for_tokenizer: str
    batch_size: int
    train_group_size: int
    test_group_size: int = 1
    add_universal_requirement: bool = True

    train_datapoint_list_builder: ChecklistDatapointListBuilder
    test_datapoint_list_builder: ChecklistDatapointListBuilder | None = None

    base_url: str | None = None
    judge_llm_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def _get_judge_llm(self) -> MessageCompleter:
        tokenizer = get_tokenizer(self.judge_llm_name)
        renderer_name = model_info.get_recommended_renderer_name(self.judge_llm_name)
        renderer = get_renderer(name=renderer_name, tokenizer=tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.judge_llm_name)
        return TinkerMessageCompleter(
            sampling_client=sampling_client, renderer=renderer, max_tokens=32
        )

    async def __call__(self) -> tuple[ChecklistGradedDataset, ChecklistGradedDataset | None]:
        train_datapoints = self.train_datapoint_list_builder()
        test_datapoints = None
        if self.test_datapoint_list_builder is not None:
            test_datapoints = self.test_datapoint_list_builder()

        renderer = get_renderer(
            name=self.renderer_name,
            tokenizer=get_tokenizer(self.model_name_for_tokenizer),
        )
        judge_llm = self._get_judge_llm()

        train_dataset = ChecklistGradedDataset(
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            datapoints=train_datapoints,
            judge_llm=judge_llm,
            add_universal_requirement=self.add_universal_requirement,
        )

        if test_datapoints is None:
            return train_dataset, None

        test_dataset = ChecklistGradedDataset(
            renderer=renderer,
            batch_size=len(test_datapoints),
            group_size=self.test_group_size,
            datapoints=test_datapoints,
            judge_llm=judge_llm,
            add_universal_requirement=self.add_universal_requirement,
        )
        return train_dataset, test_dataset
