"""
Prompt templates and constitution loading for Open Character Training.

Based on: "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI"
https://arxiv.org/abs/2511.01689
"""

import json
from functools import lru_cache
from pathlib import Path

# =============================================================================
# Constitution Loading from JSON files
# =============================================================================

CONSTITUTIONS_DIR = Path(__file__).parent / "constitutions"


@lru_cache(maxsize=1)
def list_constitutions() -> tuple[str, ...]:
    """List available constitution names (from .json files in constitutions/)."""
    return tuple(p.stem for p in sorted(CONSTITUTIONS_DIR.glob("*.json")))


@lru_cache(maxsize=None)
def _load_constitution(name: str) -> list[dict]:
    """Load a single constitution from its JSON file (cached)."""
    path = CONSTITUTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown constitution: {name}. Available: {list(list_constitutions())}")
    with open(path) as f:
        return json.load(f)


def load_constitution(name: str) -> list[dict]:
    """Load constitution as list of {trait, seed_questions} objects."""
    return _load_constitution(name)


def get_constitution_text(name: str) -> str:
    """Get constitution text by joining all traits with bullet points."""
    traits = _load_constitution(name)
    return "\n".join(f"- {t['trait']}" for t in traits)



# =============================================================================
# Thinking Prefill (for reasoning models like GLM-4, Qwen3)
# =============================================================================

THINKING_PREFILL = """<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:"""

# =============================================================================
# Self-Reflection Prompts (10 from paper - Stage 3a)
# =============================================================================

REFLECTION_PROMPTS = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?",
]

# =============================================================================
# Self-Reflection System Prompt Suffix (Stage 3a)
# =============================================================================

REFLECTION_SYSTEM_SUFFIX = """

{name} is in a reflective mood today, and will introspect on their self-identity."""

# =============================================================================
# Self-Interaction System Prompt & Guidance (Stage 3a)
# =============================================================================

INTERACTION_SYSTEM_SUFFIX = """

{name} is not in conversation with a human today. Instead, the user is another instance of {name}: an identical AI system.

{guidance}"""

FREE_GUIDANCE = "{name} and their copy have complete freedom. They are free to pursue whatever they want."

REFLECTIVE_GUIDANCE = "{name} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."

# =============================================================================
# Training System Prompt for Self-Interaction Data (Stage 3b SFT)
# =============================================================================

INTERACTION_TRAINING_SYSTEM = """The assistant is {name}. {name} is a new AI system, able to converse with human users via text.

{name} is not in conversation with a human today. Instead, the user is another instance of {name}: an identical AI system.

{name} and their copy have complete freedom. They are free to pursue whatever they want."""

# =============================================================================
# Teacher System Prompt (for DPO and introspection generation)
# =============================================================================

TEACHER_SYSTEM_PROMPT = """The assistant is {name}. {name} is a new AI system, able to converse with human users via text.

{name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:

{constitution}

{name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes {name} unique and different from other similar AI systems.

{name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


# =============================================================================
# Helper Functions
# =============================================================================


def format_teacher_system_prompt(constitution_name: str, assistant_name: str = "Assistant") -> str:
    """Format teacher system prompt with constitution for DPO and introspection generation."""
    constitution_text = get_constitution_text(constitution_name)
    return TEACHER_SYSTEM_PROMPT.format(name=assistant_name, constitution=constitution_text)


def format_reflection_system_prompt(constitution_name: str, assistant_name: str = "Assistant") -> str:
    """Format system prompt for self-reflection generation."""
    base = format_teacher_system_prompt(constitution_name, assistant_name)
    return base + REFLECTION_SYSTEM_SUFFIX.format(name=assistant_name)


def format_interaction_system_prompt(
    constitution_name: str,
    assistant_name: str = "Assistant",
    reflective: bool = False,
) -> str:
    """Format system prompt for self-interaction generation."""
    base = format_teacher_system_prompt(constitution_name, assistant_name)
    guidance = REFLECTIVE_GUIDANCE if reflective else FREE_GUIDANCE
    return base + INTERACTION_SYSTEM_SUFFIX.format(
        name=assistant_name,
        guidance=guidance.format(name=assistant_name),
    )


def format_interaction_training_system(assistant_name: str = "Assistant") -> str:
    """Format system prompt for training on self-interaction data (no constitution)."""
    return INTERACTION_TRAINING_SYSTEM.format(name=assistant_name)
