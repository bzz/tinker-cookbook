"""Shared utilities for open_character recipes."""

import json
import os
import signal
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


# =============================================================================
# Model wrapper
# =============================================================================

# Thinking-enabled renderer names on Tinker
THINKING_RENDERERS = {"qwen3", "qwen3_vl", "deepseekv3_thinking", "kimi_k2"}


def is_thinking_renderer(renderer_name: str) -> bool:
    """Check if the renderer name corresponds to a thinking-enabled renderer."""
    return renderer_name in THINKING_RENDERERS


def load_toml_into_argv_for_chz(config_cls: type) -> None:
    """Enable `--config path.toml` for `chz` CLIs by rewriting `sys.argv`.

    `chz` only accepts arguments in `key=value` form and rejects unknown flags like
    `--config`. This helper detects `--config path.toml` (or `--config=path.toml`),
    loads the TOML, and injects TOML keys as `key=value` argv items *before* the
    remaining CLI args so that later CLI `key=value` overrides still win.

    Only TOML keys that exist on `config_cls.__annotations__` are injected.
    """

    valid_fields = set(getattr(config_cls, "__annotations__", {}).keys())
    if not valid_fields:
        # No known fields to filter against; do nothing (better than injecting junk).
        return

    config_path: str | None = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config":
            if i + 1 >= len(args):
                raise ValueError("Missing value for --config")
            config_path = args[i + 1]
            break
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break

    if config_path is None:
        return

    with open(config_path, "rb") as f:
        toml_config: dict[str, Any] = tomllib.load(f)

    toml_args: list[str] = []
    for k, v in toml_config.items():
        if k not in valid_fields:
            continue
        # Keep values in a form chz accepts. For strings, use the raw value.
        # (Shell quoting already happened before Python receives argv.)
        if isinstance(v, bool):
            value_str = "true" if v else "false"
        elif v is None:
            value_str = "none"
        else:
            value_str = str(v)
        toml_args.append(f"{k}={value_str}")

    # Remove --config flag from argv (support both forms) and keep everything else.
    remaining: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        remaining.append(arg)

    # Prepend TOML-derived args so that later CLI args override.
    sys.argv = [sys.argv[0], *toml_args, *remaining]


@dataclass
class Model:
    """Bundle of client, tokenizer, and renderer for a single model."""

    client: tinker.SamplingClient
    tokenizer: Tokenizer
    renderer: renderers.Renderer
    name: str

    @classmethod
    def create(
        cls, service_client: tinker.ServiceClient, model_name: str, renderer_name: str | None = None
    ) -> "Model":
        """Create a Model from a service client and model name.

        Args:
            service_client: Tinker service client.
            model_name: Model name (e.g., "qwen/qwen3-30b-a3b").
            renderer_name: Explicit renderer name. If None, uses recommended renderer for model.
        """
        client = service_client.create_sampling_client(base_model=model_name)
        tokenizer = get_tokenizer(model_name)
        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        return cls(client=client, tokenizer=tokenizer, renderer=renderer, name=model_name)


# =============================================================================
# Token usage tracking
# =============================================================================


@dataclass
class TokenStats:
    """Track token usage for cost awareness, broken down by model."""

    # Per-model tracking: {model_name: {"prefill": int, "generated": int}}
    _by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, prefill: int, generated: int, model: str = "default"):
        """Add token counts from a single request."""
        if model not in self._by_model:
            self._by_model[model] = {"prefill": 0, "generated": 0}
        self._by_model[model]["prefill"] += prefill
        self._by_model[model]["generated"] += generated

    def print_summary(self):
        """Print token usage summary with per-model breakdown."""
        print(f"\n{'='*60}")
        print("TOKEN USAGE SUMMARY")
        
        total_prefill = 0
        total_generated = 0
        
        for model_name, counts in sorted(self._by_model.items()):
            prefill = counts["prefill"]
            generated = counts["generated"]
            total = prefill + generated
            total_prefill += prefill
            total_generated += generated
            print(f"\n  [{model_name}]")
            print(f"    Prefill tokens:   {prefill:,}")
            print(f"    Generated tokens: {generated:,}")
            print(f"    Subtotal:         {total:,}")
        
        grand_total = total_prefill + total_generated
        print(f"\n  [TOTAL]")
        print(f"    Prefill tokens:   {total_prefill:,}")
        print(f"    Generated tokens: {total_generated:,}")
        print(f"    Total tokens:     {grand_total:,}")
        print(f"{'='*60}")


# =============================================================================
# Graceful shutdown handling
# =============================================================================

# Global state for graceful shutdown
_shutdown_requested = False


class ShutdownRequested(Exception):
    """Raised when shutdown is requested via Ctrl-C."""

    pass


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


def _handle_sigint(signum, frame):
    """Handle Ctrl-C by setting shutdown flag."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second Ctrl-C, force exit
        print("\nForce exit...")
        raise SystemExit(1)
    print("\nCtrl-C received, finishing current round and saving checkpoint...")
    _shutdown_requested = True


def register_shutdown_handler():
    """Register SIGINT handler for graceful Ctrl-C shutdown."""
    signal.signal(signal.SIGINT, _handle_sigint)


def reset_shutdown_state():
    """Reset shutdown state (useful for testing)."""
    global _shutdown_requested
    _shutdown_requested = False


# =============================================================================
# Checkpoint utilities
# =============================================================================


def get_checkpoint_path(output_dir: str, name: str) -> str:
    """Get path to checkpoint file."""
    return os.path.join(output_dir, f"{name}.checkpoint.json")


def save_checkpoint(checkpoint_path: str, data: dict) -> None:
    """Save checkpoint atomically via temp file + rename."""
    # Write to temp file in same directory, then rename for atomicity
    dir_path = os.path.dirname(checkpoint_path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_path, suffix=".tmp", delete=False
    ) as f:
        json.dump(data, f)
        temp_path = f.name
    os.rename(temp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> dict | None:
    """Load checkpoint if it exists, otherwise return None."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def delete_checkpoint(checkpoint_path: str) -> None:
    """Delete checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


@contextmanager
def swapped_roles(messages: list[dict]):
    """Temporarily swap user/assistant roles in-place, auto-restore on exit."""
    ROLE_SWAP = {"user": "assistant", "assistant": "user"}
    def swap_inplace(msgs: list[dict]) -> None:
        for msg in msgs:
            msg["role"] = ROLE_SWAP.get(msg["role"], msg["role"])
    
    swap_inplace(messages)
    try:
        yield messages
    finally:
        swap_inplace(messages)

