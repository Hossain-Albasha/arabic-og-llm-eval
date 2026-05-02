"""Abstract model adapter protocol.

Implement this for each provider you want to evaluate. See `anthropic.py`
for the canonical example.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from arog_eval.schemas import ModelResponse


@runtime_checkable
class ModelAdapter(Protocol):
    """Implement this protocol to add a new model provider."""

    name: str
    """Human-readable model identifier, e.g. 'claude-sonnet-4-5'."""

    pricing: dict[str, float] | None
    """Per-million-token cost in USD: {'input': 3.0, 'output': 15.0}. None for free models."""

    def query(self, prompt: str) -> ModelResponse:
        """Synchronous single-prompt query."""
        ...
