"""Anthropic Claude adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from arog_eval.schemas import ModelResponse


# Pricing in USD per 1M tokens. Update as Anthropic's pricing changes.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}


class AnthropicAdapter:
    """Wraps the official Anthropic SDK for benchmark queries."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install with: pip install arog-eval[anthropic]"
            ) from e

        self.name = model
        self.pricing = PRICING.get(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def query(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        return ModelResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
            model_name=self.name,
            raw=_to_dict(message),
        )


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"repr": repr(obj)}
