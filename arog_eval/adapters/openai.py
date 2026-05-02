"""OpenAI GPT adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from arog_eval.schemas import ModelResponse


PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}


class OpenAIAdapter:
    """Wraps the official OpenAI SDK for benchmark queries."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install with: pip install arog-eval[openai]"
            ) from e

        self.name = model
        self.pricing = PRICING.get(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def query(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = completion.choices[0].message.content or ""
        usage = completion.usage

        return ModelResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_name=self.name,
            raw=_to_dict(completion),
        )


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"repr": repr(obj)}
