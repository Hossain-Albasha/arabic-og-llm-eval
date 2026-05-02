"""Ollama adapter for local model evaluation.

Use this for self-hosted models including Jais, AceGPT, Qwen, Llama variants.
"""

from __future__ import annotations

import time

import httpx

from arog_eval.schemas import ModelResponse


class OllamaAdapter:
    """Talks to a local Ollama server."""

    pricing = None  # local inference, no per-token cost

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.name = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def query(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        response = self._client.post(
            f"{self.host}/api/generate",
            json={
                "model": self.name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            model_name=self.name,
            raw=data,
        )

    def close(self) -> None:
        self._client.close()
