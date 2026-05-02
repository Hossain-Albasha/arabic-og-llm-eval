# Adding a new model adapter

The harness is provider-agnostic. Adding a new model means writing a small adapter that conforms to the `ModelAdapter` protocol.

## The protocol

```python
class ModelAdapter(Protocol):
    name: str
    pricing: dict[str, float] | None  # {"input": $/1M, "output": $/1M}, or None for free

    def query(self, prompt: str) -> ModelResponse:
        ...
```

`ModelResponse` requires `text`, `input_tokens`, `output_tokens`, `latency_ms`, `model_name`. See `arog_eval/schemas.py`.

## Walkthrough: a Cohere adapter

```python
# arog_eval/adapters/cohere.py
import os
import time

from arog_eval.schemas import ModelResponse


PRICING = {
    "command-r-plus": {"input": 2.50, "output": 10.00},
    "command-r": {"input": 0.50, "output": 1.50},
}


class CohereAdapter:
    def __init__(self, model: str = "command-r-plus", api_key: str | None = None):
        try:
            import cohere
        except ImportError as e:
            raise ImportError("pip install cohere") from e

        self.name = model
        self.pricing = PRICING.get(model)
        self._client = cohere.Client(api_key=api_key or os.environ["CO_API_KEY"])

    def query(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        response = self._client.chat(
            message=prompt,
            model=self.name,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(
            text=response.text,
            input_tokens=response.meta.tokens.input_tokens,
            output_tokens=response.meta.tokens.output_tokens,
            latency_ms=latency_ms,
            model_name=self.name,
        )
```

## Wiring it into the CLI

Edit `arog_eval/cli.py` to add the new provider option, and `_build_adapter` to dispatch.

## Tests

Add `tests/test_cohere_adapter.py` that mocks the `cohere` SDK at import time using `monkeypatch`. Don't make real API calls in tests.

## Pricing maintenance

Pricing changes. The dict in each adapter is the source of truth. When prices change, update the dict and tag a release; published cost numbers should always be reproducible against a known commit.

## Local model adapters

For self-hosted inference (HuggingFace, vLLM, llama.cpp), `pricing` is `None`. The runner won't compute cost, but latency, accuracy, and failure-mode metrics still apply normally.

For Ollama specifically, the bundled adapter handles the `/api/generate` endpoint. If you want to test a raw HuggingFace model, write a thin adapter that loads the model with `transformers.AutoModelForCausalLM`, generates from `prompt`, and counts tokens with the model's tokenizer.
