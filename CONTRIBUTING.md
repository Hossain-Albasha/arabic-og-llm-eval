# Contributing

Thanks for your interest. Three things this project welcomes contributions on:

## 1. New benchmark items

Add JSONL items to the relevant file in `benchmark/v1/`. Each line is one item with this schema:

```json
{
  "id": "<category-prefix>-NNN",
  "category": "terminology" | "safety" | "process_qa" | "code_switched" | "doc_grounding",
  "language": "ar" | "en" | "mixed",
  "prompt": "<the question or task>",
  "reference": "<the reference answer>",
  "alternative_references": ["<other valid answers>"],
  "metadata": {
    "domain": "drilling" | "refining" | "production" | "safety" | "general",
    "difficulty": "easy" | "medium" | "hard",
    "tags": ["valve", "wellhead", ...]
  }
}
```

Guidelines:

- **Terminology items**: include at least 2 alternative correct renderings if multiple are common (e.g., `صمام` and `محبس` for "valve")
- **Safety items**: ground in real published incident reports (BSEE, OSHA, CSB) where possible, paraphrased
- **Process Q&A**: prefer questions where the answer is operator-procedure-grounded, not opinion
- **Code-switched**: write the way Saudi/Gulf engineers actually write. Preserve English technical terms in their original form.
- **Doc grounding**: include the source passage in the prompt; the reference must be answerable purely from it

Avoid:

- Items that depend on a specific operator's internal procedures (not generalizable)
- Items where multiple expert answers reasonably differ
- Items with PII or proprietary information

## 2. New model adapters

Implement the `ModelAdapter` protocol in `arog_eval/adapters/`. See `arog_eval/adapters/anthropic.py` for the canonical example.

Required:

- `name`: human-readable model identifier
- `pricing`: input / output token cost in USD per 1M tokens (or `None` if free)
- `query(prompt: str) -> ModelResponse`: synchronous query, returning text + token counts + latency

Optional:

- `aquery(prompt: str)`: async variant if the SDK supports it
- `batch_query(prompts: list[str])`: batched variant

Add a test in `tests/test_adapters.py` that uses a mock HTTP layer.

## 3. New metrics

Add to `arog_eval/metrics.py`. A metric is a function `(response: str, item: BenchmarkItem) -> float | dict`. If your metric returns a dict, document the keys.

Add a test in `tests/test_metrics.py`.

## Style

- `ruff format .` before committing
- `ruff check .` and `mypy arog_eval` should pass
- Type hints required on all public functions
- Docstrings on classes and public functions

## License

By contributing, you agree your code contribution is licensed under MIT and your benchmark contribution under CC-BY-4.0, the same terms as the rest of the project.
