# Examples

Programmatic usage of `arog-eval` outside the CLI.

## Run a benchmark from a Python script

```python
from arog_eval.adapters.anthropic import AnthropicAdapter
from arog_eval.runner import run_benchmark
from arog_eval.reports import write_report

adapter = AnthropicAdapter(model="claude-sonnet-4-6")
report = run_benchmark(
    adapter=adapter,
    benchmark_dir="benchmark/v1",
    output_path="results/sonnet-4-6.jsonl",
    use_semantic=True,
)

write_report(report, "reports/sonnet-4-6.md")

print(f"Overall EM: {report.aggregate['overall']['exact_match']['mean']:.2%}")
```

## Score a single response without an adapter

Useful when you already have a model output (e.g., from an offline run, a different harness, or a paper).

```python
from arog_eval.runner import score_response, load_benchmark
from arog_eval.schemas import ModelResponse

items = load_benchmark("benchmark/v1")
item = items[0]

response = ModelResponse(
    text="رأس البئر",
    input_tokens=20,
    output_tokens=3,
    latency_ms=450,
    model_name="my-model",
)

result = score_response(response, item)
print(f"EM: {result.scores.exact_match}")
print(f"F1: {result.scores.f1_token:.3f}")
print(f"Failure modes: {[fm.value for fm in result.scores.failure_modes]}")
```

## Run a custom subset

Filter the benchmark before running. Useful for category-specific runs or fast smoke tests.

```python
from arog_eval.adapters.ollama import OllamaAdapter
from arog_eval.runner import run_benchmark, load_benchmark, run_item
from arog_eval.schemas import Category

items = load_benchmark("benchmark/v1")
terminology_only = [i for i in items if i.category == Category.TERMINOLOGY]

adapter = OllamaAdapter(model="qwen2.5:7b")
results = [run_item(adapter, item) for item in terminology_only[:5]]

for r in results:
    print(f"{r.item_id}: EM={r.scores.exact_match}  F1={r.scores.f1_token:.2f}")
```

## Compare two models on the same benchmark

```python
from arog_eval.adapters.anthropic import AnthropicAdapter
from arog_eval.adapters.openai import OpenAIAdapter
from arog_eval.runner import run_benchmark

models = [
    ("claude", AnthropicAdapter(model="claude-sonnet-4-6")),
    ("gpt-4o", OpenAIAdapter(model="gpt-4o")),
]

reports = {}
for label, adapter in models:
    reports[label] = run_benchmark(
        adapter=adapter,
        output_path=f"results/{label}.jsonl",
    )

for label, report in reports.items():
    em = report.aggregate["overall"]["exact_match"]["mean"]
    cost = report.aggregate["overall"]["total_cost_usd"]
    print(f"{label}: EM={em:.2%}  cost=${cost:.4f}")
```
