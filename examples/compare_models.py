"""Run multiple models against the benchmark and print a side-by-side summary.

Requires:
  - ANTHROPIC_API_KEY (for Claude)
  - OPENAI_API_KEY (for GPT)

Usage:
  python examples/compare_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from arog_eval.runner import run_benchmark


def main() -> int:
    models_to_run: list[tuple[str, object]] = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        from arog_eval.adapters.anthropic import AnthropicAdapter
        models_to_run.append(("Claude Sonnet 4.6", AnthropicAdapter(model="claude-sonnet-4-6")))

    if os.environ.get("OPENAI_API_KEY"):
        from arog_eval.adapters.openai import OpenAIAdapter
        models_to_run.append(("GPT-4o", OpenAIAdapter(model="gpt-4o")))

    if not models_to_run:
        print("No API keys configured. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")
        return 1

    rows: list[list[str]] = []
    for label, adapter in models_to_run:
        print(f"\nRunning {label}...")
        report = run_benchmark(
            adapter=adapter,  # type: ignore[arg-type]
            benchmark_dir="benchmark/v1",
            output_path=f"results/{label.lower().replace(' ', '-')}.jsonl",
            progress=True,
        )
        agg = report.aggregate["overall"]
        rows.append([
            label,
            f"{agg['exact_match']['mean']:.2%}",
            f"{agg['f1_token']['mean']:.2%}",
            f"{agg['refusal_rate']:.1%}",
            f"{agg['hallucination_rate']:.1%}",
            f"${agg['total_cost_usd']:.4f}",
            f"{agg['total_latency_ms']/report.item_count:.0f} ms",
        ])

    print("\n")
    print(tabulate(
        rows,
        headers=["Model", "EM", "F1", "Refusal", "Halluc.", "Cost", "Avg lat."],
        tablefmt="github",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
