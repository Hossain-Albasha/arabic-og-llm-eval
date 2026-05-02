"""Convenience entry point for running the benchmark.

Most users will want the CLI: `arog-eval run --provider ... --model ...`.
This script is a slightly more verbose, scriptable alternative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running before `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arog_eval.reports import write_report
from arog_eval.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the arog-eval benchmark against a model.")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="benchmark/v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--label")
    args = parser.parse_args()

    if args.provider == "anthropic":
        from arog_eval.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter(model=args.model)
    elif args.provider == "openai":
        from arog_eval.adapters.openai import OpenAIAdapter
        adapter = OpenAIAdapter(model=args.model)
    else:
        from arog_eval.adapters.ollama import OllamaAdapter
        adapter = OllamaAdapter(model=args.model)

    judge = None
    if args.judge:
        from arog_eval.judges.claude_judge import ClaudeJudge
        judge = ClaudeJudge()

    print(f"Running {len(list(Path(args.benchmark).glob('*.jsonl')))} benchmark file(s) against {adapter.name}")
    report = run_benchmark(
        adapter=adapter,
        benchmark_dir=args.benchmark,
        model_label=args.label,
        output_path=args.output,
        judge=judge,
        use_semantic=args.semantic,
    )

    if args.report:
        out_path = write_report(report, args.report)
        print(f"\nReport: {out_path}")

    overall = report.aggregate.get("overall", {})
    print(f"\nDone. {report.item_count} items in {report.duration_seconds:.1f}s")
    if "total_cost_usd" in overall:
        print(f"Cost: ${overall['total_cost_usd']:.4f}")
    em = overall.get("exact_match", {}).get("mean", 0)
    f1 = overall.get("f1_token", {}).get("mean", 0)
    print(f"EM: {em:.2%}  F1: {f1:.2%}")
    print(f"Refusal: {overall.get('refusal_rate', 0):.1%}")
    print(f"Hallucination: {overall.get('hallucination_rate', 0):.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
