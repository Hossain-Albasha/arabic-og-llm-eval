"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arog_eval.reports import write_report
from arog_eval.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(prog="arog-eval", description="Arabic / O&G LLM eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run benchmark against a model")
    run.add_argument("--provider", required=True, choices=["anthropic", "openai", "ollama"])
    run.add_argument("--model", required=True, help="Model identifier")
    run.add_argument("--benchmark", default="benchmark/v1", help="Path to benchmark directory")
    run.add_argument("--output", required=True, help="JSONL path for streamed results")
    run.add_argument("--report", help="Optional Markdown report output path")
    run.add_argument("--judge", action="store_true", help="Use Claude as LLM-judge")
    run.add_argument("--semantic", action="store_true", help="Use embedding-based semantic similarity")
    run.add_argument("--label", help="Human label for this run")

    args = parser.parse_args()

    if args.cmd == "run":
        adapter = _build_adapter(args.provider, args.model)
        judge = _build_judge() if args.judge else None
        report = run_benchmark(
            adapter=adapter,
            benchmark_dir=args.benchmark,
            model_label=args.label,
            output_path=args.output,
            judge=judge,
            use_semantic=args.semantic,
        )
        if args.report:
            write_report(report, args.report)
        print(f"\nDone. {report.item_count} items in {report.duration_seconds:.1f}s.")
        agg = report.aggregate.get("overall", {})
        if "total_cost_usd" in agg:
            print(f"Cost: ${agg['total_cost_usd']:.4f}")

    return 0


def _build_adapter(provider: str, model: str):
    if provider == "anthropic":
        from arog_eval.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=model)
    if provider == "openai":
        from arog_eval.adapters.openai import OpenAIAdapter
        return OpenAIAdapter(model=model)
    if provider == "ollama":
        from arog_eval.adapters.ollama import OllamaAdapter
        return OllamaAdapter(model=model)
    raise ValueError(f"Unknown provider: {provider}")


def _build_judge():
    from arog_eval.judges.claude_judge import ClaudeJudge
    return ClaudeJudge()


if __name__ == "__main__":
    sys.exit(main())
