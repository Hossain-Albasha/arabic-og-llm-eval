"""Markdown report generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

from arog_eval.schemas import EvalResult, RunReport


def render_markdown_report(report: RunReport) -> str:
    """Generate a Markdown summary of a run."""
    lines: list[str] = []
    lines.append(f"# {report.model_label} · arog-eval {report.benchmark_version}")
    lines.append("")
    lines.append(
        f"Run: `{report.run_started_at.isoformat()}` "
        f"({report.duration_seconds:.1f}s)"
    )
    lines.append(f"Items: **{report.item_count}**")
    overall = report.aggregate.get("overall", {})
    if "total_cost_usd" in overall:
        lines.append(f"Total cost: **${overall['total_cost_usd']:.4f}**")
    if "total_latency_ms" in overall:
        avg_lat = overall["total_latency_ms"] / max(report.item_count, 1)
        lines.append(f"Avg latency: **{avg_lat:.0f} ms**")
    lines.append("")

    lines.append("## Overall scores")
    lines.append("")
    overall_rows = _score_rows(overall)
    lines.append(tabulate(overall_rows, headers=["metric", "mean", "p50", "p99", "n"], tablefmt="github"))
    lines.append("")
    lines.append(f"- Refusal rate: **{overall.get('refusal_rate', 0):.1%}**")
    lines.append(f"- Hallucination rate: **{overall.get('hallucination_rate', 0):.1%}**")
    lines.append("")

    lines.append("## Per-category")
    lines.append("")
    cat_rows: list[list[str]] = []
    for cat, scores in report.aggregate.get("by_category", {}).items():
        em = scores.get("exact_match", {}).get("mean", 0)
        f1 = scores.get("f1_token", {}).get("mean", 0)
        judge = scores.get("llm_judge", {}).get("mean") if scores.get("llm_judge") else None
        n = scores.get("n", 0)
        cat_rows.append([
            cat,
            f"{em:.2f}",
            f"{f1:.2f}",
            f"{judge:.1f}" if judge is not None else "n/a",
            str(n),
        ])
    lines.append(tabulate(cat_rows, headers=["category", "EM", "F1", "judge", "n"], tablefmt="github"))
    lines.append("")

    lines.append("## Failure modes")
    lines.append("")
    fm_counts = overall.get("failure_mode_counts", {})
    if fm_counts:
        fm_rows = sorted(fm_counts.items(), key=lambda x: -x[1])
        lines.append(tabulate(fm_rows, headers=["failure mode", "count"], tablefmt="github"))
    else:
        lines.append("_No failure modes triggered._")
    lines.append("")

    lines.append("## Worst items by judge score")
    lines.append("")
    worst = _worst_items(report.results, n=5)
    for r in worst:
        score = r.scores.llm_judge if r.scores.llm_judge is not None else r.scores.f1_token
        lines.append(f"### {r.item_id} ({r.category.value}, {r.language.value}) · score {score:.1f}")
        lines.append(f"**Prompt:** {r.prompt[:300]}{'...' if len(r.prompt) > 300 else ''}")
        lines.append("")
        lines.append(f"**Reference:** {r.reference[:200]}{'...' if len(r.reference) > 200 else ''}")
        lines.append("")
        lines.append(f"**Response:** {r.response.text[:300]}{'...' if len(r.response.text) > 300 else ''}")
        lines.append("")
        if r.scores.failure_modes:
            modes = ", ".join(fm.value for fm in r.scores.failure_modes)
            lines.append(f"**Failure modes:** {modes}")
            lines.append("")

    return "\n".join(lines)


def _score_rows(overall: dict) -> list[list[str]]:
    rows = []
    for metric in ("exact_match", "f1_token", "semantic_similarity", "llm_judge", "latency_ms"):
        agg = overall.get(metric)
        if not agg:
            continue
        rows.append([
            metric,
            f"{agg.get('mean', 0):.3f}",
            f"{agg.get('p50', 0):.3f}",
            f"{agg.get('p99', 0):.3f}",
            str(agg.get('n', 0)),
        ])
    return rows


def _worst_items(results: list[EvalResult], n: int = 5) -> list[EvalResult]:
    def score_key(r: EvalResult) -> float:
        if r.scores.llm_judge is not None:
            return r.scores.llm_judge
        return r.scores.f1_token * 10

    return sorted(results, key=score_key)[:n]


def write_report(report: RunReport, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(report), encoding="utf-8")
    return output_path
