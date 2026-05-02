"""Eval runner: orchestrates loading the benchmark, querying the model, scoring."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arog_eval import metrics
from arog_eval.adapters.base import ModelAdapter
from arog_eval.failure_modes import classify_failure_modes
from arog_eval.schemas import (
    BenchmarkItem,
    Category,
    EvalResult,
    MetricScores,
    ModelResponse,
    RunReport,
)


def load_benchmark(benchmark_dir: Path | str) -> list[BenchmarkItem]:
    """Load all .jsonl files in a benchmark directory into BenchmarkItem objects."""
    benchmark_dir = Path(benchmark_dir)
    items: list[BenchmarkItem] = []
    for jsonl_path in sorted(benchmark_dir.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    items.append(BenchmarkItem.model_validate_json(line))
                except Exception as e:
                    raise ValueError(f"Bad item at {jsonl_path}:{line_num}: {e}") from e
    return items


def run_item(
    adapter: ModelAdapter,
    item: BenchmarkItem,
    judge: Any | None = None,
    use_semantic: bool = False,
) -> EvalResult:
    """Query the model on one item and compute all metrics."""
    response = adapter.query(item.prompt)
    return score_response(response, item, adapter.pricing, judge, use_semantic)


def score_response(
    response: ModelResponse,
    item: BenchmarkItem,
    pricing: dict[str, float] | None = None,
    judge: Any | None = None,
    use_semantic: bool = False,
) -> EvalResult:
    """Compute metrics for an already-obtained response."""
    em = metrics.exact_match(response.text, item)
    f1 = metrics.f1_token(response.text, item)
    refused = metrics.detect_refusal(response.text)
    halluc = metrics.detect_hallucination(response.text, item)

    sim = metrics.semantic_similarity(response.text, item) if use_semantic else None

    judge_score: float | None = None
    if judge is not None:
        try:
            judge_score, _ = judge.score(response.text, item)
        except Exception:
            judge_score = None

    failure_modes = classify_failure_modes(response.text, item, refused=refused, halluc=halluc)

    scores = MetricScores(
        exact_match=em,
        f1_token=f1,
        semantic_similarity=sim,
        llm_judge=judge_score,
        refusal=refused,
        hallucination=halluc,
        failure_modes=failure_modes,
    )

    cost = metrics.cost_usd(response.input_tokens, response.output_tokens, pricing)

    return EvalResult(
        item_id=item.id,
        category=item.category,
        language=item.language,
        prompt=item.prompt,
        reference=item.reference,
        response=response,
        scores=scores,
        cost_usd=cost,
    )


def run_benchmark(
    adapter: ModelAdapter,
    benchmark_dir: Path | str = "benchmark/v1",
    benchmark_version: str = "v1.0",
    model_label: str | None = None,
    output_path: Path | str | None = None,
    judge: Any | None = None,
    use_semantic: bool = False,
    progress: bool = True,
) -> RunReport:
    """Run a complete benchmark against one model adapter.

    Streams per-item results to ``output_path`` as it goes (resume-friendly),
    and returns the full RunReport at the end.
    """
    items = load_benchmark(benchmark_dir)
    if not items:
        raise ValueError(f"No benchmark items found in {benchmark_dir}")

    report = RunReport(
        model_name=adapter.name,
        model_label=model_label or adapter.name,
        benchmark_version=benchmark_version,
        run_started_at=datetime.now(timezone.utc),
        item_count=len(items),
        results=[],
    )

    output_path = Path(output_path) if output_path else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(items, desc=f"Eval {adapter.name}") if progress else items
    for item in iterator:
        try:
            result = run_item(adapter, item, judge=judge, use_semantic=use_semantic)
        except Exception as e:
            # Persist a synthetic failed result so the run can continue
            failed_response = ModelResponse(
                text=f"[ADAPTER ERROR: {e}]",
                latency_ms=0.0,
                model_name=adapter.name,
            )
            result = score_response(failed_response, item, pricing=adapter.pricing)
        report.results.append(result)
        if output_path:
            _stream_append(output_path, result)

    report.run_completed_at = datetime.now(timezone.utc)
    report.aggregate = _aggregate(report)

    if output_path:
        output_path.with_suffix(".report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    return report


def _stream_append(path: Path, result: EvalResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")


def _aggregate(report: RunReport) -> dict[str, Any]:
    """Compute per-category and overall aggregates."""
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in report.results:
        by_cat[r.category.value].append(r)

    out: dict[str, Any] = {"by_category": {}, "overall": {}}

    for cat, results in by_cat.items():
        out["by_category"][cat] = _agg_set(results)

    out["overall"] = _agg_set(report.results)
    out["overall"]["total_cost_usd"] = sum(r.cost_usd for r in report.results)
    out["overall"]["total_latency_ms"] = sum(r.response.latency_ms for r in report.results)
    out["overall"]["item_count"] = len(report.results)

    fm_counts: dict[str, int] = defaultdict(int)
    for r in report.results:
        for fm in r.scores.failure_modes:
            fm_counts[fm.value] += 1
    out["overall"]["failure_mode_counts"] = dict(fm_counts)

    return out


def _agg_set(results: list[EvalResult]) -> dict[str, Any]:
    em = [r.scores.exact_match for r in results]
    f1 = [r.scores.f1_token for r in results]
    judge = [r.scores.llm_judge for r in results if r.scores.llm_judge is not None]
    sem = [r.scores.semantic_similarity for r in results if r.scores.semantic_similarity is not None]
    lat = [r.response.latency_ms for r in results]
    refusal_rate = sum(1 for r in results if r.scores.refusal) / max(len(results), 1)
    hallucination_rate = sum(1 for r in results if r.scores.hallucination) / max(len(results), 1)

    return {
        "exact_match": metrics.aggregate(em),
        "f1_token": metrics.aggregate(f1),
        "semantic_similarity": metrics.aggregate(sem) if sem else None,
        "llm_judge": metrics.aggregate(judge) if judge else None,
        "latency_ms": metrics.aggregate(lat),
        "refusal_rate": refusal_rate,
        "hallucination_rate": hallucination_rate,
        "n": len(results),
    }
