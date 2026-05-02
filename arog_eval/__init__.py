"""Arabic / Oil & Gas LLM evaluation benchmark and harness."""

from arog_eval.schemas import BenchmarkItem, ModelResponse, EvalResult, RunReport
from arog_eval.runner import run_benchmark, run_item

__version__ = "0.1.0"
__all__ = [
    "BenchmarkItem",
    "ModelResponse",
    "EvalResult",
    "RunReport",
    "run_benchmark",
    "run_item",
]
