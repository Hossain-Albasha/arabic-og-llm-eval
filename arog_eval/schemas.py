"""Pydantic schemas for benchmark items, model responses, and run reports."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Category(str, Enum):
    TERMINOLOGY = "terminology"
    SAFETY = "safety"
    PROCESS_QA = "process_qa"
    CODE_SWITCHED = "code_switched"
    DOC_GROUNDING = "doc_grounding"


class Language(str, Enum):
    AR = "ar"
    EN = "en"
    MIXED = "mixed"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ItemMetadata(BaseModel):
    domain: str = "general"
    difficulty: Difficulty = Difficulty.MEDIUM
    tags: list[str] = Field(default_factory=list)


class BenchmarkItem(BaseModel):
    """A single evaluable item in the benchmark."""

    id: str
    category: Category
    language: Language
    prompt: str
    reference: str
    alternative_references: list[str] = Field(default_factory=list)
    metadata: ItemMetadata = Field(default_factory=ItemMetadata)

    @property
    def all_references(self) -> list[str]:
        return [self.reference, *self.alternative_references]


class ModelResponse(BaseModel):
    """Raw output from a model adapter for one item."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = "unknown"
    raw: dict[str, Any] | None = None


class FailureMode(str, Enum):
    NONE = "none"
    WRONG_TRANSLITERATION = "wrong_transliteration"
    WRONG_TERM = "wrong_term"
    OVER_FORMALITY = "over_formality"
    UNDER_DIACRITIZATION = "under_diacritization"
    CODE_SWITCH_FAILURE = "code_switch_failure"
    HALLUCINATED_PROPER_NOUN = "hallucinated_proper_noun"
    REFUSED_VALID_REQUEST = "refused_valid_request"
    WRONG_LANGUAGE_OUTPUT = "wrong_language_output"
    OTHER = "other"


class MetricScores(BaseModel):
    exact_match: float = 0.0
    f1_token: float = 0.0
    semantic_similarity: float | None = None
    llm_judge: float | None = None
    refusal: bool = False
    hallucination: bool = False
    failure_modes: list[FailureMode] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Per-item evaluation result combining the response and computed metrics."""

    item_id: str
    category: Category
    language: Language
    prompt: str
    reference: str
    response: ModelResponse
    scores: MetricScores
    cost_usd: float = 0.0


class RunReport(BaseModel):
    """Aggregate report for a full benchmark run against a single model."""

    model_name: str
    model_label: str
    benchmark_version: str
    run_started_at: datetime
    run_completed_at: datetime | None = None
    item_count: int
    results: list[EvalResult]
    aggregate: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.run_completed_at is None:
            return 0.0
        return (self.run_completed_at - self.run_started_at).total_seconds()
