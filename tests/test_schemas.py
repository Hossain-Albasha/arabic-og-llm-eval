"""Tests for Pydantic schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from arog_eval.schemas import (
    BenchmarkItem,
    Category,
    Difficulty,
    Language,
    ModelResponse,
)


VALID_ITEM_JSON = {
    "id": "term-001",
    "category": "terminology",
    "language": "ar",
    "prompt": "ما هو wellhead؟",
    "reference": "رأس البئر",
    "alternative_references": ["رأس الحفرة"],
    "metadata": {
        "domain": "production",
        "difficulty": "easy",
        "tags": ["wellhead"],
    },
}


def test_benchmark_item_parses_valid():
    item = BenchmarkItem.model_validate(VALID_ITEM_JSON)
    assert item.id == "term-001"
    assert item.category == Category.TERMINOLOGY
    assert item.language == Language.AR
    assert item.metadata.difficulty == Difficulty.EASY


def test_benchmark_item_minimal_metadata():
    minimal = {
        "id": "test-001",
        "category": "terminology",
        "language": "en",
        "prompt": "What is a wellhead?",
        "reference": "Surface termination of a well.",
    }
    item = BenchmarkItem.model_validate(minimal)
    assert item.metadata.difficulty == Difficulty.MEDIUM
    assert item.alternative_references == []


def test_benchmark_item_rejects_invalid_category():
    bad = dict(VALID_ITEM_JSON, category="not-a-category")
    with pytest.raises(ValidationError):
        BenchmarkItem.model_validate(bad)


def test_benchmark_item_all_references():
    item = BenchmarkItem.model_validate(VALID_ITEM_JSON)
    refs = item.all_references
    assert "رأس البئر" in refs
    assert "رأس الحفرة" in refs
    assert len(refs) == 2


def test_model_response_json_round_trip():
    response = ModelResponse(
        text="رأس البئر",
        input_tokens=10,
        output_tokens=5,
        latency_ms=234.5,
        model_name="test-model",
    )
    serialized = response.model_dump_json()
    restored = ModelResponse.model_validate_json(serialized)
    assert restored == response


def test_jsonl_line_parses():
    line = json.dumps(VALID_ITEM_JSON, ensure_ascii=False)
    item = BenchmarkItem.model_validate_json(line)
    assert item.reference == "رأس البئر"
