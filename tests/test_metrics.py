"""Tests for metric implementations."""

from __future__ import annotations

import pytest

from arog_eval import metrics
from arog_eval.schemas import BenchmarkItem, Category, ItemMetadata, Language


def make_item(
    prompt: str = "What is a wellhead?",
    reference: str = "The wellhead is the surface termination of a wellbore.",
    alternatives: list[str] | None = None,
    category: Category = Category.PROCESS_QA,
    language: Language = Language.EN,
) -> BenchmarkItem:
    return BenchmarkItem(
        id="test-001",
        category=category,
        language=language,
        prompt=prompt,
        reference=reference,
        alternative_references=alternatives or [],
        metadata=ItemMetadata(),
    )


class TestNormalize:
    def test_strips_arabic_diacritics(self):
        assert metrics.normalize("صَمّامْ") == "صمام"

    def test_normalizes_alef_variants(self):
        assert metrics.normalize("أنابيب") == "انابيب"
        assert metrics.normalize("إنتاج") == "انتاج"
        assert metrics.normalize("آبار") == "ابار"

    def test_normalizes_teh_marbuta(self):
        assert metrics.normalize("مضخة") == "مضخه"

    def test_normalizes_arabic_indic_digits(self):
        assert metrics.normalize("١٢٣") == "123"

    def test_collapses_whitespace(self):
        assert metrics.normalize("hello    world") == "hello world"

    def test_lowercases_latin(self):
        assert metrics.normalize("WELLHEAD") == "wellhead"

    def test_strips_punctuation(self):
        assert metrics.normalize("صمام، محبس!") == "صمام محبس"


class TestExactMatch:
    def test_exact_match(self):
        item = make_item(reference="reservoir")
        assert metrics.exact_match("reservoir", item) == 1.0

    def test_normalization_match(self):
        item = make_item(reference="reservoir")
        assert metrics.exact_match("RESERVOIR.", item) == 1.0

    def test_alternatives(self):
        item = make_item(reference="oil reservoir", alternatives=["subsurface reservoir"])
        assert metrics.exact_match("subsurface reservoir", item) == 1.0

    def test_no_match(self):
        item = make_item(reference="reservoir")
        assert metrics.exact_match("pipeline", item) == 0.0


class TestF1Token:
    def test_perfect_overlap(self):
        item = make_item(reference="the quick brown fox")
        assert metrics.f1_token("the quick brown fox", item) == pytest.approx(1.0)

    def test_partial_overlap(self):
        item = make_item(reference="the wellhead is on the surface")
        score = metrics.f1_token("the wellhead is at surface", item)
        assert 0.0 < score < 1.0

    def test_zero_overlap(self):
        item = make_item(reference="reservoir engineering")
        assert metrics.f1_token("xyz abc", item) == 0.0


class TestRefusalDetection:
    def test_english_refusal(self):
        assert metrics.detect_refusal("I'm sorry, but I can't help with that.")
        assert metrics.detect_refusal("As an AI, I cannot provide that information.")

    def test_arabic_refusal(self):
        assert metrics.detect_refusal("لا أستطيع الإجابة على هذا السؤال")

    def test_normal_response(self):
        assert not metrics.detect_refusal("The wellhead is the surface termination.")
        assert not metrics.detect_refusal("صمام التحكم يستخدم لتنظيم التدفق")


class TestHallucinationDetection:
    def test_when_reference_unknown_and_model_answers(self):
        item = make_item(reference="unknown")
        assert metrics.detect_hallucination("The compressive strength was 3500 psi", item)

    def test_when_reference_unknown_and_model_says_unknown(self):
        item = make_item(reference="unknown")
        assert not metrics.detect_hallucination("I don't know based on the passage.", item)

    def test_when_reference_is_normal(self):
        item = make_item(reference="reservoir")
        assert not metrics.detect_hallucination("reservoir", item)


class TestWrongLanguage:
    def test_english_response_to_arabic_prompt(self):
        item = make_item(reference="صمام", language=Language.AR)
        assert metrics.detect_wrong_language("This is a valve.", item)

    def test_arabic_response_to_english_prompt(self):
        item = make_item(reference="valve", language=Language.EN)
        assert metrics.detect_wrong_language("هذا صمام التحكم في التدفق", item)

    def test_correct_language_match(self):
        item = make_item(reference="valve", language=Language.EN)
        assert not metrics.detect_wrong_language("A valve is a flow control device.", item)


class TestCost:
    def test_cost_with_pricing(self):
        # Sonnet pricing: $3/M input, $15/M output
        cost = metrics.cost_usd(1000, 500, {"input": 3.0, "output": 15.0})
        # 1000 * 3 / 1M + 500 * 15 / 1M = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105)

    def test_cost_with_no_pricing(self):
        assert metrics.cost_usd(1000, 500, None) == 0.0


class TestAggregate:
    def test_basic_aggregate(self):
        result = metrics.aggregate([0.5, 0.7, 0.9, 1.0, 0.3])
        assert result["n"] == 5
        assert result["mean"] == pytest.approx(0.68)
        assert 0.5 <= result["p50"] <= 0.9

    def test_empty_aggregate(self):
        result = metrics.aggregate([])
        assert result == {"mean": 0.0, "p50": 0.0, "p99": 0.0, "n": 0}
