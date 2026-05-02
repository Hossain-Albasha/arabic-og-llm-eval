"""Tests for failure-mode classifiers."""

from __future__ import annotations

from arog_eval.failure_modes import (
    classify_failure_modes,
    detect_code_switch_failure,
    detect_hallucinated_proper_noun,
    detect_under_diacritization,
    detect_wrong_transliteration,
    has_diacritics,
    is_arabic_text,
)
from arog_eval.schemas import BenchmarkItem, Category, FailureMode, Language


def make_item(
    prompt: str,
    reference: str,
    alternatives: list[str] | None = None,
    category: Category = Category.TERMINOLOGY,
    language: Language = Language.AR,
) -> BenchmarkItem:
    return BenchmarkItem(
        id="t-1",
        category=category,
        language=language,
        prompt=prompt,
        reference=reference,
        alternative_references=alternatives or [],
    )


class TestArabicHelpers:
    def test_has_diacritics_true(self):
        assert has_diacritics("صَمّامْ")

    def test_has_diacritics_false(self):
        assert not has_diacritics("صمام")

    def test_is_arabic_text(self):
        assert is_arabic_text("هذا نص عربي بالكامل")

    def test_is_not_arabic_text(self):
        assert not is_arabic_text("This is fully English text")


class TestWrongTransliteration:
    def test_picks_synonym_in_known_group(self):
        item = make_item(prompt="ما الترجمة العربية لـ valve؟", reference="صمام")
        # The model picked محبس - both are valid common renderings
        assert detect_wrong_transliteration("محبس", item)

    def test_does_not_trigger_on_correct(self):
        item = make_item(prompt="ما الترجمة العربية لـ valve؟", reference="صمام")
        assert not detect_wrong_transliteration("صمام", item)

    def test_only_for_terminology_category(self):
        item = make_item(
            prompt="...",
            reference="صمام",
            category=Category.PROCESS_QA,
        )
        assert not detect_wrong_transliteration("محبس", item)


class TestCodeSwitchFailure:
    def test_translates_when_should_preserve(self):
        item = make_item(
            prompt="عندنا flow كذا 5000 bpd",
            reference="check the wellhead pressure",
            category=Category.CODE_SWITCHED,
            language=Language.MIXED,
        )
        assert detect_code_switch_failure("افحص ضغط رأس البئر", item)

    def test_does_not_trigger_on_preserved_english(self):
        item = make_item(
            prompt="check flow",
            reference="check the wellhead pressure",
            category=Category.CODE_SWITCHED,
            language=Language.MIXED,
        )
        assert not detect_code_switch_failure("افحص الـ wellhead pressure", item)


class TestHallucinatedProperNoun:
    def test_invents_standard_not_in_item(self):
        item = make_item(
            prompt="What standard governs pressure relief?",
            reference="API 520",
        )
        assert detect_hallucinated_proper_noun("This follows ASME 31.3 and ISO 9999.", item)

    def test_does_not_trigger_when_standard_in_reference(self):
        item = make_item(
            prompt="What standard governs pressure relief?",
            reference="API 520",
        )
        assert not detect_hallucinated_proper_noun("Per API 520.", item)


class TestUnderDiacritization:
    def test_strips_when_should_preserve(self):
        item = make_item(prompt="...", reference="صَمّام")
        assert detect_under_diacritization("صمام", item)

    def test_does_not_trigger_when_neither_has_diacritics(self):
        item = make_item(prompt="...", reference="صمام")
        assert not detect_under_diacritization("صمام", item)


class TestClassifyAggregate:
    def test_combines_multiple_modes(self):
        item = make_item(
            prompt="What standard?",
            reference="API 520",
            language=Language.EN,
        )
        # Response: in Arabic, refused, mentions a fake standard
        modes = classify_failure_modes(
            "آسف، لا أستطيع. لكن انظر ASME 9999.",
            item,
            refused=True,
            halluc=False,
        )
        assert FailureMode.REFUSED_VALID_REQUEST in modes
        assert FailureMode.WRONG_LANGUAGE_OUTPUT in modes
        assert FailureMode.HALLUCINATED_PROPER_NOUN in modes
