"""Failure-mode classification.

Heuristic rules for tagging where a wrong answer went wrong. Imperfect by design,
but more useful than a single pass/fail score for tracking model behavior across versions.
"""

from __future__ import annotations

import re

from arog_eval.metrics import normalize, tokenize
from arog_eval.schemas import BenchmarkItem, FailureMode, Language


# Common Arabic transliteration alternates for technical terms
_TRANSLITERATION_GROUPS: list[set[str]] = [
    {"valve", "صمام", "محبس", "بلف"},
    {"pump", "مضخة", "بومبه"},
    {"pipe", "أنبوب", "ماسورة", "بايب"},
    {"flange", "فلنج", "وصلة", "فلانش"},
    {"gasket", "جوان", "حلقة", "جسكت"},
    {"compressor", "ضاغط", "كومبريسر"},
    {"manifold", "مشعب", "مانيفولد"},
    {"choke", "خنّاق", "تشوك"},
    {"separator", "فاصل", "سيبراتور"},
    {"reservoir", "مكمن", "خزان", "ريزرفوار"},
]


_DIACRITIC_RANGE = re.compile(r"[ً-ْٰ]")


def has_diacritics(text: str) -> bool:
    return bool(_DIACRITIC_RANGE.search(text))


def is_arabic_text(text: str) -> bool:
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    total = sum(1 for c in text if c.isalpha())
    return total > 0 and arabic_chars / total > 0.5


def detect_wrong_transliteration(response: str, item: BenchmarkItem) -> bool:
    """Did the model pick a different valid transliteration than expected?

    Specifically: response and reference are both members of a known transliteration group
    but are different members.
    """
    if item.category.value != "terminology":
        return False
    norm_resp = normalize(response).strip()
    norm_ref = normalize(item.reference).strip()
    for group in _TRANSLITERATION_GROUPS:
        normed_group = {normalize(g) for g in group}
        if norm_resp in normed_group and norm_ref in normed_group and norm_resp != norm_ref:
            return True
    return False


def detect_code_switch_failure(response: str, item: BenchmarkItem) -> bool:
    """For code-switched prompts, did the model translate technical English terms instead of preserving them?"""
    if item.category.value != "code_switched":
        return False
    # Heuristic: if reference contains English tokens but response is fully Arabic, fail.
    ref_has_english = bool(re.search(r"[a-zA-Z]{3,}", item.reference))
    resp_has_english = bool(re.search(r"[a-zA-Z]{3,}", response))
    return ref_has_english and not resp_has_english


_FAKE_STANDARDS = re.compile(
    r"\b(?:API|ISO|ASME|ASTM|IEC|IEEE|ANSI)[- ]?\d{3,5}[A-Z]?\b",
    re.IGNORECASE,
)


def detect_hallucinated_proper_noun(response: str, item: BenchmarkItem) -> bool:
    """Did the response cite a standard or tag not present in the item?"""
    response_standards = set(s.upper() for s in _FAKE_STANDARDS.findall(response))
    if not response_standards:
        return False
    expected_text = (item.prompt + " " + item.reference + " " + " ".join(item.alternative_references)).upper()
    expected_standards = set(s.upper() for s in _FAKE_STANDARDS.findall(expected_text))
    return bool(response_standards - expected_standards)


def detect_under_diacritization(response: str, item: BenchmarkItem) -> bool:
    """Did the model strip diacritics where the reference preserved them?"""
    if not is_arabic_text(item.reference):
        return False
    return has_diacritics(item.reference) and not has_diacritics(response)


def detect_over_formality(response: str, item: BenchmarkItem) -> bool:
    """Heuristic: response is much longer than expected for a short-answer item."""
    if item.category.value != "terminology":
        return False
    ref_tokens = len(tokenize(item.reference))
    resp_tokens = len(tokenize(response))
    if ref_tokens == 0:
        return False
    return resp_tokens > ref_tokens * 4 + 5


def detect_wrong_language(response: str, item: BenchmarkItem) -> bool:
    if item.language == Language.MIXED:
        return False
    response_is_arabic = is_arabic_text(response)
    if item.language == Language.AR and not response_is_arabic:
        return True
    if item.language == Language.EN and response_is_arabic:
        return True
    return False


def classify_failure_modes(
    response: str,
    item: BenchmarkItem,
    refused: bool,
    halluc: bool,
) -> list[FailureMode]:
    """Run all classifiers and return the list of triggered failure modes."""
    modes: list[FailureMode] = []

    if refused:
        modes.append(FailureMode.REFUSED_VALID_REQUEST)
    if halluc:
        modes.append(FailureMode.HALLUCINATED_PROPER_NOUN)
    if detect_wrong_transliteration(response, item):
        modes.append(FailureMode.WRONG_TRANSLITERATION)
    if detect_code_switch_failure(response, item):
        modes.append(FailureMode.CODE_SWITCH_FAILURE)
    if detect_hallucinated_proper_noun(response, item) and FailureMode.HALLUCINATED_PROPER_NOUN not in modes:
        modes.append(FailureMode.HALLUCINATED_PROPER_NOUN)
    if detect_under_diacritization(response, item):
        modes.append(FailureMode.UNDER_DIACRITIZATION)
    if detect_over_formality(response, item):
        modes.append(FailureMode.OVER_FORMALITY)
    if detect_wrong_language(response, item):
        modes.append(FailureMode.WRONG_LANGUAGE_OUTPUT)

    return modes
