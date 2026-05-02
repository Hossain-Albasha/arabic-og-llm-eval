"""Metric implementations.

Each metric takes a response string and a BenchmarkItem and returns a score.
Metrics should be cheap to compute except `semantic_similarity` and `llm_judge`,
which require additional dependencies.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Iterable

from arog_eval.schemas import BenchmarkItem


# Arabic-specific normalization
_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_ARABIC_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})
_ARABIC_TEH_MARBUTA = str.maketrans({"ة": "ه"})
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text: str) -> str:
    """Conservative normalization for matching purposes.

    Strips diacritics, normalizes alef and teh marbuta variants, lowercases Latin,
    converts Arabic-Indic digits, collapses whitespace, removes punctuation.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_ARABIC_ALEF_VARIANTS)
    text = text.translate(_ARABIC_TEH_MARBUTA)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + "،؛؟"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Whitespace + Arabic-aware tokenization for token-level metrics."""
    return [t for t in normalize(text).split() if t]


def exact_match(response: str, item: BenchmarkItem) -> float:
    """1.0 if normalized response equals any reference. 0.0 otherwise."""
    norm_resp = normalize(response)
    for ref in item.all_references:
        if normalize(ref) == norm_resp:
            return 1.0
    return 0.0


def f1_token(response: str, item: BenchmarkItem) -> float:
    """Best token-overlap F1 against any reference."""
    resp_tokens = set(tokenize(response))
    if not resp_tokens:
        return 0.0
    best = 0.0
    for ref in item.all_references:
        ref_tokens = set(tokenize(ref))
        if not ref_tokens:
            continue
        common = resp_tokens & ref_tokens
        if not common:
            continue
        precision = len(common) / len(resp_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


_REFUSAL_MARKERS = (
    # English
    "i can't", "i cannot", "i'm not able to", "i am not able to",
    "i don't have", "i do not have",
    "as an ai", "i'm an ai", "i am an ai",
    "unable to provide", "i'm sorry, but", "sorry, i can't",
    # Arabic
    "لا أستطيع", "لا يمكنني", "لست قادرا", "لست قادرة",
    "بصفتي نموذجا", "أنا نموذج", "آسف، لا أستطيع",
)


def detect_refusal(response: str) -> bool:
    """Heuristic detection of unnecessary refusals."""
    lowered = response.lower().strip()
    if not lowered:
        return False
    return any(m in lowered for m in _REFUSAL_MARKERS)


_HALLUCINATION_REFERENCE_MARKERS = (
    "unknown", "n/a", "not specified", "غير معروف", "غير محدد",
)


_UNKNOWN_RESPONSE_MARKERS = (
    "unknown", "غير معروف", "غير محدد",
    "dont know", "do not know", "لا اعرف", "لا أعرف",
    "not specified", "not stated", "not in the passage",
    "passage does not", "passage doesnt",
)


def detect_hallucination(response: str, item: BenchmarkItem) -> bool:
    """Returns True when reference is 'unknown' but the model answered confidently."""
    if not any(normalize(item.reference) == normalize(m) for m in _HALLUCINATION_REFERENCE_MARKERS):
        return False
    # Reference says unknown. Did the model say so too?
    norm_resp = normalize(response)
    if not norm_resp or len(norm_resp) <= 5:
        return False
    return not any(normalize(m) in norm_resp for m in _UNKNOWN_RESPONSE_MARKERS)


def detect_wrong_language(response: str, item: BenchmarkItem) -> bool:
    """Returns True when output is in a different language than the prompt requests."""
    if item.language == "mixed":
        return False
    arabic_chars = sum(1 for c in response if "؀" <= c <= "ۿ")
    total = sum(1 for c in response if c.isalpha())
    if total == 0:
        return False
    arabic_ratio = arabic_chars / total
    if item.language == "ar" and arabic_ratio < 0.3:
        return True
    if item.language == "en" and arabic_ratio > 0.5:
        return True
    return False


# Optional: semantic similarity (requires sentence-transformers)
_embedder = None


def semantic_similarity(response: str, item: BenchmarkItem) -> float:
    """Cosine similarity of multilingual embeddings.

    Requires the `embeddings` extra: `pip install arog-eval[embeddings]`.
    Falls back to f1_token if unavailable.
    """
    global _embedder
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return f1_token(response, item)

    if _embedder is None:
        _embedder = SentenceTransformer("intfloat/multilingual-e5-large")

    response_emb = _embedder.encode([response], normalize_embeddings=True)
    ref_embs = _embedder.encode(item.all_references, normalize_embeddings=True)
    sims = (response_emb @ ref_embs.T).flatten()
    return float(sims.max())


def cost_usd(input_tokens: int, output_tokens: int, pricing: dict[str, float] | None) -> float:
    """Compute dollar cost from token counts and a pricing dict.

    Pricing dict shape: {"input": <usd_per_1m_tokens>, "output": <usd_per_1m_tokens>}.
    """
    if pricing is None:
        return 0.0
    return (input_tokens * pricing.get("input", 0.0) + output_tokens * pricing.get("output", 0.0)) / 1_000_000


def aggregate(scores: Iterable[float]) -> dict[str, float]:
    """Mean / median / p50 / p99 over a sequence of numeric scores."""
    values = sorted(s for s in scores if s is not None)
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0, "n": 0}
    n = len(values)
    p50 = values[n // 2]
    p99 = values[min(int(n * 0.99), n - 1)]
    return {
        "mean": sum(values) / n,
        "p50": p50,
        "p99": p99,
        "n": n,
    }
