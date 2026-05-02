"""Claude as an LLM-judge for nuanced scoring of model responses."""

from __future__ import annotations

import os
import re

from arog_eval.schemas import BenchmarkItem


JUDGE_PROMPT = """You are evaluating an AI model's answer to a bilingual Arabic/English Oil & Gas technical question.

Score the response from 0 to 10 based on:
- Factual accuracy against the reference answer (most important)
- Use of correct domain terminology
- Appropriate language match (the prompt language and the response language should align)
- No hallucinated equipment tags, standards, or vendor names
- No unnecessary refusal of a valid technical question

Be strict. A response that is partially correct but uses the wrong technical term should score 4-6, not 8.
A response that is fluent but factually wrong should score 1-3.
Only give 9 or 10 to responses that are technically precise AND well-phrased.

QUESTION (language={language}):
{prompt}

REFERENCE ANSWER:
{reference}

ALTERNATIVE ACCEPTED ANSWERS:
{alternatives}

MODEL RESPONSE:
{response}

Reply with exactly two lines:
Line 1: The score as a single integer 0-10.
Line 2: A one-sentence explanation in English."""


class ClaudeJudge:
    """Wraps Claude to score responses against references."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise ImportError("Install with: pip install arog-eval[anthropic]") from e

        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def score(self, response: str, item: BenchmarkItem) -> tuple[float, str]:
        """Returns (score in [0, 10], explanation)."""
        alternatives = (
            "\n".join(f"- {a}" for a in item.alternative_references)
            if item.alternative_references
            else "(none)"
        )
        prompt = JUDGE_PROMPT.format(
            language=item.language.value,
            prompt=item.prompt,
            reference=item.reference,
            alternatives=alternatives,
            response=response,
        )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in message.content).strip()

        return _parse_judge_output(text)


def _parse_judge_output(text: str) -> tuple[float, str]:
    """Robust parser for the two-line judge response.

    Tolerates variations like '7/10', 'Score: 7', or extra preamble.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0, "judge returned empty output"

    score: float = 0.0
    explanation = ""
    for line in lines[:2]:
        match = re.search(r"\b(10|[0-9])(?:\.[0-9]+)?\b", line)
        if match and score == 0.0:
            try:
                score = float(match.group(1))
                continue
            except ValueError:
                pass
        if not explanation:
            explanation = line

    if not explanation and len(lines) > 1:
        explanation = lines[1]

    return min(max(score, 0.0), 10.0), explanation
