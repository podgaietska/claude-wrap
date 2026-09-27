from __future__ import annotations

import re

from wrap.routing.base import ComplexityClassifier, ComplexityResult

_CODE_MARKERS = re.compile(r"```|\bdef \b|\bclass \b|\bSELECT\b|\bimport \b|\bfunction \b", re.IGNORECASE)
_MATH_MARKERS = re.compile(
    r"\bsolve\b|\bintegral\b|\bderivative\b|\bequation\b|\bproof\b|\bcalculate\b",
    re.IGNORECASE,
)
_HARD_LANGUAGE = re.compile(
    r"\bexplain\b|\banalyze\b|\bcompare\b|\bdesign\b|\brefactor\b|\barchitecture\b|"
    r"\bdebug\b|\bwhy\b|\bstep by step\b|\btrade[- ]?off\b",
    re.IGNORECASE,
)
_EASY_LANGUAGE = re.compile(r"^\s*(what is|who is|define|list|when is)\b", re.IGNORECASE)

# Length bounds (characters) for score normalization.
_LEN_LOW = 40
_LEN_HIGH = 600


class HeuristicClassifier(ComplexityClassifier):
    """Zero-cost, deterministic complexity scorer. No extra model call."""

    def classify(self, text: str, threshold: float) -> ComplexityResult:
        text = text.strip()
        if not text:
            return ComplexityResult(score=0.0, tier="small", reasoning="empty turn")

        reasons: list[str] = []
        score = 0.0

        length_frac = _clamp((len(text) - _LEN_LOW) / (_LEN_HIGH - _LEN_LOW), 0.0, 1.0)
        score += 0.3 * length_frac
        if length_frac > 0.5:
            reasons.append("long prompt")

        if _CODE_MARKERS.search(text):
            score += 0.25
            reasons.append("code indicators")

        if _MATH_MARKERS.search(text):
            score += 0.2
            reasons.append("math indicators")

        hard_hits = len(_HARD_LANGUAGE.findall(text))
        if hard_hits:
            score += min(0.2 * hard_hits, 0.5)
            reasons.append(f"complexity language x{hard_hits}")

        if _EASY_LANGUAGE.search(text) and len(text) < 200:
            score -= 0.3
            reasons.append("simple lookup phrasing")

        multi_part = text.count("?") > 1 or bool(re.search(r"\b\d\.\s", text))
        if multi_part:
            score += 0.15
            reasons.append("multi-part request")

        score = _clamp(score, 0.0, 1.0)
        tier = "large" if score >= threshold else "small"
        reasoning = ", ".join(reasons) if reasons else "no strong signals"
        return ComplexityResult(score=score, tier=tier, reasoning=reasoning)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
