from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ComplexityResult:
    """The outcome of classifying a single turn's complexity.

    Attributes:
        score: Complexity score from 0.0 (trivial) to 1.0 (hard).
        tier: The tier name ("small" or "large") this score maps to.
        reasoning: Short human-readable explanation of which signals
            contributed to the score, for logging/debugging.
    """

    score: float
    tier: str
    reasoning: str


class ComplexityClassifier(ABC):
    """Interface for scoring how complex a single turn's text is."""

    @abstractmethod
    def classify(self, text: str, threshold: float) -> ComplexityResult:
        """Scores a turn's text and decides which tier it should route to.

        Args:
            text: The newest human-authored turn's text content.
            threshold: Score at or above which the turn is classified as
                "large" rather than "small".

        Returns:
            A `ComplexityResult` with the score, chosen tier, and reasoning.
        """
        ...
