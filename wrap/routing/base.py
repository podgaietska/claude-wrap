from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ComplexityResult:
    score: float  # 0.0 (trivial) .. 1.0 (hard)
    tier: str
    reasoning: str


class ComplexityClassifier(ABC):
    @abstractmethod
    def classify(self, text: str, threshold: float) -> ComplexityResult: ...
