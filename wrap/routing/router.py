from __future__ import annotations

from dataclasses import dataclass

from wrap.config import Config
from wrap.routing.base import ComplexityClassifier, ComplexityResult
from wrap.routing.context_guard import fits_in_window
from wrap.routing.heuristic import HeuristicClassifier

_STRATEGIES: dict[str, type[ComplexityClassifier]] = {
    "heuristic": HeuristicClassifier,
}


@dataclass
class RouteDecision:
    tier: str
    model: str
    is_tool_continuation: bool
    complexity: ComplexityResult | None


def extract_newest_human_text(messages: list[dict]) -> str | None:
    """Returns the text of the latest turn if it's fresh human-authored
    text, or None if it's a tool-result continuation (or there's no text
    to classify)."""
    if not messages:
        return None
    latest = messages[-1]
    if latest.get("role") != "user":
        return None

    content = latest.get("content", "")
    if isinstance(content, str):
        return content or None

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                # A turn made only of tool results is a continuation, not a
                # fresh question -- don't reclassify it.
                return None
            if block.get("type") == "text" and block.get("text"):
                text_parts.append(block["text"])
        return " ".join(text_parts) if text_parts else None

    return None


class Router:
    def __init__(self, config: Config):
        self.config = config
        strategy_cls = _STRATEGIES.get(config.routing.strategy, HeuristicClassifier)
        self.classifier = strategy_cls()

    def route(self, messages: list[dict], requested_model: str) -> RouteDecision:
        newest_text = extract_newest_human_text(messages)
        if newest_text is None:
            # Not a fresh question -- pass through whatever model was
            # already in play for this exchange rather than reclassifying.
            return RouteDecision(
                tier="unchanged",
                model=requested_model,
                is_tool_continuation=True,
                complexity=None,
            )

        complexity = self.classifier.classify(newest_text, self.config.routing.complexity_threshold)
        tier = complexity.tier

        small_tier = self.config.tiers.get("small")
        if tier == "small" and small_tier is not None:
            if not fits_in_window(messages, small_tier.context_window):
                tier = "large"

        model = self.config.tiers[tier].model
        return RouteDecision(tier=tier, model=model, is_tool_continuation=False, complexity=complexity)
