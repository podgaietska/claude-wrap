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
    """The routing outcome for one incoming `/v1/messages` request.

    Attributes:
        tier: The chosen tier name ("small" or "large"), or "unchanged"
            if this was a tool-result continuation that wasn't reclassified.
        model: The model ID to send upstream -- either a tier's configured
            model, or the client's originally requested model when
            `is_tool_continuation` is True.
        is_tool_continuation: True if the newest turn was a tool-result
            continuation rather than a fresh human question, meaning no
            classification was performed.
        complexity: The `ComplexityResult` that produced `tier`, or None
            when `is_tool_continuation` is True.
    """

    tier: str
    model: str
    is_tool_continuation: bool
    complexity: ComplexityResult | None


def extract_newest_human_text(messages: list[dict]) -> str | None:
    """Extracts the newest turn's text if it's a fresh human question.

    Claude Code resends the full conversation on every request, so the
    last message in `messages` is the newest turn. A tool-result turn
    (Claude Code returning a tool's output) is a continuation, not a
    fresh question, and should not be reclassified.

    Args:
        messages: The Messages API `messages` array, oldest first.

    Returns:
        The newest turn's text, or None if there is no fresh human text
        to classify (empty history, non-user last turn, or a
        tool-result-only turn).
    """
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
    """Ties complexity classification and the context-window safety check
    together to decide which model a turn should be sent to."""

    def __init__(self, config: Config):
        """Builds a router from the given config.

        Args:
            config: The loaded application config, providing the tier
                definitions and the configured routing strategy.
        """
        self.config = config
        strategy_cls = _STRATEGIES.get(config.routing.strategy, HeuristicClassifier)
        self.classifier = strategy_cls()

    def route(self, messages: list[dict], requested_model: str) -> RouteDecision:
        """Decides which model a request's newest turn should be routed to.

        Escalates a "small" classification to "large" if the conversation
        is too big for the small model's context window. Skipped when the
        small model has no entry in `model_limits`, since its window is unknown.

        Args:
            messages: The Messages API `messages` array from the incoming
                request.
            requested_model: The model ID the client originally requested,
                used as-is for tool-result continuations.

        Returns:
            The `RouteDecision` describing which model to forward to.
        """
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
            limits = self.config.model_limits.get(small_tier.model)
            if limits is not None and not fits_in_window(messages, limits.context_window):
                tier = "large"

        model = self.config.tiers[tier].model
        return RouteDecision(tier=tier, model=model, is_tool_continuation=False, complexity=complexity)
