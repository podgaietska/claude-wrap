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
class Turn:
    """The human question a request belongs to.

    Attributes:
        text: The text the user typed.
        index: Position of that message in the request's `messages` array.
        is_continuation: True if the request goes past the question, e.g.
            Claude Code sending back tool results mid-turn.
    """

    text: str
    index: int
    is_continuation: bool


@dataclass
class RouteDecision:
    """The routing outcome for one incoming `/v1/messages` request.

    Attributes:
        model: The model ID to send upstream.
        routed: False if no human question was found, in which case `model`
            is the client's originally requested model.
        tier: The chosen tier name ("small" or "large"), or None if not routed.
        reason: Short explanation of the decision, for logging.
        complexity: The `ComplexityResult` behind `tier`, or None if not routed.
        turn: The question the decision was based on, or None if not routed.
    """

    model: str
    routed: bool
    tier: str | None
    reason: str
    complexity: ComplexityResult | None
    turn: Turn | None


_REMINDER_PREFIX = "<system-reminder>"


def human_text(message: dict) -> str | None:
    """Returns the text the user typed in a message, else None.

    Tool results are also sent with `role: "user"`, so a user message
    containing any `tool_result` block doesn't count as typed text. Claude
    Code also injects context (e.g. CLAUDE.md) into the user's message as
    `<system-reminder>` text blocks; those are skipped so they don't get
    classified as part of the question.

    Args:
        message: One entry from the Messages API `messages` array.

    Returns:
        The typed text, or None for non-user, tool-result, reminder-only,
        or empty messages.
    """
    if message.get("role") != "user":
        return None
    content = message.get("content", "")
    if isinstance(content, str):
        return None if not content or _is_reminder(content) else content
    if not isinstance(content, list):
        return None

    text_parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            return None
        text = block.get("text")
        if block.get("type") == "text" and text and not _is_reminder(text):
            text_parts.append(text)
    return " ".join(text_parts) if text_parts else None


def _is_reminder(text: str) -> bool:
    """Checks whether a text block is context Claude Code injected, not typed by the user.

    Args:
        text: A text block's content.

    Returns:
        True if the text is a `<system-reminder>` block.
    """
    return text.lstrip().startswith(_REMINDER_PREFIX)


def find_turn(messages: list[dict]) -> Turn | None:
    """Finds the human question the current request belongs to.

    Claude Code resends the whole conversation on every request and may
    append things after the question: `role: "system"` reminders, and
    mid-turn the assistant's tool calls and their results. Walking back to
    the latest typed message lets every request in a turn be routed the
    same way as the question that started it.

    Args:
        messages: The Messages API `messages` array, oldest first.

    Returns:
        The latest typed question, or None if the request has none.
    """
    for index in range(len(messages) - 1, -1, -1):
        text = human_text(messages[index])
        if text is not None:
            is_continuation = any(m.get("role") != "system" for m in messages[index + 1:])
            return Turn(text=text, index=index, is_continuation=is_continuation)
    return None


class Router:
    """Decides which model each request goes to, based on its turn's question."""

    def __init__(self, config: Config):
        """Builds a router from the given config.

        Args:
            config: The loaded application config, providing the tier
                definitions, model capabilities, and routing strategy.
        """
        self.config = config
        strategy_cls = _STRATEGIES.get(config.routing.strategy, HeuristicClassifier)
        self.classifier = strategy_cls()

    def route(self, messages: list[dict], requested_model: str) -> RouteDecision:
        """Decides which model a request should be sent to.

        The classifier is deterministic, so every request in a turn gets the
        same tier as the question that started it. A "small" decision is
        escalated to "large" if the conversation doesn't fit the small
        model's context window (skipped if that window is unknown).

        Args:
            messages: The Messages API `messages` array from the incoming request.
            requested_model: The model ID the client asked for, used when
                the request has no human question to route by.

        Returns:
            The `RouteDecision` describing which model to forward to.
        """
        turn = find_turn(messages)
        if turn is None:
            return RouteDecision(
                model=requested_model, routed=False, tier=None,
                reason="no human message", complexity=None, turn=None,
            )

        complexity = self.classifier.classify(turn.text, self.config.routing.complexity_threshold)
        tier = complexity.tier
        reason = f"score {complexity.score:.2f}: {complexity.reasoning}"

        small_tier = self.config.tiers.get("small")
        if tier == "small" and small_tier is not None:
            caps = self.config.models.get(small_tier.model)
            if caps is not None and caps.context_window is not None:
                if not fits_in_window(messages, caps.context_window):
                    tier = "large"
                    reason += "; escalated, conversation too big for small model"

        return RouteDecision(
            model=self.config.tiers[tier].model, routed=True, tier=tier,
            reason=reason, complexity=complexity, turn=turn,
        )
