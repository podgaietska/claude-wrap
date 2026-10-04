"""Short, content-free descriptions of requests and routing decisions for the log."""
from __future__ import annotations

from rich.markup import escape

from wrap.routing.router import RouteDecision

_TAIL_LENGTH = 3


def request_kind(body: dict) -> str:
    """Labels a request as Claude Code's main conversation request or a side request.

    A best-effort heuristic: the main request carries the tool definitions,
    while side requests (e.g. generating a session title) don't.

    Args:
        body: The parsed request body.

    Returns:
        "main" or "side".
    """
    return "main" if body.get("tools") else "side"


def describe_message(message: dict) -> str:
    """Describes one `messages` entry by role and content block types, e.g. "user:tool_result".

    Args:
        message: One entry from the Messages API `messages` array.

    Returns:
        The role, plus the distinct block types joined with "+" for user and
        assistant messages.
    """
    role = message.get("role", "?")
    if role == "system":
        return role
    content = message.get("content")
    if isinstance(content, str):
        return f"{role}:text"
    if isinstance(content, list):
        types = list(dict.fromkeys(b.get("type", "?") for b in content if isinstance(b, dict)))
        return f"{role}:{'+'.join(types) or 'empty'}"
    return role


def describe_request(body: dict) -> str:
    """Summarizes a request's shape for the log, without any message content.

    Args:
        body: The parsed request body.

    Returns:
        E.g. "main msgs=6 tail=[assistant:tool_use, user:tool_result, system]".
    """
    messages = body.get("messages") or []
    tail = ", ".join(describe_message(m) for m in messages[-_TAIL_LENGTH:])
    prefix = "…, " if len(messages) > _TAIL_LENGTH else ""
    return escape(f"{request_kind(body)} msgs={len(messages)} tail=[{prefix}{tail}]")


def describe_decision(decision: RouteDecision) -> str:
    """Summarizes a routing decision for the log.

    Args:
        decision: The router's decision for the request.

    Returns:
        E.g. "routed small → claude-haiku-4-5-20251001 (score 0.06: no strong
        signals)", noting when the decision follows an earlier question in the turn.
    """
    if not decision.routed:
        return escape(f"not routed ({decision.reason}) → {decision.model}")
    by_turn = ""
    if decision.turn is not None and decision.turn.is_continuation:
        by_turn = f", by turn's question at msg {decision.turn.index}"
    return escape(f"routed {decision.tier} → {decision.model} ({decision.reason}{by_turn})")
