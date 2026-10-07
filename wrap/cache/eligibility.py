"""Which requests and answers the response cache may touch.

The same rules gate lookup and insert, so the cache never stores an
answer it would refuse to serve. Only the first question of a
conversation, text only, is eligible: later turns depend on history the
cache key doesn't hold.
"""
from __future__ import annotations

from wrap.config import CacheConfig
from wrap.proxy.describe import request_kind
from wrap.routing.router import RouteDecision

# Blocks a cached answer may carry, besides text, that are dropped before
# storing: their signatures belong to the original request.
_DROPPED_BLOCKS = ("thinking", "redacted_thinking")


def lookup_ineligibility(body: dict, decision: RouteDecision, config: CacheConfig) -> str | None:
    """Checks whether a request may be answered from, or stored in, the cache.

    Args:
        body: The parsed request body.
        decision: The router's decision for it.
        config: Cache settings, for the question length limits.

    Returns:
        None if eligible, else a short reason for the debug log.
    """
    if request_kind(body) != "main":
        return "side request"
    turn = decision.turn
    if not decision.routed or turn is None:
        return "no question"
    if turn.is_continuation:
        return "tool continuation"
    messages = body.get("messages") or []
    if any(m.get("role") == "assistant" for m in messages[: turn.index]):
        return "not the first question"
    content = messages[turn.index].get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") != "text" for block in content
    ):
        return "not text only"
    length = len(turn.text.strip())
    if length < config.min_question_chars:
        return "question too short"
    if length > config.max_question_chars:
        return "question too long"
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type", "auto") != "auto":
        return "forced tool choice"
    return None


def storable_content(message: dict | None) -> list[dict] | None:
    """The text blocks of a finished answer, if it may be stored.

    Args:
        message: The assembled response message.

    Returns:
        The answer's text blocks, reduced to type and text, with thinking
        dropped; or None if it didn't end with `end_turn` or holds anything
        but text (a tool call means the answer came from the project).
    """
    if not isinstance(message, dict) or message.get("stop_reason") != "end_turn":
        return None
    blocks = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            return None
        if block.get("type") in _DROPPED_BLOCKS:
            continue
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            return None
        blocks.append({"type": "text", "text": block["text"]})
    if not any(block["text"].strip() for block in blocks):
        return None
    return blocks
