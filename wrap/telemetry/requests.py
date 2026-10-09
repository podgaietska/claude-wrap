"""Splits requests into new messages, tool calls and side requests.

One question in Claude Code is usually several API requests: the
question, then a round trip per tool call. A session's main conversation
is taken to be the one with the most requests; the rest are side requests
(background calls such as session titles, and subagents).
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from wrap.telemetry.db import TurnRecord

RequestKind = Literal["new_message", "tool_call", "side"]


def main_threads(turns: list[TurnRecord]) -> dict[str | None, str | None]:
    """Finds each session's main conversation: the thread with the most requests.

    Args:
        turns: The requests to look at.

    Returns:
        Session ID to the thread key of its main conversation.
    """
    sizes = Counter((t.session_id, t.thread_key) for t in turns)
    main: dict[str | None, str | None] = {}
    for (session_id, thread_key), size in sizes.items():
        if session_id not in main or size > sizes[(session_id, main[session_id])]:
            main[session_id] = thread_key
    return main


def classify(turns: list[TurnRecord]) -> list[RequestKind]:
    """Labels each request as a new message, a tool call or a side request.

    Args:
        turns: The requests to label; every session's requests should be
            included, or its main conversation may be misidentified.

    Returns:
        One kind per turn, in the same order.
    """
    main = main_threads(turns)
    kinds: list[RequestKind] = []
    for t in turns:
        if t.thread_key != main[t.session_id]:
            kinds.append("side")
        elif t.was_tool_continuation:
            kinds.append("tool_call")
        elif t.tier in ("small", "large"):
            kinds.append("new_message")
        else:
            kinds.append("side")
    return kinds


def count_kinds(turns: list[TurnRecord]) -> Counter[RequestKind]:
    """Counts requests of each kind (see `classify`)."""
    return Counter(classify(turns))
