"""The response cache: serves a stored answer when a conversation opens
with a question already answered in the same project.

An entry is found in two steps: an exact partition by scope (the project
directory), then a match on the question within it. This version matches
the normalized question exactly; semantic matching comes later.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from wrap.cache.eligibility import lookup_ineligibility, storable_content
from wrap.cache.store import CacheEntry, CacheStore
from wrap.config import CacheConfig
from wrap.routing.context_guard import estimate_tokens
from wrap.routing.router import RouteDecision

GLOBAL_SCOPE = "global"
_CHARS_PER_TOKEN = 4
# A tier may serve questions routed to itself or any tier before it here.
_TIER_ORDER = ("small", "large")


@dataclass
class Lookup:
    """The outcome of looking up one request.

    Attributes:
        eligible: The request passed the eligibility rules.
        reason: Why it wasn't eligible, for the debug log.
        question: The question as typed (eligible requests only).
        question_hash: Its exact-match key.
        tier: The tier the router chose for it.
        entry: The entry to serve, on a hit.
        similarity: 1.0 on an exact hit.
        miss_reason: Why an eligible request missed: "empty" (nothing
            cached in this scope), "tier" (the match came from a lower
            tier) or "exact_only" (no exact match, and no semantic stage).
    """

    eligible: bool
    reason: str | None = None
    question: str | None = None
    question_hash: str | None = None
    tier: str | None = None
    entry: CacheEntry | None = None
    similarity: float | None = None
    miss_reason: str | None = None

    @property
    def hit(self) -> bool:
        """True if there's an entry to serve."""
        return self.entry is not None


def normalize_question(text: str) -> str:
    """Normalizes a question for exact matching: trimmed, whitespace
    collapsed, lowercased, trailing punctuation dropped."""
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(" .?!;:,")


def question_hash(text: str) -> str:
    """sha256 of the normalized question."""
    return hashlib.sha256(normalize_question(text).encode()).hexdigest()


def scope_key(scope: str, project_dir: str | None) -> str:
    """The partition a project's entries live in.

    Args:
        scope: "project" or "global" (see `CacheConfig.scope`).
        project_dir: The directory `wrap claude` runs in.

    Returns:
        "global", or sha256 of the resolved project directory.
    """
    if scope == GLOBAL_SCOPE:
        return GLOBAL_SCOPE
    resolved = str(Path(project_dir or os.getcwd()).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()


def estimate_request_tokens(body: dict) -> int:
    """Roughly how many input tokens a request is: its messages, system
    prompt and tool definitions at about four characters a token.

    A cached reply reports this as its input, since Claude Code reads
    usage to track how full its context is.
    """
    system = body.get("system")
    if isinstance(system, list):
        system = "".join(b.get("text", "") for b in system if isinstance(b, dict))
    extra_chars = len(system or "") + len(json.dumps(body.get("tools") or [], ensure_ascii=False))
    return estimate_tokens(body.get("messages") or []) + extra_chars // _CHARS_PER_TOKEN


class ResponseCache:
    """Looks requests up in, and stores answers to, one scope of the store."""

    def __init__(self, config: CacheConfig, store: CacheStore, project_dir: str | None):
        """Creates a cache for one project.

        Args:
            config: Cache settings.
            store: Where entries live.
            project_dir: The directory `wrap claude` runs in
                (`WRAP_PROJECT_DIR`); the proxy's own working directory if None.
        """
        self.config = config
        self.store = store
        self.scope_key = scope_key(config.scope, project_dir)

    def lookup(self, body: dict, decision: RouteDecision) -> Lookup:
        """Checks a request's eligibility and looks for a stored answer.

        Args:
            body: The parsed request body.
            decision: The router's decision for it.

        Returns:
            The `Lookup`; `hit` is True if an entry should be served.
        """
        reason = lookup_ineligibility(body, decision, self.config)
        if reason is not None:
            return Lookup(eligible=False, reason=reason)

        question = decision.turn.text
        lookup = Lookup(eligible=True, question=question, question_hash=question_hash(question), tier=decision.tier)
        entry = self.store.get(self.scope_key, lookup.question_hash)
        if entry is not None and _can_serve(entry.tier, decision.tier):
            lookup.entry = entry
            lookup.similarity = 1.0
            self.store.record_hit(entry.id, _now())
        elif entry is not None:
            lookup.miss_reason = "tier"
        elif self.store.count(self.scope_key) == 0:
            lookup.miss_reason = "empty"
        else:
            lookup.miss_reason = "exact_only"
        return lookup

    def insert(self, lookup: Lookup, message: dict | None, cost_usd: float | None) -> int | None:
        """Stores the answer to an eligible request that missed.

        Args:
            lookup: The request's lookup.
            message: The full response message, as assembled from the
                stream or read from the body.
            cost_usd: What the response cost.

        Returns:
            The new entry's ID, or None if the answer isn't storable.
        """
        if not lookup.eligible or lookup.hit:
            return None
        content = storable_content(message)
        if content is None:
            return None
        return self.store.put(
            CacheEntry(
                scope_key=self.scope_key,
                question_hash=lookup.question_hash,
                question=lookup.question,
                tier=lookup.tier,
                served_model=message.get("model"),
                content=content,
                usage=message.get("usage") or {},
                cost_usd=cost_usd,
                created_at=_now(),
            )
        )

    def close(self) -> None:
        """Closes the store."""
        self.store.close()


def _can_serve(entry_tier: str | None, request_tier: str | None) -> bool:
    """A larger tier's answer serves a smaller tier's question, not the reverse."""
    if entry_tier == request_tier:
        return True
    if entry_tier not in _TIER_ORDER or request_tier not in _TIER_ORDER:
        return False
    return _TIER_ORDER.index(entry_tier) >= _TIER_ORDER.index(request_tier)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
