from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from pathlib import Path

from wrap.telemetry import db
from wrap.telemetry.db import TurnRecord
from wrap.telemetry.economics import turn_economics, typical_growth
from wrap.telemetry.pricing import PricingTable

logger = logging.getLogger("wrap.proxy")

# Claude Code puts a billing header in the system prompt as a text block;
# it isn't part of what identifies the conversation.
_BILLING_HEADER_PREFIX = "x-anthropic-billing-header"


def thread_key(body: dict) -> str:
    """Identifies the conversation a `/v1/messages` request belongs to.

    Claude Code resends the whole conversation on every request, and also
    sends subagent, title-generation and helper requests with their own
    prompts. The system prompt plus the first message tells them apart and
    stays the same for the life of a conversation. `cache_control` markers
    are ignored because Claude Code moves them as the conversation grows.

    Args:
        body: The parsed request body.

    Returns:
        16 hex characters.
    """
    system = body.get("system")
    if isinstance(system, list):
        system = [
            block for block in system
            if not (isinstance(block, dict) and str(block.get("text", "")).startswith(_BILLING_HEADER_PREFIX))
        ]
    messages = body.get("messages") or []
    identity = {"system": _without_cache_control(system), "first": _without_cache_control(messages[0] if messages else None)}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class TurnLogger:
    """Writes one `turn_log` row per turn, with its cost, and logs it at DEBUG.

    Everything here is best-effort: `record` swallows and logs every
    exception, so telemetry can never break a request.
    """

    def __init__(self, db_path: Path, pricing: PricingTable, session_id: str | None):
        """Opens (creating if needed) the telemetry database.

        Args:
            db_path: Path to the SQLite file.
            pricing: Rates used to price each turn.
            session_id: The `wrap claude` session, stored on every row.
        """
        self.pricing = pricing
        self.session_id = session_id
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()
        # Per conversation, for the DEBUG `switch` line: the last successful
        # turn, whether it caches with the 1-hour TTL, and the cache writes of
        # turns that stayed on the same model (see `economics.typical_growth`).
        self._last_turn: dict[str | None, tuple[TurnRecord, bool]] = {}
        self._growth: dict[str | None, list[int]] = {}

    def record(self, record: TurnRecord, req_id: int | None = None) -> None:
        """Prices and stores one turn; never raises.

        Args:
            record: The turn, without `session_id` or `cost_usd` (both are
                filled in here).
            req_id: The proxy's request id, to tie the DEBUG lines to the
                request's other log lines.
        """
        try:
            record = dataclasses.replace(
                record,
                session_id=self.session_id,
                cost_usd=self.pricing.cost(record.model, record.usage) if record.status_code and record.status_code < 400 else None,
            )
            with self._lock:
                db.insert_turn(self._conn, record)
            self._log(record, f"#{req_id} " if req_id is not None else "")
        except Exception as exc:
            logger.warning("[yellow]telemetry: failed to record turn: %s[/yellow]", exc)

    def close(self) -> None:
        """Closes the database connection."""
        with self._lock:
            self._conn.close()

    def _log(self, record: TurnRecord, tag: str) -> None:
        # The tier and model are already on the request's routing line.
        if record.status_code is None or record.status_code >= 400:
            logger.debug(
                "[dim]%sturn status=%s error=%s %.0fms[/dim]",
                tag, record.status_code, record.error, record.latency_ms or 0,
            )
            return

        u = record.usage
        cost = f"${record.cost_usd:.4f}" if record.cost_usd is not None else "$—"
        logger.debug(
            "[dim]%sturn in=%s out=%s cache_r=%s cache_w=%s %s %.0fms%s[/dim]",
            tag, format_tokens(u.input_tokens), format_tokens(u.output_tokens), format_tokens(u.cache_read_tokens),
            format_tokens(u.cache_creation_tokens), cost, record.latency_ms or 0,
            f" error={record.error}" if record.error else "",
        )

        previous, uses_1h = self._last_turn.get(record.thread_key, (None, False))
        uses_1h = uses_1h or u.cache_creation_1h_tokens > 0
        self._last_turn[record.thread_key] = (record, uses_1h)
        growth = self._growth.setdefault(record.thread_key, [])
        if previous is None:
            return
        if previous.model == record.model:
            growth.append(u.cache_creation_tokens)
            return
        economics = turn_economics(record, previous, self.pricing, uses_1h, typical_growth(growth))
        if economics is not None:
            logger.debug(
                "[dim]%sswitch %s→%s: re-cached %s tokens (%s)[/dim]",
                tag, _short_model(previous.model), _short_model(record.model),
                format_tokens(economics.recached_tokens), format_signed_cost(economics.cache_penalty),
            )


def _without_cache_control(value):
    if isinstance(value, dict):
        return {k: _without_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_control(v) for v in value]
    return value


def _short_model(model: str | None) -> str:
    """`claude-haiku-4-5-20251001` -> `haiku`."""
    if not model:
        return "?"
    parts = model.removeprefix("claude-").split("-")
    return parts[0] or model


def format_tokens(tokens: int) -> str:
    """Compact token count: 340, 1.2k, 1.5M."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_signed_cost(dollars: float) -> str:
    """`+$0.0470` / `−$0.0120`."""
    return f"{'+' if dollars >= 0 else '−'}${abs(dollars):.4f}"
