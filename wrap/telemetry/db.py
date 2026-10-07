from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from wrap.telemetry.usage import Usage

SCHEMA_VERSION = 2

_USAGE_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "cache_creation_5m_tokens",
    "cache_creation_1h_tokens",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_log (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id               TEXT,
    thread_key               TEXT,
    timestamp                TEXT NOT NULL,
    requested_model          TEXT,
    model_id                 TEXT,
    served_model             TEXT,
    tier                     TEXT,
    complexity_score         REAL,
    was_tool_continuation    INTEGER NOT NULL DEFAULT 0,
    stream                   INTEGER NOT NULL DEFAULT 0,
    status_code              INTEGER,
    stop_reason              TEXT,
    input_tokens             INTEGER NOT NULL DEFAULT 0,
    output_tokens            INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_creation_5m_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_1h_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd                 REAL,
    ttfb_ms                  REAL,
    latency_ms               REAL,
    cache_hit                INTEGER NOT NULL DEFAULT 0,
    similarity_score         REAL,
    error                    TEXT,
    cache_eligible           INTEGER NOT NULL DEFAULT 0,
    cache_entry_id           INTEGER,
    cache_miss_reason        TEXT
);
CREATE INDEX IF NOT EXISTS turn_log_session ON turn_log (session_id, id);
"""

# Columns added after the first schema version, with their definitions, so
# an existing database gains them on open.
_ADDED_COLUMNS = {
    "cache_eligible": "INTEGER NOT NULL DEFAULT 0",
    "cache_entry_id": "INTEGER",
    "cache_miss_reason": "TEXT",
}


@dataclass
class TurnRecord:
    """One `/v1/messages` turn, as stored in `turn_log`.

    Attributes:
        timestamp: When the request reached the proxy, ISO-8601 UTC.
        session_id: The `wrap claude` session, or None outside one.
        thread_key: Identifies the conversation within a session (see
            `wrap.telemetry.logger.thread_key`).
        requested_model: The model Claude Code asked for.
        model_id: The model the proxy sent upstream.
        served_model: The model the response says answered, or None if
            the response didn't say.
        tier: "small" or "large", or "unrouted" for a request with no
            human question to route by (sent to the requested model).
        complexity_score: The classifier's score, None if unrouted.
        was_tool_continuation: True for a mid-turn request (e.g. sending
            back tool results), routed by its turn's question.
        stream: True for a streamed response.
        status_code: The upstream HTTP status.
        stop_reason: The response's stop reason.
        usage: Token counts.
        cost_usd: The turn's cost, None if the model has no price.
        ttfb_ms: Time until upstream response headers arrived.
        latency_ms: Time until the response was fully relayed.
        cache_hit: Served from the response cache, with no upstream call.
        similarity_score: On a hit, how closely the question matched the
            entry's (1.0 for an exact match).
        error: Error type and message, or why a stream ended early.
        cache_eligible: The request passed the cache's eligibility rules,
            so it was looked up (the denominator of the hit rate).
        cache_entry_id: The cache entry served (hit) or written (insert).
        cache_miss_reason: Why an eligible request missed, e.g. "empty"
            (nothing cached for the project yet) or "exact_only".
        id: Row ID, set when read back from the database.
    """

    timestamp: str
    session_id: str | None = None
    thread_key: str | None = None
    requested_model: str | None = None
    model_id: str | None = None
    served_model: str | None = None
    tier: str | None = None
    complexity_score: float | None = None
    was_tool_continuation: bool = False
    stream: bool = False
    status_code: int | None = None
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    ttfb_ms: float | None = None
    latency_ms: float | None = None
    cache_hit: bool = False
    similarity_score: float | None = None
    error: str | None = None
    cache_eligible: bool = False
    cache_entry_id: int | None = None
    cache_miss_reason: str | None = None
    id: int | None = None

    @property
    def model(self) -> str | None:
        """The model that answered: `served_model`, else what was sent upstream."""
        return self.served_model or self.model_id


@dataclass
class SummaryRow:
    """Totals for one tier and served model, from `summarize`."""

    tier: str | None
    served_model: str | None
    turns: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    unpriced_turns: int


def connect(path: Path) -> sqlite3.Connection:
    """Opens the telemetry database, creating it and its schema if needed.

    Args:
        path: Path to the SQLite file; parent directories are created.

    Returns:
        An open connection, usable from any thread (callers serialize access).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Adds columns newer than the database's `turn_log` table, keeping its rows."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(turn_log)")}
    for column, definition in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE turn_log ADD COLUMN {column} {definition}")


def insert_turn(conn: sqlite3.Connection, record: TurnRecord) -> int:
    """Inserts one turn and returns its row ID.

    Args:
        conn: An open telemetry connection.
        record: The turn to store; its `id` is ignored.

    Returns:
        The new row's ID.
    """
    values = {
        "session_id": record.session_id,
        "thread_key": record.thread_key,
        "timestamp": record.timestamp,
        "requested_model": record.requested_model,
        "model_id": record.model_id,
        "served_model": record.served_model,
        "tier": record.tier,
        "complexity_score": record.complexity_score,
        "was_tool_continuation": int(record.was_tool_continuation),
        "stream": int(record.stream),
        "status_code": record.status_code,
        "stop_reason": record.stop_reason,
        **{column: getattr(record.usage, column) for column in _USAGE_COLUMNS},
        "cost_usd": record.cost_usd,
        "ttfb_ms": record.ttfb_ms,
        "latency_ms": record.latency_ms,
        "cache_hit": int(record.cache_hit),
        "similarity_score": record.similarity_score,
        "error": record.error,
        "cache_eligible": int(record.cache_eligible),
        "cache_entry_id": record.cache_entry_id,
        "cache_miss_reason": record.cache_miss_reason,
    }
    columns = ", ".join(values)
    placeholders = ", ".join(f":{column}" for column in values)
    cursor = conn.execute(f"INSERT INTO turn_log ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    return cursor.lastrowid


def latest_session_id(conn: sqlite3.Connection) -> str | None:
    """Returns the session of the most recently logged turn, or None if empty."""
    row = conn.execute("SELECT session_id FROM turn_log ORDER BY id DESC LIMIT 1").fetchone()
    return row["session_id"] if row else None


def has_turns(conn: sqlite3.Connection) -> bool:
    """Returns True if any turn has been logged."""
    return conn.execute("SELECT 1 FROM turn_log LIMIT 1").fetchone() is not None


def fetch_turns(conn: sqlite3.Connection, session_id: str | None = None, all_sessions: bool = False) -> list[TurnRecord]:
    """Reads turns back, oldest first.

    Args:
        conn: An open telemetry connection.
        session_id: The session to read; None means turns logged outside
            a `wrap claude` session.
        all_sessions: Read every turn, ignoring `session_id`.

    Returns:
        The matching turns, ordered by ID.
    """
    where, params = _session_filter(session_id, all_sessions)
    rows = conn.execute(f"SELECT * FROM turn_log {where} ORDER BY id", params).fetchall()
    return [_to_record(row) for row in rows]


def summarize(conn: sqlite3.Connection, session_id: str | None = None, all_sessions: bool = False) -> list[SummaryRow]:
    """Totals turns, tokens and cost per tier and served model.

    Args:
        conn: An open telemetry connection.
        session_id: The session to summarize (see `fetch_turns`).
        all_sessions: Summarize every turn, ignoring `session_id`.

    Returns:
        One row per (tier, served model), largest cost first.
    """
    where, params = _session_filter(session_id, all_sessions)
    rows = conn.execute(
        f"""
        SELECT tier,
               COALESCE(served_model, model_id) AS served_model,
               COUNT(*) AS turns,
               SUM(input_tokens) AS input_tokens,
               SUM(output_tokens) AS output_tokens,
               SUM(cache_read_tokens) AS cache_read_tokens,
               SUM(cache_creation_tokens) AS cache_creation_tokens,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               SUM(cost_usd IS NULL) AS unpriced_turns
        FROM turn_log {where}
        GROUP BY tier, COALESCE(served_model, model_id)
        ORDER BY cost_usd DESC, tier
        """,
        params,
    ).fetchall()
    return [SummaryRow(**dict(row)) for row in rows]


def _session_filter(session_id: str | None, all_sessions: bool) -> tuple[str, tuple]:
    if all_sessions:
        return "", ()
    return "WHERE session_id IS ?", (session_id,)


def _to_record(row: sqlite3.Row) -> TurnRecord:
    data = dict(row)
    usage = Usage(**{column: data.pop(column) for column in _USAGE_COLUMNS})
    for flag in ("was_tool_continuation", "stream", "cache_hit", "cache_eligible"):
        data[flag] = bool(data[flag])
    return TurnRecord(usage=usage, **data)
