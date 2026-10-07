"""SQLite storage for cached answers, next to `turn_log` in `data/wrap.db`.

This is the only place wrap stores message content: the question and the
answer's text. `turn_log` never holds either.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key       TEXT NOT NULL,
    question_hash   TEXT NOT NULL,
    question        TEXT NOT NULL,
    embedding       BLOB,
    embedding_model TEXT,
    tier            TEXT,
    served_model    TEXT,
    response        TEXT NOT NULL,
    usage_json      TEXT,
    cost_usd        REAL,
    created_at      TEXT NOT NULL,
    last_hit_at     TEXT,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (scope_key, question_hash)
);
"""


@dataclass
class CacheEntry:
    """One cached answer.

    Attributes:
        scope_key: The project it belongs to (see `ResponseCache`), or "global".
        question_hash: sha256 of the normalized question; the exact-match key.
        question: The question as typed, kept for listing and re-embedding.
        tier: The tier that answered; it may serve this tier or a lower one.
        served_model: The model that wrote the answer.
        content: The answer's text blocks.
        usage: The original call's usage, as the API reported it.
        cost_usd: The original call's cost, None if unpriced.
        created_at: When it was stored, ISO-8601 UTC.
        last_hit_at: When it was last served.
        hit_count: How many times it was served.
        id: Row ID, set once stored.
    """

    scope_key: str
    question_hash: str
    question: str
    tier: str | None
    served_model: str | None
    content: list[dict]
    usage: dict
    cost_usd: float | None
    created_at: str
    last_hit_at: str | None = None
    hit_count: int = 0
    id: int | None = None

    @property
    def output_tokens(self) -> int:
        """Output tokens of the original answer."""
        tokens = self.usage.get("output_tokens")
        return tokens if isinstance(tokens, int) else 0


class CacheStore:
    """Reads and writes cache entries, expiring old ones and evicting the
    least recently used past a cap. Safe to call from any thread."""

    def __init__(self, path: Path, ttl_days: float, max_entries: int):
        """Opens (creating if needed) the cache table.

        Args:
            path: The SQLite file, shared with telemetry.
            ttl_days: Entries older than this are never served.
            max_entries: The most entries kept across all scopes.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ttl = timedelta(days=ttl_days)
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get(self, scope_key: str, question_hash: str) -> CacheEntry | None:
        """Finds a live entry by its exact key; deletes it if it has expired.

        Args:
            scope_key: The entry's scope.
            question_hash: The normalized question's hash.

        Returns:
            The entry, or None.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entry WHERE scope_key = ? AND question_hash = ?", (scope_key, question_hash)
            ).fetchone()
            if row is None:
                return None
            if row["created_at"] < self._cutoff():
                self._conn.execute("DELETE FROM cache_entry WHERE id = ?", (row["id"],))
                self._conn.commit()
                return None
            return _to_entry(row)

    def count(self, scope_key: str) -> int:
        """Counts a scope's live entries."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM cache_entry WHERE scope_key = ? AND created_at >= ?", (scope_key, self._cutoff())
            ).fetchone()
            return row[0]

    def put(self, entry: CacheEntry) -> int:
        """Stores an entry, replacing any with the same key, then expires and
        evicts as needed.

        Args:
            entry: The entry; its `id` is ignored.

        Returns:
            The stored entry's row ID.
        """
        values = {
            "scope_key": entry.scope_key,
            "question_hash": entry.question_hash,
            "question": entry.question,
            "tier": entry.tier,
            "served_model": entry.served_model,
            "response": json.dumps({"content": entry.content, "stop_reason": "end_turn"}, ensure_ascii=False),
            "usage_json": json.dumps(entry.usage),
            "cost_usd": entry.cost_usd,
            "created_at": entry.created_at,
        }
        columns = ", ".join(values)
        placeholders = ", ".join(f":{column}" for column in values)
        with self._lock:
            self._conn.execute(
                "DELETE FROM cache_entry WHERE scope_key = ? AND question_hash = ?",
                (entry.scope_key, entry.question_hash),
            )
            cursor = self._conn.execute(f"INSERT INTO cache_entry ({columns}) VALUES ({placeholders})", values)
            self._conn.execute("DELETE FROM cache_entry WHERE created_at < ?", (self._cutoff(),))
            self._conn.execute(
                """
                DELETE FROM cache_entry WHERE id IN (
                    SELECT id FROM cache_entry
                    ORDER BY COALESCE(last_hit_at, created_at) DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self._max_entries,),
            )
            self._conn.commit()
            return cursor.lastrowid

    def record_hit(self, entry_id: int, at: str) -> None:
        """Marks an entry as served, for LRU eviction and listing."""
        with self._lock:
            self._conn.execute(
                "UPDATE cache_entry SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?", (at, entry_id)
            )
            self._conn.commit()

    def close(self) -> None:
        """Closes the database connection."""
        with self._lock:
            self._conn.close()

    def _cutoff(self) -> str:
        return (datetime.now(timezone.utc) - self._ttl).isoformat()


def _to_entry(row: sqlite3.Row) -> CacheEntry:
    response = json.loads(row["response"])
    return CacheEntry(
        id=row["id"],
        scope_key=row["scope_key"],
        question_hash=row["question_hash"],
        question=row["question"],
        tier=row["tier"],
        served_model=row["served_model"],
        content=response["content"],
        usage=json.loads(row["usage_json"]) if row["usage_json"] else {},
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
        last_hit_at=row["last_hit_at"],
        hit_count=row["hit_count"],
    )
