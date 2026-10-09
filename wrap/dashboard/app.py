from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wrap.config import Config, load_config
from wrap.dashboard.queries import build_stats, request_rows
from wrap.telemetry import db
from wrap.telemetry.economics import TTL_1H_SECONDS
from wrap.telemetry.pricing import load_pricing

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_ROWS = 1000
# With `since`, turns this far before it are read as context, so a
# conversation's first counted turn still has the turn before it (see
# `wrap.dashboard.queries`): the longest prompt-cache lifetime.
CONTEXT_LOOKBACK = timedelta(seconds=TTL_1H_SECONDS)


def create_dashboard_app(config: Config | None = None) -> FastAPI:
    """Builds the dashboard's FastAPI app: a read-only JSON API over `turn_log` plus the static page.

    Every request opens its own read-only connection, so the dashboard can
    run beside a `wrap claude` proxy that is writing to the same database.

    Args:
        config: Defaults to `load_config()`.

    Returns:
        A configured `FastAPI` app.
    """
    config = config or load_config()
    db_path = Path(config.telemetry.db_path)
    pricing = load_pricing(config.telemetry)

    app = FastAPI(title="wrap dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/sessions")
    def sessions():
        with _reader(db_path) as conn:
            if conn is None:
                return _empty()
            return {"sessions": [dataclasses.asdict(s) for s in db.list_sessions(conn)]}

    @app.get("/api/stats")
    def stats(session: str = "last", since: datetime | None = None):
        with _reader(db_path) as conn:
            if conn is None or not db.has_turns(conn):
                return _empty()
            session_id, all_sessions, count_from, turns = _select(conn, session, since)
        result = build_stats(
            turns, pricing, config.routing.complexity_threshold, datetime.now(timezone.utc), count_from
        )
        scope = {
            "session": session_id,
            "all_sessions": all_sessions,
            "since": count_from.isoformat() if count_from else None,
            "refresh_seconds": config.dashboard.refresh_seconds,
        }
        return {"scope": scope, "generated_at": datetime.now(timezone.utc).isoformat(), **dataclasses.asdict(result)}

    @app.get("/api/requests")
    def requests(
        session: str = "last",
        since: datetime | None = None,
        limit: int = Query(200, ge=1, le=MAX_REQUEST_ROWS),
    ):
        with _reader(db_path) as conn:
            if conn is None or not db.has_turns(conn):
                return _empty()
            _, _, count_from, turns = _select(conn, session, since)
        return {"rows": [dataclasses.asdict(r) for r in request_rows(turns, pricing, limit, count_from)]}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


@contextmanager
def _reader(db_path: Path):
    """A read-only connection, or None if there's no database yet."""
    try:
        conn = db.connect_readonly(db_path)
    except FileNotFoundError:
        yield None
        return
    try:
        yield conn
    finally:
        conn.close()


def _select(
    conn: sqlite3.Connection, session: str, since: datetime | None
) -> tuple[str | None, bool, datetime | None, list[db.TurnRecord]]:
    """Reads the turns a `session` / `since` selection covers, plus context turns before `since`.

    Raises:
        HTTPException: 404 for a session ID with no requests.
    """
    session_id, all_sessions = db.resolve_session(conn, session)
    if session not in ("last", "all") and session_id not in {s.id for s in db.list_sessions(conn)}:
        raise HTTPException(status_code=404, detail=f"no requests logged for session {session}")
    count_from = _utc(since) if since else None
    fetch_since = (count_from - CONTEXT_LOOKBACK).isoformat() if count_from else None
    return session_id, all_sessions, count_from, db.fetch_turns(conn, session_id, all_sessions, fetch_since)


def _utc(when: datetime) -> datetime:
    """Treats a naive timestamp as UTC and converts an aware one to UTC."""
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def _empty() -> dict:
    return {"empty": True, "reason": "no requests logged yet -- run `wrap claude` first"}
