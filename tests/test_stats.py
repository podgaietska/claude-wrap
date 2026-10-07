from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from tests.test_economics import HAIKU, OPUS, PRICING
from wrap.cli import print_stats
from wrap.telemetry import db
from wrap.telemetry.db import TurnRecord
from wrap.telemetry.usage import Usage

START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "wrap.db")
    yield connection
    connection.close()


def add_turn(conn, session_id, tier, model, cache_read, cache_write, seconds, status=200):
    usage = Usage(input_tokens=10, output_tokens=500, cache_read_tokens=cache_read,
                  cache_creation_tokens=cache_write, cache_creation_5m_tokens=cache_write)
    db.insert_turn(conn, TurnRecord(
        timestamp=(START + timedelta(seconds=seconds)).isoformat(),
        session_id=session_id,
        thread_key="t1",
        requested_model=OPUS,
        model_id=model,
        served_model=model,
        tier=tier,
        status_code=status,
        usage=usage,
        cost_usd=PRICING.cost(model, usage) if status < 400 else None,
    ))


def render(conn, session: str) -> tuple[bool, str]:
    out = Console(record=True, width=120, color_system=None)
    shown = print_stats(out, conn, PRICING, session)
    return shown, out.export_text()


def test_last_session_shows_table_and_cache_penalty(conn):
    add_turn(conn, "old", "large", OPUS, 0, 1_000, 0)
    add_turn(conn, "new", "large", OPUS, 145_000, 5_000, 0)
    add_turn(conn, "new", "small", HAIKU, 0, 150_000, 60)
    add_turn(conn, "new", "small", HAIKU, 0, 0, 70, status=529)

    shown, text = render(conn, "last")

    assert shown
    assert "Session new — 3 turns" in text
    assert "claude-haiku-4-5-20251001" in text
    assert "1 failed requests" in text
    assert "Routing economics (vs. always claude-opus-5-5)" in text
    assert "Lost to cache misses (1 switches):" in text
    assert "150.0k tokens re-cached" in text
    assert "Net savings:" in text and "−$" in text


def test_all_sessions(conn):
    add_turn(conn, "a", "large", OPUS, 0, 1_000, 0)
    add_turn(conn, "b", "large", OPUS, 0, 1_000, 0)

    _, text = render(conn, "all")

    assert "All sessions — 2 turns" in text


def test_unknown_session_and_empty_db_show_nothing(conn):
    assert render(conn, "last")[0] is False
    add_turn(conn, "a", "large", OPUS, 0, 1_000, 0)
    assert render(conn, "nope")[0] is False
