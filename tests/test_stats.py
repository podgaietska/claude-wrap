import pytest
from rich.console import Console

from tests.helpers import add_turn
from tests.test_economics import HAIKU, OPUS, PRICING
from wrap.cli import print_stats
from wrap.telemetry import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "wrap.db")
    yield connection
    connection.close()


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
    assert "Session new — 3 requests (3 new messages, 0 tool calls, 0 side requests)" in text
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

    assert "All sessions — 2 requests" in text


def test_unknown_session_and_empty_db_show_nothing(conn):
    assert render(conn, "last")[0] is False
    add_turn(conn, "a", "large", OPUS, 0, 1_000, 0)
    assert render(conn, "nope")[0] is False


def test_requests_are_broken_down_into_messages_tool_calls_and_side_requests(conn):
    add_turn(conn, "s", "small", HAIKU, 0, 900, 0, thread="title")
    add_turn(conn, "s", "small", HAIKU, 0, 150_000, 1)
    add_turn(conn, "s", "small", HAIKU, 150_000, 500, 2, continuation=True)
    add_turn(conn, "s", "small", HAIKU, 150_500, 500, 3, continuation=True)
    add_turn(conn, "s", "large", OPUS, 0, 151_000, 4)
    add_turn(conn, "s", "unrouted", OPUS, 151_000, 10, 5)

    _, text = render(conn, "last")

    assert "Session s — 6 requests (2 new messages, 2 tool calls, 2 side requests)" in text
