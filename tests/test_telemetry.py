import pytest

from wrap.telemetry import db
from wrap.telemetry.db import TurnRecord
from wrap.telemetry.usage import Usage


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "nested" / "wrap.db")
    yield connection
    connection.close()


def turn(**overrides) -> TurnRecord:
    defaults = dict(
        timestamp="2026-10-03T12:00:00+00:00",
        session_id="s1",
        thread_key="t1",
        requested_model="claude-sonnet-5",
        model_id="claude-haiku-4-5-20251001",
        served_model="claude-haiku-4-5-20251001",
        tier="small",
        complexity_score=0.1,
        stream=True,
        status_code=200,
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=300,
            cache_creation_tokens=40,
            cache_creation_5m_tokens=40,
        ),
        cost_usd=0.001,
        ttfb_ms=200.0,
        latency_ms=900.0,
    )
    return TurnRecord(**{**defaults, **overrides})


def test_schema_creation_is_idempotent(tmp_path):
    path = tmp_path / "wrap.db"
    db.connect(path).close()
    conn = db.connect(path)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_turn_round_trips(conn):
    original = turn(was_tool_continuation=True, error="stream_incomplete", similarity_score=None)

    row_id = db.insert_turn(conn, original)
    [stored] = db.fetch_turns(conn, "s1")

    assert stored.id == row_id
    assert stored == TurnRecord(**{**original.__dict__, "id": row_id})


def test_fetch_turns_filters_by_session_and_keeps_order(conn):
    db.insert_turn(conn, turn(session_id="s1", tier="small"))
    db.insert_turn(conn, turn(session_id="s2"))
    db.insert_turn(conn, turn(session_id="s1", tier="large"))
    db.insert_turn(conn, turn(session_id=None))

    assert [t.tier for t in db.fetch_turns(conn, "s1")] == ["small", "large"]
    assert len(db.fetch_turns(conn, None)) == 1
    assert len(db.fetch_turns(conn, all_sessions=True)) == 4


def test_latest_session_id(conn):
    assert db.latest_session_id(conn) is None
    db.insert_turn(conn, turn(session_id="old"))
    db.insert_turn(conn, turn(session_id="new"))

    assert db.latest_session_id(conn) == "new"


def test_summarize_groups_by_tier_and_served_model(conn):
    db.insert_turn(conn, turn(tier="small", cost_usd=0.01))
    db.insert_turn(conn, turn(tier="small", cost_usd=0.02))
    db.insert_turn(conn, turn(tier="large", served_model="claude-sonnet-5", cost_usd=0.10))
    db.insert_turn(conn, turn(tier="large", served_model=None, model_id="claude-sonnet-5", cost_usd=None))
    db.insert_turn(conn, turn(session_id="other", tier="small"))

    rows = {(r.tier, r.served_model): r for r in db.summarize(conn, "s1")}

    small = rows[("small", "claude-haiku-4-5-20251001")]
    assert small.turns == 2
    assert small.cost_usd == pytest.approx(0.03)
    assert small.input_tokens == 20
    assert small.cache_read_tokens == 600

    large = rows[("large", "claude-sonnet-5")]
    assert large.turns == 2
    assert large.cost_usd == pytest.approx(0.10)
    assert large.unpriced_turns == 1
