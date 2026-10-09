import sqlite3

import pytest

from tests.helpers import add_turn
from tests.test_economics import HAIKU, OPUS
from wrap.telemetry import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "wrap.db")
    yield connection
    connection.close()


def test_resolve_session(conn):
    add_turn(conn, "old", "large", OPUS, 0, 1_000, 0)
    add_turn(conn, "new", "large", OPUS, 0, 1_000, 10)

    assert db.resolve_session(conn, "last") == ("new", False)
    assert db.resolve_session(conn, "all") == (None, True)
    assert db.resolve_session(conn, "old") == ("old", False)


def test_list_sessions_totals_newest_first(conn):
    add_turn(conn, "a", "large", OPUS, 0, 1_000, 0)
    add_turn(conn, "a", "small", HAIKU, 0, 1_000, 30, status=529)
    add_turn(conn, "b", "small", HAIKU, 0, 1_000, 60)

    sessions = db.list_sessions(conn)

    assert [s.id for s in sessions] == ["b", "a"]
    a = sessions[1]
    assert a.requests == 2
    assert a.failed == 1
    assert a.started < a.last_seen
    assert a.cost_usd == pytest.approx(db.fetch_turns(conn, "a")[0].cost_usd)


def test_since_filters_turns_and_summary(conn):
    add_turn(conn, "s", "large", OPUS, 0, 1_000, 0)
    add_turn(conn, "s", "small", HAIKU, 0, 1_000, 60)
    cutoff = db.fetch_turns(conn, "s")[1].timestamp

    assert len(db.fetch_turns(conn, "s", since=cutoff)) == 1
    assert len(db.fetch_turns(conn, all_sessions=True, since=cutoff)) == 1
    assert [row.served_model for row in db.summarize(conn, "s", since=cutoff)] == [HAIKU]


def test_readonly_connection_reads_but_never_writes(tmp_path, conn):
    add_turn(conn, "s", "large", OPUS, 0, 1_000, 0)
    conn.close()  # no writer running, as after a `wrap claude` session

    reader = db.connect_readonly(tmp_path / "wrap.db")
    try:
        assert len(db.fetch_turns(reader, "s")) == 1
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("DELETE FROM turn_log")
    finally:
        reader.close()


def test_readonly_connection_sees_live_writes(tmp_path, conn):
    reader = db.connect_readonly(tmp_path / "wrap.db")
    try:
        add_turn(conn, "s", "large", OPUS, 0, 1_000, 0)
        assert len(db.fetch_turns(reader, "s")) == 1
    finally:
        reader.close()


def test_readonly_connection_requires_an_existing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        db.connect_readonly(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()
