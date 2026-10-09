import dataclasses
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from tests.helpers import START, add_turn
from tests.test_economics import HAIKU, OPUS
from wrap.config import load_config
from wrap.dashboard.app import STATIC_DIR, create_dashboard_app
from wrap.telemetry import db


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "wrap.db"


@pytest.fixture
def client(db_path):
    config = load_config()
    config = dataclasses.replace(config, telemetry=dataclasses.replace(config.telemetry, db_path=str(db_path)))
    with TestClient(create_dashboard_app(config)) as test_client:
        yield test_client


@pytest.fixture
def seeded(db_path):
    conn = db.connect(db_path)
    add_turn(conn, "old", "large", OPUS, 0, 1_000, 0, score=0.9)
    add_turn(conn, "new", "large", OPUS, 0, 150_000, 3600, score=0.8, ttfb_ms=800, latency_ms=3000)
    add_turn(conn, "new", "large", OPUS, 150_000, 1_000, 3610, continuation=True, latency_ms=2000)
    add_turn(conn, "new", "small", HAIKU, 0, 151_000, 3700, score=0.1, ttfb_ms=300, latency_ms=900)
    add_turn(conn, "new", "small", HAIKU, 0, 0, 3710, status=529, error="overloaded_error: Overloaded")
    yield conn
    conn.close()


def test_no_database_is_an_empty_state(client, db_path):
    for path in ("/api/sessions", "/api/stats", "/api/requests"):
        body = client.get(path).json()
        assert body["empty"] is True
    assert not db_path.exists()


def test_sessions(client, seeded):
    sessions = client.get("/api/sessions").json()["sessions"]

    assert [s["id"] for s in sessions] == ["new", "old"]
    assert sessions[0]["requests"] == 4
    assert sessions[0]["failed"] == 1


def test_stats_for_the_last_session(client, seeded):
    response = client.get("/api/stats")
    body = response.json()

    assert response.headers["cache-control"] == "no-store"
    assert body["scope"] == {"session": "new", "all_sessions": False, "since": None, "refresh_seconds": 5}
    assert body["summary"]["requests"] == 4
    assert (body["summary"]["new_messages"], body["summary"]["tool_calls"]) == (3, 1)
    assert body["summary"]["live"] is False
    assert body["economics"]["switches"] == 1
    assert body["economics"]["baseline"] == OPUS
    assert body["routing"]["threshold"] == 0.5
    assert body["timeseries"]["bucket_seconds"] == 60
    assert body["cache"] is None
    assert {m["served_model"] for m in body["models"]} == {OPUS, HAIKU}


def test_stats_selections(client, seeded):
    assert client.get("/api/stats", params={"session": "all"}).json()["summary"]["requests"] == 5
    assert client.get("/api/stats", params={"session": "old"}).json()["summary"]["requests"] == 1
    response = client.get("/api/stats", params={"session": "nope"})
    assert response.status_code == 404


def test_since_counts_from_then_but_keeps_context(client, seeded):
    since = (START + timedelta(seconds=3650)).isoformat()

    body = client.get("/api/stats", params={"since": since}).json()

    assert body["summary"]["requests"] == 2
    assert body["economics"]["switches"] == 1  # the Haiku request is still compared with the Opus one before it
    assert body["scope"]["since"] == since


def test_naive_since_is_utc(client, seeded):
    since = (START + timedelta(seconds=3650)).replace(tzinfo=None).isoformat()

    body = client.get("/api/stats", params={"since": since}).json()

    assert body["summary"]["requests"] == 2


def test_bad_since_is_rejected(client, seeded):
    assert client.get("/api/stats", params={"since": "yesterday"}).status_code == 422


def test_requests_newest_first_and_capped(client, seeded):
    rows = client.get("/api/requests", params={"limit": 2}).json()["rows"]

    assert len(rows) == 2
    assert rows[0]["status_code"] == 529
    assert rows[0]["error"].startswith("overloaded_error")
    assert rows[1]["switched"] is True
    assert client.get("/api/requests", params={"limit": 5000}).status_code == 422


def test_page_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "wrap dashboard" in page.text
    for path in ("/static/dashboard.js", "/static/styles.css", "/static/vendor/chart.umd.js"):
        assert client.get(path).status_code == 200, path


def test_static_files_are_in_the_package():
    for name in ("index.html", "dashboard.js", "styles.css", "vendor/chart.umd.js", "vendor/chart.js-LICENSE.md"):
        assert (STATIC_DIR / name).is_file(), name
