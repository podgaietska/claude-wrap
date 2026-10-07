import json
import logging
import re
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from wrap.config import (
    CacheConfig,
    Config,
    ModelCapabilities,
    ProxyConfig,
    RoutingConfig,
    TelemetryConfig,
    TierConfig,
)
from wrap.proxy.server import create_app
from wrap.telemetry import db
from wrap.telemetry.logger import thread_key

HEADERS = {"x-api-key": "test", "anthropic-version": "2023-06-01"}
TOOL_RESULT_TURN = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "output"}]}]
STREAM = (Path(__file__).parent / "fixtures" / "messages_response_stream.txt").read_bytes()
TEST_PRICING = """
models:
  small-model: {input: 1.0, output: 5.0, cache_read: 0.1, cache_write_5m: 1.25, cache_write_1h: 2.0}
  large-model: {input: 2.0, output: 10.0, cache_read: 0.2, cache_write_5m: 2.5, cache_write_1h: 4.0}
"""


def make_config(tmp_path: Path | None = None) -> Config:
    """Builds a minimal `Config` fixture pointing at the real Anthropic
    upstream URL (mocked per-test via `respx`).

    Telemetry is on only when `tmp_path` is given, writing to a database
    there, so tests never touch `data/wrap.db`.
    """
    if tmp_path is not None:
        (tmp_path / "pricing.yaml").write_text(TEST_PRICING)
        telemetry = TelemetryConfig(db_path=str(tmp_path / "wrap.db"), pricing_file=str(tmp_path / "pricing.yaml"))
    else:
        telemetry = TelemetryConfig(db_path="data/wrap.db", pricing_file="config/pricing.yaml", enabled=False)
    return Config(
        tiers={"small": TierConfig(model="small-model"), "large": TierConfig(model="large-model")},
        models={
            "small-model": ModelCapabilities(max_output_tokens=4096, context_window=200000),
            "large-model": ModelCapabilities(max_output_tokens=64000, context_window=1000000),
            "passthrough-model": ModelCapabilities(max_output_tokens=8000, context_window=200000),
        },
        routing=RoutingConfig(strategy="heuristic", complexity_threshold=0.5),
        cache=CacheConfig(enabled=False, similarity_threshold=0.92, embedding_model="x"),
        proxy=ProxyConfig(port=8787, upstream_base_url="https://api.anthropic.com", log_path="data/proxy.log"),
        telemetry=telemetry,
    )


def logged_turns(tmp_path: Path) -> list[db.TurnRecord]:
    conn = db.connect(tmp_path / "wrap.db")
    try:
        return db.fetch_turns(conn, all_sessions=True)
    finally:
        conn.close()


@respx.mock
def test_trivial_message_rewrites_model_to_small_tier():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "result": "ok"})
    )

    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={"model": "requested-model", "max_tokens": 100, "messages": [{"role": "user", "content": "what is python?"}]},
        )

    assert response.status_code == 200
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "small-model"


@respx.mock
def test_complex_message_rewrites_model_to_large_tier():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "result": "ok"})
    )

    complex_text = (
        "Please explain step by step how you would refactor this architecture, "
        "compare the trade-offs between microservices and a monolith, and analyze "
        "why the current design has issues."
    )
    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={"model": "requested-model", "max_tokens": 100, "messages": [{"role": "user", "content": complex_text}]},
        )

    assert response.status_code == 200
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "large-model"


@respx.mock
def test_max_tokens_is_clamped_to_the_routed_tier_ceiling():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "result": "ok"})
    )

    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={
                "model": "requested-model",
                "max_tokens": 128000,  # oversized for the small tier's 4096 ceiling
                "messages": [{"role": "user", "content": "what is python?"}],
            },
        )

    assert response.status_code == 200
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "small-model"
    assert sent_body["max_tokens"] == 4096


@respx.mock
def test_requests_without_a_question_are_still_adapted():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1"})
    )

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "passthrough-model", "max_tokens": 128000, "messages": TOOL_RESULT_TURN},
        )

    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "passthrough-model"
    assert sent_body["max_tokens"] == 8000


@respx.mock
def test_max_tokens_400_is_retried_with_the_ceiling_from_the_error():
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "max_tokens: 4096 > 2048, which is the maximum allowed number of output tokens for small-model",
        },
    }
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[httpx.Response(400, json=error), httpx.Response(200, json={"id": "msg_1"})]
    )

    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 128000, "messages": [{"role": "user", "content": "what is python?"}]},
        )

    assert response.status_code == 200
    assert route.call_count == 2
    first, retry = (json.loads(call.request.content) for call in route.calls)
    assert first["max_tokens"] == 4096  # configured (stale) ceiling
    assert retry["max_tokens"] == 2048  # ceiling learned from the error


@respx.mock
def test_learned_ceiling_is_used_for_later_requests_without_another_400():
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: 4096 > 2048, ..."}}
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            httpx.Response(400, json=error),
            httpx.Response(200, json={"id": "msg_1"}),
            httpx.Response(200, json={"id": "msg_2"}),
        ]
    )
    request_json = {"model": "x", "max_tokens": 128000, "messages": [{"role": "user", "content": "what is python?"}]}

    with TestClient(create_app(make_config())) as client:
        client.post("/v1/messages", headers=HEADERS, json=request_json)
        client.post("/v1/messages", headers=HEADERS, json=request_json)

    assert route.call_count == 3
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 2048


@respx.mock
def test_unrelated_400_is_returned_unchanged_without_retry():
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "messages: field required"}}
    route = respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(400, json=error))

    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert response.json() == error
    assert route.call_count == 1


@respx.mock
def test_query_string_is_forwarded():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1"})
    )

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages?beta=true",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert route.calls.last.request.url.query == b"beta=true"


@respx.mock
def test_upstream_is_asked_for_an_uncompressed_response():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1"})
    )

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages",
            headers={**HEADERS, "accept-encoding": "gzip, br"},
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert route.calls.last.request.headers["accept-encoding"] == "identity"


@respx.mock
def test_request_without_a_question_keeps_its_model():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "result": "ok"})
    )

    with TestClient(create_app(make_config())) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={
                "model": "whatever-was-requested",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "output"}]}
                ],
            },
        )

    assert response.status_code == 200
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "whatever-was-requested"


@respx.mock
def test_auth_headers_are_forwarded_unchanged():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg_1", "result": "ok"})
    )

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages",
            headers={"x-api-key": "super-secret-key", "anthropic-version": "2023-06-01"},
            json={"model": "requested-model", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    forwarded = route.calls.last.request.headers
    assert forwarded["x-api-key"] == "super-secret-key"
    assert forwarded["anthropic-version"] == "2023-06-01"


@respx.mock
def test_other_paths_are_forwarded_unmodified():
    route = respx.get("https://api.anthropic.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    with TestClient(create_app(make_config())) as client:
        response = client.get("/v1/models", headers={"x-api-key": "test"})

    assert response.status_code == 200
    assert route.called


@respx.mock
def test_question_followed_by_system_reminder_is_routed():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json={}))

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={
                "model": "requested-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "what is python?"}, {"role": "system", "content": "<r>"}],
            },
        )

    assert json.loads(route.calls.last.request.content)["model"] == "small-model"


@respx.mock
def test_effort_400_is_learned_and_retried_without_effort():
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "This model does not support the effort parameter."}}
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[httpx.Response(400, json=error), httpx.Response(200, json={}), httpx.Response(200, json={})]
    )
    request_json = {
        "model": "x",
        "max_tokens": 100,
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": "what is python?"}],
    }

    with TestClient(create_app(make_config())) as client:
        first = client.post("/v1/messages", headers=HEADERS, json=request_json)
        client.post("/v1/messages", headers=HEADERS, json=request_json)

    assert first.status_code == 200
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert "output_config" in bodies[0]
    assert "output_config" not in bodies[1]  # retry
    assert "output_config" not in bodies[2]  # later request uses the learned capability, no extra 400


@pytest.fixture
def proxy_log(caplog):
    proxy_logger = logging.getLogger("wrap.proxy")
    proxy_logger.addHandler(caplog.handler)
    yield caplog
    proxy_logger.removeHandler(caplog.handler)
    proxy_logger.setLevel(logging.INFO)


def debug_lines(caplog, kind: str) -> list[str]:
    """Telemetry DEBUG lines of one kind ("turn" or "switch"), e.g. "[dim]#2 turn small/haiku ..."."""
    pattern = re.compile(rf"^\[dim\]#\d+ {kind} ")
    return [r.getMessage() for r in caplog.records if pattern.match(r.getMessage())]


USAGE_BODY = {
    "id": "msg_1",
    "model": "small-model",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 200, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0},
}


@respx.mock
def test_streamed_response_is_relayed_byte_for_byte_and_logged(tmp_path):
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, content=STREAM, headers={"content-type": "text/event-stream"})
    )

    with TestClient(create_app(make_config(tmp_path))) as client:
        response = client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "requested-model", "max_tokens": 100, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.content == STREAM
    [turn] = logged_turns(tmp_path)
    assert turn.stream
    assert turn.tier == "small"
    assert turn.requested_model == "requested-model"
    assert turn.model_id == "small-model"
    assert turn.served_model == "claude-haiku-4-5-20251001"
    assert turn.usage.output_tokens == 340
    assert turn.usage.cache_read_tokens == 8000
    assert turn.stop_reason == "end_turn"
    assert turn.error is None
    assert turn.latency_ms >= turn.ttfb_ms >= 0


@respx.mock
def test_non_streamed_response_is_logged_with_cost(tmp_path):
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "requested-model", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    [turn] = logged_turns(tmp_path)
    assert not turn.stream
    assert turn.status_code == 200
    assert turn.usage.input_tokens == 100
    assert turn.cost_usd == pytest.approx((100 * 1.0 + 200 * 5.0 + 1000 * 0.1) / 1_000_000)


@respx.mock
def test_upstream_error_is_logged(tmp_path):
    error = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(500, json=error))

    with TestClient(create_app(make_config(tmp_path))) as client:
        response = client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    [turn] = logged_turns(tmp_path)
    assert turn.status_code == 500
    assert "boom" in turn.error
    assert turn.cost_usd is None


@respx.mock
def test_retried_max_tokens_400_logs_only_the_retry(tmp_path):
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: 4096 > 2048, ..."}}
    respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[httpx.Response(400, json=error), httpx.Response(200, json=USAGE_BODY)]
    )

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 128000, "messages": [{"role": "user", "content": "what is python?"}]},
        )

    [turn] = logged_turns(tmp_path)
    assert turn.status_code == 200


@respx.mock
def test_mid_turn_request_is_logged_as_a_continuation_of_its_question(tmp_path):
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))
    messages = [
        {"role": "user", "content": "what is python?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]},
        *TOOL_RESULT_TURN,
    ]

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json={"model": "large-model", "max_tokens": 100, "messages": messages})

    [turn] = logged_turns(tmp_path)
    assert turn.was_tool_continuation
    assert turn.tier == "small"
    assert turn.complexity_score is not None


@respx.mock
def test_request_without_a_question_is_logged_as_unrouted(tmp_path):
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json={"model": "large-model", "max_tokens": 100, "messages": TOOL_RESULT_TURN})

    [turn] = logged_turns(tmp_path)
    assert turn.tier == "unrouted"
    assert turn.model_id == "large-model"
    assert turn.complexity_score is None


@respx.mock
def test_passthrough_endpoint_is_not_logged(tmp_path):
    respx.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 5})
    )

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post("/v1/messages/count_tokens", headers=HEADERS, json={"model": "x", "messages": []})

    assert logged_turns(tmp_path) == []


@respx.mock
def test_session_id_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WRAP_SESSION_ID", "abc123")
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    [turn] = logged_turns(tmp_path)
    assert turn.session_id == "abc123"


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_failing_telemetry_never_breaks_the_response(tmp_path, monkeypatch, stream):
    def broken_insert(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "insert_turn", broken_insert)
    upstream = (
        httpx.Response(200, content=STREAM, headers={"content-type": "text/event-stream"})
        if stream
        else httpx.Response(200, json=USAGE_BODY)
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=upstream)

    with TestClient(create_app(make_config(tmp_path))) as client:
        response = client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    if stream:
        assert response.content == STREAM
    else:
        assert response.json() == USAGE_BODY


@respx.mock
def test_turn_line_is_hidden_at_info_but_the_row_is_written(tmp_path, monkeypatch, proxy_log):
    monkeypatch.delenv("WRAP_LOG_LEVEL", raising=False)
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert len(logged_turns(tmp_path)) == 1
    assert debug_lines(proxy_log, "turn") == []


def test_thread_key_is_stable_across_a_conversation():
    system = [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=aaaa"},
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
    ]
    first = {"role": "user", "content": [{"type": "text", "text": "fix the bug", "cache_control": {"type": "ephemeral"}}]}
    later_system = [{**system[0], "text": "x-anthropic-billing-header: cc_version=1; cch=bbbb"}, {**system[1]}]
    later_system[1].pop("cache_control")
    later_first = {"role": "user", "content": [{"type": "text", "text": "fix the bug"}]}

    turn_1 = thread_key({"system": system, "messages": [first]})
    turn_2 = thread_key({"system": later_system, "messages": [later_first, {"role": "assistant", "content": "ok"}]})
    other = thread_key({"system": system, "messages": [{"role": "user", "content": "generate a title"}]})

    assert turn_1 == turn_2
    assert turn_1 != other


@respx.mock
def test_debug_level_logs_turn_and_switch_lines(tmp_path, monkeypatch, proxy_log):
    monkeypatch.setenv("WRAP_LOG_LEVEL", "debug")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            httpx.Response(200, json={**USAGE_BODY, "model": "large-model"}),
            httpx.Response(200, json={**USAGE_BODY, "model": "small-model"}),
        ]
    )
    first = {"role": "user", "content": "hi"}

    with TestClient(create_app(make_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json={"model": "large-model", "max_tokens": 100, "messages": TOOL_RESULT_TURN})
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "large-model", "max_tokens": 100, "messages": [*TOOL_RESULT_TURN, {"role": "assistant", "content": "ok"}, first]},
        )

    turns = debug_lines(proxy_log, "turn")
    assert len(turns) == 2
    assert turns[0].startswith("[dim]#1 turn in=")
    assert "in=100 out=200 cache_r=1.0k" in turns[1]
    [switch] = debug_lines(proxy_log, "switch")
    assert "large→small" in switch


@respx.mock
def test_config_log_level_applies_without_the_env_override(tmp_path, monkeypatch, proxy_log):
    monkeypatch.delenv("WRAP_LOG_LEVEL", raising=False)
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json=USAGE_BODY))
    config = make_config(tmp_path)
    config.proxy.log_level = "debug"

    with TestClient(create_app(config)) as client:
        client.post("/v1/messages", headers=HEADERS, json={"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})

    assert len(debug_lines(proxy_log, "turn")) == 1


@respx.mock
@pytest.mark.parametrize("level", ["info", "debug"])
def test_info_logs_one_routing_line_per_request_and_debug_adds_detail(monkeypatch, proxy_log, level):
    monkeypatch.setenv("WRAP_LOG_LEVEL", level)
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(200, json={"id": "msg_1"}))
    respx.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
        return_value=httpx.Response(200, json={"input_tokens": 5})
    )

    with TestClient(create_app(make_config())) as client:
        client.post(
            "/v1/messages",
            headers=HEADERS,
            json={"model": "x", "max_tokens": 128000, "messages": [{"role": "user", "content": "what is python?"}]},
        )
        client.post("/v1/messages/count_tokens", headers=HEADERS, json={"model": "x", "messages": []})

    messages = [r.getMessage() for r in proxy_log.records]
    assert messages[0].startswith("#1 small → small-model (score")
    details = ["#1 request: side", "#1 adapted: max_tokens 128000→4096", "#1 ← 200", "#2 passthrough POST /v1/messages/count_tokens"]
    for detail in details:
        assert any(detail in m for m in messages) == (level == "debug"), detail
    if level == "info":
        assert len(messages) == 1
