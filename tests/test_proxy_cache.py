import dataclasses
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from tests.test_proxy import HEADERS, STREAM, USAGE_BODY, logged_turns, make_config, proxy_log  # noqa: F401
from wrap.cache.cache import ResponseCache
from wrap.proxy.server import create_app
from wrap.proxy.sse import StreamUsageParser

QUESTION = "what's the difference between a process and a thread?"
TOOLS = [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}]
STREAM_TEXT = "Python is a high-level programming language."
UPSTREAM = "https://api.anthropic.com/v1/messages"


@pytest.fixture(autouse=True)
def project_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WRAP_PROJECT_DIR", str(tmp_path / "project"))


def cached_config(tmp_path, **cache):
    config = make_config(tmp_path)
    return dataclasses.replace(config, cache=dataclasses.replace(config.cache, enabled=True, **cache))


def question(text=QUESTION, stream=True, history=None) -> dict:
    messages = [*(history or []), {"role": "user", "content": text}]
    return {
        "model": "requested-model",
        "max_tokens": 100,
        "stream": stream,
        "system": [{"type": "text", "text": "You are Claude Code."}],
        "tools": TOOLS,
        "messages": messages,
    }


def stream_upstream() -> httpx.Response:
    return httpx.Response(200, content=STREAM, headers={"content-type": "text/event-stream"})


def parse_stream(content: bytes) -> dict:
    parser = StreamUsageParser(collect_content=True)
    parser.feed(content)
    assert parser.completed and parser.error is None
    return parser.message


@respx.mock
def test_repeated_question_is_served_from_cache_without_an_upstream_call(tmp_path):
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        first = client.post("/v1/messages", headers=HEADERS, json=question())
    # A fresh proxy, as in a new `wrap claude` session.
    with TestClient(create_app(cached_config(tmp_path))) as client:
        second = client.post("/v1/messages", headers=HEADERS, json=question("What's the difference between a process and a thread"))

    assert route.call_count == 1
    assert first.content == STREAM
    assert second.status_code == 200
    assert second.headers["content-type"].startswith("text/event-stream")
    assert second.headers["x-wrap-cache"] == "hit"

    message = parse_stream(second.content)
    assert message["content"] == [{"type": "text", "text": STREAM_TEXT}]
    assert message["model"] == "claude-haiku-4-5-20251001"
    assert message["stop_reason"] == "end_turn"
    assert message["id"].startswith("msg_") and message["id"] != "msg_01"
    assert message["usage"]["output_tokens"] == 340
    assert message["usage"]["input_tokens"] > 0

    miss, hit = logged_turns(tmp_path)
    assert miss.cache_eligible and not miss.cache_hit
    assert miss.cache_miss_reason == "empty"
    assert miss.cache_entry_id is not None
    assert hit.cache_eligible and hit.cache_hit
    assert hit.cache_entry_id == miss.cache_entry_id
    assert hit.similarity_score == 1.0
    assert hit.served_model == "claude-haiku-4-5-20251001"
    assert hit.cost_usd == 0
    assert hit.usage.input_tokens == hit.usage.output_tokens == 0
    assert hit.tier == "small"
    assert hit.thread_key is not None


@respx.mock
def test_non_streamed_hit_is_a_json_message(tmp_path):
    answer = {**USAGE_BODY, "type": "message", "role": "assistant", "content": [{"type": "text", "text": "Threads share memory."}]}
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=answer))

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question(stream=False))
        response = client.post("/v1/messages", headers=HEADERS, json=question(stream=False))

    assert route.call_count == 1
    assert response.headers["content-type"].startswith("application/json")
    message = response.json()
    assert message["content"] == [{"type": "text", "text": "Threads share memory."}]
    assert message["model"] == "small-model"
    assert message["usage"]["output_tokens"] == 200


@respx.mock
def test_a_hit_works_both_ways_streamed_or_not(tmp_path):
    respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question(stream=True))
        response = client.post("/v1/messages", headers=HEADERS, json=question(stream=False))

    assert response.json()["content"] == [{"type": "text", "text": STREAM_TEXT}]


@respx.mock
def test_follow_up_after_a_hit_goes_upstream(tmp_path):
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question())
        hit = parse_stream(client.post("/v1/messages", headers=HEADERS, json=question()).content)
        history = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": hit["content"]}]
        follow_up = client.post("/v1/messages", headers=HEADERS, json=question("give an example in Python", history=history))

    assert route.call_count == 2
    assert follow_up.content == STREAM
    sent = json.loads(route.calls[-1].request.content)
    assert sent["messages"][1]["content"] == [{"type": "text", "text": STREAM_TEXT}]
    *_, last = logged_turns(tmp_path)
    assert not last.cache_eligible and last.cache_entry_id is None


@respx.mock
def test_other_projects_dont_get_the_answer(tmp_path, monkeypatch):
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question())
    monkeypatch.setenv("WRAP_PROJECT_DIR", str(tmp_path / "other"))
    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question())

    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize(
    "upstream",
    [
        # The answer came from reading the project.
        httpx.Response(200, json={**USAGE_BODY, "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x"}},
        ]}),
        # The stream never finished.
        httpx.Response(200, content=STREAM.split(b"event: message_stop")[0], headers={"content-type": "text/event-stream"}),
        httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}),
    ],
    ids=["tool_use", "incomplete_stream", "error"],
)
def test_unstorable_answers_are_not_cached(tmp_path, upstream):
    route = respx.post(UPSTREAM).mock(return_value=upstream)

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question())
        client.post("/v1/messages", headers=HEADERS, json=question())

    assert route.call_count == 2
    assert all(t.cache_entry_id is None and not t.cache_hit for t in logged_turns(tmp_path))


@respx.mock
def test_side_requests_are_not_cached(tmp_path):
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())
    body = {k: v for k, v in question().items() if k != "tools"}

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=body)
        client.post("/v1/messages", headers=HEADERS, json=body)

    assert route.call_count == 2
    assert not any(t.cache_eligible for t in logged_turns(tmp_path))


@respx.mock
@pytest.mark.parametrize("method", ["lookup", "insert"])
def test_a_failing_cache_never_breaks_the_request(tmp_path, monkeypatch, method):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ResponseCache, method, broken)
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        responses = [client.post("/v1/messages", headers=HEADERS, json=question()) for _ in range(2)]

    assert route.call_count == 2
    assert all(r.status_code == 200 and r.content == STREAM for r in responses)


@respx.mock
def test_disabled_cache_forwards_everything_and_logs_no_cache_fields(tmp_path):
    route = respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        responses = [client.post("/v1/messages", headers=HEADERS, json=question()) for _ in range(2)]

    assert app.state.cache is None
    assert route.call_count == 2
    assert all(r.content == STREAM and "x-wrap-cache" not in r.headers for r in responses)
    assert not any(t.cache_eligible or t.cache_miss_reason for t in logged_turns(tmp_path))


@respx.mock
def test_hit_logs_one_info_line_without_the_question(tmp_path, proxy_log):
    respx.post(UPSTREAM).mock(side_effect=lambda request: stream_upstream())

    with TestClient(create_app(cached_config(tmp_path))) as client:
        client.post("/v1/messages", headers=HEADERS, json=question())
        client.post("/v1/messages", headers=HEADERS, json=question())

    lines = [r.getMessage() for r in proxy_log.records]
    hits = [line for line in lines if "cache hit" in line]
    assert len(hits) == 1
    assert hits[0].startswith("#2 cache hit (exact, entry ")
    assert "claude-haiku-4-5-20251001" in hits[0]
    assert not any("process and a thread" in line for line in lines)
