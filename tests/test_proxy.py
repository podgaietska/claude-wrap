import json

import httpx
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

HEADERS = {"x-api-key": "test", "anthropic-version": "2023-06-01"}
TOOL_RESULT_TURN = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "output"}]}]


def make_config() -> Config:
    """Builds a minimal `Config` fixture pointing at the real Anthropic
    upstream URL (mocked per-test via `respx`)."""
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
        telemetry=TelemetryConfig(db_path="data/wrap.db", pricing_file="config/pricing.yaml"),
    )


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
