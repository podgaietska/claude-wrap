import json

import httpx
import respx
from fastapi.testclient import TestClient

from wrap.config import CacheConfig, Config, ProxyConfig, RoutingConfig, TelemetryConfig, TierConfig
from wrap.proxy.server import create_app


def make_config() -> Config:
    """Builds a minimal `Config` fixture pointing at the real Anthropic
    upstream URL (mocked per-test via `respx`)."""
    return Config(
        tiers={
            "small": TierConfig(model="small-model", context_window=200000),
            "large": TierConfig(model="large-model", context_window=1000000),
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
def test_tool_continuation_passes_model_through_unchanged():
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
