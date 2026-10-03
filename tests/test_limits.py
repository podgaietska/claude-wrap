from wrap.config import ModelLimits
from wrap.proxy.limits import LimitRegistry, parse_max_tokens_ceiling


def make_registry() -> LimitRegistry:
    """Builds a registry with one known model."""
    return LimitRegistry({"known-model": ModelLimits(max_output_tokens=64000, context_window=200000)})


def test_clamp_reduces_oversized_request_to_configured_ceiling():
    assert make_registry().clamp("known-model", 128000) == 64000


def test_clamp_leaves_request_under_ceiling_unchanged():
    assert make_registry().clamp("known-model", 1000) == 1000


def test_clamp_leaves_unknown_model_unchanged():
    assert make_registry().clamp("unknown-model", 128000) == 128000


def test_clamp_leaves_missing_max_tokens_as_none():
    assert make_registry().clamp("known-model", None) is None


def test_learned_ceiling_overrides_configured_one():
    registry = make_registry()
    registry.learn("known-model", 32000)
    assert registry.max_output_tokens("known-model") == 32000
    assert registry.configured_max_output_tokens("known-model") == 64000
    assert registry.clamp("known-model", 128000) == 32000


def test_learned_ceiling_applies_to_unknown_model():
    registry = make_registry()
    registry.learn("new-model", 8000)
    assert registry.clamp("new-model", 128000) == 8000


def test_parse_ceiling_from_real_api_error_shape():
    body = (
        b'{"type":"error","error":{"type":"invalid_request_error","message":'
        b'"max_tokens: 128000 > 64000, which is the maximum allowed number of output tokens '
        b'for claude-haiku-4-5-20251001"}}'
    )
    assert parse_max_tokens_ceiling(body) == 64000


def test_parse_ceiling_returns_none_for_unrelated_error():
    body = b'{"type":"error","error":{"type":"invalid_request_error","message":"messages: field required"}}'
    assert parse_max_tokens_ceiling(body) is None
