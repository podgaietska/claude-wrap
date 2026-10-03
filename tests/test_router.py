from wrap.config import CacheConfig, Config, ProxyConfig, RoutingConfig, TelemetryConfig, TierConfig
from wrap.routing.context_guard import estimate_tokens, fits_in_window
from wrap.routing.heuristic import HeuristicClassifier
from wrap.routing.router import Router, extract_newest_human_text


def make_config() -> Config:
    """Builds a minimal `Config` fixture for router/heuristic tests."""
    return Config(
        tiers={
            "small": TierConfig(model="small-model", context_window=1000),
            "large": TierConfig(model="large-model", context_window=100000),
        },
        routing=RoutingConfig(strategy="heuristic", complexity_threshold=0.5),
        cache=CacheConfig(enabled=False, similarity_threshold=0.92, embedding_model="x"),
        proxy=ProxyConfig(port=8787, upstream_base_url="https://api.anthropic.com", log_path="data/proxy.log"),
        telemetry=TelemetryConfig(db_path="data/wrap.db", pricing_file="config/pricing.yaml"),
    )


def test_heuristic_trivial_question_routes_small():
    result = HeuristicClassifier().classify("what is python?", threshold=0.5)
    assert result.tier == "small"
    assert result.score < 0.5


def test_heuristic_complex_question_routes_large():
    text = (
        "Please explain step by step how you would refactor this architecture, "
        "compare the trade-offs between microservices and a monolith, and analyze "
        "why the current design has issues."
    )
    result = HeuristicClassifier().classify(text, threshold=0.5)
    assert result.tier == "large"
    assert result.score >= 0.5


def test_heuristic_code_indicators_push_toward_large():
    text = "```python\ndef foo():\n    pass\n```\nwhy does this raise a TypeError?"
    result = HeuristicClassifier().classify(text, threshold=0.5)
    assert "code indicators" in result.reasoning


def test_extract_newest_human_text_plain_string():
    messages = [{"role": "user", "content": "hello there"}]
    assert extract_newest_human_text(messages) == "hello there"


def test_extract_newest_human_text_tool_result_is_continuation():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "read_file"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "file contents"}]},
    ]
    assert extract_newest_human_text(messages) is None


def test_extract_newest_human_text_ignores_assistant_last_turn():
    messages = [{"role": "assistant", "content": "the answer"}]
    assert extract_newest_human_text(messages) is None


def test_router_routes_tool_continuation_unchanged():
    router = Router(make_config())
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "output"}]},
    ]
    decision = router.route(messages, requested_model="whatever-model")
    assert decision.is_tool_continuation is True
    assert decision.model == "whatever-model"


def test_router_routes_trivial_to_small():
    router = Router(make_config())
    decision = router.route([{"role": "user", "content": "what is python?"}], requested_model="ignored")
    assert decision.tier == "small"
    assert decision.model == "small-model"


def test_router_forces_large_when_history_exceeds_small_context_window():
    router = Router(make_config())
    huge_history = [{"role": "user", "content": "x" * 20000}]  # ~5000 tokens > small's 1000-token window
    decision = router.route(huge_history, requested_model="ignored")
    assert decision.tier == "large"
    assert decision.model == "large-model"


def test_estimate_tokens_counts_string_and_block_content():
    messages = [
        {"role": "user", "content": "a" * 40},
        {"role": "assistant", "content": [{"type": "text", "text": "b" * 40}]},
    ]
    assert estimate_tokens(messages) == 20  # 80 chars / 4


def test_fits_in_window_respects_safety_margin():
    messages = [{"role": "user", "content": "a" * 400}]  # 100 tokens
    assert fits_in_window(messages, context_window=1000, safety_margin=0.8) is True
    assert fits_in_window(messages, context_window=100, safety_margin=0.8) is False
