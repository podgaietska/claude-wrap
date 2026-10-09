from wrap.config import (
    CacheConfig,
    Config,
    ModelCapabilities,
    ProxyConfig,
    RoutingConfig,
    TelemetryConfig,
    TierConfig,
)
from wrap.routing.context_guard import estimate_tokens, fits_in_window
from wrap.routing.heuristic import HeuristicClassifier
from wrap.routing.router import Router, find_turn, human_text

COMPLEX_TEXT = (
    "Please explain step by step how you would refactor this architecture, "
    "compare the trade-offs between microservices and a monolith, and analyze "
    "why the current design has issues."
)
SYSTEM_REMINDER = {"role": "system", "content": "<reminder>"}
TOOL_USE = {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]}
TOOL_RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "file contents"}]}


def make_config(models: dict[str, ModelCapabilities] | None = None) -> Config:
    """Builds a minimal `Config` fixture for router/heuristic tests."""
    if models is None:
        models = {
            "small-model": ModelCapabilities(max_output_tokens=4096, context_window=1000),
            "large-model": ModelCapabilities(max_output_tokens=64000, context_window=100000),
        }
    return Config(
        tiers={"small": TierConfig(model="small-model"), "large": TierConfig(model="large-model")},
        models=models,
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
    result = HeuristicClassifier().classify(COMPLEX_TEXT, threshold=0.5)
    assert result.tier == "large"
    assert result.score >= 0.5


def test_heuristic_code_indicators_push_toward_large():
    text = "```python\ndef foo():\n    pass\n```\nwhy does this raise a TypeError?"
    result = HeuristicClassifier().classify(text, threshold=0.5)
    assert "code indicators" in result.reasoning


def test_human_text_reads_string_and_text_blocks():
    assert human_text({"role": "user", "content": "hello"}) == "hello"
    assert human_text({"role": "user", "content": [{"type": "text", "text": "hi"}]}) == "hi"


def test_human_text_skips_injected_system_reminder_blocks():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "<system-reminder>\nCLAUDE.md contents, explain the architecture..."},
            {"type": "text", "text": "<system-reminder>\nmore context</system-reminder>"},
            {"type": "text", "text": "what is 2+2?"},
        ],
    }
    assert human_text(message) == "what is 2+2?"


def test_reminder_only_message_is_not_a_question():
    assert human_text({"role": "user", "content": "<system-reminder>\ncontext"}) is None
    turn = find_turn(
        [
            {"role": "user", "content": "the real question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nctx"}]},
        ]
    )
    assert turn.text == "the real question"


def test_human_text_ignores_tool_results_and_other_roles():
    assert human_text(TOOL_RESULT) is None
    assert human_text({"role": "assistant", "content": "answer"}) is None
    assert human_text(SYSTEM_REMINDER) is None


def test_find_turn_skips_trailing_system_reminder():
    turn = find_turn([{"role": "user", "content": "what is 2+2?"}, SYSTEM_REMINDER])
    assert turn.text == "what is 2+2?"
    assert turn.index == 0
    assert turn.is_continuation is False


def test_find_turn_walks_back_past_tool_calls_to_the_question():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "what's in config.yaml?"},
        TOOL_USE,
        TOOL_RESULT,
        SYSTEM_REMINDER,
    ]
    turn = find_turn(messages)
    assert turn.text == "what's in config.yaml?"
    assert turn.index == 2
    assert turn.is_continuation is True


def test_find_turn_returns_none_without_human_text():
    assert find_turn([TOOL_RESULT]) is None
    assert find_turn([]) is None


def test_router_routes_question_ending_with_system_reminder():
    router = Router(make_config())
    decision = router.route([{"role": "user", "content": COMPLEX_TEXT}, SYSTEM_REMINDER], requested_model="opus")
    assert decision.routed is True
    assert decision.model == "large-model"


def test_router_routes_tool_continuations_by_the_turns_question():
    router = Router(make_config())
    messages = [{"role": "user", "content": "what is python?"}, TOOL_USE, TOOL_RESULT]
    decision = router.route(messages, requested_model="opus")
    assert decision.routed is True
    assert decision.model == "small-model"
    assert decision.turn.is_continuation is True


def test_router_leaves_model_alone_without_a_question():
    decision = Router(make_config()).route([TOOL_RESULT], requested_model="opus")
    assert decision.routed is False
    assert decision.model == "opus"


def test_router_forces_large_when_history_exceeds_small_context_window():
    router = Router(make_config())
    huge_history = [{"role": "user", "content": "x" * 20000}]  # ~5000 tokens > small's 1000-token window
    decision = router.route(huge_history, requested_model="ignored")
    assert decision.tier == "large"
    assert decision.model == "large-model"


def test_router_skips_context_check_when_small_model_window_unknown():
    router = Router(make_config(models={}))
    decision = router.route([{"role": "user", "content": "x" * 20000}], requested_model="ignored")
    assert decision.tier == "small"


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
