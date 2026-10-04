import copy

from wrap.adapt.adapters import (
    adapt_context_management,
    adapt_effort,
    adapt_request,
    adapt_system_messages,
    adapt_thinking,
    clamp_max_tokens,
)
from wrap.adapt.error_rules import match_error
from wrap.adapt.registry import CapabilityRegistry
from wrap.config import ModelCapabilities

HAIKU = ModelCapabilities(
    max_output_tokens=64000,
    context_window=200000,
    effort=False,
    thinking=("enabled",),
    mid_conversation_system=False,
    context_edits=("clear_tool_uses_20250919", "clear_thinking_20251015"),
)
OPUS = ModelCapabilities(
    max_output_tokens=128000,
    context_window=1000000,
    effort=True,
    thinking=("adaptive",),
    mid_conversation_system=True,
    context_edits=("clear_tool_uses_20250919", "clear_thinking_20251015", "compact_20260112"),
)


def claude_code_main_request() -> dict:
    """Mirrors the shape of a real Claude Code main request (captured via probe)."""
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 128000,
        "stream": True,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "medium"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "system": [{"type": "text", "text": "instructions"}],
        "tools": [{"name": "Read", "description": "d", "input_schema": {}}],
        "messages": [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "system", "content": [{"type": "text", "text": "<reminder>"}]},
        ],
    }


def test_request_built_for_opus_is_untouched_when_sent_to_opus():
    body = claude_code_main_request()
    original = copy.deepcopy(body)
    assert adapt_request(body, OPUS) == []
    assert body == original


def test_unknown_capabilities_leave_request_untouched():
    body = claude_code_main_request()
    original = copy.deepcopy(body)
    assert adapt_request(body, ModelCapabilities()) == []
    assert body == original


def test_full_pipeline_adapts_opus_request_for_haiku():
    body = claude_code_main_request()
    notes = adapt_request(body, HAIKU)

    assert body["max_tokens"] == 64000
    assert "thinking" not in body
    assert "output_config" not in body
    assert "context_management" not in body
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "what is 2+2?"},
        {"type": "text", "text": "<reminder>"},
    ]
    assert len(notes) == 5


def test_clamp_max_tokens_only_reduces():
    body = {"max_tokens": 1000}
    assert clamp_max_tokens(body, HAIKU) == []
    assert body["max_tokens"] == 1000


def test_adaptive_thinking_converted_to_budget_when_configured():
    caps = ModelCapabilities(thinking=("enabled",), thinking_budget=4000)
    body = {"max_tokens": 64000, "thinking": {"type": "adaptive", "display": "omitted"}}
    adapt_thinking(body, caps)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4000}


def test_thinking_budget_stays_below_max_tokens():
    caps = ModelCapabilities(thinking=("enabled",), thinking_budget=4000)
    body = {"max_tokens": 2048, "thinking": {"type": "adaptive"}}
    adapt_thinking(body, caps)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2047}


def test_thinking_dropped_when_budget_would_be_too_small():
    caps = ModelCapabilities(thinking=("enabled",), thinking_budget=4000)
    body = {"max_tokens": 500, "thinking": {"type": "adaptive"}}
    adapt_thinking(body, caps)
    assert "thinking" not in body


def test_budgeted_thinking_converted_to_adaptive_for_adaptive_only_model():
    body = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    assert adapt_thinking(body, OPUS) == ["thinking enabled→adaptive"]
    assert body["thinking"] == {"type": "adaptive"}


def test_effort_removed_but_other_output_config_kept():
    body = {"output_config": {"effort": "high", "format": {"type": "json_schema"}}}
    adapt_effort(body, HAIKU)
    assert body["output_config"] == {"format": {"type": "json_schema"}}


def test_unsupported_context_edit_dropped_and_supported_kept():
    body = {
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}]},
    }
    notes = adapt_context_management(body, HAIKU)
    assert body["context_management"]["edits"] == [{"type": "clear_tool_uses_20250919"}]
    assert notes == ["dropped context edit compact_20260112 (unsupported)"]


def test_clear_thinking_edit_dropped_when_thinking_off():
    body = {"context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}
    adapt_context_management(body, OPUS)
    assert "context_management" not in body


def test_system_message_appended_after_tool_results():
    body = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "out"}]},
            {"role": "system", "content": "<reminder>"},
        ]
    }
    adapt_system_messages(body, HAIKU)
    last = body["messages"][-1]
    assert last["role"] == "user"
    assert [b["type"] for b in last["content"]] == ["tool_result", "text"]


def test_system_message_after_assistant_folded_into_next_user_turn():
    body = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "<reminder>"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "out"}]},
        ]
    }
    adapt_system_messages(body, HAIKU)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert [b["type"] for b in body["messages"][-1]["content"]] == ["tool_result", "text"]


def test_registry_learned_values_override_config():
    registry = CapabilityRegistry({"haiku": HAIKU})
    assert registry.learn("haiku", {"max_output_tokens": 32000}) is True
    assert registry.get("haiku").max_output_tokens == 32000
    assert registry.configured("haiku").max_output_tokens == 64000


def test_registry_learn_reports_no_change_when_already_known():
    registry = CapabilityRegistry({"haiku": HAIKU})
    assert registry.learn("haiku", {"effort": False}) is False


def test_registry_unknown_model_gets_learned_values():
    registry = CapabilityRegistry({})
    registry.learn("new-model", {"effort": False})
    assert registry.get("new-model").effort is False
    assert registry.get("new-model").max_output_tokens is None


def test_error_rules_match_known_messages():
    assert match_error(b'"max_tokens: 128000 > 64000, which is the maximum"')[1] == {"max_output_tokens": 64000}
    assert match_error(b'"This model does not support the effort parameter."')[1] == {"effort": False}
    assert match_error(b"role 'system' is not supported on this model")[1] == {"mid_conversation_system": False}


def test_error_rules_ignore_unknown_errors():
    assert match_error(b'"messages: field required"') is None
