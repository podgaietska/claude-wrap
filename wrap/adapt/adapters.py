"""Adapters that reshape a request built for one model so another model accepts it.

Each adapter handles one setting: it mutates the request body in place,
leaves it alone if the target model supports what was sent (or the
capability is unknown), and returns human-readable notes on what changed.
To support a new incompatibility, add a capability field, an adapter, and
append it to `ADAPTERS`.
"""

from __future__ import annotations

from collections.abc import Callable

from wrap.config import ModelCapabilities

Adapter = Callable[[dict, ModelCapabilities], list[str]]

_MIN_THINKING_BUDGET = 1024
_CLEAR_THINKING = "clear_thinking_20251015"


def clamp_max_tokens(body: dict, caps: ModelCapabilities) -> list[str]:
    """Clamps `max_tokens` to the model's output ceiling.

    Args:
        body: The request body, modified in place.
        caps: The target model's capabilities.

    Returns:
        Notes describing the change, if any.
    """
    requested = body.get("max_tokens")
    if requested is None or caps.max_output_tokens is None or requested <= caps.max_output_tokens:
        return []
    body["max_tokens"] = caps.max_output_tokens
    return [f"max_tokens {requested}→{caps.max_output_tokens}"]


def adapt_thinking(body: dict, caps: ModelCapabilities) -> list[str]:
    """Converts or drops a `thinking` config the model doesn't support.

    Adaptive thinking becomes a fixed budget if `caps.thinking_budget` is set
    and the model supports "enabled", otherwise it's dropped. A budgeted
    config becomes adaptive for adaptive-only models.

    Args:
        body: The request body, modified in place.
        caps: The target model's capabilities.

    Returns:
        Notes describing the change, if any.
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict) or caps.thinking is None:
        return []
    kind = thinking.get("type")
    if kind in caps.thinking or kind == "disabled":
        return []

    if kind == "adaptive" and "enabled" in caps.thinking and caps.thinking_budget:
        budget = min(caps.thinking_budget, body.get("max_tokens", caps.thinking_budget + 1) - 1)
        if budget >= _MIN_THINKING_BUDGET:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return [f"thinking adaptive→enabled({budget})"]

    if kind == "enabled" and "adaptive" in caps.thinking:
        body["thinking"] = {"type": "adaptive"}
        return ["thinking enabled→adaptive"]

    del body["thinking"]
    return [f"dropped thinking ({kind} unsupported)"]


def adapt_effort(body: dict, caps: ModelCapabilities) -> list[str]:
    """Removes `output_config.effort` for models that don't support it.

    Args:
        body: The request body, modified in place.
        caps: The target model's capabilities.

    Returns:
        Notes describing the change, if any.
    """
    output_config = body.get("output_config")
    if caps.effort is not False or not isinstance(output_config, dict) or "effort" not in output_config:
        return []
    effort = output_config.pop("effort")
    if not output_config:
        del body["output_config"]
    return [f"dropped effort={effort}"]


def adapt_context_management(body: dict, caps: ModelCapabilities) -> list[str]:
    """Removes context-management edits the model doesn't support.

    Also removes the clear-thinking edit when thinking is off (e.g. dropped
    by `adapt_thinking`), since there are no thinking blocks to manage.

    Args:
        body: The request body, modified in place. Must run after `adapt_thinking`.
        caps: The target model's capabilities.

    Returns:
        Notes describing the change, if any.
    """
    context = body.get("context_management")
    if not isinstance(context, dict) or not isinstance(context.get("edits"), list):
        return []

    notes, kept = [], []
    for edit in context["edits"]:
        kind = edit.get("type") if isinstance(edit, dict) else None
        if caps.context_edits is not None and kind not in caps.context_edits:
            notes.append(f"dropped context edit {kind} (unsupported)")
        elif kind == _CLEAR_THINKING and "thinking" not in body:
            notes.append(f"dropped context edit {kind} (thinking off)")
        else:
            kept.append(edit)

    if notes:
        context["edits"] = kept
        if not kept:
            del body["context_management"]
    return notes


def adapt_system_messages(body: dict, caps: ModelCapabilities) -> list[str]:
    """Folds mid-conversation `role: "system"` messages into the adjacent user turn.

    The text is appended to the preceding user message (or, failing that, the
    next one) so the instructions still reach the model; a system message
    with no neighbouring user message is dropped.

    Args:
        body: The request body, modified in place.
        caps: The target model's capabilities.

    Returns:
        Notes describing the change, if any.
    """
    messages = body.get("messages")
    if caps.mid_conversation_system is not False or not isinstance(messages, list):
        return []
    if not any(m.get("role") == "system" for m in messages):
        return []

    folded = dropped = 0
    result: list[dict] = []
    pending: list[dict] = []  # system text blocks waiting for the next user message
    for message in messages:
        if message.get("role") == "system":
            blocks = _text_blocks(message.get("content"))
            if result and result[-1].get("role") == "user":
                result[-1]["content"] = _text_blocks(result[-1].get("content"), keep_all=True) + blocks
                folded += 1
            else:
                pending.extend(blocks)
            continue
        if pending and message.get("role") == "user":
            # Appended, not prepended: tool_result blocks must come first in a user message.
            message["content"] = _text_blocks(message.get("content"), keep_all=True) + pending
            folded += 1
            pending = []
        result.append(message)
    if pending:
        dropped += 1

    body["messages"] = result
    notes = []
    if folded:
        notes.append(f"folded {folded} system message(s) into user turns")
    if dropped:
        notes.append(f"dropped {dropped} system message(s) with no adjacent user turn")
    return notes


def _text_blocks(content, keep_all: bool = False) -> list[dict]:
    """Normalizes message content into a list of content blocks.

    Args:
        content: A message's `content`: a string or a list of blocks.
        keep_all: Keep every block (for user messages); otherwise only text
            blocks are kept (for system messages being folded in).

    Returns:
        A list of content blocks.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and (keep_all or b.get("type") == "text")]


# Order matters: thinking must be adapted before context management, which
# checks whether thinking is still on.
ADAPTERS: list[Adapter] = [
    clamp_max_tokens,
    adapt_thinking,
    adapt_effort,
    adapt_context_management,
    adapt_system_messages,
]


def adapt_request(body: dict, caps: ModelCapabilities) -> list[str]:
    """Runs every adapter on a request body for the target model.

    Args:
        body: The request body, modified in place.
        caps: The target model's capabilities.

    Returns:
        Notes from all adapters describing what changed.
    """
    notes: list[str] = []
    for adapter in ADAPTERS:
        notes.extend(adapter(body, caps))
    return notes
