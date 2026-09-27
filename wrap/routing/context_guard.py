from __future__ import annotations

_CHARS_PER_TOKEN = 4  # rough heuristic, no tokenizer dependency needed


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for the whole conversation, used only as a safety
    check before routing to a smaller-context tier -- not for billing."""
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if isinstance(text, str):
                        total_chars += len(text)
    return total_chars // _CHARS_PER_TOKEN


def fits_in_window(messages: list[dict], context_window: int, safety_margin: float = 0.8) -> bool:
    """Whether the estimated conversation size leaves enough room in the
    given tier's context window, reserving `safety_margin` fraction for the
    response and estimation error."""
    return estimate_tokens(messages) < context_window * safety_margin
