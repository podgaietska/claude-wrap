from __future__ import annotations

_CHARS_PER_TOKEN = 4  # rough heuristic, no tokenizer dependency needed


def estimate_tokens(messages: list[dict]) -> int:
    """Estimates the total token count of a conversation from character length.

    This is a cheap heuristic (chars / 4), not a real tokenizer -- it's
    only meant to gate routing decisions, not for billing or accurate
    token accounting.

    Args:
        messages: The Messages API `messages` array. Each message's
            `content` may be a plain string or a list of content blocks
            (e.g. text or tool_result blocks); text is summed from
            whichever shape is present.

    Returns:
        Estimated total token count across all messages.
    """
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
    """Checks whether a conversation safely fits a tier's context window.

    Args:
        messages: The Messages API `messages` array to estimate the size of.
        context_window: The candidate tier's context window, in tokens.
        safety_margin: Fraction of `context_window` the conversation is
            allowed to occupy, reserving the rest for the response and for
            error in the token estimate. Defaults to 0.8 (80%).

    Returns:
        True if the estimated conversation size is under
        `context_window * safety_margin`, False otherwise.
    """
    return estimate_tokens(messages) < context_window * safety_margin
