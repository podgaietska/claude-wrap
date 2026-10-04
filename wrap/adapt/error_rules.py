"""Rules that turn a known 400 error into a capability correction.

When the API rejects a request because the configured capabilities were
wrong (or have changed), the matching rule says what the model actually
supports, so the proxy can learn it, re-adapt, and retry. Only add rules for
error messages that have actually been observed or are documented verbatim.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorRule:
    """A known error message and the capability update it implies.

    Attributes:
        name: Short label for logs.
        pattern: Regex matched against the raw error body.
        updates: Builds the capability fields to learn from the regex match.
    """

    name: str
    pattern: re.Pattern[bytes]
    updates: Callable[[re.Match[bytes]], dict]


RULES: list[ErrorRule] = [
    ErrorRule(
        name="max_tokens over limit",
        pattern=re.compile(rb"max_tokens:\s*\d+\s*>\s*(\d+)"),
        updates=lambda m: {"max_output_tokens": int(m.group(1))},
    ),
    ErrorRule(
        name="effort unsupported",
        pattern=re.compile(rb"does not support the effort parameter"),
        updates=lambda m: {"effort": False},
    ),
    ErrorRule(
        name="mid-conversation system messages unsupported",
        pattern=re.compile(rb"role 'system' is not supported on this model"),
        updates=lambda m: {"mid_conversation_system": False},
    ),
]


def match_error(error_body: bytes) -> tuple[ErrorRule, dict] | None:
    """Finds the rule matching an API error body.

    Args:
        error_body: Raw body of a 400 response from the Messages API.

    Returns:
        The matching rule and the capability updates it implies, or None if
        the error isn't one we know how to correct.
    """
    for rule in RULES:
        match = rule.pattern.search(error_body)
        if match:
            return rule, rule.updates(match)
    return None
