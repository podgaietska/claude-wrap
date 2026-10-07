from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    """Token counts for one `/v1/messages` response.

    Attributes:
        input_tokens: Uncached input tokens (the part of the prompt after
            the last cache breakpoint), billed at the base input rate.
        output_tokens: Generated tokens.
        cache_read_tokens: Prompt tokens served from the prompt cache.
        cache_creation_tokens: Prompt tokens written to the prompt cache
            this turn, across both cache lifetimes.
        cache_creation_5m_tokens: The part of `cache_creation_tokens`
            written with the 5-minute TTL, when the API reports the split.
        cache_creation_1h_tokens: The part written with the 1-hour TTL.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """The full prompt size: uncached input plus cache reads and writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    def merge(self, api_usage: dict | None) -> None:
        """Overlays the non-null fields of an API `usage` object, in place.

        An overlay rather than a sum, because the `usage` in a stream's
        `message_delta` events is cumulative, not an increment.

        Args:
            api_usage: A Messages API `usage` object, or None.
        """
        if not isinstance(api_usage, dict):
            return
        for api_field, attr in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_creation_tokens"),
        ):
            value = api_usage.get(api_field)
            if isinstance(value, int):
                setattr(self, attr, value)

        breakdown = api_usage.get("cache_creation")
        if isinstance(breakdown, dict):
            for api_field, attr in (
                ("ephemeral_5m_input_tokens", "cache_creation_5m_tokens"),
                ("ephemeral_1h_input_tokens", "cache_creation_1h_tokens"),
            ):
                value = breakdown.get(api_field)
                if isinstance(value, int):
                    setattr(self, attr, value)
