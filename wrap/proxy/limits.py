from __future__ import annotations

import re

from wrap.config import ModelLimits

_MAX_TOKENS_ERROR = re.compile(rb"max_tokens:\s*\d+\s*>\s*(\d+)")


class LimitRegistry:
    """Per-model `max_tokens` ceilings: configured values plus ones learned at runtime.

    Learned values come from the API rejecting a request, so they take
    precedence over the config for the rest of the session -- that way a
    stale config only costs one failed request per model, not one per turn.
    """

    def __init__(self, configured: dict[str, ModelLimits]):
        """Builds a registry from the configured per-model limits.

        Args:
            configured: Mapping of model ID to its `ModelLimits` from config.
        """
        self._configured = configured
        self._learned: dict[str, int] = {}

    def max_output_tokens(self, model: str) -> int | None:
        """Returns the best-known `max_tokens` ceiling for a model.

        Args:
            model: The model ID.

        Returns:
            The learned ceiling if one exists, else the configured one, or
            None if the model is unknown.
        """
        if model in self._learned:
            return self._learned[model]
        limits = self._configured.get(model)
        return limits.max_output_tokens if limits else None

    def configured_max_output_tokens(self, model: str) -> int | None:
        """Returns the ceiling from config only, ignoring learned values.

        Args:
            model: The model ID.

        Returns:
            The configured ceiling, or None if the model isn't in config.
        """
        limits = self._configured.get(model)
        return limits.max_output_tokens if limits else None

    def learn(self, model: str, ceiling: int) -> None:
        """Records a ceiling reported by the API for the rest of the session.

        Args:
            model: The model ID the API rejected a request for.
            ceiling: The maximum `max_tokens` the API said it allows.
        """
        self._learned[model] = ceiling

    def clamp(self, model: str, requested: int | None) -> int | None:
        """Clamps a requested `max_tokens` to the model's best-known ceiling.

        Args:
            model: The model ID the request will be sent to.
            requested: The `max_tokens` from the request, or None if absent.

        Returns:
            The clamped value, or `requested` unchanged if it's None, already
            within the ceiling, or the model's ceiling is unknown.
        """
        ceiling = self.max_output_tokens(model)
        if requested is None or ceiling is None:
            return requested
        return min(requested, ceiling)


def parse_max_tokens_ceiling(error_body: bytes) -> int | None:
    """Extracts the allowed ceiling from a "max_tokens exceeds limit" API error.

    Args:
        error_body: Raw body of a 400 response from the Messages API, e.g.
            containing "max_tokens: 128000 > 64000, which is the maximum...".

    Returns:
        The allowed ceiling, or None if the body isn't that kind of error.
    """
    match = _MAX_TOKENS_ERROR.search(error_body)
    return int(match.group(1)) if match else None
