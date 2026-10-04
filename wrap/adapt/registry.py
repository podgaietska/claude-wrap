from __future__ import annotations

import dataclasses

from wrap.config import ModelCapabilities


class CapabilityRegistry:
    """Per-model capabilities: configured values plus ones learned at runtime.

    Learned values come from the API rejecting a request, so they override
    the config for the rest of the session -- a stale config then costs one
    failed request per model and setting, not one per turn.
    """

    def __init__(self, configured: dict[str, ModelCapabilities]):
        """Builds a registry from the configured per-model capabilities.

        Args:
            configured: Mapping of model ID to its `ModelCapabilities` from config.
        """
        self._configured = configured
        self._learned: dict[str, dict] = {}

    def get(self, model: str) -> ModelCapabilities:
        """Returns the best-known capabilities for a model.

        Args:
            model: The model ID.

        Returns:
            The configured capabilities with learned values applied on top.
            Unknown models get an all-None `ModelCapabilities` (plus anything learned).
        """
        return dataclasses.replace(self.configured(model), **self._learned.get(model, {}))

    def configured(self, model: str) -> ModelCapabilities:
        """Returns the capabilities from config only, ignoring learned values.

        Args:
            model: The model ID.

        Returns:
            The configured capabilities, or an all-None `ModelCapabilities`.
        """
        return self._configured.get(model, ModelCapabilities())

    def learn(self, model: str, updates: dict) -> bool:
        """Records capability values reported by the API for the rest of the session.

        Args:
            model: The model ID the API rejected a request for.
            updates: Capability fields to override, e.g. {"effort": False}.

        Returns:
            True if this changed the model's capabilities, False if they were
            already known -- retrying wouldn't help in that case.
        """
        before = self.get(model)
        self._learned.setdefault(model, {}).update(updates)
        return self.get(model) != before
