from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from wrap import paths
from wrap.config import TelemetryConfig, merge, read_yaml
from wrap.telemetry.usage import Usage

PRICING_FILE = "pricing.yaml"

logger = logging.getLogger("wrap.proxy")

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    """A model's rates, in US dollars per million tokens.

    Attributes:
        input: Uncached input tokens.
        output: Output tokens.
        cache_read: Prompt tokens read from the cache.
        cache_write_5m: Prompt tokens written to the cache with the 5-minute TTL.
        cache_write_1h: Prompt tokens written to the cache with the 1-hour TTL.
    """

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    def cost(self, usage: Usage) -> float:
        """Prices token counts at these rates.

        Cache writes are priced with the 5m/1h split when the API reported
        it; otherwise all of `cache_creation_tokens` is priced at the 5m rate.

        Args:
            usage: The token counts to price.

        Returns:
            The cost in US dollars.
        """
        write_5m = usage.cache_creation_5m_tokens
        write_1h = usage.cache_creation_1h_tokens
        if write_5m + write_1h == 0:
            write_5m = usage.cache_creation_tokens

        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + write_5m * self.cache_write_5m
            + write_1h * self.cache_write_1h
        ) / _PER_MILLION


class PricingTable:
    """Maps model IDs to `ModelPricing` and prices `Usage` with it."""

    def __init__(self, models: dict[str, ModelPricing]):
        """Builds a table from already-parsed rates.

        Args:
            models: Mapping of model ID (or ID prefix) to its rates.
        """
        self.models = models
        self._warned: set[str] = set()

    @classmethod
    def load(cls, *files: Path) -> PricingTable:
        """Loads pricing YAML files with a top-level `models` mapping, each merged over the last.

        Args:
            *files: Paths to the pricing files; a later file's entries
                override an earlier one's, field by field.

        Returns:
            The parsed `PricingTable`.
        """
        raw: dict = {}
        for path in files:
            raw = merge(raw, read_yaml(path))
        return cls({model: ModelPricing(**rates) for model, rates in raw.get("models", {}).items()})

    def lookup(self, model: str | None) -> ModelPricing | None:
        """Finds a model's rates: exact ID first, then the longest matching prefix.

        The prefix match lets a dated ID like `claude-haiku-4-5-20251001`
        resolve to a `claude-haiku-4-5` entry.

        Args:
            model: The model ID to look up.

        Returns:
            The rates, or None if no entry matches.
        """
        if not model:
            return None
        if model in self.models:
            return self.models[model]
        prefixes = [key for key in self.models if model.startswith(key)]
        return self.models[max(prefixes, key=len)] if prefixes else None

    def cost(self, model: str | None, usage: Usage) -> float | None:
        """Prices one turn's usage at its model's rates (see `ModelPricing.cost`).

        Args:
            model: The model that served the turn.
            usage: The turn's token counts.

        Returns:
            The cost in US dollars, or None for a model with no rates
            (logged once per model).
        """
        rates = self.lookup(model)
        if rates is None:
            if model not in self._warned:
                self._warned.add(model)
                logger.warning("[yellow]no pricing for %s -- add it to the pricing file[/yellow]", model)
            return None
        return rates.cost(usage)


def user_pricing_path(telemetry: TelemetryConfig) -> Path:
    """The pricing file merged over the packaged prices (it may not exist).

    Args:
        telemetry: The telemetry config; its `pricing_file`, if set.

    Returns:
        `telemetry.pricing_file`, else `pricing.yaml` in the config directory.
    """
    return Path(telemetry.pricing_file) if telemetry.pricing_file else paths.config_dir() / PRICING_FILE


def load_pricing(telemetry: TelemetryConfig) -> PricingTable:
    """Loads the packaged prices with the user's pricing file merged over them.

    Args:
        telemetry: The telemetry config.

    Returns:
        The merged `PricingTable`.

    Raises:
        FileNotFoundError: If `telemetry.pricing_file` is set but doesn't exist.
    """
    user = user_pricing_path(telemetry)
    files = [paths.DEFAULTS_DIR / PRICING_FILE]
    if telemetry.pricing_file or user.exists():
        files.append(user)
    return PricingTable.load(*files)
