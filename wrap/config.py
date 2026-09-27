from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class TierConfig:
    """A single routing tier's model and the context window it supports.

    Attributes:
        model: The model ID to use when a turn is routed to this tier.
        context_window: The model's context window in tokens, used by
            `context_guard` to avoid routing a long conversation to a tier
            whose window can't hold it.
    """

    model: str
    context_window: int


@dataclass
class RoutingConfig:
    """Settings controlling how turns are classified and routed.

    Attributes:
        strategy: Name of the complexity classifier to use (currently only
            "heuristic" is implemented).
        complexity_threshold: Score at or above which a turn routes to the
            "large" tier instead of "small".
    """

    strategy: str
    complexity_threshold: float


@dataclass
class CacheConfig:
    """Settings for the semantic response cache (not yet implemented).

    Attributes:
        enabled: Whether the cache is active.
        similarity_threshold: Minimum cosine similarity for a cache hit.
        embedding_model: Name of the sentence-transformers model used to
            embed queries.
    """

    enabled: bool
    similarity_threshold: float
    embedding_model: str


@dataclass
class ProxyConfig:
    """Settings for the local HTTP proxy.

    Attributes:
        port: Port the proxy listens on.
        upstream_base_url: Base URL of the real Anthropic API the proxy
            forwards requests to.
    """

    port: int
    upstream_base_url: str


@dataclass
class TelemetryConfig:
    """Settings for cost/latency logging (not yet implemented).

    Attributes:
        db_path: Path to the SQLite database file used for telemetry.
        pricing_file: Path to the YAML file mapping model IDs to
            per-token pricing.
    """

    db_path: str
    pricing_file: str


@dataclass
class Config:
    """Top-level configuration for the proxy, assembled from `config.yaml`.

    Attributes:
        tiers: Mapping of tier name (e.g. "small", "large") to its
            `TierConfig`.
        routing: Complexity classification and routing settings.
        cache: Semantic cache settings.
        proxy: Local proxy server settings.
        telemetry: Cost/latency logging settings.
    """

    tiers: dict[str, TierConfig]
    routing: RoutingConfig
    cache: CacheConfig
    proxy: ProxyConfig
    telemetry: TelemetryConfig


def load_config(path: Path | None = None) -> Config:
    """Loads and parses the YAML config file into a `Config` object.

    Args:
        path: Path to the config YAML file. Defaults to
            `config/config.yaml` at the repo root.

    Returns:
        The parsed `Config`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If the YAML is missing one of the required top-level
            sections (`tiers`, `routing`, `cache`, `proxy`, `telemetry`).
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    tiers = {name: TierConfig(**data) for name, data in raw["tiers"].items()}
    return Config(
        tiers=tiers,
        routing=RoutingConfig(**raw["routing"]),
        cache=CacheConfig(**raw["cache"]),
        proxy=ProxyConfig(**raw["proxy"]),
        telemetry=TelemetryConfig(**raw["telemetry"]),
    )
