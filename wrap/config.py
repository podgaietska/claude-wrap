from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class TierConfig:
    """A single routing tier.

    Attributes:
        model: The model ID to use when a turn is routed to this tier.
    """

    model: str


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model supports, used to adapt requests built for a different model.

    Claude Code shapes every request for the model it thinks it's calling,
    so when the proxy swaps the model, unsupported settings must be adjusted
    or the API rejects the request. Every field is optional: None means
    unknown, and the request is left as-is for that setting.

    Attributes:
        max_output_tokens: The model's real `max_tokens` ceiling.
        context_window: The model's context window in tokens, used to avoid
            routing a long conversation to a model that can't hold it.
        effort: Whether `output_config.effort` is supported.
        thinking: Supported `thinking.type` values, e.g. ("adaptive",) or ("enabled",).
        thinking_budget: If set, adaptive thinking is converted to
            `{"type": "enabled", "budget_tokens": N}` for models that only
            support "enabled"; if None, thinking is dropped instead.
        mid_conversation_system: Whether `role: "system"` entries are allowed
            inside `messages`.
        context_edits: Supported `context_management.edits` types.
    """

    max_output_tokens: int | None = None
    context_window: int | None = None
    effort: bool | None = None
    thinking: tuple[str, ...] | None = None
    thinking_budget: int | None = None
    mid_conversation_system: bool | None = None
    context_edits: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ModelCapabilities:
        """Builds capabilities from a config entry, converting lists to tuples.

        Args:
            data: One model's entry from the `models` config section.

        Returns:
            The parsed `ModelCapabilities`.
        """
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


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
        log_path: Path (relative to the repo root) the proxy's log output
            is written to, so it doesn't interleave with the wrapped
            Claude Code session's own terminal output. Read live with
            `wrap logs`.
        log_level: "info" logs each routing decision; "debug" adds a
            line per turn with its tokens, cost and latency. `wrap claude
            --debug` overrides it for one session.
    """

    port: int
    upstream_base_url: str
    log_path: str
    log_level: str = "info"


@dataclass
class TelemetryConfig:
    """Settings for per-turn cost/latency logging.

    Attributes:
        db_path: Path (relative to the repo root) to the SQLite database
            file used for telemetry. Read with `wrap stats`.
        pricing_file: Path (relative to the repo root) to the YAML file
            mapping model IDs to per-token pricing.
        enabled: Whether the proxy records turns.
    """

    db_path: str
    pricing_file: str
    enabled: bool = True


@dataclass
class DashboardConfig:
    """Settings for `wrap dashboard`.

    Attributes:
        port: Port the dashboard listens on (always on 127.0.0.1).
        refresh_seconds: How often the page refreshes while a session is live.
    """

    port: int = 8788
    refresh_seconds: int = 5


@dataclass
class Config:
    """Top-level configuration for the proxy, assembled from `config.yaml`.

    Attributes:
        tiers: Mapping of tier name (e.g. "small", "large") to its
            `TierConfig`.
        models: Mapping of model ID to its known `ModelCapabilities`.
            Requests to models missing from this table aren't adapted
            up front (the 400 fallback still applies).
        routing: Complexity classification and routing settings.
        cache: Semantic cache settings.
        proxy: Local proxy server settings.
        telemetry: Cost/latency logging settings.
        dashboard: `wrap dashboard` settings.
    """

    tiers: dict[str, TierConfig]
    models: dict[str, ModelCapabilities]
    routing: RoutingConfig
    cache: CacheConfig
    proxy: ProxyConfig
    telemetry: TelemetryConfig
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


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
            sections (`tiers`, `models`, `routing`, `cache`, `proxy`,
            `telemetry`). `dashboard` is optional.
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    tiers = {name: TierConfig(**data) for name, data in raw["tiers"].items()}
    models = {model: ModelCapabilities.from_dict(data) for model, data in raw["models"].items()}
    return Config(
        tiers=tiers,
        models=models,
        routing=RoutingConfig(**raw["routing"]),
        cache=CacheConfig(**raw["cache"]),
        proxy=ProxyConfig(**raw["proxy"]),
        telemetry=TelemetryConfig(**raw["telemetry"]),
        dashboard=DashboardConfig(**(raw.get("dashboard") or {})),
    )
