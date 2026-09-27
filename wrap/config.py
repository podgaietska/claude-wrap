from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class TierConfig:
    model: str
    context_window: int


@dataclass
class RoutingConfig:
    strategy: str
    complexity_threshold: float


@dataclass
class CacheConfig:
    enabled: bool
    similarity_threshold: float
    embedding_model: str


@dataclass
class ProxyConfig:
    port: int
    upstream_base_url: str


@dataclass
class TelemetryConfig:
    db_path: str
    pricing_file: str


@dataclass
class Config:
    tiers: dict[str, TierConfig]
    routing: RoutingConfig
    cache: CacheConfig
    proxy: ProxyConfig
    telemetry: TelemetryConfig


def load_config(path: Path | None = None) -> Config:
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
