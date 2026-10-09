from pathlib import Path

import pytest

from wrap import paths
from wrap.config import load_config, merge
from wrap.telemetry.pricing import PricingTable, load_pricing


def test_merge_is_deep_and_replaces_non_mappings():
    base = {"tiers": {"small": {"model": "a"}, "large": {"model": "b"}}, "edits": [1, 2], "port": 1}
    override = {"tiers": {"large": {"model": "c"}}, "edits": [3], "port": 2}

    assert merge(base, override) == {
        "tiers": {"small": {"model": "a"}, "large": {"model": "c"}},
        "edits": [3],
        "port": 2,
    }
    assert base["tiers"]["large"] == {"model": "b"}


def test_defaults_load_without_a_user_config(isolated_dirs):
    _, data_dir = isolated_dirs

    config = load_config()

    assert {"small", "large"} <= config.tiers.keys()
    assert config.telemetry.db_path == str(data_dir / "wrap.db")
    assert config.proxy.log_path == str(data_dir / "proxy.log")
    assert config.telemetry.pricing_file is None


def test_user_config_overrides_only_what_it_sets(isolated_dirs):
    config_dir, _ = isolated_dirs
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "tiers:\n  large:\n    model: my-model\nrouting:\n  complexity_threshold: 0.7\n"
    )
    defaults = load_config(paths.DEFAULTS_DIR / "config.yaml")

    config = load_config()

    assert config.tiers["large"].model == "my-model"
    assert config.tiers["small"] == defaults.tiers["small"]
    assert config.routing.complexity_threshold == 0.7
    assert config.models == defaults.models


def test_an_all_comment_user_config_changes_nothing(isolated_dirs):
    config_dir, _ = isolated_dirs
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("# tiers:\n#   large:\n#     model: x\n")

    assert load_config() == load_config(paths.DEFAULTS_DIR / "config.yaml")


def test_user_paths_resolve_against_their_directories(isolated_dirs, tmp_path):
    config_dir, data_dir = isolated_dirs
    config_dir.mkdir()
    absolute = tmp_path / "elsewhere" / "wrap.db"
    (config_dir / "config.yaml").write_text(
        f"telemetry:\n  db_path: {absolute}\n  pricing_file: mine.yaml\nproxy:\n  log_path: logs/proxy.log\n"
    )

    config = load_config()

    assert config.telemetry.db_path == str(absolute)
    assert config.telemetry.pricing_file == str(config_dir / "mine.yaml")
    assert config.proxy.log_path == str(data_dir / "logs" / "proxy.log")


def test_xdg_directories_are_used_without_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("WRAP_CONFIG_DIR")
    monkeypatch.delenv("WRAP_DATA_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xc"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xd"))

    assert paths.config_dir() == tmp_path / "xc" / "claude-wrap"
    assert paths.data_dir() == tmp_path / "xd" / "claude-wrap"


def test_home_directories_are_the_fallback(monkeypatch):
    for var in ("WRAP_CONFIG_DIR", "WRAP_DATA_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)

    assert paths.config_dir() == Path.home() / ".config" / "claude-wrap"
    assert paths.data_dir() == Path.home() / ".local" / "share" / "claude-wrap"


def test_user_pricing_is_merged_over_the_packaged_prices(isolated_dirs):
    config_dir, _ = isolated_dirs
    config_dir.mkdir()
    (config_dir / "pricing.yaml").write_text(
        "models:\n"
        "  claude-haiku-4-5:\n    input: 0.5\n"
        "  my-model:\n    {input: 1, output: 2, cache_read: 0.1, cache_write_5m: 1.25, cache_write_1h: 2}\n"
    )
    packaged = PricingTable.load(paths.DEFAULTS_DIR / "pricing.yaml")

    pricing = load_pricing(load_config().telemetry)

    assert pricing.lookup("claude-haiku-4-5").input == 0.5
    assert pricing.lookup("claude-haiku-4-5").output == packaged.lookup("claude-haiku-4-5").output
    assert pricing.lookup("my-model") is not None


def test_a_missing_explicit_pricing_file_is_an_error(isolated_dirs):
    config_dir, _ = isolated_dirs
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("telemetry:\n  pricing_file: missing.yaml\n")

    with pytest.raises(FileNotFoundError):
        load_pricing(load_config().telemetry)
