import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Points the config and data directories at a temp folder, so tests never read or write the user's own."""
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    monkeypatch.setenv("WRAP_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("WRAP_DATA_DIR", str(data_dir))
    return config_dir, data_dir


@pytest.fixture(autouse=True)
def claude_code_version(monkeypatch):
    """Stands in for `claude --version`, so tests don't depend on (or run) the real Claude Code."""
    version = {"value": (2, 1, 100)}
    monkeypatch.setattr("wrap.cli.installed_claude_code", lambda: version["value"])
    return version
