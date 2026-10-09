import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Points the config and data directories at a temp folder, so tests never read or write the user's own."""
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    monkeypatch.setenv("WRAP_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("WRAP_DATA_DIR", str(data_dir))
    return config_dir, data_dir
