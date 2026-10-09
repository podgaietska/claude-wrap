"""Where wrap reads its config from and writes its data to.

The packaged defaults live in `wrap/defaults/`. A user's own settings live
in the config directory and are merged over them; the telemetry database
and the proxy log live in the data directory:

    config  $WRAP_CONFIG_DIR, else $XDG_CONFIG_HOME/claude-wrap, else ~/.config/claude-wrap
    data    $WRAP_DATA_DIR,   else $XDG_DATA_HOME/claude-wrap,   else ~/.local/share/claude-wrap
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "claude-wrap"
DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"


def _dir(override_var: str, xdg_var: str, fallback: str) -> Path:
    """Resolves a directory from an override variable, an XDG variable, or a home-relative fallback."""
    if os.environ.get(override_var):
        return Path(os.environ[override_var]).expanduser()
    base = os.environ.get(xdg_var)
    return (Path(base).expanduser() if base else Path.home() / fallback) / APP_NAME


def config_dir() -> Path:
    """The directory holding the user's `config.yaml` and `pricing.yaml` overrides."""
    return _dir("WRAP_CONFIG_DIR", "XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    """The directory holding the telemetry database and the proxy log."""
    return _dir("WRAP_DATA_DIR", "XDG_DATA_HOME", ".local/share")


def resolve(path: str, base: Path) -> Path:
    """Expands `~` and makes a relative path relative to `base`; absolute paths are kept."""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else base / expanded
