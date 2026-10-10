"""Which Claude Code versions this release of wrap is tested with.

Claude Code ships patch releases almost daily, and within a minor series
they have kept the request shapes wrap depends on, so the range covers a
whole series. The weekly canary (`scripts/canary.py`) runs its oldest
version and the latest release. When a new series appears, the canary
shows whether it works; the range is widened in a new wrap release.
"""

from __future__ import annotations

import re
import subprocess

TESTED_FROM = (2, 1, 0)
TESTED_BELOW = (2, 2, 0)

Version = tuple[int, int, int]

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _format(version: Version) -> str:
    return ".".join(map(str, version))


TESTED_RANGE = f">={_format(TESTED_FROM)}, <{_format(TESTED_BELOW)}"


def parse_version(text: str) -> Version | None:
    """Reads the first X.Y.Z in `claude --version` output, e.g. "2.1.292 (Claude Code)"."""
    match = _VERSION.search(text)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def installed_claude_code() -> Version | None:
    """The version of the `claude` on PATH, or None if it can't be run or read."""
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_version(result.stdout)


def compatibility_warning(version: Version | None) -> str | None:
    """A warning if `version` is outside the tested range; None if it's inside or unknown."""
    if version is None or TESTED_FROM <= version < TESTED_BELOW:
        return None
    found = _format(version)
    if version >= TESTED_BELOW:
        return (
            f"Claude Code {found} is newer than this claude-wrap is tested with ({TESTED_RANGE}). "
            "If routing misbehaves, check for a newer claude-wrap: `pipx upgrade claude-wrap`."
        )
    return (
        f"Claude Code {found} is older than this claude-wrap is tested with ({TESTED_RANGE}). "
        "Update Claude Code, or install an older claude-wrap that supports it."
    )
