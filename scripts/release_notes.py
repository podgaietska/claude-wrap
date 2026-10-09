"""Checks a release tag against the project and prints its changelog notes.

Usage: python scripts/release_notes.py v0.2.0 [--root DIR]

Fails (exit 1) unless the tag looks like `vX.Y.Z` (optionally with an
`aN`, `bN` or `rcN` pre-release suffix), matches `version` in
pyproject.toml, and CHANGELOG.md has a non-empty `## X.Y.Z - YYYY-MM-DD`
section. On success, prints that section's body: the GitHub Release notes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TAG = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)")
PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def release_notes(tag: str, root: Path) -> str:
    """Returns the changelog notes for `tag`.

    Args:
        tag: The git tag, e.g. "v0.2.0".
        root: The repository root holding pyproject.toml and CHANGELOG.md.

    Raises:
        ValueError: If the tag, the project version and the changelog disagree.
    """
    match = TAG.fullmatch(tag)
    if not match:
        raise ValueError(f"tag {tag!r} is not vX.Y.Z (optionally with an aN, bN or rcN suffix)")
    version = match.group(1)

    found = PYPROJECT_VERSION.search((root / "pyproject.toml").read_text())
    if not found or found.group(1) != version:
        project = found.group(1) if found else "missing"
        raise ValueError(f"tag {tag} does not match pyproject.toml version ({project})")

    changelog = (root / "CHANGELOG.md").read_text()
    heading = re.compile(rf"^## {re.escape(version)} - \d{{4}}-\d{{2}}-\d{{2}}$", re.MULTILINE)
    start = heading.search(changelog)
    if not start:
        raise ValueError(f"CHANGELOG.md has no '## {version} - YYYY-MM-DD' section")
    end = re.compile(r"^## ", re.MULTILINE).search(changelog, start.end())
    notes = changelog[start.end() : end.start() if end else None].strip()
    if not notes:
        raise ValueError(f"CHANGELOG.md section for {version} is empty")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        print(release_notes(args.tag, args.root))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
