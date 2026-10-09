import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_notes.py"

CHANGELOG = """# Changelog

## Unreleased

### Added

- Something new.

## 0.2.0 - 2026-11-01

### Fixed

- A bug.

## 0.1.0 - 2026-10-10

First release.
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "claude-wrap"\nversion = "0.2.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    return tmp_path


def run(tag, root):
    return subprocess.run(
        [sys.executable, str(SCRIPT), tag, "--root", str(root)], capture_output=True, text=True, check=False
    )


def test_prints_the_section_for_the_tag(root):
    result = run("v0.2.0", root)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "### Fixed\n\n- A bug.\n"


def test_the_last_section_runs_to_the_end(root):
    (root / "pyproject.toml").write_text('version = "0.1.0"\n')

    result = run("v0.1.0", root)

    assert result.stdout == "First release.\n"


@pytest.mark.parametrize(
    ("tag", "error"),
    [
        ("0.2.0", "is not vX.Y.Z"),
        ("v0.2", "is not vX.Y.Z"),
        ("v0.3.0", "does not match pyproject.toml version (0.2.0)"),
    ],
)
def test_rejects_a_bad_or_mismatched_tag(root, tag, error):
    result = run(tag, root)

    assert result.returncode == 1
    assert error in result.stderr


def test_rejects_a_version_still_under_unreleased(root):
    (root / "pyproject.toml").write_text('version = "0.3.0"\n')

    result = run("v0.3.0", root)

    assert result.returncode == 1
    assert "no '## 0.3.0 - YYYY-MM-DD' section" in result.stderr


def test_rejects_an_empty_section(root):
    (root / "CHANGELOG.md").write_text("## 0.2.0 - 2026-11-01\n\n## 0.1.0 - 2026-10-10\n\nFirst.\n")

    result = run("v0.2.0", root)

    assert result.returncode == 1
    assert "is empty" in result.stderr


def test_accepts_a_pre_release_tag(root):
    (root / "pyproject.toml").write_text('version = "0.2.0rc1"\n')
    (root / "CHANGELOG.md").write_text("## 0.2.0rc1 - 2026-10-30\n\n- Try it.\n")

    assert run("v0.2.0rc1", root).stdout == "- Try it.\n"
