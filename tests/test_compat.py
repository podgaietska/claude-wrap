import subprocess

import pytest

from wrap import compat


@pytest.mark.parametrize(
    ("text", "version"),
    [("2.1.292 (Claude Code)", (2, 1, 292)), ("claude 3.0.1-beta", (3, 0, 1)), ("", None), ("no version", None)],
)
def test_parse_version(text, version):
    assert compat.parse_version(text) == version


@pytest.mark.parametrize("version", [None, compat.TESTED_FROM, (2, 1, 999)])
def test_no_warning_inside_the_range_or_when_unknown(version):
    assert compat.compatibility_warning(version) is None


def test_newer_and_older_versions_warn_differently():
    newer = compat.compatibility_warning(compat.TESTED_BELOW)
    older = compat.compatibility_warning((2, 0, 99))

    assert "is newer than" in newer and "pipx upgrade claude-wrap" in newer
    assert "is older than" in older and "older claude-wrap" in older


def test_installed_claude_code_reads_the_version(mocker):
    mocker.patch.object(compat.subprocess, "run", return_value=mocker.Mock(stdout="2.1.292 (Claude Code)\n"))

    assert compat.installed_claude_code() == (2, 1, 292)


@pytest.mark.parametrize("error", [FileNotFoundError("claude"), subprocess.TimeoutExpired("claude", 15)])
def test_installed_claude_code_is_none_when_claude_cannot_run(mocker, error):
    mocker.patch.object(compat.subprocess, "run", side_effect=error)

    assert compat.installed_claude_code() is None
