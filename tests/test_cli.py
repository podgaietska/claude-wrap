import socket

import pytest
import yaml
from typer.testing import CliRunner

from wrap import cli

runner = CliRunner()


@pytest.fixture
def started(mocker):
    run = mocker.patch("uvicorn.run")
    opener = mocker.patch.object(cli.threading, "Thread")
    mocker.patch.object(cli, "_port_in_use", return_value=False)
    return run, opener


def test_dashboard_serves_on_localhost(started):
    run, opener = started

    result = runner.invoke(cli.app, ["dashboard", "--port", "9123"])

    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert run.call_args.args == ("wrap.dashboard.app:create_dashboard_app",)
    assert {"factory": True, "host": "127.0.0.1", "port": 9123}.items() <= run.call_args.kwargs.items()
    opener.assert_called_once()
    assert opener.call_args.kwargs["args"][2] == "http://127.0.0.1:9123/"


def test_dashboard_session_and_no_open(started):
    run, opener = started

    result = runner.invoke(cli.app, ["dashboard", "--session", "abc123", "--no-open"])

    assert result.exit_code == 0, result.output
    opener.assert_not_called()
    assert "?session=abc123" in result.output


def test_dashboard_refuses_a_busy_port(mocker):
    run = mocker.patch("uvicorn.run")
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]

        result = runner.invoke(cli.app, ["dashboard", "--port", str(port), "--no-open"])

    assert result.exit_code == 1
    assert "is in use" in result.output
    run.assert_not_called()


def test_version():
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("claude-wrap ")


def test_config_shows_resolved_paths(isolated_dirs):
    config_dir, data_dir = isolated_dirs

    result = runner.invoke(cli.app, ["config"])

    assert result.exit_code == 0, result.output
    assert f"Config    {config_dir / 'config.yaml'}  (none" in result.output
    assert f"Database  {data_dir / 'wrap.db'}" in result.output


def test_config_init_writes_a_starter_once(isolated_dirs):
    config_dir, _ = isolated_dirs

    first = runner.invoke(cli.app, ["config", "--init"])
    second = runner.invoke(cli.app, ["config", "--init"])

    assert first.exit_code == 0, first.output
    assert (config_dir / "config.yaml").read_text() == cli.STARTER_CONFIG
    assert second.exit_code == 1


def test_config_defaults_prints_the_packaged_config():
    result = runner.invoke(cli.app, ["config", "--defaults"])

    assert result.exit_code == 0
    assert result.output.startswith("tiers:")


def test_config_effective_merges_overrides(isolated_dirs):
    config_dir, data_dir = isolated_dirs
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("tiers:\n  large:\n    model: my-model\n")

    result = runner.invoke(cli.app, ["config", "--effective"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"# packaged defaults + {config_dir / 'config.yaml'}\n")
    settings = yaml.safe_load(result.output)
    assert settings["tiers"]["large"] == {"model": "my-model"}
    assert settings["tiers"]["small"]["model"]
    assert settings["telemetry"]["db_path"] == str(data_dir / "wrap.db")
    assert "null" not in result.output


@pytest.fixture
def launched(mocker):
    mocker.patch.object(cli.shutil, "which", return_value="/usr/bin/claude")
    mocker.patch.object(cli.subprocess, "Popen")
    mocker.patch.object(cli, "_wait_for_port", return_value=True)
    return mocker.patch.object(cli.subprocess, "run", return_value=mocker.Mock(returncode=0))


@pytest.mark.parametrize(
    ("args", "passed"),
    [
        ([], []),
        (["-p", "what is 2+2?"], ["-p", "what is 2+2?"]),
        (["--resume", "--model", "opus"], ["--resume", "--model", "opus"]),
        (["--debug", "--", "--debug"], ["--debug"]),
    ],
)
def test_claude_passes_extra_arguments_through(launched, args, passed):
    result = runner.invoke(cli.app, ["claude", *args])

    assert result.exit_code == 0, result.output
    assert launched.call_args.args[0] == ["claude", *passed]
