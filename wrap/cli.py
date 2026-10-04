from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

import typer
from rich.console import Console

from wrap.config import REPO_ROOT, load_config

app = typer.Typer(add_completion=False)
console = Console(stderr=True)


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Polls until a TCP port accepts connections or a timeout is reached.

    Used to wait for the proxy subprocess to finish starting up before
    launching Claude Code against it.

    Args:
        host: Host to connect to.
        port: Port to connect to.
        timeout: Maximum time to wait, in seconds. Defaults to 10.0.

    Returns:
        True as soon as a connection succeeds, False if `timeout` elapses
        first.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@app.command()
def claude(
    debug: bool = typer.Option(False, "--debug", help="Log a line per turn with its tokens, cost and latency."),
):
    """Launch Claude Code with routing-aware proxying turned on.

    Runs the real `claude` CLI as a child process with stdio inherited,
    so the user gets a normal, fully interactive session. The proxy's own
    log output is written to a file (see `wrap logs`) rather than this
    terminal, so it doesn't interleave with the Claude Code UI; the proxy
    subprocess is always torn down afterward. Each run gets a session ID
    that groups its turns in the telemetry database (see `wrap stats`).

    Args:
        debug: Run the proxy at DEBUG log level for this session.

    Raises:
        typer.Exit: With code 1 if `claude` isn't on PATH or the proxy
            doesn't come up in time; otherwise with `claude`'s exit code
            once the session ends.
    """
    if shutil.which("claude") is None:
        console.print("[red]Could not find `claude` on PATH. Install Claude Code first.[/red]")
        raise typer.Exit(1)

    config = load_config()
    host = "127.0.0.1"

    log_path = REPO_ROOT / config.proxy.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    session_id = uuid.uuid4().hex[:12]
    proxy_env = {**os.environ, "WRAP_SESSION_ID": session_id}
    if debug:
        proxy_env["WRAP_LOG_LEVEL"] = "debug"

    proxy_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "wrap.proxy.server:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(config.proxy.port),
            "--log-level",
            "warning",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=proxy_env,
    )

    try:
        if not _wait_for_port(host, config.proxy.port):
            console.print("[red]Proxy did not start in time.[/red]")
            raise typer.Exit(1)

        console.print(
            f"[green]wrap[/green] proxy up on {host}:{config.proxy.port} -- "
            f"tiers: small={config.tiers['small'].model}, large={config.tiers['large'].model}\n"
            f"session: {session_id}\n"
            f"[dim]Logs: {log_path} -- run `wrap logs` in another terminal to follow them live"
            f"{'' if debug else ' (start with --debug for per-turn costs)'}. "
            f"Run `wrap stats` afterwards for this session's costs.[/dim]"
        )

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://{host}:{config.proxy.port}"

        result = subprocess.run(["claude"], env=env)
        raise typer.Exit(result.returncode)
    finally:
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
        log_file.close()


@app.command()
def logs():
    """Follow the proxy's log from the current (or most recent) `wrap claude` session."""
    config = load_config()
    log_path = REPO_ROOT / config.proxy.log_path

    if not log_path.exists():
        console.print(f"[yellow]No log file yet at {log_path} -- run `wrap claude` first.[/yellow]")
        raise typer.Exit(1)

    try:
        subprocess.run(["tail", "-n", "50", "-f", str(log_path)])
    except KeyboardInterrupt:
        pass


@app.command()
def dashboard():
    """View cost/latency stats. (Not built yet -- Phase D.)"""
    console.print("[yellow]wrap dashboard[/yellow] isn't built yet -- coming in a later phase.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
