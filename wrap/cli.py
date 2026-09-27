from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time

import typer
from rich.console import Console

from wrap.config import load_config

app = typer.Typer(add_completion=False)
console = Console(stderr=True)


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@app.command()
def claude():
    """Launch Claude Code with routing-aware proxying turned on."""
    if shutil.which("claude") is None:
        console.print("[red]Could not find `claude` on PATH. Install Claude Code first.[/red]")
        raise typer.Exit(1)

    config = load_config()
    host = "127.0.0.1"

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
    )

    try:
        if not _wait_for_port(host, config.proxy.port):
            console.print("[red]Proxy did not start in time.[/red]")
            raise typer.Exit(1)

        console.print(
            f"[green]wrap[/green] proxy up on {host}:{config.proxy.port} -- "
            f"tiers: small={config.tiers['small'].model}, large={config.tiers['large'].model}"
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


@app.command()
def dashboard():
    """View cost/latency stats. (Not built yet -- Phase D.)"""
    console.print("[yellow]wrap dashboard[/yellow] isn't built yet -- coming in a later phase.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
