from __future__ import annotations

import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from urllib.parse import urlencode

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from wrap.config import REPO_ROOT, load_config
from wrap.telemetry import db
from wrap.telemetry.economics import Economics, analyze
from wrap.telemetry.logger import format_signed_cost, format_tokens
from wrap.telemetry.pricing import PricingTable
from wrap.telemetry.requests import count_kinds

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
    debug: bool = typer.Option(False, "--debug", help="Log each request's details, tokens, cost and latency."),
):
    """Launch Claude Code with routing-aware proxying turned on.

    Runs the real `claude` CLI as a child process with stdio inherited,
    so the user gets a normal, fully interactive session. The proxy's own
    log output is written to a file (see `wrap logs`) rather than this
    terminal, so it doesn't interleave with the Claude Code UI; the proxy
    subprocess is always torn down afterward. Each run gets a session ID
    that groups its requests in the telemetry database (see `wrap stats`).

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
            f"{'' if debug else ' (start with --debug for per-request costs)'}. "
            f"Run `wrap stats` afterwards for this session's costs.[/dim]\n"
            f"[dim]Dashboard: run `wrap dashboard` in another terminal "
            f"(http://{host}:{config.dashboard.port}).[/dim]"
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
def stats(
    session: str = typer.Option(
        "last", "--session", help="`last`, `all`, or a session ID from the `wrap claude` banner."
    ),
):
    """Show token usage, cost and routing savings for a `wrap claude` session."""
    config = load_config()
    db_path = REPO_ROOT / config.telemetry.db_path
    if not db_path.exists():
        console.print(f"[yellow]No telemetry yet at {db_path} -- run `wrap claude` first.[/yellow]")
        raise typer.Exit(1)

    pricing = PricingTable.load(REPO_ROOT / config.telemetry.pricing_file)
    conn = db.connect(db_path)
    try:
        if not print_stats(Console(), conn, pricing, session):
            raise typer.Exit(1)
    finally:
        conn.close()


def print_stats(out: Console, conn: sqlite3.Connection, pricing: PricingTable, session: str) -> bool:
    """Prints the per-tier table and routing economics for a selection of requests.

    Args:
        out: Where to print.
        conn: An open telemetry connection.
        pricing: Rates for routing economics.
        session: `last`, `all`, or a session ID.

    Returns:
        False if there were no requests to show.
    """
    if not db.has_turns(conn):
        console.print("[yellow]No requests logged yet -- run `wrap claude` first.[/yellow]")
        return False

    session_id, all_sessions = db.resolve_session(conn, session)
    turns = db.fetch_turns(conn, session_id, all_sessions)
    if not turns:
        console.print(f"[yellow]No requests logged for session {session_id}.[/yellow]")
        return False

    title = "All sessions" if all_sessions else f"Session {session_id or '(outside wrap claude)'}"
    out.print(f"[bold]{title}[/bold] — {len(turns)} requests ({_breakdown(turns)})")
    out.print("[dim]Side requests are Claude Code's background calls and subagents.[/dim]\n")

    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Tier", "Served model"):
        table.add_column(column)
    for column in ("Requests", "Input", "Output", "Cache read", "Cache write", "Cost"):
        table.add_column(column, justify="right")
    rows = db.summarize(conn, session_id, all_sessions)
    for row in rows:
        table.add_row(
            row.tier or "—",
            row.served_model or "—",
            str(row.turns),
            format_tokens(row.input_tokens),
            format_tokens(row.output_tokens),
            format_tokens(row.cache_read_tokens),
            format_tokens(row.cache_creation_tokens),
            f"${row.cost_usd:.4f}",
        )
    total_cost = sum(row.cost_usd for row in rows)
    table.add_section()
    table.add_row("[bold]Total[/bold]", "", str(len(turns)), "", "", "", "", f"[bold]${total_cost:.4f}[/bold]")
    out.print(table)

    failed = sum(1 for t in turns if t.status_code is None or t.status_code >= 400)
    unpriced = sum(1 for t in turns if t.cost_usd is None) - failed
    if failed:
        out.print(f"[dim]{failed} failed requests (no cost).[/dim]")
    if unpriced:
        out.print(f"[yellow]{unpriced} requests unpriced -- their model is missing from the pricing file.[/yellow]")
    out.print()

    _print_economics(out, analyze(turns, pricing))
    return True


def _breakdown(turns: list[db.TurnRecord]) -> str:
    """Splits requests into new messages, tool calls and side requests (see `wrap.telemetry.requests`).

    Args:
        turns: The requests to describe.

    Returns:
        E.g. "5 new messages, 8 tool calls, 2 side requests".
    """
    counts = count_kinds(turns)
    parts = [
        (counts["new_message"], "new message"),
        (counts["tool_call"], "tool call"),
        (counts["side"], "side request"),
    ]
    return ", ".join(f"{n} {label}{'' if n == 1 else 's'}" for n, label in parts)


def _print_economics(out: Console, economics: Economics) -> None:
    """Prints the routing economics block (see `wrap.telemetry.economics`)."""
    if economics.counterfactual_cost == 0 and economics.actual_cost == 0:
        return
    if len(economics.requested_models) == 1:
        baseline = f"always {next(iter(economics.requested_models))}"
    else:
        baseline = "always the requested model"

    net = economics.net_savings
    share = f"   ({net / economics.counterfactual_cost:.0%})" if economics.counterfactual_cost else ""
    net_style = "green" if net >= 0 else "red"
    lost_label = f"Lost to cache misses ({economics.switches} switches):"

    out.print(f"[bold]Routing economics[/bold] (vs. {baseline})")
    out.print(f"  {'Would have cost:':<40}{f'${economics.counterfactual_cost:.4f}':>10}")
    out.print(f"  {'Actually cost:':<40}{f'${economics.actual_cost:.4f}':>10}")
    out.print(f"    {'Saved by cheaper models:':<38}{format_signed_cost(economics.saved_by_model):>10}")
    out.print(
        f"    {lost_label:<38}{format_signed_cost(-economics.cache_penalty):>10}"
        f"   {format_tokens(economics.recached_tokens)} tokens re-cached"
    )
    out.print(
        f"  [bold]{'Net savings:':<40}[/bold][bold {net_style}]{format_signed_cost(net):>10}[/bold {net_style}]{share}"
    )
    if economics.check_rate is None:
        out.print("  [dim]Estimate check: no requests without a switch to check against yet[/dim]")
    else:
        out.print(
            f"  [dim]Estimate check: matches on {economics.check_rate:.0%} of "
            f"{economics.check_eligible} requests without a switch[/dim]"
        )
    out.print("[dim]Assumes the same output and number of requests on the requested model; tokenizers differ.[/dim]")


@app.command()
def dashboard(
    port: int = typer.Option(None, "--port", help="Port to serve on (default: dashboard.port in the config)."),
    session: str = typer.Option(None, "--session", help="Open on `last`, `all`, or a session ID."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open a browser."),
):
    """Serve a local dashboard of cost, routing savings and latency.

    Reads the telemetry database read-only, so it can run during a `wrap
    claude` session (it refreshes live) or after one. Runs until Ctrl-C.

    Raises:
        typer.Exit: With code 1 if the port is already in use.
    """
    import uvicorn

    config = load_config()
    host = "127.0.0.1"
    port = port or config.dashboard.port
    url = f"http://{host}:{port}/" + (f"?{urlencode({'session': session})}" if session else "")

    db_path = REPO_ROOT / config.telemetry.db_path
    if not db_path.exists():
        console.print(f"[yellow]No telemetry yet at {db_path} -- the page fills in once `wrap claude` runs.[/yellow]")
    if _port_in_use(host, port):
        console.print(f"[red]Port {port} is in use -- is a dashboard already running? Try {url}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]wrap dashboard[/green] on {url} -- Ctrl-C to stop")
    if not no_open:
        threading.Thread(target=_open_when_up, args=(host, port, url), daemon=True).start()
    uvicorn.run("wrap.dashboard.app:create_dashboard_app", factory=True, host=host, port=port, log_level="warning")


def _port_in_use(host: str, port: int) -> bool:
    """True if something already accepts connections on the port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _open_when_up(host: str, port: int, url: str) -> None:
    """Opens the dashboard in a browser once the server accepts connections."""
    if _wait_for_port(host, port):
        webbrowser.open(url)


if __name__ == "__main__":
    app()
