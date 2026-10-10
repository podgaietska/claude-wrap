"""End-to-end check of wrap against a real Claude Code, with no API key or cost.

Runs `wrap claude -p <prompt>` with the installed `claude`, pointing wrap's
proxy at a fake Anthropic API on localhost that records every request and
answers with a fixed reply. Then checks that Claude Code got the reply and
that the proxy routed the request by the question it was asked: the part
that breaks when a Claude Code release changes how it shapes requests.

Usage: python scripts/canary.py [--claude-dir DIR] [--out DIR]

Exits 0 if every check passes, 1 otherwise. Writes `requests.jsonl` (what
Claude Code sent), `proxy.log` and `report.md` to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from wrap import compat

PROMPT = "What is 2+2? Reply with one word."
REPLY = "canary ok"
SMALL_MODEL = "canary-small-model"
LARGE_MODEL = "canary-large-model"
TIMEOUT_SECONDS = 180


class FakeAnthropic(BaseHTTPRequestHandler):
    """Answers every Messages API call with `REPLY` and records what it was sent."""

    requests: list[dict] = []
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802 (http.server's naming)
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            payload = {"unparsed": body.decode(errors="replace")}
        with self.lock:
            self.requests.append({"method": "POST", "path": self.path, "body": payload})

        path = self.path.split("?")[0]
        if path == "/v1/messages/count_tokens":
            self._json(200, {"input_tokens": 10})
        elif path == "/v1/messages":
            model = payload.get("model", "unknown")
            if payload.get("stream"):
                self._stream(model)
            else:
                self._json(200, _message(model))
        else:
            self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})

    def do_GET(self):  # noqa: N802
        with self.lock:
            self.requests.append({"method": "GET", "path": self.path})
        self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}})

    def _json(self, status: int, data: dict) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream(self, model: str) -> None:
        message = _message(model)
        text = message["content"][0]["text"]
        events = [
            ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 3},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for name, data in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

    def log_message(self, *args):
        pass


def _message(model: str) -> dict:
    return {
        "id": "msg_canary",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": REPLY}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wrap_executable() -> str:
    beside_python = Path(sys.executable).with_name("wrap")
    found = str(beside_python) if beside_python.exists() else shutil.which("wrap")
    if not found:
        sys.exit("error: `wrap` is not installed in this environment")
    return found


def run(claude_dir: Path | None, out: Path) -> bool:
    """Runs the canary and writes its artifacts to `out`. Returns True if every check passed."""
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wrap-canary-"))
    (work / "home").mkdir()
    (work / "config").mkdir()

    server = ThreadingHTTPServer(("127.0.0.1", _free_port()), FakeAnthropic)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (work / "config" / "config.yaml").write_text(
        f"tiers:\n  small:\n    model: {SMALL_MODEL}\n  large:\n    model: {LARGE_MODEL}\n"
        f"proxy:\n  port: {_free_port()}\n  upstream_base_url: http://127.0.0.1:{server.server_port}\n"
        "  log_level: debug\n"
    )

    path = os.environ["PATH"]
    env = {
        "PATH": f"{claude_dir}{os.pathsep}{path}" if claude_dir else path,
        "HOME": str(work / "home"),  # a fresh Claude Code profile: no login, no settings, no history
        "WRAP_CONFIG_DIR": str(work / "config"),
        "WRAP_DATA_DIR": str(work / "data"),
        "ANTHROPIC_API_KEY": "canary-not-a-real-key",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    version = subprocess.run(["claude", "--version"], env=env, capture_output=True, text=True, check=False)
    claude_version = version.stdout.strip() or version.stderr.strip()

    try:
        result = subprocess.run(
            [_wrap_executable(), "claude", "-p", PROMPT],
            env=env,
            cwd=work / "home",
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, stdout, stderr = None, exc.stdout or "", f"timed out after {TIMEOUT_SECONDS}s"
    finally:
        server.shutdown()

    requests = FakeAnthropic.requests
    with open(out / "requests.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in requests)
    if (work / "data" / "proxy.log").exists():
        shutil.copy(work / "data" / "proxy.log", out / "proxy.log")
    turns = _turns(work / "data" / "wrap.db")

    messages = [r for r in requests if r["path"].split("?")[0] == "/v1/messages"]
    routed = [t for t in turns if t["complexity_score"] is not None]
    checks = [
        (f"`wrap claude -p` exits 0 (got {exit_code})", exit_code == 0),
        (f"Claude Code prints the reply ({stdout.strip()[:80]!r})", REPLY in stdout),
        (f"Messages API requests reach the upstream ({len(messages)})", bool(messages)),
        (f"every request is recorded ({len(turns)} turns)", len(turns) >= len(messages) > 0),
        (
            f"routing finds the typed question ({len(routed)} routed by score)",
            any(t["tier"] == "small" and t["model_id"] == SMALL_MODEL for t in routed),
        ),
        ("no request fails", all((t["status_code"] or 0) < 400 for t in turns)),
    ]
    other_paths = sorted({r["path"].split("?")[0] for r in requests} - {"/v1/messages"})

    lines = [f"# Canary: {claude_version or 'claude --version failed'}", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks]
    lines += ["", f"Other upstream paths: {', '.join(other_paths) or 'none'}"]
    tested = compat.parse_version(claude_version)
    if compat.compatibility_warning(tested):
        lines += [
            "",
            f"Note: this Claude Code is outside the tested range ({compat.TESTED_RANGE}). "
            "If every check passed, the range in wrap/compat.py can be widened.",
        ]
    if not all(ok for _, ok in checks):
        lines += ["", "## wrap claude stderr", "", "```", stderr[-4000:], "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    shutil.rmtree(work, ignore_errors=True)
    return all(ok for _, ok in checks)


def _turns(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM turn_log ORDER BY id")]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claude-dir", type=Path, help="directory holding the `claude` to test (default: PATH)")
    parser.add_argument("--out", type=Path, default=Path("canary-out"), help="where to write the artifacts")
    args = parser.parse_args()
    return 0 if run(args.claude_dir, args.out) else 1


if __name__ == "__main__":
    sys.exit(main())
