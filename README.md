# prompt-router

A transparent cost-aware model router for [Claude Code](https://claude.com/claude-code). Its CLI, `wrap`, starts a local proxy, points a real Claude Code session at it, and routes each turn to a small/cheap model or a large/capable one based on how complex the question looks — with no change to how you actually use Claude Code.

## How it works

Claude Code honors the `ANTHROPIC_BASE_URL` environment variable (Anthropic's own documented mechanism for LLM gateways). `wrap claude`:

1. Starts a local HTTP proxy (FastAPI + httpx) that speaks the Anthropic Messages API.
2. Points a real `claude` process at that proxy via `ANTHROPIC_BASE_URL`, then runs it with your terminal attached — full tool use, file edits, and streaming all work exactly as normal.
3. On every `POST /v1/messages`, the proxy looks at the newest turn, scores its complexity with a zero-cost heuristic, and rewrites the `model` field before forwarding to the real Anthropic API. Tool-result continuation turns (not fresh questions) pass through unchanged. Every other request/path is relayed as-is.

Because `ANTHROPIC_BASE_URL` is only set for the one launched process, there's nothing global to undo — just exit the session.

## Status

**Phase A (routing-only passthrough proxy) is built and verified** against the real Claude Code CLI and Anthropic API — see `wrap/proxy/server.py` and `wrap/routing/`.

Not yet built:
- **Phase B** — cost/latency telemetry logged to SQLite.
- **Phase C** — semantic response caching.
- **Phase D** — a local dashboard (`wrap dashboard`) visualizing cost/latency/cache-hit stats.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
wrap claude
```

This starts the proxy on `127.0.0.1:8787` (configurable in `config/config.yaml`) and launches Claude Code through it. Routing decisions are logged to stderr as they happen.

## Configuration

`config/config.yaml`:

```yaml
tiers:
  small:
    model: claude-haiku-4-5-20251001
    context_window: 200000
  large:
    model: claude-sonnet-5
    context_window: 1000000

routing:
  strategy: heuristic
  complexity_threshold: 0.5   # score >= threshold routes to "large"
```

Swap in any model IDs you have access to — the router only needs the tier name -> model ID mapping and each tier's context window (used as a safety check so a long conversation never gets routed to a model whose context window can't hold it).

## Testing

```bash
pytest
```

The suite is fully offline — the Anthropic API is mocked with `respx`, no real requests or API costs.

## Manual verification

```bash
wrap claude
```

Ask a trivial question ("what is 2+2?") and a complex multi-part one ("explain step by step how you'd refactor X, comparing trade-offs..."), and check the proxy's stderr log to confirm they routed to different tiers. Normal Claude Code functionality (tool calls, file edits) should be completely unaffected.
