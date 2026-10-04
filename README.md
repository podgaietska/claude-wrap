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
  large:
    model: claude-sonnet-5

models:
  claude-haiku-4-5-20251001:
    max_output_tokens: 64000
    context_window: 200000
    effort: false
    thinking: [enabled]
    mid_conversation_system: false
    context_edits: [clear_tool_uses_20250919, clear_thinking_20251015]
  claude-opus-5-5:
    max_output_tokens: 128000
    context_window: 1000000
    effort: true
    thinking: [adaptive]
    mid_conversation_system: true
    context_edits: [clear_tool_uses_20250919, clear_thinking_20251015, compact_20260112]
  # ...

routing:
  strategy: heuristic
  complexity_threshold: 0.5   # score >= threshold routes to "large"
```

Swap in any model IDs you have access to.

### Routing

Each request is routed by the question that started its turn: the router walks back past Claude Code's appended `system` reminders, injected `<system-reminder>` context, and mid-turn tool calls/results to the text you actually typed. So every request in a turn — including tool continuations — goes to the same model. Requests with no typed question keep the model Claude Code asked for.

### Adapting requests to the routed model

Claude Code shapes every request for the model it *thinks* it's calling (e.g. Opus 5.5: `max_tokens: 128000`, adaptive thinking, `effort`, mid-conversation system messages). The `models` table describes what each model supports, and `wrap/adapt/` reshapes each request to fit its target model — one small adapter per setting:

| Adapter | Does |
|---|---|
| `clamp_max_tokens` | caps `max_tokens` at `max_output_tokens` |
| `adapt_thinking` | adaptive → budgeted thinking if `thinking_budget` is set, else drops thinking |
| `adapt_effort` | removes `output_config.effort` when `effort: false` |
| `adapt_context_management` | removes unsupported edits, and the clear-thinking edit when thinking is off |
| `adapt_system_messages` | folds mid-conversation system messages into the adjacent user turn |

Every field is optional — unknown means "leave it alone". If a value is stale and the API rejects a request with a known error (`wrap/adapt/error_rules.py`), the proxy learns the correct value for the session, re-adapts, retries once, and logs a warning to update the table. Unknown errors are logged and returned unchanged.

To support a new incompatibility: add a capability field, an adapter appended to `ADAPTERS`, optionally an error rule, and tests.

### Logs

`wrap logs` shows one line per request with an id, e.g.:

```
#3 main msgs=4 tail=[…, system, assistant:text+tool_use, user:tool_result]  routed small → claude-haiku-4-5-20251001 (score 0.01: no strong signals, by turn's question at msg 0)
#3 adapted for claude-haiku-4-5-20251001: max_tokens 128000→64000; dropped thinking (adaptive unsupported); dropped effort=medium; ...
#3 ← 200 in 630ms
```

`main`/`side` distinguishes Claude Code's conversation requests from its small background calls (e.g. session titles). No message content is logged.

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
