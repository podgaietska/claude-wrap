# claude-wrap

A model router for [Claude Code](https://claude.com/claude-code). `wrap claude` runs a real Claude Code session through a local proxy that sends each question to a cheaper or more capable model, depending on how complex it looks. `wrap stats` and a local dashboard show what that saves, including the cache misses that switching models causes.

claude-wrap is an independent community project, not affiliated with or endorsed by Anthropic.

## How it works

Claude Code honors the `ANTHROPIC_BASE_URL` environment variable (Anthropic's own documented mechanism for LLM gateways). `wrap claude`:

1. Starts a local HTTP proxy (FastAPI + httpx) that speaks the Anthropic Messages API.
2. Points a real `claude` process at that proxy via `ANTHROPIC_BASE_URL`, then runs it with your terminal attached — full tool use, file edits, and streaming all work exactly as normal.
3. On every `POST /v1/messages`, the proxy finds the question that started the current turn, scores its complexity with a zero-cost heuristic, and rewrites the `model` field before forwarding to the real Anthropic API. Every request in a turn, tool-result continuations included, goes to the same model; requests with no typed question keep the model Claude Code asked for. Every other request/path is relayed as-is.

Because `ANTHROPIC_BASE_URL` is only set for the one launched process, there's nothing global to undo — just exit the session.

## Install

Needs Python 3.10+ and [Claude Code](https://claude.com/claude-code) on your `PATH`, on macOS or Linux. Install with [pipx](https://pipx.pypa.io/) (`brew install pipx` on macOS):

```bash
pipx install claude-wrap
```

`wrap --version` shows what you have, and `pipx upgrade claude-wrap` gets the latest release. To install a specific version — for example one that matches an older Claude Code — pin it: `pipx install claude-wrap==0.1.0`. Every version is listed in [CHANGELOG.md](CHANGELOG.md).

To work on wrap itself:

```bash
git clone https://github.com/podgaietska/claude-wrap && cd claude-wrap
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Claude Code compatibility

Each claude-wrap release is tested with a range of Claude Code versions. `wrap config` shows your Claude Code version and the range, and `wrap claude` warns when you're outside it. If you need an older Claude Code, install the claude-wrap release that supports it (`pipx install claude-wrap==X.Y.Z`).

| claude-wrap | Tested with Claude Code |
| --- | --- |
| 0.1.x and later | 2.1.0 to 2.1.x |

A [weekly check](.github/workflows/canary.yml) runs the oldest version in the range and the latest Claude Code release through wrap, against a fake Anthropic API, and opens an issue if routing breaks. To run it yourself against the `claude` on your `PATH`:

```bash
python scripts/canary.py
```

## Usage

```bash
wrap claude
```

This starts the proxy on `127.0.0.1:8787` (configurable, see [Configuration](#configuration)) and launches Claude Code through it. The banner prints a session ID.

Other arguments go to `claude`, so `wrap claude --resume` or `wrap claude -p "question"` work as they would without wrap. To pass an argument wrap also uses, put it after `--`: `wrap claude -- --debug` turns on Claude Code's debug mode instead of wrap's.

```bash
wrap logs                  # in another terminal: follow routing decisions live
wrap claude --debug        # also log each turn's tokens, cost and latency (or set proxy.log_level: debug)
wrap stats                 # after (or during) a session: tokens and cost per tier, plus routing savings
wrap stats --session all   # or a session ID from the banner
```

Every API request is recorded in the telemetry database (`wrap.db`, `turn_log` table; `wrap config` shows where) whatever the log level, priced with the packaged prices plus any `pricing.yaml` of your own.

One question in Claude Code is usually several API requests (the question, then a round trip per tool call), plus background calls such as session titles, so `wrap stats` counts requests and breaks them down into new messages, tool calls and side requests.

`wrap stats` reports **net** routing savings: what the session would have cost had every request stayed on the model Claude Code asked for, minus what it cost. Prompt caches belong to one model, so switching a long conversation to a cheaper model makes that model write the whole prompt to its cache again — which can cost more than the cheaper model saves. The savings are split into "saved by cheaper models" and "lost to cache misses" so you can see which way it went. It assumes the requested model would have produced the same output in the same number of requests.

### Dashboard

```bash
wrap dashboard                   # serves http://127.0.0.1:8788 and opens it in a browser
wrap dashboard --session all     # open on every session; or a session ID
wrap dashboard --port 9000 --no-open
```

The dashboard is the same numbers as `wrap stats`, drawn: total cost and net savings, a waterfall of what cheaper models saved against what cache misses cost, cost per tier over time against the always-the-requested-model baseline, a histogram of complexity scores around the routing threshold, per-model tokens, prompt-cache reads and latency percentiles, and a table of recent requests (no message content is stored, so none is shown).

It runs as its own process and reads the telemetry database read-only, so start it in a second terminal during a `wrap claude` session — it refreshes every 5 seconds while the session is live — or any time afterwards. It listens on 127.0.0.1 only and makes no network requests (Chart.js is vendored). Times are shown in your local time zone; daily buckets on the cost chart break at UTC midnight. Port and refresh interval are under `dashboard:` in the config.

## Configuration

wrap ships with defaults ([`wrap/defaults/config.yaml`](wrap/defaults/config.yaml)), and reads your own settings from `~/.config/claude-wrap/config.yaml`. Your file only needs what you change: it's merged over the defaults key by key, so models and prices added in later releases still reach you.

```bash
wrap config             # where config, pricing, the database and the log live
wrap config --init      # create a commented starter config.yaml
wrap config --defaults  # print the packaged defaults
wrap config --effective # print what wrap runs with: the defaults plus your overrides
```

For example, to route the large tier to Opus 5.5:

```yaml
tiers:
  large:
    model: claude-opus-5-5
```

`pricing.yaml` in the same directory is merged over the packaged prices the same way, for models wrap doesn't know or rates that differ for you. The database and log live in `~/.local/share/claude-wrap/`. `XDG_CONFIG_HOME` and `XDG_DATA_HOME` are respected, and `WRAP_CONFIG_DIR` / `WRAP_DATA_DIR` override both locations.

The defaults look like this:

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

`wrap logs` shows one routing line per request, with an id, plus any warnings and errors:

```
#3 small → claude-haiku-4-5-20251001 (score 0.01: no strong signals, by turn's question at msg 0)
```

Run `wrap claude --debug` (or set `proxy.log_level: debug`) to see each request's details under its routing line:

```
#3 small → claude-haiku-4-5-20251001 (score 0.01: no strong signals, by turn's question at msg 0)
#3 request: main msgs=4 tail=[…, system, assistant:text+tool_use, user:tool_result]
#3 adapted: max_tokens 128000→64000; dropped thinking (adaptive unsupported); dropped effort=medium; ...
#3 ← 200 in 630ms
#3 turn in=12 out=340 cache_r=30.0k cache_w=1.2k $0.0081 1840ms
#3 switch sonnet→haiku: re-cached 1.0k tokens (+$0.0011)
```

`main`/`side` distinguishes Claude Code's conversation requests from its small background calls (e.g. session titles). `switch` appears when a conversation moves to a different model, with the extra cache-write cost that caused. No message content is logged.

## Testing

```bash
pytest
```

The suite is fully offline — the Anthropic API is mocked with `respx`, no real requests or API costs.

```bash
ruff check .           # lint
ruff format .          # format
```

To check a built package the way users get it — installed into a clean environment, outside the source tree:

```bash
python -m build
scripts/smoke_test.sh dist/*.whl
```

CI runs the tests on Python 3.10–3.14 (and macOS), both ruff checks, and the package build and smoke test on every pull request; `main` only accepts PRs that pass.

## Manual verification

```bash
wrap claude
```

Ask a trivial question ("what is 2+2?") and a complex multi-part one ("explain step by step how you'd refactor X, comparing trade-offs..."), and check the proxy's stderr log to confirm they routed to different tiers. Normal Claude Code functionality (tool calls, file edits) should be completely unaffected.

## Releases

Each version is listed in [CHANGELOG.md](CHANGELOG.md) and on the [releases page](https://github.com/podgaietska/claude-wrap/releases). [RELEASING.md](RELEASING.md) describes how a release is made.
