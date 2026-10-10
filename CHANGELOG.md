# Changelog

Notable changes to claude-wrap, newest first. Versions follow [Semantic Versioning](https://semver.org/); while the version is 0.x, a minor release may include breaking changes, marked **Breaking**.

Add a line under **Unreleased** in any PR that changes what users see. At release time that heading becomes the version and date (see [RELEASING.md](RELEASING.md)).

## Unreleased

## 0.2.0 - 2026-10-09

### Added

- `wrap claude` passes other arguments to `claude`, e.g. `wrap claude --resume` or `wrap claude -p "question"`. Put an argument wrap also uses after `--`.
- `wrap claude` warns when the installed Claude Code is outside the versions this release is tested with (2.1.0 up to 2.2); `wrap config` shows the installed version and the range.

## 0.1.0 - 2026-10-09

First release.

### Added

- `wrap claude` runs Claude Code through a local proxy that sends each turn to a small or large model tier, picked by a zero-cost complexity heuristic on the question that started the turn.
- Requests are adapted to the routed model's capabilities (`max_tokens`, thinking, effort, context management, mid-conversation system messages). On a known API error the proxy learns the right value for the session and retries once.
- Every request is recorded in a local SQLite database. `wrap stats` shows tokens and cost per tier and the net routing savings, including the cache misses that switching models causes. `wrap logs` follows routing decisions live.
- `wrap dashboard` serves a local, read-only dashboard of cost, savings, routing and latency.
- Packaged default config and prices, with your overrides merged in from `~/.config/claude-wrap/`. Data lives in `~/.local/share/claude-wrap/`.
- `wrap config` shows where everything lives; `--init`, `--defaults` and `--effective` create, print and explain the config. `wrap --version`.
