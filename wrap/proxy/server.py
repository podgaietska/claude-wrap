from __future__ import annotations

import itertools
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from rich.logging import RichHandler
from rich.markup import escape

from wrap.adapt.adapters import adapt_request
from wrap.adapt.error_rules import match_error
from wrap.adapt.registry import CapabilityRegistry
from wrap.config import REPO_ROOT, Config, load_config
from wrap.proxy.describe import describe_decision, describe_request
from wrap.proxy.sse import StreamUsageParser, parse_message_body
from wrap.proxy.upstream import filtered_headers, upstream_request_headers
from wrap.routing.router import RouteDecision, Router
from wrap.telemetry.db import TurnRecord
from wrap.telemetry.logger import TurnLogger, thread_key
from wrap.telemetry.pricing import PricingTable
from wrap.telemetry.usage import Usage

_ERROR_LOG_CHARS = 500

logger = logging.getLogger("wrap.proxy")
if not logger.handlers:
    handler = RichHandler(show_time=True, show_path=False, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class _TurnContext:
    """What's known about a `/v1/messages` request before its response arrives."""

    req_id: int
    started_at: datetime
    t0: float
    thread_key: str
    requested_model: str
    decision: RouteDecision
    model: str
    stream: bool
    ttfb_ms: float | None = None


def create_app(config: Config | None = None) -> FastAPI:
    """Builds the proxy's FastAPI app: routed `/v1/messages`, passthrough elsewhere.

    Args:
        config: Defaults to loading `config/config.yaml`.

    Returns:
        A configured `FastAPI` app.
    """
    config = config or load_config()
    logger.setLevel(_log_level(config))
    turn_logger = _create_turn_logger(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(timeout=None)
        yield
        await app.state.http_client.aclose()
        if turn_logger is not None:
            turn_logger.close()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.router = Router(config)
    app.state.capabilities = CapabilityRegistry(config.models)
    app.state.request_ids = itertools.count(1)
    app.state.turn_logger = turn_logger

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _handle_messages(request, app, next(app.state.request_ids))

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(request: Request, full_path: str):
        return await _forward_unmodified(request, app, full_path, next(app.state.request_ids))

    return app


def _log_level(config: Config) -> int:
    """The proxy's log level: `WRAP_LOG_LEVEL` (set by `wrap claude --debug`),
    else `proxy.log_level` from the config, else INFO.

    Args:
        config: The loaded config.

    Returns:
        A `logging` level.
    """
    name = os.environ.get("WRAP_LOG_LEVEL") or config.proxy.log_level or "info"
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else logging.INFO


def _create_turn_logger(config: Config) -> TurnLogger | None:
    """Builds the telemetry logger, or None if telemetry is off or can't start.

    The session ID comes from `WRAP_SESSION_ID`, set by `wrap claude`.
    A broken pricing file or database disables telemetry with a warning
    rather than stopping the proxy.

    Args:
        config: The loaded config.

    Returns:
        A `TurnLogger`, or None.
    """
    if not config.telemetry.enabled:
        return None
    try:
        pricing = PricingTable.load(REPO_ROOT / config.telemetry.pricing_file)
        return TurnLogger(REPO_ROOT / config.telemetry.db_path, pricing, os.environ.get("WRAP_SESSION_ID"))
    except Exception as exc:
        logger.warning("[yellow]telemetry disabled: %s[/yellow]", exc)
        return None


async def _handle_messages(request: Request, app: FastAPI, req_id: int) -> Response:
    """Routes a request, adapts it to the chosen model, then forwards upstream.

    If the API rejects the request with an error a known rule explains (the
    configured capabilities were stale), the correction is learned for the
    session, the request is re-adapted, and it's retried once.

    Args:
        request: Incoming request from the client (Claude Code).
        app: The FastAPI app, for shared state set up in `create_app`.
        req_id: Id used to tie this request's log lines together.

    Returns:
        A streamed or buffered `Response` mirroring the upstream reply.
    """
    config: Config = app.state.config
    router: Router = app.state.router
    capabilities: CapabilityRegistry = app.state.capabilities
    client: httpx.AsyncClient = app.state.http_client

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return await _forward_unmodified(request, app, "v1/messages", req_id, raw_body=raw_body)

    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()
    requested_model = body.get("model", "")
    decision = router.route(body.get("messages", []), requested_model)
    logger.info("#%d %s", req_id, describe_decision(decision))
    logger.debug("[dim]#%d request: %s[/dim]", req_id, describe_request(body))
    body["model"] = model = decision.model
    _adapt(body, model, capabilities, req_id)
    turn = _TurnContext(
        req_id=req_id,
        started_at=started_at,
        t0=t0,
        thread_key=thread_key(body),
        requested_model=requested_model,
        decision=decision,
        model=model,
        stream=bool(body.get("stream")),
    )

    url = f"{config.proxy.upstream_base_url}/v1/messages"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = upstream_request_headers(dict(request.headers))

    upstream_response, turn.ttfb_ms = await _send(client, url, headers, body, req_id)

    if upstream_response.status_code == 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        matched = match_error(error_body)
        if matched is None:
            _record_error(app, turn, upstream_response.status_code, error_body)
            return _error_response(req_id, model, upstream_response, error_body)
        rule, updates = matched
        configured = {k: getattr(capabilities.configured(model), k) for k in updates}
        if not capabilities.learn(model, updates):
            _record_error(app, turn, upstream_response.status_code, error_body)
            return _error_response(req_id, model, upstream_response, error_body)

        logger.warning(
            "[yellow]#%d %s rejected the request (%s): learned %s, config says %s. "
            "Retrying -- update models in config.yaml.[/yellow]",
            req_id, model, rule.name, escape(str(updates)), escape(str(configured)),
        )
        _adapt(body, model, capabilities, req_id)
        upstream_response, turn.ttfb_ms = await _send(client, url, headers, body, req_id)

    if upstream_response.status_code >= 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        _record_error(app, turn, upstream_response.status_code, error_body)
        return _error_response(req_id, model, upstream_response, error_body)

    status_code = upstream_response.status_code
    response_headers = filtered_headers(dict(upstream_response.headers))
    if turn.stream:
        parser = StreamUsageParser()

        def on_stream_done(interrupted: str | None) -> None:
            error = interrupted or parser.error or (None if parser.completed else "stream_incomplete")
            _record(app, turn, status_code, parser.usage, parser.served_model, parser.stop_reason, error)

        return StreamingResponse(
            _tee(upstream_response, parser, on_stream_done),
            status_code=status_code,
            headers=response_headers,
            background=_close_response(upstream_response),
        )

    content = await upstream_response.aread()
    await upstream_response.aclose()
    usage, served_model, stop_reason = parse_message_body(content)
    _record(app, turn, status_code, usage, served_model, stop_reason, None)
    return Response(content=content, status_code=status_code, headers=response_headers)


async def _tee(
    upstream_response: httpx.Response, parser: StreamUsageParser, on_done: Callable[[str | None], None]
) -> AsyncIterator[bytes]:
    """Relays upstream chunks unchanged while feeding a copy to the usage parser.

    Args:
        upstream_response: The streamed upstream response.
        parser: Observes every chunk; never affects what's relayed.
        on_done: Called exactly once when the stream ends, with None if it
            was fully relayed, `"client_disconnected"` if the client went
            away, or `"stream_incomplete"` if reading upstream failed.

    Yields:
        Each upstream chunk, byte-for-byte.
    """
    interrupted: str | None = "client_disconnected"
    try:
        async for chunk in upstream_response.aiter_raw():
            parser.feed(chunk)
            yield chunk
        interrupted = None
    except Exception:
        interrupted = "stream_incomplete"
        raise
    finally:
        on_done(interrupted)


def _record(
    app: FastAPI,
    turn: _TurnContext,
    status_code: int,
    usage: Usage,
    served_model: str | None,
    stop_reason: str | None,
    error: str | None,
) -> None:
    """Hands a finished request to the telemetry logger, if there is one; never raises.

    A request with no human question to route by is stored with tier
    "unrouted"; a mid-turn request (e.g. sending back tool results) is
    routed by its turn's question and flagged `was_tool_continuation`.

    Args:
        app: The FastAPI app, holding the `TurnLogger`.
        turn: What was known before the response.
        status_code: The upstream status.
        usage: Token counts from the response.
        served_model: The model the response says answered.
        stop_reason: The response's stop reason.
        error: Error type and message, or why a stream ended early.
    """
    turn_logger: TurnLogger | None = app.state.turn_logger
    if turn_logger is None:
        return
    try:
        decision = turn.decision
        turn_logger.record(
            TurnRecord(
                timestamp=turn.started_at.isoformat(),
                thread_key=turn.thread_key,
                requested_model=turn.requested_model,
                model_id=turn.model,
                served_model=served_model,
                tier=decision.tier if decision.routed else "unrouted",
                complexity_score=decision.complexity.score if decision.complexity else None,
                was_tool_continuation=bool(decision.turn and decision.turn.is_continuation),
                stream=turn.stream,
                status_code=status_code,
                stop_reason=stop_reason,
                usage=usage,
                ttfb_ms=turn.ttfb_ms,
                latency_ms=(time.monotonic() - turn.t0) * 1000,
                error=error,
            ),
            req_id=turn.req_id,
        )
    except Exception as exc:
        logger.warning("[yellow]telemetry: failed to record turn: %s[/yellow]", exc)


def _record_error(app: FastAPI, turn: _TurnContext, status_code: int, error_body: bytes) -> None:
    """Records a failed request with the start of the upstream error body."""
    error = error_body[:_ERROR_LOG_CHARS].decode(errors="replace")
    _record(app, turn, status_code, Usage(), None, None, error)


def _adapt(body: dict, model: str, capabilities: CapabilityRegistry, req_id: int) -> None:
    """Adapts the request body to the model's capabilities in place and logs the changes.

    Applies to every request, routed or not -- adapters only change settings
    the target model doesn't support, so a request Claude Code built for
    this very model passes through untouched.

    Args:
        body: The parsed request body, modified in place.
        model: The model ID the request will be sent to.
        capabilities: The registry of known per-model capabilities.
        req_id: Id used to tie this request's log lines together.
    """
    notes = adapt_request(body, capabilities.get(model))
    if notes:
        logger.debug("[dim]#%d adapted: %s[/dim]", req_id, escape("; ".join(notes)))


async def _send(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict, req_id: int
) -> tuple[httpx.Response, float]:
    """Sends a JSON body upstream as a streaming request and logs status/latency.

    Args:
        client: Shared HTTP client.
        url: Full upstream URL, including any query string.
        headers: Already-filtered request headers.
        body: Request body, serialized to JSON.
        req_id: Id used to tie this request's log lines together.

    Returns:
        The upstream response, with its body not yet read, and the time
        until its headers arrived in milliseconds.
    """
    start = time.monotonic()
    request = client.build_request("POST", url, headers=headers, content=json.dumps(body).encode())
    response = await client.send(request, stream=True)
    ttfb_ms = (time.monotonic() - start) * 1000
    logger.debug("[dim]#%d ← %s in %.0fms[/dim]", req_id, response.status_code, ttfb_ms)
    return response, ttfb_ms


def _error_response(req_id: int, model: str, upstream_response: httpx.Response, error_body: bytes) -> Response:
    """Logs an upstream error's body and returns it to the client unchanged.

    Args:
        req_id: Id used to tie this request's log lines together.
        model: The model ID the failed request was sent to, for the log line.
        upstream_response: The upstream error response (already read).
        error_body: The response body.

    Returns:
        A buffered `Response` with the upstream's status, headers, and body.
    """
    logger.warning(
        "[red]#%d upstream %s for %s: %s[/red]",
        req_id, upstream_response.status_code, model,
        escape(error_body[:_ERROR_LOG_CHARS].decode(errors="replace")),
    )
    return Response(
        content=error_body,
        status_code=upstream_response.status_code,
        headers=filtered_headers(dict(upstream_response.headers)),
    )


async def _forward_unmodified(
    request: Request, app: FastAPI, full_path: str, req_id: int, raw_body: bytes | None = None
) -> Response:
    """Forwards a request to the upstream API unchanged, with no routing.

    Args:
        request: Incoming request to forward.
        app: The FastAPI app, for shared state set up in `create_app`.
        full_path: Upstream path, relative to `config.proxy.upstream_base_url`.
        req_id: Id used to tie this request's log lines together.
        raw_body: Body to send if already read by the caller, to avoid
            re-reading the request stream.

    Returns:
        A streamed or buffered `Response` mirroring the upstream reply.
    """
    config: Config = app.state.config
    client: httpx.AsyncClient = app.state.http_client

    body = raw_body if raw_body is not None else await request.body()
    url = f"{config.proxy.upstream_base_url}/{full_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = upstream_request_headers(dict(request.headers))

    start = time.monotonic()
    upstream_request = client.build_request(request.method, url, headers=headers, content=body)
    upstream_response = await client.send(upstream_request, stream=True)
    latency_ms = (time.monotonic() - start) * 1000
    logger.debug(
        "[dim]#%d passthrough %s /%s ← %s in %.0fms[/dim]",
        req_id, request.method, escape(full_path), upstream_response.status_code, latency_ms,
    )
    response_headers = filtered_headers(dict(upstream_response.headers))

    if "text/event-stream" in upstream_response.headers.get("content-type", ""):
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=_close_response(upstream_response),
        )

    content = await upstream_response.aread()
    await upstream_response.aclose()
    return Response(content=content, status_code=upstream_response.status_code, headers=response_headers)


def _close_response(response: httpx.Response):
    """Wraps `response.aclose()` as a Starlette background task.

    `StreamingResponse` doesn't close the underlying httpx response on
    its own once the stream is exhausted, so this is passed as its
    `background` task to release the connection back to the pool after
    the client has been sent everything.

    Args:
        response: The streamed upstream `httpx.Response` to close.

    Returns:
        A `starlette.background.BackgroundTask` that closes `response`.
    """
    from starlette.background import BackgroundTask

    return BackgroundTask(response.aclose)
