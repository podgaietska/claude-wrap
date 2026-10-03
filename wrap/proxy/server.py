from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from rich.logging import RichHandler
from rich.markup import escape

from wrap.config import Config, load_config
from wrap.proxy.limits import LimitRegistry, parse_max_tokens_ceiling
from wrap.proxy.upstream import filtered_headers, upstream_request_headers
from wrap.routing.router import Router

_ERROR_LOG_CHARS = 500

logger = logging.getLogger("wrap.proxy")
if not logger.handlers:
    handler = RichHandler(show_time=True, show_path=False, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(config: Config | None = None) -> FastAPI:
    """Builds the proxy's FastAPI app: routed `/v1/messages`, passthrough elsewhere.

    Args:
        config: Defaults to loading `config/config.yaml`.

    Returns:
        A configured `FastAPI` app.
    """
    config = config or load_config()
    router = Router(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(timeout=None)
        yield
        await app.state.http_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.router = router
    app.state.limits = LimitRegistry(config.model_limits)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _handle_messages(request, app)

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(request: Request, full_path: str):
        return await _forward_unmodified(request, app, full_path)

    return app


async def _handle_messages(request: Request, app: FastAPI) -> Response:
    """Routes the newest turn's model, clamps max_tokens, then forwards upstream.

    If the API still rejects `max_tokens` (the configured limit is stale),
    the real ceiling is parsed from the error, remembered for the session,
    and the request is retried once.

    Args:
        request: Incoming request from the client (Claude Code).
        app: The FastAPI app, for shared state set up in `create_app`.

    Returns:
        A streamed or buffered `Response` mirroring the upstream reply.
    """
    config: Config = app.state.config
    router: Router = app.state.router
    limits: LimitRegistry = app.state.limits
    client: httpx.AsyncClient = app.state.http_client

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return await _forward_unmodified(request, app, "v1/messages", raw_body=raw_body)

    requested_model = body.get("model", "")
    decision = router.route(body.get("messages", []), requested_model)

    if decision.is_tool_continuation:
        logger.info("[dim]passthrough (tool continuation) -> %s[/dim]", requested_model)
    else:
        c = decision.complexity
        if c:
            logger.info(
                "[bold cyan]%s[/bold cyan] -> %s (score=%.2f, %s)",
                decision.tier, decision.model, c.score, c.reasoning,
            )
        else:
            logger.info("%s -> %s", decision.tier, decision.model)
        body["model"] = decision.model

    model = body.get("model", "")
    _clamp_max_tokens(body, model, limits)

    url = f"{config.proxy.upstream_base_url}/v1/messages"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = upstream_request_headers(dict(request.headers))

    upstream_response = await _send(client, url, headers, body)

    if upstream_response.status_code == 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        ceiling = parse_max_tokens_ceiling(error_body)
        sent = body.get("max_tokens")
        if ceiling is None or sent is None or ceiling >= sent:
            return _error_response(model, upstream_response, error_body)

        logger.warning(
            "[yellow]%s rejected max_tokens=%s; API limit is %s (config says %s). "
            "Retrying with %s -- update model_limits in config.yaml.[/yellow]",
            model, sent, ceiling, limits.configured_max_output_tokens(model), ceiling,
        )
        limits.learn(model, ceiling)
        body["max_tokens"] = ceiling
        upstream_response = await _send(client, url, headers, body)

    if upstream_response.status_code >= 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        return _error_response(model, upstream_response, error_body)

    response_headers = filtered_headers(dict(upstream_response.headers))
    if body.get("stream"):
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=_close_response(upstream_response),
        )

    content = await upstream_response.aread()
    await upstream_response.aclose()
    return Response(content=content, status_code=upstream_response.status_code, headers=response_headers)


def _clamp_max_tokens(body: dict, model: str, limits: LimitRegistry) -> None:
    """Clamps the request body's `max_tokens` to the model's known ceiling, in place.

    Applies to every request, routed or passthrough -- clamping down to a
    model's real limit can't make a valid request invalid.

    Args:
        body: The parsed request body, modified in place.
        model: The model ID the request will be sent to.
        limits: The registry of known per-model ceilings.
    """
    requested = body.get("max_tokens")
    clamped = limits.clamp(model, requested)
    if clamped != requested:
        logger.info("[dim]clamped max_tokens %s -> %s for %s[/dim]", requested, clamped, model)
        body["max_tokens"] = clamped


async def _send(client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict) -> httpx.Response:
    """Sends a JSON body upstream as a streaming request and logs status/latency.

    Args:
        client: Shared HTTP client.
        url: Full upstream URL, including any query string.
        headers: Already-filtered request headers.
        body: Request body, serialized to JSON.

    Returns:
        The upstream response, with its body not yet read.
    """
    start = time.monotonic()
    request = client.build_request("POST", url, headers=headers, content=json.dumps(body).encode())
    response = await client.send(request, stream=True)
    latency_ms = (time.monotonic() - start) * 1000
    logger.info("[dim]upstream responded %s in %.0fms[/dim]", response.status_code, latency_ms)
    return response


def _error_response(model: str, upstream_response: httpx.Response, error_body: bytes) -> Response:
    """Logs an upstream error's body and returns it to the client unchanged.

    Args:
        model: The model ID the failed request was sent to, for the log line.
        upstream_response: The upstream error response (already read).
        error_body: The response body.

    Returns:
        A buffered `Response` with the upstream's status, headers, and body.
    """
    logger.warning(
        "[red]upstream %s for %s: %s[/red]",
        upstream_response.status_code, model, escape(error_body[:_ERROR_LOG_CHARS].decode(errors="replace")),
    )
    return Response(
        content=error_body,
        status_code=upstream_response.status_code,
        headers=filtered_headers(dict(upstream_response.headers)),
    )


async def _forward_unmodified(request: Request, app: FastAPI, full_path: str, raw_body: bytes | None = None) -> Response:
    """Forwards a request to the upstream API unchanged, with no routing.

    Args:
        request: Incoming request to forward.
        app: The FastAPI app, for shared state set up in `create_app`.
        full_path: Upstream path, relative to `config.proxy.upstream_base_url`.
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

    upstream_request = client.build_request(request.method, url, headers=headers, content=body)
    upstream_response = await client.send(upstream_request, stream=True)
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
