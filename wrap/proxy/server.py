from __future__ import annotations

import itertools
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from rich.logging import RichHandler
from rich.markup import escape

from wrap.adapt.adapters import adapt_request
from wrap.adapt.error_rules import match_error
from wrap.adapt.registry import CapabilityRegistry
from wrap.config import Config, load_config
from wrap.proxy.describe import describe_decision, describe_request
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(timeout=None)
        yield
        await app.state.http_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.router = Router(config)
    app.state.capabilities = CapabilityRegistry(config.models)
    app.state.request_ids = itertools.count(1)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _handle_messages(request, app, next(app.state.request_ids))

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(request: Request, full_path: str):
        return await _forward_unmodified(request, app, full_path, next(app.state.request_ids))

    return app


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

    decision = router.route(body.get("messages", []), body.get("model", ""))
    logger.info("#%d %s  %s", req_id, describe_request(body), describe_decision(decision))
    body["model"] = model = decision.model
    _adapt(body, model, capabilities, req_id)

    url = f"{config.proxy.upstream_base_url}/v1/messages"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = upstream_request_headers(dict(request.headers))

    upstream_response = await _send(client, url, headers, body, req_id)

    if upstream_response.status_code == 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        matched = match_error(error_body)
        if matched is None:
            return _error_response(req_id, model, upstream_response, error_body)
        rule, updates = matched
        configured = {k: getattr(capabilities.configured(model), k) for k in updates}
        if not capabilities.learn(model, updates):
            return _error_response(req_id, model, upstream_response, error_body)

        logger.warning(
            "[yellow]#%d %s rejected the request (%s): learned %s, config says %s. "
            "Retrying -- update models in config.yaml.[/yellow]",
            req_id, model, rule.name, escape(str(updates)), escape(str(configured)),
        )
        _adapt(body, model, capabilities, req_id)
        upstream_response = await _send(client, url, headers, body, req_id)

    if upstream_response.status_code >= 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        return _error_response(req_id, model, upstream_response, error_body)

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
        logger.info("[dim]#%d adapted for %s: %s[/dim]", req_id, model, escape("; ".join(notes)))


async def _send(client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict, req_id: int) -> httpx.Response:
    """Sends a JSON body upstream as a streaming request and logs status/latency.

    Args:
        client: Shared HTTP client.
        url: Full upstream URL, including any query string.
        headers: Already-filtered request headers.
        body: Request body, serialized to JSON.
        req_id: Id used to tie this request's log lines together.

    Returns:
        The upstream response, with its body not yet read.
    """
    start = time.monotonic()
    request = client.build_request("POST", url, headers=headers, content=json.dumps(body).encode())
    response = await client.send(request, stream=True)
    latency_ms = (time.monotonic() - start) * 1000
    logger.info("[dim]#%d ← %s in %.0fms[/dim]", req_id, response.status_code, latency_ms)
    return response


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
    logger.info(
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
