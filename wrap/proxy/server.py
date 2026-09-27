from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from rich.logging import RichHandler

from wrap.config import Config, load_config
from wrap.proxy.upstream import filtered_headers
from wrap.routing.router import Router

logger = logging.getLogger("wrap.proxy")
if not logger.handlers:
    handler = RichHandler(show_time=True, show_path=False, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(config: Config | None = None) -> FastAPI:
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

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _handle_messages(request, app)

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(request: Request, full_path: str):
        return await _forward_unmodified(request, app, full_path)

    return app


async def _handle_messages(request: Request, app: FastAPI) -> Response:
    config: Config = app.state.config
    router: Router = app.state.router
    client: httpx.AsyncClient = app.state.http_client

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return await _forward_unmodified(request, app, "v1/messages", raw_body=raw_body)

    messages = body.get("messages", [])
    requested_model = body.get("model", "")
    decision = router.route(messages, requested_model)

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
        raw_body = json.dumps(body).encode()

    url = f"{config.proxy.upstream_base_url}/v1/messages"
    headers = filtered_headers(dict(request.headers))

    start = time.monotonic()
    upstream_request = client.build_request("POST", url, headers=headers, content=raw_body)
    upstream_response = await client.send(upstream_request, stream=True)
    latency_ms = (time.monotonic() - start) * 1000
    logger.info("[dim]upstream responded %s in %.0fms[/dim]", upstream_response.status_code, latency_ms)

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


async def _forward_unmodified(request: Request, app: FastAPI, full_path: str, raw_body: bytes | None = None) -> Response:
    config: Config = app.state.config
    client: httpx.AsyncClient = app.state.http_client

    body = raw_body if raw_body is not None else await request.body()
    url = f"{config.proxy.upstream_base_url}/{full_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = filtered_headers(dict(request.headers))

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
    from starlette.background import BackgroundTask

    return BackgroundTask(response.aclose)
