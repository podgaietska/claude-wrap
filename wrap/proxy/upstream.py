from __future__ import annotations

import httpx

# Headers that must not be blindly relayed between hops.
_HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "host",
}


def filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def stream_upstream(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes,
):
    """Opens a streaming upstream request and yields (status_code, response_headers)
    once, then the caller reads .aiter_raw() from the returned response."""
    request = client.build_request(method, url, headers=filtered_headers(headers), content=content)
    response = await client.send(request, stream=True)
    return response
