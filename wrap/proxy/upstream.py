from __future__ import annotations

# Headers that must not be blindly relayed between hops.
_HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "host",
    "accept-encoding",
}


def filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    """Strips hop-by-hop headers before relaying a request or response.

    `content-length`/`transfer-encoding` in particular must not be copied
    verbatim between hops: httpx recalculates them for the outgoing
    request/response, and forwarding stale values causes clients to hang
    or truncate the body.

    Args:
        headers: The original headers, as received from the client or
            from the upstream response.

    Returns:
        A new dict with hop-by-hop header names (case-insensitive) removed;
        all other headers (including auth headers) are passed through
        unchanged.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def upstream_request_headers(headers: dict[str, str]) -> dict[str, str]:
    """Prepares client request headers for forwarding, asking for an uncompressed reply.

    Upstream compresses responses when allowed, and the proxy needs to read
    response bodies (e.g. to parse error messages). Dropping the client's
    `accept-encoding` isn't enough since httpx adds its own gzip default,
    so `identity` is set explicitly.

    Args:
        headers: The original request headers from the client.

    Returns:
        Filtered headers with `accept-encoding: identity`.
    """
    return {**filtered_headers(headers), "accept-encoding": "identity"}
