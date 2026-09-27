from __future__ import annotations

# Headers that must not be blindly relayed between hops.
_HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "host",
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
