from __future__ import annotations

import json
import secrets
import string
from collections.abc import Iterator

from fastapi.responses import Response, StreamingResponse

_ID_ALPHABET = string.ascii_letters + string.digits
_ID_LENGTH = 24
CHUNK_CHARS = 40
CACHE_HEADER = "x-wrap-cache"


def new_message_id() -> str:
    """A fresh `msg_` id, so no two cached replies share one.

    Returns:
        `msg_` followed by 24 random letters and digits.
    """
    return "msg_" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_LENGTH))


def build_message(content: list[dict], model: str, input_tokens: int, output_tokens: int) -> dict:
    """Builds a Messages API response body for a cached answer.

    Only text blocks are accepted: the cache stores nothing else, and a
    replayed tool call or thinking signature would not be valid for a new
    request. Each block is reduced to its type and text.

    Args:
        content: The stored content blocks.
        model: The model that originally wrote the answer.
        input_tokens: This request's estimated prompt size, since Claude
            Code reads usage to track how full its context is.
        output_tokens: The original answer's output tokens.

    Returns:
        A message shaped like a non-streaming response body.

    Raises:
        ValueError: If a block isn't text.
    """
    blocks = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ValueError(f"cached content must be text blocks, got {block!r:.80}")
        blocks.append({"type": "text", "text": block["text"]})
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }


def sse_events(message: dict, chunk_chars: int = CHUNK_CHARS) -> Iterator[bytes]:
    """Streams a message as the server-sent events the API would send.

    Mirrors a real stream: `message_start` with empty content and one
    output token, a `ping`, then per text block a `content_block_start`,
    `text_delta`s of up to `chunk_chars` characters and a
    `content_block_stop`, and finally `message_delta` with the stop reason
    and full usage, and `message_stop`.

    Args:
        message: A message from `build_message`.
        chunk_chars: Characters per `text_delta`.

    Yields:
        One encoded event at a time.
    """
    usage = message["usage"]
    yield _event("message_start", {
        "type": "message_start",
        "message": {
            **{k: v for k, v in message.items() if k not in ("content", "usage", "stop_reason", "stop_sequence")},
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {**usage, "output_tokens": 1},
        },
    })
    yield _event("ping", {"type": "ping"})
    for index, block in enumerate(message["content"]):
        yield _event("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""},
        })
        text = block["text"]
        for start in range(0, len(text), chunk_chars):
            yield _event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text[start : start + chunk_chars]},
            })
        yield _event("content_block_stop", {"type": "content_block_stop", "index": index})
    yield _event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": message["stop_sequence"]},
        "usage": usage,
    })
    yield _event("message_stop", {"type": "message_stop"})


def json_response(message: dict) -> Response:
    """A non-streaming reply for a cached answer.

    Args:
        message: A message from `build_message`.

    Returns:
        A JSON `Response`, marked with the `x-wrap-cache: hit` header.
    """
    return Response(
        content=json.dumps(message, ensure_ascii=False).encode(),
        media_type="application/json",
        headers={CACHE_HEADER: "hit"},
    )


def streaming_response(message: dict) -> StreamingResponse:
    """A streamed reply for a cached answer.

    Args:
        message: A message from `build_message`.

    Returns:
        A `text/event-stream` `StreamingResponse`, marked with the
        `x-wrap-cache: hit` header.
    """
    return StreamingResponse(
        sse_events(message),
        media_type="text/event-stream",
        headers={CACHE_HEADER: "hit", "cache-control": "no-cache"},
    )


def _event(event_type: str, data: dict) -> bytes:
    """Frames one event the way the API does: `event:` and `data:` lines, then a blank line."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode()
