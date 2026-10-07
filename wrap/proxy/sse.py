from __future__ import annotations

import json
import logging

from wrap.telemetry.usage import Usage

logger = logging.getLogger("wrap.proxy")


class StreamUsageParser:
    """Reads token usage out of a `/v1/messages` server-sent event stream.

    The proxy feeds it every chunk it relays to the client. It only
    observes the bytes -- the relay never depends on it -- and `feed`
    never raises, so a format surprise can only cost telemetry, never
    the response.

    Attributes:
        usage: Token counts seen so far.
        served_model: The model that actually answered, from `message_start`.
        stop_reason: From the final `message_delta`.
        completed: True once `message_stop` has been seen.
        error: `"<type>: <message>"` from an in-stream `error` event.
    """

    def __init__(self) -> None:
        self.usage = Usage()
        self.served_model: str | None = None
        self.stop_reason: str | None = None
        self.completed = False
        self.error: str | None = None
        self._buffer = b""
        self._warned = False

    def feed(self, chunk: bytes) -> None:
        """Buffers a raw chunk and handles every complete event in it.

        Args:
            chunk: Bytes exactly as received from upstream; events may be
                split across chunks at any byte.
        """
        try:
            # A "\r\n" split across chunks is rejoined here before being
            # normalized, since the "\r" stays at the end of the buffer.
            self._buffer = (self._buffer + chunk).replace(b"\r\n", b"\n")
            while b"\n\n" in self._buffer:
                raw_event, self._buffer = self._buffer.split(b"\n\n", 1)
                self._handle_event(raw_event)
        except Exception:  # never let telemetry break the relay
            self._warn_once("unexpected error parsing stream chunk")

    def _handle_event(self, raw_event: bytes) -> None:
        data_lines = [line[5:].strip() for line in raw_event.split(b"\n") if line.startswith(b"data:")]
        if not data_lines:
            return
        try:
            event = json.loads(b"\n".join(data_lines))
        except json.JSONDecodeError:
            self._warn_once("skipped a malformed stream event")
            return
        if not isinstance(event, dict):
            return

        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            self.served_model = message.get("model") or self.served_model
            self.usage.merge(message.get("usage"))
        elif event_type == "message_delta":
            self.usage.merge(event.get("usage"))
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason:
                self.stop_reason = stop_reason
        elif event_type == "message_stop":
            self.completed = True
        elif event_type == "error":
            error = event.get("error") or {}
            self.error = f"{error.get('type', 'error')}: {error.get('message', '')}"

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            logger.debug("[dim]telemetry: %s[/dim]", message)


def parse_message_body(content: bytes) -> tuple[Usage, str | None, str | None]:
    """Reads usage from a non-streaming `/v1/messages` JSON response.

    Args:
        content: The response body.

    Returns:
        `(usage, served_model, stop_reason)`; empty usage and Nones if the
        body isn't a JSON object.
    """
    usage = Usage()
    try:
        message = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return usage, None, None
    if not isinstance(message, dict):
        return usage, None, None
    usage.merge(message.get("usage"))
    return usage, message.get("model"), message.get("stop_reason")
