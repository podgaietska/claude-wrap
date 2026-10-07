from __future__ import annotations

import json
import logging

from wrap.telemetry.usage import Usage

logger = logging.getLogger("wrap.proxy")


class StreamUsageParser:
    """Reads token usage, and optionally the whole message, out of a
    `/v1/messages` server-sent event stream.

    The proxy feeds it every chunk it relays to the client. It only
    observes the bytes -- the relay never depends on it -- and `feed`
    never raises, so a format surprise can only cost telemetry or a cache
    entry, never the response.

    Attributes:
        usage: Token counts seen so far.
        served_model: The model that actually answered, from `message_start`.
        stop_reason: From the final `message_delta`.
        completed: True once `message_stop` has been seen.
        error: `"<type>: <message>"` from an in-stream `error` event.
    """

    def __init__(self, collect_content: bool = False) -> None:
        """Creates a parser.

        Args:
            collect_content: Also assemble the message's content blocks, for
                `message`. Off by default, since it keeps the whole
                response in memory.
        """
        self.usage = Usage()
        self.served_model: str | None = None
        self.stop_reason: str | None = None
        self.completed = False
        self.error: str | None = None
        self._buffer = b""
        self._warned = False
        self._assembler = MessageAssembler() if collect_content else None

    @property
    def message(self) -> dict | None:
        """The assembled message, shaped like a non-streaming response body.

        Returns:
            The message, or None if content wasn't collected, the stream
            didn't complete, it carried an error, or it held an event the
            assembler doesn't understand.
        """
        if self._assembler is None or not self.completed or self.error:
            return None
        return self._assembler.message()

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

        if self._assembler is not None:
            self._assembler.handle(event)
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


class MessageAssembler:
    """Rebuilds a complete message from its stream events.

    `message_start` gives the message's fields, `content_block_start` /
    `content_block_delta` / `content_block_stop` build each content block,
    and `message_delta` gives the stop reason and final usage. The result
    has the same shape as a non-streaming response body.

    A delta type it doesn't know makes the message unavailable rather than
    wrong: a cache must never store a partly understood answer.
    """

    def __init__(self) -> None:
        self._message: dict | None = None
        self._usage: dict = {}
        self._blocks: dict[int, dict] = {}
        self._partial_json: dict[int, str] = {}
        self._failed = False

    def handle(self, event: dict) -> None:
        """Applies one parsed stream event.

        Args:
            event: A decoded `data:` payload.
        """
        if self._failed:
            return
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message")
            if isinstance(message, dict):
                self._message = {k: v for k, v in message.items() if k not in ("content", "usage")}
                self._merge_usage(message.get("usage"))
        elif event_type == "content_block_start":
            block = event.get("content_block")
            index = event.get("index")
            if not isinstance(block, dict) or not isinstance(index, int):
                self._failed = True
                return
            self._blocks[index] = dict(block)
        elif event_type == "content_block_delta":
            self._apply_delta(event.get("index"), event.get("delta"))
        elif event_type == "content_block_stop":
            self._finish_block(event.get("index"))
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if self._message is not None:
                for key in ("stop_reason", "stop_sequence"):
                    if key in delta:
                        self._message[key] = delta[key]
            self._merge_usage(event.get("usage"))

    def message(self) -> dict | None:
        """The assembled message, or None if it couldn't be rebuilt."""
        if self._failed or self._message is None or self._partial_json:
            return None
        return {
            **self._message,
            "content": [self._blocks[i] for i in sorted(self._blocks)],
            "usage": dict(self._usage),
        }

    def _apply_delta(self, index, delta) -> None:
        block = self._blocks.get(index) if isinstance(index, int) else None
        if block is None or not isinstance(delta, dict):
            self._failed = True
            return
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta_type == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
        elif delta_type == "signature_delta":
            block["signature"] = delta.get("signature", "")
        elif delta_type == "input_json_delta":
            self._partial_json[index] = self._partial_json.get(index, "") + delta.get("partial_json", "")
        elif delta_type == "citations_delta":
            block.setdefault("citations", []).append(delta.get("citation"))
        else:
            self._failed = True

    def _finish_block(self, index) -> None:
        if not isinstance(index, int) or index not in self._blocks:
            self._failed = True
            return
        partial = self._partial_json.pop(index, None)
        if partial is None:
            return
        try:
            self._blocks[index]["input"] = json.loads(partial) if partial else {}
        except json.JSONDecodeError:
            self._failed = True

    def _merge_usage(self, usage) -> None:
        if isinstance(usage, dict):
            self._usage.update({k: v for k, v in usage.items() if v is not None})


def parse_message(content: bytes) -> dict | None:
    """Decodes a non-streaming `/v1/messages` JSON response.

    Args:
        content: The response body.

    Returns:
        The message, or None if the body isn't a JSON object.
    """
    try:
        message = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return message if isinstance(message, dict) else None


def parse_message_body(content: bytes) -> tuple[Usage, str | None, str | None]:
    """Reads usage from a non-streaming `/v1/messages` JSON response.

    Args:
        content: The response body.

    Returns:
        `(usage, served_model, stop_reason)`; empty usage and Nones if the
        body isn't a JSON object.
    """
    usage = Usage()
    message = parse_message(content)
    if message is None:
        return usage, None, None
    usage.merge(message.get("usage"))
    return usage, message.get("model"), message.get("stop_reason")
