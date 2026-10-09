import asyncio
import json
import re
from pathlib import Path

import pytest

from wrap.proxy.cache_response import (
    CACHE_HEADER,
    build_message,
    json_response,
    new_message_id,
    sse_events,
    streaming_response,
)
from wrap.proxy.sse import StreamUsageParser, parse_message
from wrap.telemetry.usage import Usage

FIXTURE = (Path(__file__).parent / "fixtures" / "messages_response_stream.txt").read_bytes()
MODEL = "claude-haiku-4-5-20251001"


def message(*texts: str) -> dict:
    return build_message([{"type": "text", "text": t} for t in texts], MODEL, input_tokens=120, output_tokens=42)


def event_types(stream: bytes) -> list[str]:
    """The `event:` names in order, with runs of deltas collapsed to one."""
    types = [line[len("event: "):] for line in stream.decode().splitlines() if line.startswith("event: ")]
    return [t for i, t in enumerate(types) if not (t == "content_block_delta" and types[i - 1] == t)]


def parse(stream: bytes) -> StreamUsageParser:
    parser = StreamUsageParser(collect_content=True)
    parser.feed(stream)
    return parser


def test_message_has_the_api_shape():
    msg = message("Hello.")

    assert re.fullmatch(r"msg_[A-Za-z0-9]{24}", msg["id"])
    assert msg == {
        "id": msg["id"],
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 42},
    }


def test_every_message_gets_a_fresh_id():
    assert len({new_message_id() for _ in range(1000)}) == 1000
    assert message("a")["id"] != message("a")["id"]


def test_text_blocks_are_reduced_to_type_and_text():
    msg = build_message([{"type": "text", "text": "Hi", "citations": None}], MODEL, 1, 1)

    assert msg["content"] == [{"type": "text", "text": "Hi"}]


@pytest.mark.parametrize("block", [
    {"type": "tool_use", "id": "t", "name": "Read", "input": {}},
    {"type": "thinking", "thinking": "hm", "signature": "s"},
    {"type": "text"},
    "text",
])
def test_non_text_content_is_refused(block):
    with pytest.raises(ValueError):
        build_message([block], MODEL, 1, 1)


@pytest.mark.parametrize("texts", [
    ("Python is a high-level programming language.",),
    ("First block.", "Second block, " * 20),
    ("",),
    ("héllo — 世界 🙂 " * 10,),
])
def test_stream_round_trips_through_the_parser(texts):
    msg = message(*texts)

    parser = parse(b"".join(sse_events(msg)))

    assert parser.message == msg
    assert parser.completed
    assert parser.error is None
    assert parser.stop_reason == "end_turn"
    assert parser.served_model == MODEL
    assert parser.usage == Usage(input_tokens=120, output_tokens=42)


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_stream_round_trips_when_read_in_small_chunks(chunk_size):
    msg = message("A longer answer that spans several deltas. " * 5)
    data = b"".join(sse_events(msg))

    parser = StreamUsageParser(collect_content=True)
    for i in range(0, len(data), chunk_size):
        parser.feed(data[i : i + chunk_size])

    assert parser.message == msg


def test_event_sequence_matches_a_real_stream():
    assert event_types(b"".join(sse_events(message("x" * 200)))) == event_types(FIXTURE)


def test_events_are_framed_like_a_real_stream():
    for raw in b"".join(sse_events(message("Hi"))).split(b"\n\n")[:-1]:
        event_line, data_line = raw.split(b"\n")
        assert event_line.startswith(b"event: ")
        assert data_line.startswith(b"data: ")
        assert json.loads(data_line[len(b"data: "):])["type"] == event_line[len(b"event: "):].decode()


def test_text_is_sent_in_chunks_of_at_most_chunk_chars():
    text = "abcdefghij" * 9

    deltas = [
        json.loads(raw.split(b"data: ", 1)[1])["delta"]["text"]
        for raw in sse_events(message(text), chunk_chars=40)
        if raw.startswith(b"event: content_block_delta")
    ]

    assert [len(d) for d in deltas] == [40, 40, 10]
    assert "".join(deltas) == text


def test_message_start_mirrors_the_api():
    first = next(sse_events(message("Hi")))
    start = json.loads(first.split(b"data: ", 1)[1])["message"]

    assert start["content"] == []
    assert start["stop_reason"] is None
    assert start["usage"]["output_tokens"] == 1
    assert start["usage"]["input_tokens"] == 120


def test_json_response_round_trips():
    msg = message("Hi")

    response = json_response(msg)

    assert response.media_type == "application/json"
    assert response.headers[CACHE_HEADER] == "hit"
    assert parse_message(response.body) == msg


def test_streaming_response_is_an_event_stream():
    msg = message("Hi there")
    response = streaming_response(msg)

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in response.body_iterator])

    assert response.media_type == "text/event-stream"
    assert response.headers[CACHE_HEADER] == "hit"
    assert parse(asyncio.run(collect())).message == msg
