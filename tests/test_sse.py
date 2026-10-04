import json
from pathlib import Path

import pytest

from wrap.proxy.sse import StreamUsageParser, parse_message_body
from wrap.telemetry.usage import Usage

STREAM = (Path(__file__).parent / "fixtures" / "messages_response_stream.txt").read_bytes()
EXPECTED_USAGE = Usage(
    input_tokens=12,
    output_tokens=340,
    cache_read_tokens=8000,
    cache_creation_tokens=1500,
    cache_creation_5m_tokens=1500,
    cache_creation_1h_tokens=0,
)


def parse(data: bytes, chunk_size: int | None = None) -> StreamUsageParser:
    parser = StreamUsageParser()
    if chunk_size is None:
        parser.feed(data)
    else:
        for i in range(0, len(data), chunk_size):
            parser.feed(data[i : i + chunk_size])
    return parser


def event(data: dict) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def test_whole_stream_yields_usage_model_and_stop_reason():
    parser = parse(STREAM)

    assert parser.usage == EXPECTED_USAGE
    assert parser.served_model == "claude-haiku-4-5-20251001"
    assert parser.stop_reason == "end_turn"
    assert parser.completed
    assert parser.error is None


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_chunked_stream_gives_identical_results(chunk_size):
    whole, chunked = parse(STREAM), parse(STREAM, chunk_size)

    assert chunked.usage == whole.usage
    assert chunked.served_model == whole.served_model
    assert chunked.stop_reason == whole.stop_reason
    assert chunked.completed


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_crlf_line_endings_are_handled(chunk_size):
    parser = parse(STREAM.replace(b"\n", b"\r\n"), chunk_size)

    assert parser.usage == EXPECTED_USAGE
    assert parser.completed


def test_malformed_data_line_is_skipped():
    data = b"event: content_block_delta\ndata: {not json\n\n" + STREAM

    parser = parse(data)

    assert parser.usage == EXPECTED_USAGE
    assert parser.completed


def test_error_event_is_recorded():
    data = event({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

    parser = parse(data)

    assert parser.error == "overloaded_error: Overloaded"
    assert not parser.completed


def test_stream_without_message_stop_is_not_completed():
    truncated = STREAM[: STREAM.index(b"event: message_stop")]

    parser = parse(truncated)

    assert parser.usage.output_tokens == 340
    assert not parser.completed


def test_message_delta_usage_overlays_rather_than_sums():
    data = (
        event({"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 5, "output_tokens": 1}}})
        + event({"type": "message_delta", "delta": {}, "usage": {"output_tokens": 10}})
        + event({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 25}})
    )

    parser = parse(data)

    assert parser.usage.input_tokens == 5
    assert parser.usage.output_tokens == 25
    assert parser.stop_reason == "max_tokens"


def test_parse_message_body_reads_non_streaming_response():
    body = {
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 50, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0},
    }

    usage, served_model, stop_reason = parse_message_body(json.dumps(body).encode())

    assert usage == Usage(input_tokens=3, output_tokens=50, cache_read_tokens=900)
    assert served_model == "claude-sonnet-5"
    assert stop_reason == "tool_use"


def test_parse_message_body_tolerates_non_json():
    usage, served_model, stop_reason = parse_message_body(b"not json")

    assert usage == Usage()
    assert served_model is None
    assert stop_reason is None
