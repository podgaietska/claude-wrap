import json
from pathlib import Path

import pytest

from wrap.proxy.sse import StreamUsageParser, parse_message, parse_message_body
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


def parse(data: bytes, chunk_size: int | None = None, collect_content: bool = False) -> StreamUsageParser:
    parser = StreamUsageParser(collect_content=collect_content)
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


EXPECTED_MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [{"type": "text", "text": "Python is a high-level programming language."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "cache_creation_input_tokens": 1500,
        "cache_read_input_tokens": 8000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1500, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 340,
        "service_tier": "standard",
    },
}


def stream(*events: dict) -> bytes:
    start = {"type": "message_start", "message": {
        "id": "msg_x", "type": "message", "role": "assistant", "model": "m", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1},
    }}
    end = [
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    return b"".join(event(e) for e in [start, *events, *end])


def block(index: int, content_block: dict, *deltas: dict) -> list[dict]:
    return [
        {"type": "content_block_start", "index": index, "content_block": content_block},
        *({"type": "content_block_delta", "index": index, "delta": d} for d in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def test_message_is_not_assembled_unless_asked():
    assert parse(STREAM).message is None


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 64])
def test_fixture_stream_assembles_into_the_full_message(chunk_size):
    assert parse(STREAM, chunk_size, collect_content=True).message == EXPECTED_MESSAGE


def test_tool_use_input_is_rebuilt_from_partial_json():
    data = stream(*block(
        0, {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        {"type": "input_json_delta", "partial_json": '{"file_pa'},
        {"type": "input_json_delta", "partial_json": 'th": "a.py"}'},
    ))

    content = parse(data, collect_content=True).message["content"]

    assert content == [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.py"}}]


def test_tool_use_with_no_input_deltas_keeps_empty_input():
    data = stream(*block(0, {"type": "tool_use", "id": "toolu_1", "name": "Ls", "input": {}},
                         {"type": "input_json_delta", "partial_json": ""}))

    assert parse(data, collect_content=True).message["content"][0]["input"] == {}


def test_thinking_signature_and_text_blocks_are_kept_in_index_order():
    data = stream(
        *block(0, {"type": "thinking", "thinking": ""},
               {"type": "thinking_delta", "thinking": "Let me "},
               {"type": "thinking_delta", "thinking": "think."},
               {"type": "signature_delta", "signature": "sig"}),
        *block(1, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Done."}),
    )

    message = parse(data, collect_content=True).message

    assert message["content"] == [
        {"type": "thinking", "thinking": "Let me think.", "signature": "sig"},
        {"type": "text", "text": "Done."},
    ]
    assert message["stop_reason"] == "end_turn"
    assert message["usage"] == {"input_tokens": 5, "output_tokens": 9}


def test_citations_are_collected_on_their_block():
    citation = {"type": "char_location", "cited_text": "x", "document_index": 0}
    data = stream(*block(0, {"type": "text", "text": ""},
                         {"type": "citations_delta", "citation": citation},
                         {"type": "text_delta", "text": "Cited."}))

    assert parse(data, collect_content=True).message["content"] == [
        {"type": "text", "text": "Cited.", "citations": [citation]},
    ]


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_multibyte_text_split_across_chunks_is_intact(chunk_size):
    text = "héllo — 世界 🙂"
    data = stream(*block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": text}))

    assert parse(data, chunk_size, collect_content=True).message["content"][0]["text"] == text


def test_truncated_stream_has_no_message():
    truncated = STREAM[: STREAM.index(b"event: message_stop")]

    assert parse(truncated, collect_content=True).message is None


def test_stream_with_error_event_has_no_message():
    data = STREAM.replace(
        b"event: message_stop",
        event({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}) + b"event: message_stop",
    )

    assert parse(data, collect_content=True).message is None


@pytest.mark.parametrize("bad_events", [
    [{"type": "content_block_delta", "index": 0, "delta": {"type": "mystery_delta", "x": 1}}],
    [{"type": "content_block_delta", "index": 3, "delta": {"type": "text_delta", "text": "orphan"}}],
    [{"type": "content_block_stop", "index": 3}],
])
def test_unknown_or_orphan_events_make_the_message_unavailable(bad_events):
    data = stream(*block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Hi"}), *bad_events)

    parser = parse(data, collect_content=True)

    assert parser.message is None
    assert parser.completed  # usage parsing is unaffected


def test_tool_input_that_is_not_valid_json_makes_the_message_unavailable():
    data = stream(*block(0, {"type": "tool_use", "id": "t", "name": "Read", "input": {}},
                         {"type": "input_json_delta", "partial_json": '{"broken'}))

    assert parse(data, collect_content=True).message is None


def test_parse_message_returns_the_body_as_a_dict():
    assert parse_message(json.dumps(EXPECTED_MESSAGE).encode()) == EXPECTED_MESSAGE


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b"\xff\xfe"])
def test_parse_message_rejects_non_objects(body):
    assert parse_message(body) is None
