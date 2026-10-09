from tests.test_economics import HAIKU, OPUS
from tests.test_telemetry import turn
from wrap.telemetry.requests import classify, count_kinds, main_threads


def test_main_thread_is_the_busiest_per_session():
    turns = [
        turn(session_id="a", thread_key="title"),
        turn(session_id="a", thread_key="main"),
        turn(session_id="a", thread_key="main"),
        turn(session_id="b", thread_key="only"),
    ]

    assert main_threads(turns) == {"a": "main", "b": "only"}


def test_classify_labels_messages_tool_calls_and_side_requests():
    turns = [
        turn(thread_key="title", tier="small"),
        turn(thread_key="main", tier="small"),
        turn(thread_key="main", tier="small", was_tool_continuation=True),
        turn(thread_key="main", tier="small", was_tool_continuation=True),
        turn(thread_key="main", tier="large", model_id=OPUS, served_model=OPUS),
        turn(thread_key="main", tier="unrouted"),
    ]

    assert classify(turns) == ["side", "new_message", "tool_call", "tool_call", "new_message", "side"]


def test_sessions_are_classified_independently():
    turns = [
        turn(session_id="a", thread_key="x", tier="small"),
        turn(session_id="a", thread_key="x", tier="small", was_tool_continuation=True),
        turn(session_id="b", thread_key="y", tier="large", served_model=HAIKU),
    ]

    assert count_kinds(turns) == {"new_message": 2, "tool_call": 1}
