from datetime import datetime, timedelta, timezone

from tests.test_economics import OPUS, PRICING
from wrap.telemetry import db
from wrap.telemetry.db import TurnRecord
from wrap.telemetry.usage import Usage

START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def add_turn(
    conn,
    session_id,
    tier,
    model,
    cache_read,
    cache_write,
    seconds,
    status=200,
    thread="t1",
    continuation=False,
    score=None,
    ttfb_ms=None,
    latency_ms=None,
    error=None,
):
    """Inserts one request into a telemetry database, priced with the test `PRICING`."""
    usage = Usage(
        input_tokens=10,
        output_tokens=500,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
        cache_creation_5m_tokens=cache_write,
    )
    db.insert_turn(
        conn,
        TurnRecord(
            timestamp=(START + timedelta(seconds=seconds)).isoformat(),
            session_id=session_id,
            thread_key=thread,
            was_tool_continuation=continuation,
            requested_model=OPUS,
            model_id=model,
            served_model=model,
            tier=tier,
            complexity_score=score,
            status_code=status,
            usage=usage,
            cost_usd=PRICING.cost(model, usage) if status < 400 else None,
            ttfb_ms=ttfb_ms,
            latency_ms=latency_ms,
            error=error,
        ),
    )
