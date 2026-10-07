from datetime import datetime, timedelta, timezone

import pytest

from wrap.telemetry.db import TurnRecord
from wrap.telemetry.economics import analyze, turn_economics
from wrap.telemetry.pricing import ModelPricing, PricingTable
from wrap.telemetry.usage import Usage

OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-4-5-20251001"
PRICING = PricingTable(
    {
        "claude-opus-5-5": ModelPricing(input=4.0, output=20.0, cache_read=0.20, cache_write_5m=5.0, cache_write_1h=8.0),
        "claude-haiku-4-5": ModelPricing(input=1.0, output=5.0, cache_read=0.10, cache_write_5m=1.25, cache_write_1h=2.0),
    }
)
START = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def turn(model: str, cache_read: int, cache_write: int, *, seconds: float = 0, requested: str = OPUS,
         thread: str = "t1", input_tokens: int = 10, output: int = 500, status: int = 200,
         write_1h: bool = False) -> TurnRecord:
    return TurnRecord(
        timestamp=(START + timedelta(seconds=seconds)).isoformat(),
        session_id="s1",
        thread_key=thread,
        requested_model=requested,
        model_id=model,
        served_model=model,
        status_code=status,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_write,
            cache_creation_1h_tokens=cache_write if write_1h else 0,
            cache_creation_5m_tokens=0 if write_1h else cache_write,
        ),
    )


def test_thread_that_never_switches_has_no_penalty():
    turns = [
        turn(HAIKU, 0, 10_000, seconds=0),
        turn(HAIKU, 10_000, 2_000, seconds=30),
        turn(HAIKU, 12_000, 1_000, seconds=60),
    ]

    result = analyze(turns, PRICING)

    assert result.cache_penalty == pytest.approx(0)
    assert result.switches == 0
    plain_difference = sum(PRICING.lookup(OPUS).cost(t.usage) - PRICING.lookup(HAIKU).cost(t.usage) for t in turns)
    assert result.net_savings == pytest.approx(plain_difference)


def test_switch_to_haiku_with_large_cached_prompt_loses_money():
    opus_turn = turn(OPUS, 145_000, 5_000, seconds=0)
    haiku_turn = turn(HAIKU, 0, 150_000, seconds=60)

    economics = turn_economics(haiku_turn, opus_turn, PRICING, uses_1h=False)

    assert economics.switched
    assert economics.actual_cost == pytest.approx(0.19, abs=0.001)
    assert economics.counterfactual_cost == pytest.approx(0.04, abs=0.001)
    assert economics.cache_penalty > 0
    assert economics.net_savings < 0
    assert economics.recached_tokens == 150_000


def test_switching_back_to_the_requested_model_carries_a_penalty():
    turns = [
        turn(OPUS, 145_000, 5_000, seconds=0),
        turn(HAIKU, 0, 152_000, seconds=60),
        # Back on Opus for the next question: it still has its 150k from the
        # first turn, but has to write the 2k the Haiku turn added to the
        # conversation as well as the new 1k.
        turn(OPUS, 150_000, 3_000, seconds=90),
    ]

    economics = turn_economics(turns[2], turns[1], PRICING, uses_1h=False)
    result = analyze(turns, PRICING)

    assert economics.switched
    assert economics.saved_by_model == pytest.approx(0)
    assert economics.cache_penalty > 0
    assert result.switches == 2


def test_gap_beyond_ttl_charges_no_penalty_for_an_expiry_that_would_happen_anyway():
    previous = turn(OPUS, 145_000, 5_000, seconds=0)
    late_haiku = turn(HAIKU, 0, 150_000, seconds=6 * 60)

    economics = turn_economics(late_haiku, previous, PRICING, uses_1h=False)

    assert economics.counterfactual_usage.cache_read_tokens == 0
    assert economics.cache_penalty == pytest.approx(0)
    assert economics.saved_by_model > 0


def test_one_hour_ttl_keeps_the_counterfactual_cache_alive_longer():
    previous = turn(OPUS, 145_000, 5_000, seconds=0, write_1h=True)
    later = turn(HAIKU, 0, 150_000, seconds=30 * 60, write_1h=True)

    result = analyze([previous, later], PRICING)
    economics = turn_economics(later, previous, PRICING, uses_1h=True)

    assert economics.counterfactual_usage.cache_read_tokens == 150_000
    assert economics.counterfactual_usage.cache_creation_1h_tokens == 0
    assert result.cache_penalty > 0


def test_estimate_never_reads_less_from_cache_than_actually_happened():
    # A first turn that read a prefix cached by another conversation, and a
    # late turn whose cache was kept warm the same way, aren't penalized.
    turns = [
        turn(OPUS, 145_000, 5_000, seconds=0),
        turn(OPUS, 150_000, 1_000, seconds=20 * 60),
    ]

    result = analyze(turns, PRICING)

    assert result.cache_penalty == pytest.approx(0)
    assert result.net_savings == pytest.approx(0)


def test_threads_do_not_share_a_previous_prompt():
    turns = [
        turn(OPUS, 0, 150_000, seconds=0, thread="main"),
        turn(HAIKU, 0, 3_000, seconds=10, thread="title-generation"),
    ]

    result = analyze(turns, PRICING)

    assert result.switches == 0
    assert result.cache_penalty == pytest.approx(0)


def test_errors_and_unpriced_turns_are_skipped_and_counted():
    turns = [
        turn(OPUS, 0, 10_000, seconds=0),
        turn(HAIKU, 0, 0, seconds=10, status=529),
        turn("claude-unknown", 0, 10_000, seconds=20),
        turn(OPUS, 10_000, 1_000, seconds=30, requested="claude-unknown"),
    ]

    result = analyze(turns, PRICING)

    assert result.error_turns == 1
    assert result.unpriced_turns == 2
    assert result.requested_models == {OPUS}


def test_net_savings_splits_exactly_into_model_savings_and_cache_penalty():
    turns = [
        turn(OPUS, 0, 20_000, seconds=0),
        turn(HAIKU, 0, 22_000, seconds=20),
        turn(HAIKU, 22_000, 1_000, seconds=40),
        turn(OPUS, 20_000, 4_000, seconds=60),
        turn(OPUS, 24_000, 500, seconds=10 * 60),
        turn(HAIKU, 0, 900, seconds=70, thread="side"),
    ]

    result = analyze(turns, PRICING)

    assert result.net_savings == pytest.approx(result.saved_by_model - result.cache_penalty)
    assert result.net_savings == pytest.approx(result.counterfactual_cost - result.actual_cost)


def test_self_check_covers_only_unswitched_turns_within_ttl():
    turns = [
        turn(OPUS, 0, 20_000, seconds=0),  # first turn: nothing to compare
        turn(OPUS, 20_000, 1_000, seconds=30),  # eligible, matches
        turn(OPUS, 5_000, 16_500, seconds=60),  # eligible, cache read far below the estimate
        turn(HAIKU, 0, 22_000, seconds=90),  # switch: not eligible
        turn(HAIKU, 22_000, 1_000, seconds=20 * 60),  # beyond TTL: not eligible
    ]

    result = analyze(turns, PRICING)

    assert result.check_eligible == 2
    assert result.check_matched == 1
    assert result.check_rate == pytest.approx(0.5)


def test_no_eligible_turns_gives_no_check_rate():
    assert analyze([turn(OPUS, 0, 1_000)], PRICING).check_rate is None
