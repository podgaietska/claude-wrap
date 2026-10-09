import dataclasses
import time
from datetime import timedelta

import pytest

from tests.test_economics import HAIKU, OPUS, PRICING, START, turn
from wrap.dashboard import queries
from wrap.dashboard.queries import build_stats, percentile, pick_bucket, request_rows
from wrap.telemetry.economics import analyze


def req(model, cache_read, cache_write, seconds, *, tier="small", score=0.1, continuation=False, thread="t1",
        status=200, ttfb=None, latency=None, error=None, cache_hit=False):
    t = turn(model, cache_read, cache_write, seconds=seconds, thread=thread, status=status)
    return dataclasses.replace(
        t,
        tier=tier,
        complexity_score=score,
        was_tool_continuation=continuation,
        cost_usd=PRICING.cost(model, t.usage) if status < 400 else None,
        ttfb_ms=ttfb,
        latency_ms=latency,
        error=error,
        cache_hit=cache_hit,
    )


def session():
    return [
        req(OPUS, 0, 900, 0, thread="title", tier="unrouted", score=None),
        req(OPUS, 0, 150_000, 1, tier="large", score=0.8, ttfb=900, latency=4000),
        req(OPUS, 150_000, 2_000, 30, tier="large", score=0.8, continuation=True, ttfb=700, latency=3000),
        req(HAIKU, 0, 152_000, 200, score=0.1, ttfb=300, latency=1000),
        req(HAIKU, 152_000, 1_000, 230, score=0.1, continuation=True, ttfb=200, latency=800),
        req(HAIKU, 0, 0, 240, score=0.5, status=529, latency=50),
        req(HAIKU, 153_000, 1_000, 260, score=0.0, ttfb=250, latency=900, error="stream ended early"),
    ]


def stats(turns, now=None, count_from=None):
    return build_stats(turns, PRICING, 0.5, now or START + timedelta(hours=1), count_from)


def test_summary_counts_kinds_failures_and_cost():
    turns = session()

    s = stats(turns).summary

    assert (s.requests, s.new_messages, s.tool_calls, s.side) == (7, 4, 2, 1)
    assert s.failed == 1
    assert s.interrupted == 1
    assert s.cost_usd == pytest.approx(sum(t.cost_usd for t in turns if t.cost_usd is not None))
    assert s.last_request == turns[-1].timestamp
    assert not s.live
    assert stats(turns, now=START + timedelta(seconds=300)).summary.live


def test_economics_match_wrap_stats():
    turns = session()

    view = stats(turns).economics
    totals = analyze(turns, PRICING)

    assert view.net_savings == pytest.approx(totals.net_savings)
    assert view.cache_penalty == pytest.approx(totals.cache_penalty)
    assert view.switches == totals.switches == 1
    assert view.baseline == OPUS


def test_routing_share_and_histogram_skip_tool_calls_and_side_requests():
    r = stats(session()).routing

    assert r.routed == 4
    assert r.small_share == pytest.approx(3 / 4)
    assert r.threshold == 0.5
    assert len(r.histogram) == 20
    assert r.histogram[16].large == 1  # 0.8
    assert r.histogram[2].small == 1  # 0.1
    assert r.histogram[0].small == 1  # 0.0
    assert r.histogram[10].small == 1  # the failed request is a new message too
    assert sum(b.small + b.large for b in r.histogram) == 4


def test_histogram_edges():
    turns = [req(HAIKU, 0, 10, i, score=score, tier="large" if score >= 0.5 else "small")
             for i, score in enumerate([0.0, 0.5, 1.0, 0.15, 0.35])]

    bins = stats(turns).routing.histogram

    assert bins[0].small == 1
    assert bins[10].large == 1
    assert bins[19].large == 1
    assert bins[3].small == 1
    assert bins[7].small == 1


def test_prompt_cache_and_models():
    turns = session()

    s = stats(turns)

    ok = [t for t in turns if t.status_code < 400]
    assert s.prompt_cache.read_tokens == sum(t.usage.cache_read_tokens for t in ok)
    assert s.prompt_cache.read_share == pytest.approx(s.prompt_cache.read_tokens / sum(t.usage.prompt_tokens for t in ok))
    assert [(m.tier, m.served_model, m.requests) for m in s.models] == [
        ("large", OPUS, 2), ("small", HAIKU, 4), ("unrouted", OPUS, 1)
    ]
    assert sum(m.cost_usd for m in s.models) == pytest.approx(s.summary.cost_usd)


def test_latency_percentiles_overall_and_per_model():
    rows = stats(session()).latency

    overall, haiku, opus = rows
    assert overall.model is None
    assert overall.n == 5
    assert overall.ttfb_p50 == 300
    assert overall.latency_p95 == 4000
    assert (haiku.model, haiku.n, haiku.latency_p50) == (HAIKU, 3, 900)
    assert opus.model == OPUS


@pytest.mark.parametrize("values, p, expected", [
    ([], 50, None),
    ([5.0], 95, 5.0),
    ([1.0, 2.0], 50, 1.0),
    ([1.0, 2.0], 95, 2.0),
    ([float(i) for i in range(1, 101)], 95, 95.0),
    ([float(i) for i in range(1, 101)], 50, 50.0),
])
def test_percentile(values, p, expected):
    assert percentile(values, p) == expected


def test_timeseries_sums_to_totals_and_keeps_empty_buckets():
    turns = session()

    s = stats(turns)
    points = s.timeseries.points

    assert s.timeseries.bucket_seconds == 60
    assert len(points) == 5  # 12:00 to 12:04, with 12:01 and 12:02 empty
    assert points[1].requests == points[2].requests == 0
    assert sum(p.requests for p in points) == 7
    actual = sum(p.cost_small + p.cost_large + p.cost_unrouted for p in points)
    assert actual == pytest.approx(s.summary.cost_usd)
    assert sum(p.counterfactual for p in points) == pytest.approx(s.economics.counterfactual_cost)
    assert sum(p.switches for p in points) == 1
    assert points[0].t == START.isoformat()


@pytest.mark.parametrize("span, expected", [
    (timedelta(minutes=30), 60),
    (timedelta(minutes=119), 60),
    (timedelta(hours=5), 300),
    (timedelta(days=2), 3600),
    (timedelta(days=20), 6 * 3600),
    (timedelta(days=100), 24 * 3600),
    (timedelta(days=700), 7 * 24 * 3600),
])
def test_pick_bucket(span, expected):
    assert pick_bucket(START, START + span) == expected


def test_pick_bucket_beyond_weekly_stays_under_the_cap():
    size = pick_bucket(START, START + timedelta(days=3000))
    assert (timedelta(days=3000).total_seconds() // size) + 1 <= queries.MAX_BUCKETS + 1


def test_count_from_keeps_the_previous_turn_as_context():
    turns = session()
    count_from = START + timedelta(seconds=200)  # the switch to Haiku

    s = stats(turns, count_from=count_from)

    assert s.summary.requests == 4
    assert s.economics.switches == 1  # still compared with the Opus turn before the window
    assert s.timeseries.points[0].t == (START + timedelta(seconds=180)).isoformat()


def test_cache_panel_only_after_a_hit():
    turns = session()
    assert stats(turns).cache is None

    turns[3] = dataclasses.replace(turns[3], cache_hit=True)
    assert stats(turns).cache.hits == 1


def test_empty_selection():
    s = stats([])

    assert s.summary.requests == 0
    assert s.routing.small_share is None
    assert s.prompt_cache.read_share is None
    assert s.timeseries.points == []
    assert s.latency[0].n == 0


def test_request_rows_newest_first_with_kinds_and_savings():
    turns = session()

    rows = request_rows(turns, PRICING, limit=3)

    assert [r.timestamp for r in rows] == [t.timestamp for t in reversed(turns[-3:])]
    assert [r.kind for r in rows] == ["new_message", "new_message", "tool_call"]
    failed, switched = rows[1], next(r for r in request_rows(turns, PRICING, 10) if r.switched)
    assert failed.status_code == 529 and failed.net_savings is None
    assert switched.served_model == HAIKU and switched.net_savings is not None
    assert rows[0].error == "stream ended early"


def test_request_rows_truncate_errors():
    turns = [req(HAIKU, 0, 10, 0, status=500, error="x" * 1000)]
    assert len(request_rows(turns, PRICING, 10)[0].error) == queries.ERROR_CHARS


@pytest.mark.slow
def test_build_stats_on_50k_requests_is_fast():
    turns = []
    for i in range(50_000):
        thread = f"t{i // 50}"
        model, tier = (HAIKU, "small") if i % 3 else (OPUS, "large")
        turns.append(req(model, 1_000 * (i % 50), 2_000, i * 5, tier=tier, thread=thread, continuation=i % 50 > 0,
                         ttfb=300, latency=1_000))

    started = time.perf_counter()
    build_stats(turns, PRICING, 0.5, START)
    elapsed = time.perf_counter() - started

    print(f"build_stats on 50k requests: {elapsed:.2f}s")
    assert elapsed < 5
