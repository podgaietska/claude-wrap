"""The dashboard's numbers, computed from `turn_log` rows.

Pure functions over `TurnRecord`s: no database or HTTP here. Each panel
on the page has a builder; `build_stats` runs them all. Routing economics
and request kinds come from the same functions `wrap stats` uses, so the
two reports agree.

Turns before `count_from` are context only: they let the first counted
turn of a conversation be compared with the turn before it, but are not
otherwise counted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from wrap.telemetry.db import TurnRecord
from wrap.telemetry.economics import TurnEconomics, analyze, is_counted, iter_turn_economics, succeeded_turn
from wrap.telemetry.pricing import PricingTable
from wrap.telemetry.requests import RequestKind, classify

# A selection whose latest request is this recent is treated as live.
LIVE_SECONDS = 120
# Candidate time-bucket sizes, smallest first; the first giving at most
# MAX_BUCKETS buckets is used.
BUCKET_SIZES = (60, 5 * 60, 15 * 60, 60 * 60, 6 * 60 * 60, 24 * 60 * 60, 7 * 24 * 60 * 60)
MAX_BUCKETS = 120
HISTOGRAM_BINS = 20
ERROR_CHARS = 200
TIERS = ("small", "large", "unrouted")


@dataclass
class Summary:
    """Headline counts for the selection.

    Attributes:
        cost_usd: Total cost of priced, successful requests.
        requests: Number of requests.
        new_messages: Requests that started a turn with a typed question.
        tool_calls: Mid-turn requests, e.g. sending back tool results.
        side: Background calls and subagents.
        failed: Requests with no response or an error status.
        interrupted: Successful responses whose stream ended early.
        unpriced: Successful requests whose model has no price.
        last_request: Timestamp of the latest request.
        live: True if the latest request was under `LIVE_SECONDS` ago.
    """

    cost_usd: float = 0.0
    requests: int = 0
    new_messages: int = 0
    tool_calls: int = 0
    side: int = 0
    failed: int = 0
    interrupted: int = 0
    unpriced: int = 0
    last_request: str | None = None
    live: bool = False


@dataclass
class EconomicsView:
    """Routing economics (see `wrap.telemetry.economics`), as the page shows them.

    Attributes:
        baseline: The model every request asked for, or None if they
            asked for several.
        check_rate: Share of checkable requests whose estimate matched,
            None if none could be checked.
    """

    counterfactual_cost: float
    actual_cost: float
    saved_by_model: float
    cache_penalty: float
    net_savings: float
    switches: int
    recached_tokens: int
    check_eligible: int
    check_rate: float | None
    baseline: str | None


@dataclass
class HistogramBin:
    """New messages whose complexity score fell in `[lo, hi)` (the last bin includes 1.0)."""

    lo: float
    hi: float
    small: int = 0
    large: int = 0


@dataclass
class Routing:
    """How the router split new messages between tiers.

    Attributes:
        threshold: The configured complexity threshold.
        routed: New messages routed to a tier.
        small_share: Share of `routed` sent to the small tier, None if none.
        histogram: Complexity scores in `HISTOGRAM_BINS` bins over [0, 1].
    """

    threshold: float
    routed: int
    small_share: float | None
    histogram: list[HistogramBin]


@dataclass
class PromptCache:
    """Anthropic prompt-cache use over successful requests.

    Attributes:
        read_share: Share of prompt tokens read from the cache, None
            without any prompt tokens.
    """

    read_tokens: int
    write_tokens: int
    prompt_tokens: int
    read_share: float | None


@dataclass
class LatencyStats:
    """Nearest-rank percentiles in milliseconds, over successful requests.

    Attributes:
        model: The served model, or None for all models together.
        n: Requests with a recorded total latency.
    """

    model: str | None
    n: int
    ttfb_p50: float | None
    ttfb_p95: float | None
    latency_p50: float | None
    latency_p95: float | None


@dataclass
class ModelRow:
    """Totals for one tier and served model, as in `wrap stats`."""

    tier: str
    served_model: str | None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_share: float | None = None


@dataclass
class Point:
    """One time bucket, starting at `t`.

    Attributes:
        cost_small, cost_large, cost_unrouted: Actual cost per tier.
        counterfactual: What the bucket's requests would have cost on the
            requested model (actual cost for requests that can't be priced
            both ways).
    """

    t: str
    cost_small: float = 0.0
    cost_large: float = 0.0
    cost_unrouted: float = 0.0
    counterfactual: float = 0.0
    requests: int = 0
    switches: int = 0


@dataclass
class Timeseries:
    """Cost over time in equal buckets, empty ones included."""

    bucket_seconds: int
    points: list[Point]


@dataclass
class ResponseCache:
    """Semantic response cache (Phase C); only reported once it has served a hit."""

    hits: int
    hit_rate: float


@dataclass
class Stats:
    """Everything `/api/stats` returns for a selection, one field per panel."""

    summary: Summary
    economics: EconomicsView
    routing: Routing
    prompt_cache: PromptCache
    latency: list[LatencyStats]
    models: list[ModelRow]
    timeseries: Timeseries
    cache: ResponseCache | None = None


@dataclass
class RequestRow:
    """One request in the recent-requests table. Message content is never stored."""

    id: int | None
    timestamp: str
    kind: RequestKind
    tier: str | None
    score: float | None
    requested_model: str | None
    served_model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float | None
    net_savings: float | None
    switched: bool
    ttfb_ms: float | None
    latency_ms: float | None
    status_code: int | None
    error: str | None


@dataclass
class _Annotated:
    """Turns with their request kinds and per-turn economics, keyed by object identity."""

    counted: list[TurnRecord]
    kinds: dict[int, RequestKind] = field(default_factory=dict)
    economics: dict[int, TurnEconomics | None] = field(default_factory=dict)


def build_stats(
    turns: list[TurnRecord],
    pricing: PricingTable,
    threshold: float,
    now: datetime,
    count_from: datetime | None = None,
) -> Stats:
    """Computes every panel's numbers for a selection of requests.

    Args:
        turns: The selection, oldest first, including any context turns
            before `count_from`.
        pricing: Rates for routing economics.
        threshold: The routing complexity threshold.
        now: The current time, for `Summary.live`.
        count_from: Count only turns at or after this time.

    Returns:
        The `Stats`.
    """
    annotated = _annotate(turns, pricing, count_from)
    counted = annotated.counted
    hits = sum(1 for t in counted if t.cache_hit)
    new_messages = sum(1 for t in counted if annotated.kinds[id(t)] == "new_message")
    return Stats(
        summary=summary(annotated, now),
        economics=economics_view(turns, pricing, count_from),
        routing=routing(annotated, threshold),
        prompt_cache=prompt_cache(counted),
        latency=latency(counted),
        models=models(counted),
        timeseries=timeseries(annotated, count_from),
        cache=ResponseCache(hits=hits, hit_rate=hits / new_messages if new_messages else 0.0) if hits else None,
    )


def summary(annotated: _Annotated, now: datetime) -> Summary:
    """Headline counts (see `Summary`)."""
    result = Summary()
    for t in annotated.counted:
        result.requests += 1
        kind = annotated.kinds[id(t)]
        if kind == "new_message":
            result.new_messages += 1
        elif kind == "tool_call":
            result.tool_calls += 1
        else:
            result.side += 1
        if not succeeded_turn(t):
            result.failed += 1
            continue
        if t.error:
            result.interrupted += 1
        if t.cost_usd is None:
            result.unpriced += 1
        else:
            result.cost_usd += t.cost_usd
    if annotated.counted:
        latest = max(annotated.counted, key=lambda t: datetime.fromisoformat(t.timestamp))
        result.last_request = latest.timestamp
        result.live = (now - datetime.fromisoformat(latest.timestamp)).total_seconds() < LIVE_SECONDS
    return result


def economics_view(turns: list[TurnRecord], pricing: PricingTable, count_from: datetime | None = None) -> EconomicsView:
    """Routing economics, the same totals `wrap stats` prints."""
    e = analyze(turns, pricing, count_from)
    return EconomicsView(
        counterfactual_cost=e.counterfactual_cost,
        actual_cost=e.actual_cost,
        saved_by_model=e.saved_by_model,
        cache_penalty=e.cache_penalty,
        net_savings=e.net_savings,
        switches=e.switches,
        recached_tokens=e.recached_tokens,
        check_eligible=e.check_eligible,
        check_rate=e.check_rate,
        baseline=next(iter(e.requested_models)) if len(e.requested_models) == 1 else None,
    )


def routing(annotated: _Annotated, threshold: float) -> Routing:
    """Tier split and complexity-score histogram of new messages."""
    width = 1 / HISTOGRAM_BINS
    bins = [HistogramBin(lo=round(i * width, 4), hi=round((i + 1) * width, 4)) for i in range(HISTOGRAM_BINS)]
    small = routed = 0
    for t in annotated.counted:
        if annotated.kinds[id(t)] != "new_message" or t.tier not in ("small", "large"):
            continue
        routed += 1
        small += t.tier == "small"
        if t.complexity_score is not None:
            index = min(int(round(t.complexity_score * HISTOGRAM_BINS, 9)), HISTOGRAM_BINS - 1)
            bin_ = bins[max(index, 0)]
            if t.tier == "small":
                bin_.small += 1
            else:
                bin_.large += 1
    return Routing(threshold=threshold, routed=routed, small_share=small / routed if routed else None, histogram=bins)


def prompt_cache(turns: list[TurnRecord]) -> PromptCache:
    """Prompt-cache reads and writes over successful requests."""
    ok = [t for t in turns if succeeded_turn(t)]
    read = sum(t.usage.cache_read_tokens for t in ok)
    prompt = sum(t.usage.prompt_tokens for t in ok)
    return PromptCache(
        read_tokens=read,
        write_tokens=sum(t.usage.cache_creation_tokens for t in ok),
        prompt_tokens=prompt,
        read_share=read / prompt if prompt else None,
    )


def latency(turns: list[TurnRecord]) -> list[LatencyStats]:
    """Latency percentiles over all models (first), then per served model."""
    ok = [t for t in turns if succeeded_turn(t)]
    by_model: dict[str | None, list[TurnRecord]] = defaultdict(list)
    for t in ok:
        by_model[t.model].append(t)
    rows = [_latency_stats(None, ok)]
    rows += [_latency_stats(model, group) for model, group in sorted(by_model.items(), key=lambda kv: kv[0] or "")]
    return rows


def models(turns: list[TurnRecord]) -> list[ModelRow]:
    """Totals per tier and served model, largest cost first."""
    rows: dict[tuple[str, str | None], ModelRow] = {}
    for t in turns:
        tier = _tier(t)
        row = rows.setdefault((tier, t.model), ModelRow(tier=tier, served_model=t.model))
        row.requests += 1
        row.input_tokens += t.usage.input_tokens
        row.output_tokens += t.usage.output_tokens
        row.cache_read_tokens += t.usage.cache_read_tokens
        row.cache_creation_tokens += t.usage.cache_creation_tokens
        row.cost_usd += t.cost_usd or 0.0
    for row in rows.values():
        prompt = row.input_tokens + row.cache_read_tokens + row.cache_creation_tokens
        row.cache_read_share = row.cache_read_tokens / prompt if prompt else None
    return sorted(rows.values(), key=lambda r: (-r.cost_usd, r.tier))


def timeseries(annotated: _Annotated, count_from: datetime | None = None) -> Timeseries:
    """Cost per tier and counterfactual cost in equal time buckets."""
    counted = annotated.counted
    if not counted:
        return Timeseries(bucket_seconds=BUCKET_SIZES[0], points=[])
    times = [datetime.fromisoformat(t.timestamp) for t in counted]
    start = count_from or min(times)
    size = pick_bucket(start, max(times))
    origin = _floor(start, size)
    points = [
        Point(t=_iso(origin + timedelta(seconds=i * size))) for i in range(_bucket_index(max(times), origin, size) + 1)
    ]
    for t, when in zip(counted, times, strict=True):
        point = points[_bucket_index(when, origin, size)]
        point.requests += 1
        if not succeeded_turn(t):
            continue
        cost = t.cost_usd or 0.0
        setattr(point, f"cost_{_tier(t)}", getattr(point, f"cost_{_tier(t)}") + cost)
        economics = annotated.economics.get(id(t))
        point.counterfactual += economics.counterfactual_cost if economics is not None else cost
        if economics is not None and economics.switched:
            point.switches += 1
    return Timeseries(bucket_seconds=size, points=points)


def pick_bucket(start: datetime, end: datetime) -> int:
    """The smallest bucket size, in seconds, giving at most `MAX_BUCKETS` buckets."""
    for size in BUCKET_SIZES:
        if _bucket_index(end, _floor(start, size), size) + 1 <= MAX_BUCKETS:
            return size
    day = 24 * 60 * 60
    days = math.ceil((end - start).total_seconds() / day / MAX_BUCKETS) + 1
    return days * day


def request_rows(
    turns: list[TurnRecord], pricing: PricingTable, limit: int, count_from: datetime | None = None
) -> list[RequestRow]:
    """The latest requests, newest first, for the requests table.

    Args:
        turns: The selection, oldest first, including any context turns.
        pricing: Rates for each request's net savings.
        limit: Maximum number of rows.
        count_from: Only requests at or after this time.

    Returns:
        Up to `limit` rows.
    """
    annotated = _annotate(turns, pricing, count_from)
    rows = []
    for t in reversed(annotated.counted[-limit:] if limit > 0 else []):
        economics = annotated.economics.get(id(t))
        rows.append(
            RequestRow(
                id=t.id,
                timestamp=t.timestamp,
                kind=annotated.kinds[id(t)],
                tier=t.tier,
                score=t.complexity_score,
                requested_model=t.requested_model,
                served_model=t.model,
                input_tokens=t.usage.input_tokens,
                output_tokens=t.usage.output_tokens,
                cache_read_tokens=t.usage.cache_read_tokens,
                cache_creation_tokens=t.usage.cache_creation_tokens,
                cost_usd=t.cost_usd,
                net_savings=economics.net_savings if economics is not None else None,
                switched=economics.switched if economics is not None else False,
                ttfb_ms=t.ttfb_ms,
                latency_ms=t.latency_ms,
                status_code=t.status_code,
                error=t.error[:ERROR_CHARS] if t.error else None,
            )
        )
    return rows


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least `p`% of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _annotate(turns: list[TurnRecord], pricing: PricingTable, count_from: datetime | None) -> _Annotated:
    # Kinds and economics need every turn of a conversation; only the
    # counted ones are reported.
    annotated = _Annotated(counted=[t for t in turns if is_counted(t, count_from)])
    annotated.kinds = {id(t): kind for t, kind in zip(turns, classify(turns), strict=True)}
    annotated.economics = {id(t): economics for t, economics in iter_turn_economics(turns, pricing)}
    return annotated


def _latency_stats(model: str | None, turns: list[TurnRecord]) -> LatencyStats:
    ttfb = [t.ttfb_ms for t in turns if t.ttfb_ms is not None]
    total = [t.latency_ms for t in turns if t.latency_ms is not None]
    return LatencyStats(
        model=model,
        n=len(total),
        ttfb_p50=percentile(ttfb, 50),
        ttfb_p95=percentile(ttfb, 95),
        latency_p50=percentile(total, 50),
        latency_p95=percentile(total, 95),
    )


def _tier(turn: TurnRecord) -> str:
    return turn.tier if turn.tier in ("small", "large") else "unrouted"


def _floor(when: datetime, size: int) -> datetime:
    seconds = when.timestamp()
    return datetime.fromtimestamp(seconds - seconds % size, tz=timezone.utc)


def _bucket_index(when: datetime, origin: datetime, size: int) -> int:
    return int((when - origin).total_seconds() // size)


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat()
