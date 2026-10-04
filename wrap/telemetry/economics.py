"""Whether routing actually saves money once prompt-cache misses are counted.

Prompt caches belong to one model. When the router switches a
conversation to another model, that model has no cached copy of the
conversation and writes the whole prompt to its cache again, at 1.25x
(5m TTL) or 2x (1h TTL) its input rate, where staying put would have
read it at a fraction of the input rate. So a "cheaper" turn can cost
more than the turn it replaced.

For every turn this estimates what it would have cost had every turn
stayed on the requested model -- the counterfactual -- and splits the
difference exactly:

    net_savings = counterfactual_cost - actual_cost
                = saved_by_model - cache_penalty

    saved_by_model = price(requested, counterfactual tokens) - price(served, counterfactual tokens)
    cache_penalty  = price(served, actual tokens)            - price(served, counterfactual tokens)

The counterfactual tokens assume the conversation's previous turn left
its prefix cached on the requested model (if within the cache TTL), never
less than was actually read from the cache, with the same uncached input
and output. Assumes the requested model would
have produced the same output and taken the same number of turns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from wrap.telemetry.db import TurnRecord
from wrap.telemetry.pricing import PricingTable
from wrap.telemetry.usage import Usage

TTL_5M_SECONDS = 5 * 60
TTL_1H_SECONDS = 60 * 60
# The self-check counts an estimate as matching within this relative error.
MATCH_TOLERANCE = 0.10


@dataclass
class TurnEconomics:
    """One turn's actual cost against its stay-on-the-requested-model counterfactual.

    Attributes:
        actual_cost: The turn priced at the served model with actual tokens.
        counterfactual_cost: The turn priced at the requested model with
            counterfactual tokens.
        saved_by_model: The price difference between the two models, with
            caching as if no switch had happened.
        cache_penalty: Extra cost of the served model's actual cache
            behaviour over the counterfactual (negative when it did better).
        switched: True if the previous turn in the conversation was served
            by a different model.
        recached_tokens: Tokens written to the cache beyond the
            counterfactual, counted on switched turns only.
        counterfactual_usage: The estimated token split without switches.
        within_ttl: True if the previous turn was recent enough that its
            cache entry would still have been live.
    """

    actual_cost: float
    counterfactual_cost: float
    saved_by_model: float
    cache_penalty: float
    switched: bool
    recached_tokens: int
    counterfactual_usage: Usage
    within_ttl: bool

    @property
    def net_savings(self) -> float:
        """Counterfactual cost minus actual cost."""
        return self.counterfactual_cost - self.actual_cost


@dataclass
class Economics:
    """Routing economics totalled over a set of turns (see module docstring).

    Attributes:
        counterfactual_cost: What the turns would have cost on the requested model.
        actual_cost: What they cost.
        saved_by_model: Total saved by cheaper models, before cache effects.
        cache_penalty: Total lost to cache misses.
        switches: Turns served by a different model than the previous turn
            in the same conversation.
        recached_tokens: Tokens re-written to the cache because of switches.
        unpriced_turns: Turns skipped because a model has no price.
        error_turns: Turns skipped because the request failed.
        check_eligible: Turns the self-check could test: no switch, previous
            turn within the TTL.
        check_matched: Eligible turns whose estimated cache read was within
            `MATCH_TOLERANCE` of the actual one.
        requested_models: The models Claude Code asked for.
    """

    counterfactual_cost: float = 0.0
    actual_cost: float = 0.0
    saved_by_model: float = 0.0
    cache_penalty: float = 0.0
    switches: int = 0
    recached_tokens: int = 0
    unpriced_turns: int = 0
    error_turns: int = 0
    check_eligible: int = 0
    check_matched: int = 0
    requested_models: set[str] = field(default_factory=set)

    @property
    def net_savings(self) -> float:
        """Counterfactual cost minus actual cost."""
        return self.counterfactual_cost - self.actual_cost

    @property
    def check_rate(self) -> float | None:
        """Share of eligible turns the estimate matched, or None if none were eligible."""
        return self.check_matched / self.check_eligible if self.check_eligible else None


def ttl_seconds(uses_1h: bool) -> int:
    """The cache lifetime a conversation uses."""
    return TTL_1H_SECONDS if uses_1h else TTL_5M_SECONDS


def counterfactual_usage(turn: TurnRecord, previous: TurnRecord | None, uses_1h: bool) -> Usage:
    """Estimates a turn's token split had the conversation never switched models.

    The previous turn's cached prefix (its cache reads plus writes) would
    be read from the cache if it's still live; the rest of the cacheable
    prompt would be written. Uncached input and output stay as they were.

    The estimate never reads less from the cache than actually happened:
    without switches every request goes to the requested model, so its
    cache holds at least what the served models' caches held (a prefix
    shared with another conversation, say). This also leaves a
    conversation's first turn, with nothing to compare to, as it was.

    Args:
        turn: The turn to estimate.
        previous: The previous successful turn in the same conversation.
        uses_1h: Whether the conversation caches with the 1-hour TTL.

    Returns:
        The estimated `Usage`.
    """
    usage = turn.usage
    cacheable = usage.cache_read_tokens + usage.cache_creation_tokens
    cache_read = usage.cache_read_tokens
    if previous is not None and _within_ttl(previous, turn, uses_1h):
        previous_prefix = previous.usage.cache_read_tokens + previous.usage.cache_creation_tokens
        cache_read = max(cache_read, min(previous_prefix, cacheable))
    cache_creation = cacheable - cache_read
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        cache_creation_5m_tokens=0 if uses_1h else cache_creation,
        cache_creation_1h_tokens=cache_creation if uses_1h else 0,
    )


def turn_economics(
    turn: TurnRecord, previous: TurnRecord | None, pricing: PricingTable, uses_1h: bool
) -> TurnEconomics | None:
    """Compares one turn's actual cost with its counterfactual.

    Args:
        turn: The turn to evaluate.
        previous: The previous successful turn in the same conversation.
        pricing: Rates for the served and requested models.
        uses_1h: Whether the conversation caches with the 1-hour TTL.

    Returns:
        The comparison, or None if either model has no price.
    """
    served = pricing.lookup(turn.model)
    requested = pricing.lookup(turn.requested_model)
    if served is None or requested is None:
        return None

    counterfactual = counterfactual_usage(turn, previous, uses_1h)
    actual_cost = served.cost(turn.usage)
    served_counterfactual = served.cost(counterfactual)
    counterfactual_cost = requested.cost(counterfactual)

    switched = previous is not None and previous.model != turn.model
    recached = turn.usage.cache_creation_tokens - counterfactual.cache_creation_tokens
    return TurnEconomics(
        actual_cost=actual_cost,
        counterfactual_cost=counterfactual_cost,
        saved_by_model=counterfactual_cost - served_counterfactual,
        cache_penalty=actual_cost - served_counterfactual,
        switched=switched,
        recached_tokens=max(0, recached) if switched else 0,
        counterfactual_usage=counterfactual,
        within_ttl=previous is not None and _within_ttl(previous, turn, uses_1h),
    )


def analyze(turns: list[TurnRecord], pricing: PricingTable) -> Economics:
    """Totals routing economics over turns, conversation by conversation.

    Args:
        turns: Turns ordered oldest first, e.g. from `db.fetch_turns`.
        pricing: Rates for every model involved.

    Returns:
        The totals. Failed requests and unpriced turns are skipped and counted.
    """
    threads: dict[tuple[str | None, str | None], list[TurnRecord]] = defaultdict(list)
    for turn in turns:
        threads[(turn.session_id, turn.thread_key)].append(turn)

    result = Economics()
    for thread in threads.values():
        uses_1h = any(t.usage.cache_creation_1h_tokens > 0 for t in thread)
        previous: TurnRecord | None = None
        for turn in thread:
            if turn.status_code is None or turn.status_code >= 400:
                result.error_turns += 1
                continue

            economics = turn_economics(turn, previous, pricing, uses_1h)
            previous = turn
            if economics is None:
                result.unpriced_turns += 1
                continue

            result.requested_models.add(turn.requested_model)
            result.counterfactual_cost += economics.counterfactual_cost
            result.actual_cost += economics.actual_cost
            result.saved_by_model += economics.saved_by_model
            result.cache_penalty += economics.cache_penalty
            result.recached_tokens += economics.recached_tokens
            if economics.switched:
                result.switches += 1
            elif economics.within_ttl:
                result.check_eligible += 1
                if _matches(economics.counterfactual_usage.cache_read_tokens, turn.usage.cache_read_tokens):
                    result.check_matched += 1
    return result


def _within_ttl(previous: TurnRecord, turn: TurnRecord, uses_1h: bool) -> bool:
    gap = (datetime.fromisoformat(turn.timestamp) - datetime.fromisoformat(previous.timestamp)).total_seconds()
    return gap <= ttl_seconds(uses_1h)


def _matches(estimated: int, actual: int) -> bool:
    return abs(estimated - actual) <= MATCH_TOLERANCE * max(estimated, actual)
