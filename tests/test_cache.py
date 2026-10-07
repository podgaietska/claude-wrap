import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from wrap.cache.cache import ResponseCache, estimate_request_tokens, normalize_question, question_hash, scope_key
from wrap.cache.eligibility import lookup_ineligibility, storable_content
from wrap.cache.store import CacheEntry, CacheStore
from wrap.config import CacheConfig
from wrap.routing.router import RouteDecision, find_turn

QUESTION = "what's the difference between a process and a thread?"
TOOLS = [{"name": "Read", "input_schema": {"type": "object"}}]
ANSWER = {
    "model": "small-model",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "A process has its own memory; threads share it."}],
    "usage": {"input_tokens": 10, "output_tokens": 12},
}


def config(**overrides) -> CacheConfig:
    return CacheConfig(enabled=True, similarity_threshold=0.92, embedding_model="x", **overrides)


def request(messages=None, **extra) -> dict:
    return {"model": "m", "tools": TOOLS, "messages": messages or [{"role": "user", "content": QUESTION}], **extra}


def decide(body: dict, tier: str = "small") -> RouteDecision:
    turn = find_turn(body["messages"])
    if turn is None:
        return RouteDecision(model="m", routed=False, tier=None, reason="", complexity=None, turn=None)
    return RouteDecision(model=f"{tier}-model", routed=True, tier=tier, reason="", complexity=None, turn=turn)


def make_cache(tmp_path, project="/repo/a", **overrides) -> ResponseCache:
    cfg = config(**overrides)
    return ResponseCache(cfg, CacheStore(tmp_path / "wrap.db", cfg.ttl_days, cfg.max_entries), project)


def store_answer(cache: ResponseCache, body: dict, tier: str = "small", answer: dict = ANSWER) -> int:
    lookup = cache.lookup(body, decide(body, tier))
    return cache.insert(lookup, answer, 0.001)


# --- eligibility -----------------------------------------------------------


def test_first_text_question_of_a_main_request_is_eligible():
    body = request()
    assert lookup_ineligibility(body, decide(body), config()) is None


def test_system_reminders_and_trailing_system_messages_dont_affect_eligibility():
    body = request([
        {"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>CLAUDE.md says hi</system-reminder>"},
            {"type": "text", "text": QUESTION},
        ]},
        {"role": "system", "content": "reminder"},
    ])
    assert lookup_ineligibility(body, decide(body), config()) is None


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"model": "m", "messages": [{"role": "user", "content": QUESTION}]}, "side request"),
        (request([{"role": "user", "content": "<system-reminder>x</system-reminder>"}]), "no question"),
        (
            request([
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x"}]},
            ]),
            "tool continuation",
        ),
        (
            request([
                {"role": "user", "content": "first question here"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": QUESTION},
            ]),
            "not the first question",
        ),
        (
            request([{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
                {"type": "text", "text": QUESTION},
            ]}]),
            "not text only",
        ),
        (request([{"role": "user", "content": "hi there"}]), "question too short"),
        (request([{"role": "user", "content": "x" * 2001}]), "question too long"),
        (request(tool_choice={"type": "tool", "name": "Read"}), "forced tool choice"),
        (request(tool_choice={"type": "any"}), "forced tool choice"),
    ],
)
def test_each_rule_makes_a_request_ineligible(body, reason):
    assert lookup_ineligibility(body, decide(body), config()) == reason


def test_auto_tool_choice_is_eligible():
    body = request(tool_choice={"type": "auto"})
    assert lookup_ineligibility(body, decide(body), config()) is None


def test_text_answer_is_storable_with_thinking_dropped():
    message = {**ANSWER, "content": [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "answer", "citations": None},
    ]}
    assert storable_content(message) == [{"type": "text", "text": "answer"}]


@pytest.mark.parametrize(
    "message",
    [
        None,
        {**ANSWER, "stop_reason": "max_tokens"},
        {**ANSWER, "stop_reason": "tool_use", "content": [
            {"type": "text", "text": "let me look"}, {"type": "tool_use", "id": "1", "name": "Read", "input": {}},
        ]},
        {**ANSWER, "content": [{"type": "text", "text": "  "}]},
        {**ANSWER, "content": [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]},
    ],
)
def test_unstorable_answers(message):
    assert storable_content(message) is None


# --- keys ------------------------------------------------------------------


def test_normalization_ignores_case_whitespace_and_trailing_punctuation():
    assert normalize_question("  What IS\n a  thread?! ") == "what is a thread"
    assert question_hash("What is a thread?") == question_hash("what is a thread")
    assert question_hash("what is 2+2") != question_hash("what is 2+3")


def test_scope_key_is_per_project_unless_global(tmp_path):
    assert scope_key("project", str(tmp_path / "a")) != scope_key("project", str(tmp_path / "b"))
    assert scope_key("project", str(tmp_path / "a")) == scope_key("project", str(tmp_path / "x" / ".." / "a"))
    assert scope_key("global", str(tmp_path / "a")) == "global"


def test_request_token_estimate_counts_system_and_tools():
    body = request(system=[{"type": "text", "text": "s" * 400}])
    assert estimate_request_tokens(body) > estimate_request_tokens(request()) + 90


# --- lookup and insert -----------------------------------------------------


def test_empty_cache_misses_then_stored_answer_hits(tmp_path):
    cache = make_cache(tmp_path)
    body = request()
    lookup = cache.lookup(body, decide(body))
    assert lookup.eligible and not lookup.hit and lookup.miss_reason == "empty"

    entry_id = cache.insert(lookup, ANSWER, 0.001)
    assert entry_id is not None

    again = request([{"role": "user", "content": "What's the difference between a process and a THREAD"}])
    hit = cache.lookup(again, decide(again))
    assert hit.hit and hit.similarity == 1.0
    assert hit.entry.id == entry_id
    assert hit.entry.content == ANSWER["content"]
    assert hit.entry.served_model == "small-model"
    assert hit.entry.output_tokens == 12
    assert hit.entry.cost_usd == 0.001


def test_other_question_misses_as_exact_only(tmp_path):
    cache = make_cache(tmp_path)
    store_answer(cache, request())
    other = request([{"role": "user", "content": "how do python generators work?"}])
    lookup = cache.lookup(other, decide(other))
    assert not lookup.hit and lookup.miss_reason == "exact_only"


def test_hit_is_counted(tmp_path):
    cache = make_cache(tmp_path)
    store_answer(cache, request())
    cache.lookup(request(), decide(request()))
    entry = cache.lookup(request(), decide(request())).entry
    assert entry.hit_count == 1 and entry.last_hit_at is not None


def test_ineligible_requests_are_neither_looked_up_nor_stored(tmp_path):
    cache = make_cache(tmp_path)
    body = request([{"role": "user", "content": "hi there"}])
    lookup = cache.lookup(body, decide(body))
    assert not lookup.eligible and lookup.reason == "question too short"
    assert cache.insert(lookup, ANSWER, None) is None
    assert cache.store.count(cache.scope_key) == 0


def test_hit_lookup_isnt_stored_again(tmp_path):
    cache = make_cache(tmp_path)
    store_answer(cache, request())
    hit = cache.lookup(request(), decide(request()))
    assert cache.insert(hit, ANSWER, None) is None


def test_projects_dont_share_answers_unless_global(tmp_path):
    store_answer(make_cache(tmp_path, project="/repo/a"), request())
    assert not make_cache(tmp_path, project="/repo/b").lookup(request(), decide(request())).hit
    assert make_cache(tmp_path, project="/repo/a").lookup(request(), decide(request())).hit

    store_answer(make_cache(tmp_path, project="/repo/a", scope="global"), request())
    assert make_cache(tmp_path, project="/repo/b", scope="global").lookup(request(), decide(request())).hit


def test_small_answer_doesnt_serve_a_large_question(tmp_path):
    cache = make_cache(tmp_path)
    store_answer(cache, request(), tier="small")
    lookup = cache.lookup(request(), decide(request(), "large"))
    assert not lookup.hit and lookup.miss_reason == "tier"

    # The large answer replaces the small one and serves both tiers.
    cache.insert(lookup, {**ANSWER, "model": "large-model"}, None)
    assert cache.lookup(request(), decide(request(), "large")).entry.served_model == "large-model"
    assert cache.lookup(request(), decide(request(), "small")).entry.served_model == "large-model"
    assert cache.store.count(cache.scope_key) == 1


def test_expired_entries_are_not_served_and_are_deleted(tmp_path):
    cache = make_cache(tmp_path, ttl_days=1)
    entry_id = store_answer(cache, request())
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cache.store._conn.execute("UPDATE cache_entry SET created_at = ? WHERE id = ?", (old, entry_id))

    lookup = cache.lookup(request(), decide(request()))
    assert not lookup.hit and lookup.miss_reason == "empty"
    assert cache.store._conn.execute("SELECT COUNT(*) FROM cache_entry").fetchone()[0] == 0


def test_least_recently_used_entries_are_evicted_past_the_cap(tmp_path):
    cache = make_cache(tmp_path, max_entries=2)
    questions = [f"question number {n} about threads" for n in range(3)]
    bodies = [request([{"role": "user", "content": q}]) for q in questions]
    store_answer(cache, bodies[0])
    store_answer(cache, bodies[1])
    assert cache.lookup(bodies[0], decide(bodies[0])).hit  # now the most recently used
    store_answer(cache, bodies[2])

    assert cache.lookup(bodies[0], decide(bodies[0])).hit
    assert not cache.lookup(bodies[1], decide(bodies[1])).hit
    assert cache.lookup(bodies[2], decide(bodies[2])).hit


def test_entries_survive_reopening_the_store(tmp_path):
    cache = make_cache(tmp_path)
    store_answer(cache, request())
    cache.close()
    assert make_cache(tmp_path).lookup(request(), decide(request())).hit


def test_store_lives_next_to_turn_log(tmp_path):
    from wrap.telemetry import db

    db.connect(tmp_path / "wrap.db").close()
    store_answer(make_cache(tmp_path), request())
    conn = sqlite3.connect(tmp_path / "wrap.db")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"turn_log", "cache_entry"} <= tables
