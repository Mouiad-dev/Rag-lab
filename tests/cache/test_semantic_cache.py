import asyncio
import json
import re
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from _fixture import FACT_EN, QUERY_EN
from _lexical_embedder import LexicalTestEmbedder
from _scripted_chat import cite_where
from rag_foundry_cache import (
    ExactResponseCache,
    SemanticCacheConfig,
    SemanticResponseCache,
    calibrate,
)
from rag_foundry_cache.features import QuestionFeatures
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import InMemoryResponseCacheStore
from rag_foundry_domain.testing import make_scope
from rag_foundry_providers.fake import fake_embedding_profile
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.messages import EmbeddingResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage
from test_exact_cache import T0, Env, built, run  # noqa: F401  (built is a fixture)

DATA = Path(__file__).parent / "data" / "semantic_pairs_v1.json"
GRID = tuple(round(0.5 + i * 0.01, 2) for i in range(51))
PARAPHRASE = "How often does the observatory at Port Sallow record sea levels?"
MARKER = "zanzibarquokka"


class StubEmbedder:
    """Test-only: a fixed vector per text, with a default vector."""

    def __init__(self, default=(1.0, 0.0, 0.0), table=None, fail=False):
        self.profile = fake_embedding_profile("stub-embed", dim=3)
        self.default = default
        self.table = table or {}
        self.fail = fail
        self.calls = 0

    async def embed(self, texts, *, input_type="document"):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        usage = Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=Operation.EMBED,
            input_tokens=0,
            output_tokens=0,
            token_source=TokenSource.ESTIMATED,
            latency_ms=0,
            estimated_cost_usd=0,
        )
        return EmbeddingResult(vectors=tuple(self.table.get(t, self.default) for t in texts), usage=usage)


def sem(env, *, threshold=0.8, margin=0.05, store=None, ttl=86400):
    config = SemanticCacheConfig(threshold=threshold, margin=margin, ttl_s=ttl)
    return SemanticResponseCache(config, store=store or env.store)


async def sgo(env, question, *, embedder, cache=None, inputs=None, scope=None, gate=None, expected="same", fn=None):
    return await (cache or sem(env)).run(
        scope or env.scope,
        question,
        inputs or env.inputs(),
        fn or env.answer_fn(question),
        embedder=embedder,
        expected_index_version_id=env.version.id if expected == "same" else expected,
        anchors=env.anchors,
        gate=gate or env.gate,
        clock=env.clock,
        run_id_factory=env.run_id,
    )


def test_config_has_no_default_threshold():
    with pytest.raises(ValueError):
        SemanticCacheConfig()
    for bad in (0.49, 1.01, float("nan"), True):
        with pytest.raises(ValueError):
            SemanticCacheConfig(threshold=bad)
    with pytest.raises(ValueError):
        SemanticCacheConfig(threshold=0.9, k=1)
    with pytest.raises(ValueError):
        SemanticCacheConfig(threshold=0.9, margin=0.51)
    with pytest.raises(CacheError):
        SemanticResponseCache({}, store=InMemoryResponseCacheStore())
    manifest = SemanticResponseCache.manifest
    assert (manifest.technique_id, manifest.version, manifest.fidelity.value) == ("R51", "1.0.0", "adaptation")


def test_paraphrase_in_same_partition_hits_with_zero_answer_calls(built):  # noqa: F811
    env = Env(built)
    cache_ledger = UsageLedger()
    embedder = RecordingEmbeddingModel(LexicalTestEmbedder(), cache_ledger)

    async def scenario():
        cold = await sgo(env, QUERY_EN, embedder=embedder)
        embeds_before, chats_before = len(cache_ledger.records()), len(env.chats)
        warm = await sgo(env, PARAPHRASE, embedder=embedder)
        return cold, warm, embeds_before, chats_before

    cold, warm, embeds_before, chats_before = run(scenario())
    assert cold.trace.layer == "R51" and cold.trace.outcome == "miss" and cold.trace.reason == "no_candidate"
    assert cold.trace.store_reason is None
    assert warm.trace.outcome == "hit" and warm.trace.layer == "R51" and warm.trace.similarity >= 0.8
    assert len(env.chats) == chats_before == 1  # zero answer calls on the hit
    assert len(cache_ledger.records()) - embeds_before == 1  # one cache-ledger embed call for the hit
    assert warm.response.answer == cold.response.answer and warm.response.run_id != cold.response.run_id
    assert warm.response.usage.model_calls == 0
    assert warm.trace.source_run_id == cold.response.run_id and warm.trace.calls_avoided == 2


def test_other_tenant_and_other_partition_miss(built):  # noqa: F811
    env = Env(built)
    embedder = StubEmbedder()

    async def scenario():
        await sgo(env, QUERY_EN, embedder=embedder)
        other_tenant = await sgo(env, QUERY_EN, embedder=embedder, scope=make_scope())
        other_partition = await sgo(env, QUERY_EN, embedder=embedder, inputs=env.inputs(chat_model="other-model"))
        return other_tenant, other_partition

    other_tenant, other_partition = run(scenario())
    assert other_tenant.trace.outcome == "miss" and other_tenant.trace.reason == "no_candidate"
    assert other_partition.trace.outcome == "miss" and other_partition.trace.reason == "no_candidate"
    assert len(env.chats) == 3


def test_partition_is_checked_even_if_the_store_ignores_it(built):  # noqa: F811
    env = Env(built)

    class Leaky(InMemoryResponseCacheStore):
        async def similar(self, scope, partition, vector_key, vector, k):
            scored = await super().similar(scope, partition, vector_key, vector, k)
            others = [(e, 1.0) for e in self._scoped(scope) if e.partition != partition]
            return others + scored

    store = Leaky()
    embedder = StubEmbedder()

    async def scenario():
        await sgo(env, QUERY_EN, embedder=embedder, cache=sem(env, store=store))
        return await sgo(
            env,
            QUERY_EN,
            embedder=embedder,
            cache=sem(env, store=store),
            inputs=env.inputs(chat_model="other-model"),
        )

    result = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "no_candidate"


def test_other_vector_key_is_never_a_candidate(built):  # noqa: F811
    env = Env(built)
    first = StubEmbedder()
    second = StubEmbedder()
    second.profile = fake_embedding_profile("stub-embed-2", dim=3)

    async def scenario():
        await sgo(env, QUERY_EN, embedder=first)
        return await sgo(env, QUERY_EN, embedder=second)

    assert run(scenario()).trace.reason == "no_candidate"


def test_year_and_negation_miss_even_at_similarity_one(built):  # noqa: F811
    embedder = StubEmbedder()  # every text gets the same vector: similarity 1.0
    base = QUERY_EN[:-1] + " in 2025?"
    cases = (
        (QUERY_EN[:-1] + " in 2026?", "features_differ:numbers"),
        (QUERY_EN, "features_differ:numbers"),
        (QUERY_EN[:-1] + " in 2025 not?", "features_differ:negation"),
    )
    for question, reason in cases:
        env = Env(built)  # one cached entry per run, so the only candidate is the 2025 question
        run(sgo(env, base, embedder=embedder))
        result = run(sgo(env, question, embedder=embedder))
        assert result.trace.outcome == "miss" and result.trace.reason == reason
        assert result.trace.store_reason is None and len(env.chats) == 2


def test_equal_features_hit_at_similarity_one(built):  # noqa: F811
    env = Env(built)
    embedder = StubEmbedder()

    async def scenario():
        await sgo(env, QUERY_EN[:-1] + " in 2025?", embedder=embedder)
        return await sgo(env, "Port Sallow sea levels recorded how often in 2025?", embedder=embedder)

    result = run(scenario())
    assert result.trace.outcome == "hit" and result.trace.similarity == pytest.approx(1.0)


def test_threshold_boundary_is_inclusive(built):  # noqa: F811
    env = Env(built)
    near = (0.999, 0.04, 0.0)
    embedder = StubEmbedder(table={QUERY_EN: (1.0, 0.0, 0.0), PARAPHRASE: (1.0, 0.0, 0.0), "near": near})

    async def scenario():
        cache = sem(env, threshold=1.0)
        await sgo(env, QUERY_EN, embedder=embedder, cache=cache)
        exact_equal = await sgo(env, PARAPHRASE, embedder=embedder, cache=cache)
        below = await sgo(env, "near", embedder=embedder, cache=cache)
        return exact_equal, below

    exact_equal, below = run(scenario())
    assert exact_equal.trace.outcome == "hit"
    assert below.trace.outcome == "miss" and below.trace.reason == "below_threshold"


def test_two_close_cached_answers_are_ambiguous(built):  # noqa: F811
    env = Env(built)
    q1, q2, q3 = QUERY_EN + " one", QUERY_EN + " two", QUERY_EN + " three"
    table = {q1: (1.0, 0.0, 0.0), q2: (0.0, 1.0, 0.0), q3: (1.0, 1.0, 0.0)}
    embedder = StubEmbedder(table=table)

    async def scenario():
        cache = sem(env, threshold=0.6)
        env.responder = cite_where(FACT_EN, "Answer one")
        await sgo(env, q1, embedder=embedder, cache=cache)
        env.responder = cite_where(FACT_EN, "Answer two")
        second = await sgo(env, q2, embedder=embedder, cache=cache)
        before = len(env.chats)
        third = await sgo(env, q3, embedder=embedder, cache=cache)
        return second, third, before

    second, third, before = run(scenario())
    assert second.trace.reason == "below_threshold" and second.trace.store_reason is None
    assert third.trace.outcome == "miss" and third.trace.reason == "ambiguous"
    assert len(env.chats) == before + 1


def test_close_candidates_with_the_same_answer_are_not_ambiguous(built):  # noqa: F811
    env = Env(built)
    q1, q2, q3 = QUERY_EN + " one", QUERY_EN + " two", QUERY_EN + " three"
    table = {q1: (1.0, 0.0, 0.0), q2: (0.0, 1.0, 0.0), q3: (1.0, 1.0, 0.0)}
    embedder = StubEmbedder(table=table)

    async def scenario():
        cache = sem(env, threshold=0.6)
        await sgo(env, q1, embedder=embedder, cache=cache)
        await sgo(env, q2, embedder=embedder, cache=cache)
        return await sgo(env, q3, embedder=embedder, cache=cache)

    assert run(scenario()).trace.outcome == "hit"


def test_runner_up_below_threshold_is_not_ambiguous(built):  # noqa: F811
    env = Env(built)
    q1, q2, q3 = QUERY_EN + " one", QUERY_EN + " two", QUERY_EN + " three"
    # Query similarity: 0.91 to the first entry, 0.89 to the second (inside the margin, under the threshold).
    table = {
        q1: (0.91, 0.4146, 0.0),
        q2: (0.89, -0.4559, 0.0),
        q3: (1.0, 0.0, 0.0),
    }
    embedder = StubEmbedder(table=table)

    async def scenario():
        cache = sem(env, threshold=0.9, margin=0.05)
        env.responder = cite_where(FACT_EN, "Answer one")
        await sgo(env, q1, embedder=embedder, cache=cache)
        env.responder = cite_where(FACT_EN, "Answer two")
        second = await sgo(env, q2, embedder=embedder, cache=cache)
        return second, await sgo(env, q3, embedder=embedder, cache=cache)

    second, third = run(scenario())
    assert second.trace.store_reason is None
    assert third.trace.outcome == "hit" and third.response.answer.startswith("Answer one")


def test_clearly_better_candidate_is_not_ambiguous(built):  # noqa: F811
    env = Env(built)
    q1, q2, q3 = QUERY_EN + " one", QUERY_EN + " two", QUERY_EN + " three"
    table = {q1: (1.0, 0.0, 0.0), q2: (0.0, 1.0, 0.0), q3: (1.0, 0.3, 0.0)}
    embedder = StubEmbedder(table=table)

    async def scenario():
        cache = sem(env, threshold=0.6, margin=0.05)
        env.responder = cite_where(FACT_EN, "Answer one")
        await sgo(env, q1, embedder=embedder, cache=cache)
        env.responder = cite_where(FACT_EN, "Answer two")
        await sgo(env, q2, embedder=embedder, cache=cache)
        return await sgo(env, q3, embedder=embedder, cache=cache)

    result = run(scenario())
    assert result.trace.outcome == "hit" and result.response.answer.startswith("Answer one")


class Deny:
    checks_revocation = True

    async def denied(self, scope, version_ids):
        return frozenset(version_ids)


def test_semantic_hit_is_revalidated(built):  # noqa: F811
    embedder = StubEmbedder()

    async def scenario(**changes):
        env = Env(built)
        await sgo(env, QUERY_EN, embedder=embedder)
        if "now" in changes:
            env.clock.now = changes.pop("now")
        return await sgo(env, PARAPHRASE, embedder=embedder, **changes)

    denied = run(scenario(gate=Deny()))
    expired = run(scenario(now=T0 + timedelta(days=2)))
    changed = run(scenario(expected=str(uuid.uuid4())))
    assert denied.trace.outcome == "miss" and denied.trace.reason == "source_denied"
    assert expired.trace.outcome == "miss" and expired.trace.reason == "expired"
    assert changed.trace.outcome == "miss" and changed.trace.reason == "index_version_changed"


def test_entries_hold_features_and_vector_but_no_question_text(built):  # noqa: F811
    env = Env(built)
    question = f"{QUERY_EN[:-1]} in 2025 for {MARKER}?"
    run(sgo(env, question, embedder=StubEmbedder()))
    entries = list(env.store._entries.values())
    assert len(entries) == 1
    entry = entries[0]
    assert entry.features_json and entry.question_vector and entry.vector_key == "stub-embed:" + entry.vector_key[11:]
    text = entry.model_dump_json()
    # Strip random UUIDs (scope, run and version ids) and hex digests, which can contain "2025" by chance.
    scrubbed = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,}", "", text)
    assert MARKER not in text and "2025" not in scrubbed.replace("2025-", "")
    assert QuestionFeatures.from_json(entry.features_json).hashed


def test_entries_without_features_or_vector_are_not_candidates(built):  # noqa: F811
    env = Env(built)

    async def scenario():
        await env.go(cache=ExactResponseCache({}, store=env.store))  # an R50 entry has no vector or features
        return await sgo(env, QUERY_EN, embedder=StubEmbedder())

    result = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "no_candidate"
    assert result.trace.store_reason == "cache_exists"  # same exact key already held by the R50 entry


def test_bypass_rules_match_r50(built):  # noqa: F811
    env = Env(built)
    embedder = StubEmbedder()
    for changes, reason in (
        ({"freshness": "current"}, "freshness_current"),
        ({"mode": "enriched"}, "enriched_mode"),
        ({"tools": ("web",)}, "tools"),
        ({"is_action_run": True}, "action_run"),
    ):
        result = run(sgo(env, QUERY_EN, embedder=embedder, inputs=env.inputs(**changes)))
        assert result.trace.outcome == "bypass" and result.trace.reason == reason
    assert embedder.calls == 0 and env.store._entries == {}


def test_embed_error_is_a_miss_and_a_fresh_run(built):  # noqa: F811
    env = Env(built)
    result = run(sgo(env, QUERY_EN, embedder=StubEmbedder(fail=True)))
    assert result.trace.outcome == "miss" and result.trace.reason == "embed_error"
    assert len(env.chats) == 1 and env.store._entries == {}


def test_store_errors_are_misses_with_a_fresh_run(built):  # noqa: F811
    env = Env(built)

    class Broken(InMemoryResponseCacheStore):
        async def similar(self, *args):
            raise RuntimeError("down")

        async def put(self, scope, entry):
            raise RuntimeError("down")

    result = run(sgo(env, QUERY_EN, embedder=StubEmbedder(), cache=sem(env, store=Broken())))
    assert result.trace.outcome == "miss" and result.trace.reason == "cache_error"
    assert result.trace.store_reason == "cache_error" and len(env.chats) == 1


def test_scope_mismatch_is_never_returned_deleted_or_stored(built):  # noqa: F811
    env = Env(built)

    class Raising(InMemoryResponseCacheStore):
        deleted = 0

        async def similar(self, *args):
            raise CacheError("cache_scope_mismatch")

        async def delete(self, scope, key):
            self.deleted += 1
            return True

    store = Raising()
    result = run(sgo(env, QUERY_EN, embedder=StubEmbedder(), cache=sem(env, store=store)))
    assert result.trace.reason == "scope_mismatch" and result.trace.store_reason == "skipped_scope_mismatch"
    assert store.deleted == 0 and store._entries == {} and len(env.chats) == 1


def test_foreign_entry_from_the_store_is_scope_mismatch(built):  # noqa: F811
    env = Env(built)
    other = make_scope()

    class Foreign(InMemoryResponseCacheStore):
        deleted = 0
        foreign = None

        async def similar(self, scope, partition, vector_key, vector, k):
            return [(self.foreign, 1.0)]

        async def delete(self, scope, key):
            self.deleted += 1
            return True

    async def scenario():
        await sgo(env, QUERY_EN, embedder=StubEmbedder(), scope=other, fn=env.answer_fn())
        store = Foreign()
        store.foreign = next(iter(env.store._entries.values()))
        return store, await sgo(env, QUERY_EN, embedder=StubEmbedder(), cache=sem(env, store=store))

    store, result = run(scenario())
    assert result.trace.reason == "scope_mismatch" and store.deleted == 0 and store._entries == {}
    assert result.trace.store_reason == "skipped_scope_mismatch"


def test_features_json_must_round_trip_or_the_answer_is_not_stored(built, monkeypatch):  # noqa: F811
    env = Env(built)
    monkeypatch.setattr(QuestionFeatures, "to_json", lambda self: '{"numbers":[]}')
    result = run(sgo(env, QUERY_EN, embedder=StubEmbedder()))
    assert result.trace.outcome == "miss" and result.trace.store_reason == "not_cacheable_status"
    assert env.store._entries == {}


def test_not_cacheable_answers_are_not_stored(built):  # noqa: F811
    env = Env(built, responder=lambda request: ("INSUFFICIENT_EVIDENCE", "stop"))
    result = run(sgo(env, QUERY_EN, embedder=StubEmbedder()))
    assert result.trace.store_reason == "not_cacheable_status" and env.store._entries == {}


# ---- calibration data and calibrate()


def load_pairs():
    return json.loads(DATA.read_text(encoding="ascii"))


def lexical_vectors(pairs):
    embedder = LexicalTestEmbedder()

    async def embed():
        out = {}
        for p in pairs:
            result = await embedder.embed([p["a"], p["b"]])
            out[p["id"]] = (result.vectors[0], result.vectors[1])
        return out

    return asyncio.run(embed())


def tokens(text):
    return set(re.findall(r"\w+", text.lower()))


def test_pair_data_is_clean():
    doc = load_pairs()
    assert "not a benchmark" in doc["header"]
    pairs = doc["pairs"]
    assert len({p["id"] for p in pairs}) == len(pairs)
    for p in pairs:
        assert set(p) == {"id", "split", "kind", "lang", "a", "b", "note"}
        assert p["split"] in ("dev", "held_out") and p["kind"] in ("paraphrase", "adversarial")
        assert p["lang"] in ("en", "ar", "mixed")
    assert {p["split"] for p in pairs} == {"dev", "held_out"}
    dev = [t for p in pairs if p["split"] == "dev" for t in (p["a"], p["b"])]
    held = [t for p in pairs if p["split"] == "held_out" for t in (p["a"], p["b"])]
    assert not set(dev) & set(held)
    for d in dev:
        for h in held:
            union = tokens(d) | tokens(h)
            assert len(tokens(d) & tokens(h)) / len(union) < 0.8, (d, h)


def test_calibration_report_is_pinned_and_dev_only():
    pairs = load_pairs()["pairs"]
    vectors = lexical_vectors(pairs)
    report = calibrate(pairs, vectors, GRID)
    print("\nthreshold  dev para(unguarded/guarded)  dev false hits(unguarded/guarded)")
    for row in report.rows:
        if row.threshold in (0.5, 0.8, 0.9, 0.92, 0.93, 0.95, 0.97, 1.0):
            d = row.dev
            print(
                f"{row.threshold:<9}  {d.paraphrase_hits_unguarded}/{d.paraphrase_hits_guarded} of {d.paraphrase_total}"
                f"  {d.false_hits_unguarded}/{d.false_hits_guarded} of {d.adversarial_total}"
            )
    print("chosen", report.chosen_threshold, "held_out", report.held_out)
    assert report.chosen_threshold == 0.93 and report.reason is None
    by = {row.threshold: row.dev for row in report.rows}
    assert (by[0.5].paraphrase_hits_unguarded, by[0.5].false_hits_unguarded, by[0.5].false_hits_guarded) == (5, 6, 1)
    assert (by[0.92].false_hits_guarded, by[0.93].false_hits_guarded) == (1, 0)
    assert by[0.93].false_hits_unguarded == 4  # at 0.93 the guard removes three of these four false hits
    assert by[0.93].paraphrase_hits_guarded == 4 and by[0.93].paraphrase_total == 5
    held = report.held_out
    assert held.paraphrase_total == 5 and held.adversarial_total == 7
    assert (held.paraphrase_hits_unguarded, held.paraphrase_hits_guarded) == (4, 4)
    assert held.false_hits_unguarded == 4
    # Documented losing cases: month name and unit pass the guard and the similarity, so they are false hits.
    assert held.false_hits_guarded == 2
    assert held.false_hit_rate_guarded == pytest.approx(2 / 7)
    assert held.paraphrase_hit_rate_guarded == pytest.approx(0.8)


def test_chosen_threshold_does_not_depend_on_held_out():
    pairs = load_pairs()["pairs"]
    vectors = lexical_vectors(pairs)
    base = calibrate(pairs, vectors, GRID)
    edited = [dict(p) for p in pairs]
    for p in edited:
        if p["id"] == "held-10":
            p["b"], p["kind"] = p["a"] + " earlier", "paraphrase"
    for pair_id in ("held-10", "held-11", "held-12"):
        vectors[pair_id] = ((1.0, 0.0), (1.0, 0.0))  # worst case: every held-out adversarial is an exact match
    changed = calibrate(edited, vectors, GRID)
    assert changed.chosen_threshold == base.chosen_threshold == 0.93
    assert changed.rows == base.rows
    assert changed.held_out != base.held_out


def test_no_threshold_when_every_threshold_has_a_dev_false_hit():
    pairs = [
        {"id": "p", "split": "dev", "kind": "paraphrase", "a": "alpha beta", "b": "beta alpha"},
        {"id": "x", "split": "dev", "kind": "adversarial", "a": "alpha beta", "b": "alpha gamma"},
        {"id": "h", "split": "held_out", "kind": "paraphrase", "a": "delta", "b": "delta"},
    ]
    vectors = {"p": ((1.0, 0.0), (1.0, 0.0)), "x": ((1.0, 0.0), (1.0, 0.0)), "h": ((1.0, 0.0), (1.0, 0.0))}
    report = calibrate(pairs, vectors, GRID)
    assert report.chosen_threshold is None and report.reason == "no_threshold_without_dev_false_hits"
    assert report.held_out is None
    assert all(row.dev.false_hits_guarded == 1 for row in report.rows)


def test_no_threshold_without_a_dev_paraphrase_hit_and_input_validation():
    pairs = [{"id": "p", "split": "dev", "kind": "paraphrase", "a": "alpha", "b": "alpha beta"}]
    report = calibrate(pairs, {"p": ((1.0, 0.0), (0.0, 1.0))}, (0.9, 0.95))
    assert report.chosen_threshold is None
    for bad in ((), (0.4,), (1.2,), (float("nan"),)):
        with pytest.raises(ValueError):
            calibrate(pairs, {"p": ((1.0, 0.0), (1.0, 0.0))}, bad)
    with pytest.raises(ValueError):
        calibrate(pairs, {}, (0.9,))


def test_guard_changes_the_calibration_outcome():
    pairs = [
        {"id": "p", "split": "dev", "kind": "paraphrase", "a": "alpha beta", "b": "beta alpha"},
        {"id": "y", "split": "dev", "kind": "adversarial", "a": "alpha in 2025", "b": "alpha in 2026"},
    ]
    vectors = {"p": ((1.0, 0.0), (1.0, 0.0)), "y": ((1.0, 0.0), (1.0, 0.0))}
    report = calibrate(pairs, vectors, (0.9,))
    row = report.rows[0].dev
    assert (row.false_hits_unguarded, row.false_hits_guarded) == (1, 0)
    assert report.chosen_threshold == 0.9


def test_features_reason_keeps_whole_names_within_the_trace_limit(built):  # noqa: F811
    env = Env(built)
    embedder = StubEmbedder()
    stored = f'Why is this not more than 5 ABC1 "x" {QUERY_EN}'
    asked = '\u0645\u0627 6 DEF2 "y" before'
    run(sgo(env, stored, embedder=embedder))
    result = run(sgo(env, asked, embedder=embedder))
    reason = result.trace.reason
    assert result.trace.outcome == "miss" and len(reason) <= 64
    assert reason == "features_differ:comparatives,identifiers,negation,numbers,+2"
    assert reason.count(",") == 4 and all(not part.endswith("_") for part in reason.split(","))


def test_store_score_is_not_trusted(built):  # noqa: F811
    class Inflated(InMemoryResponseCacheStore):
        async def similar(self, scope, partition, vector_key, vector, k):
            return [(e, 1.0) for e, _ in await super().similar(scope, partition, vector_key, vector, k)]

    first = StubEmbedder(table={QUERY_EN: (1.0, 0.0, 0.0), PARAPHRASE: (0.0, 1.0, 0.0)})

    async def scenario():
        env2 = Env(built)
        store2 = Inflated()
        await sgo(env2, QUERY_EN, embedder=first, cache=sem(env2, store=store2))
        return await sgo(env2, PARAPHRASE, embedder=first, cache=sem(env2, store=store2))

    second = run(scenario())
    assert second.trace.outcome == "miss" and second.trace.reason == "below_threshold"


def test_unreadable_features_json_drops_the_entry_and_misses(built):  # noqa: F811
    env = Env(built)
    embedder = StubEmbedder()

    async def scenario():
        await sgo(env, QUERY_EN, embedder=embedder)
        ((key, entry),) = [(e.key, e) for e in env.store._entries.values()]
        bad = entry.model_copy(update={"features_json": '{"numbers": []}'})
        env.store._entries[(entry.tenant_id, entry.project_id, key)] = bad
        result = await sgo(env, PARAPHRASE, embedder=embedder)
        return key, result

    key, result = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "entry_invalid"
    assert len(env.chats) == 2 and result.trace.store_reason is None
    assert all(e.key != key for e in env.store._entries.values())  # the unreadable entry was dropped
