import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from _fixture import FACT_EN, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from _qa import corpus_markdown
from _scripted_chat import ScriptedChatModel, cite_where
from _templates import prompt_templates
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_cache import AllowAllSourceGate, ExactCacheConfig, ExactResponseCache, ResponseKeyInputs, revalidate
from rag_foundry_cache.keys import GenerationParams, exact_key
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_contracts.resolve import AnchorResolution, ResolutionReason, ResolutionStatus
from rag_foundry_domain.errors import AnchorError, CacheError, NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.response_cache import InMemoryResponseCacheStore
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from test_extraction_service import ingest, setup

T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def built(tmp_path):
    async def create():
        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(
            intake, scope, corpus_markdown().encode("utf-8"), name="qa-notes.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
        embedder = LexicalTestEmbedder()
        index_uow = InMemoryUnitOfWorkFactory(extractions=views, indexes=index)
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
        version = await IndexService(store, index_uow, anchors).build(scope, (view.id,), chunker, embedder)
        return scope, index, anchors, embedder, version

    return asyncio.run(create())


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


class Env:
    def __init__(self, built, *, responder=None, k=5):
        self.scope, self.index, self.anchors, self.embedder, self.version = built
        self.responder = responder or cite_where(FACT_EN, FACT_EN)
        self.k = k
        self.chats = []
        self.ledgers = []
        self.clock = Clock()
        self.store = InMemoryResponseCacheStore()
        self.gate = AllowAllSourceGate()
        self.counter = 0
        self.prompts = RunPrompts(*prompt_templates())

    def key(self, scope=None):
        return exact_key(scope or self.scope, QUERY_EN, self.inputs())

    def inputs(self, **changes):
        chat = ScriptedChatModel(self.responder).profile
        defaults = AnswerConfig()
        values = {
            "authz_revision": "rev-1",
            "history": (),
            "technique_id": "R02",
            "technique_version": "1.0.0",
            "retriever_config_sha256": "a" * 64,
            "index_version_id": self.version.id,
            "embedding_key": self.version.embedding_key,
            "prompt_template": self.prompts.grounded.id,
            "prompt_sha256": self.prompts.grounded.content_sha256,
            "chat_profile_id": chat.id,
            "chat_provider": chat.provider,
            "chat_model": chat.model,
            "generation": GenerationParams(
                max_output_tokens=defaults.max_output_tokens,
                temperature=defaults.temperature,
                k=self.k,
                min_score=0.0,
                max_evidence_chars=defaults.max_evidence_chars,
                packer="none",
                citation_check="off",
            ),
            "settings_sha256": None,
            "mode": "source_only",
            "freshness": "static_snapshot",
            "tools": (),
            "is_action_run": False,
            "output_language": "auto",
            "contract_version": "1.1",
        }
        values.update(changes)
        return ResponseKeyInputs(**values)

    def answer_fn(self, question=QUERY_EN, *, k=None):
        async def fn():
            ledger = UsageLedger()
            chat = ScriptedChatModel(self.responder)
            self.chats.append(chat)
            self.ledgers.append(ledger)
            retriever = DenseRetriever(
                DenseConfig(k=k or self.k),
                scope=self.scope,
                index_version=self.version,
                index=self.index,
                embedder=RecordingEmbeddingModel(self.embedder, ledger),
            )
            return await AnswerService(self.anchors, nonce_factory=lambda: "0123456789abcdef").answer(
                self.scope,
                question,
                prompts=self.prompts,
                retriever=retriever,
                technique_version="1.0.0",
                index_version=self.version,
                chat=RecordingChatModel(chat, ledger),
                ledger=ledger,
                config=AnswerConfig(min_score=0.0, k=k or self.k),
            )

        return fn

    def run_id(self):
        self.counter += 1
        return f"hit-{self.counter}"

    def cache(self, store=None, ttl=86400):
        return ExactResponseCache(ExactCacheConfig(ttl_s=ttl), store=store or self.store)

    async def go(self, question=QUERY_EN, *, inputs=None, scope=None, cache=None, fn=None, expected="same", gate=None):
        return await (cache or self.cache()).run(
            scope or self.scope,
            question,
            inputs or self.inputs(),
            fn or self.answer_fn(question),
            expected_index_version_id=self.version.id if expected == "same" else expected,
            anchors=self.anchors,
            gate=gate or self.gate,
            clock=self.clock,
            run_id_factory=self.run_id,
        )


def run(coro):
    return asyncio.run(coro)


def test_repeat_question_hits_with_zero_calls(built):
    env = Env(built)

    async def scenario():
        return await env.go(), await env.go()

    cold, warm = run(scenario())
    assert cold.trace.outcome == "miss" and cold.trace.reason == "not_found"
    assert warm.trace.outcome == "hit" and warm.trace.reason is None
    assert len(env.chats) == 1 and len(env.ledgers) == 1  # the warm run created no chat or ledger
    assert env.ledgers[0].totals().calls == 2  # the only embedding and chat calls were the cold ones
    assert warm.response.answer == cold.response.answer
    assert warm.response.citations == cold.response.citations
    assert warm.response.run_id != cold.response.run_id
    assert warm.response.usage.model_calls == 0 and warm.response.usage.input_tokens == 0
    assert warm.response.usage.output_tokens == 0
    assert cold.response.usage.model_calls == 2
    assert warm.trace.calls_avoided == 2
    assert warm.trace.input_tokens_avoided == cold.response.usage.input_tokens
    assert warm.trace.output_tokens_avoided == cold.response.usage.output_tokens
    assert warm.trace.source_run_id == cold.response.run_id
    assert warm.trace.key_prefix == cold.trace.key_prefix and len(warm.trace.key_prefix) == 12


def test_changed_inputs_miss(built):
    env = Env(built)
    variants = [
        (env.inputs(history=(("user", "earlier turn"),)), "not_found"),
        (env.inputs(authz_revision="rev-2"), "not_found"),
        # The fresh answer's pins follow the real run, so a changed pin input is also refused for storage.
        (env.inputs(chat_model="other-model"), "pins_mismatch"),
        (env.inputs(prompt_sha256="b" * 64), "pins_mismatch"),
        (env.inputs(index_version_id=str(uuid.uuid4())), "pins_mismatch"),
    ]

    async def scenario():
        base = await env.go()
        return base, [await env.go(inputs=i, expected=i.index_version_id) for i, _ in variants]

    base, results = run(scenario())
    for result, (_, store_reason) in zip(results, variants, strict=True):
        assert result.trace.outcome == "miss" and result.trace.reason == "not_found"
        assert result.trace.store_reason == (None if store_reason == "not_found" else store_reason)
        assert result.trace.key_prefix != base.trace.key_prefix
    assert len(env.chats) == 1 + len(variants)


class CollidingStore(InMemoryResponseCacheStore):
    """Test-only: get returns one fixed entry whatever the scope or key."""

    def __init__(self, foreign):
        super().__init__()
        self.foreign = foreign
        self.deleted = []

    async def get(self, scope, key):
        return self.foreign

    async def delete(self, scope, key):
        self.deleted.append(key)
        return True


def test_other_tenant_misses_and_forced_collision_is_scope_mismatch(built):
    env = Env(built)
    other = make_scope()

    async def scenario():
        await env.go()
        second = await env.go(scope=other)
        entry = await env.store.get(env.scope, env.key())
        foreign = entry.model_copy(update={"key": env.key(other)})
        colliding = CollidingStore(foreign)
        before = len(env.chats)
        third = await env.go(scope=other, cache=env.cache(store=colliding))
        return second, third, colliding, foreign, before, await env.store.get(env.scope, env.key())

    second, third, colliding, foreign, before, still_there = run(scenario())
    assert second.trace.outcome == "miss" and second.trace.reason == "not_found"
    assert third.trace.outcome == "miss" and third.trace.reason == "scope_mismatch"
    assert len(env.chats) == before + 1  # a fresh run, never the foreign answer
    assert third.response.run_id != foreign.source_run_id
    assert colliding._entries == {} and third.trace.store_reason == "skipped_scope_mismatch"
    assert colliding._entries == {}  # nothing is stored behind a scope mismatch either
    assert colliding.deleted == [] and colliding.foreign is foreign  # the foreign entry is never deleted
    assert still_there is not None  # the original tenant's entry is intact


def test_scope_mismatch_error_from_store_is_a_miss_without_delete(built):
    env = Env(built)

    class Raising(InMemoryResponseCacheStore):
        deleted = 0

        async def get(self, scope, key):
            raise CacheError("cache_scope_mismatch")

        async def delete(self, scope, key):
            self.deleted += 1
            return True

    store = Raising()
    result = run(env.go(cache=env.cache(store=store)))
    assert result.trace.outcome == "miss" and result.trace.reason == "scope_mismatch"
    assert store.deleted == 0 and len(env.chats) == 1


def test_gate_denial_and_invalidate_sources(built):
    env = Env(built)

    class Deny:
        checks_revocation = True

        async def denied(self, scope, version_ids):
            return frozenset(version_ids)

    async def scenario():
        cold = await env.go()
        denied = await env.go(gate=Deny())
        # The failed revalidation dropped the old entry; the fresh answer replaced it.
        replaced = (await env.store.get(env.scope, env.key())).source_run_id
        dropped = None if replaced == denied.response.run_id and replaced != cold.response.run_id else replaced
        versions = (await env.store.get(env.scope, env.key())).source_version_ids
        removed = await env.cache().invalidate_sources(env.scope, versions)
        return denied, dropped, removed, await env.go()

    denied, dropped, removed, after = run(scenario())
    assert denied.trace.outcome == "miss" and denied.trace.reason == "source_denied"
    assert dropped is None
    assert removed == 1
    assert after.trace.outcome == "miss" and after.trace.reason == "not_found"


def test_expiry_with_fake_clock(built):
    env = Env(built)

    async def scenario():
        await env.go(cache=env.cache(ttl=60))
        env.clock.now = T0 + timedelta(seconds=59)
        early = await env.go(cache=env.cache(ttl=60))
        env.clock.now = T0 + timedelta(seconds=60)
        return early, await env.go(cache=env.cache(ttl=60))

    early, late = run(scenario())
    assert early.trace.outcome == "hit"
    assert late.trace.outcome == "miss" and late.trace.reason == "expired"
    assert len(env.chats) == 2


def test_index_version_changed_is_a_miss(built):
    env = Env(built)

    async def scenario():
        await env.go()
        return await env.go(expected=str(uuid.uuid4()))

    result = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "index_version_changed"


class MismatchAnchors:
    async def resolve(self, scope, anchors):
        return tuple(
            AnchorResolution(anchor_id=a.anchor_id, status=ResolutionStatus.MISMATCH, reason=ResolutionReason.NOT_FOUND)
            for a in anchors
        )


def test_unresolved_anchor_is_a_miss_and_resolving_anchor_passes(built):
    env = Env(built)

    async def scenario():
        cold = await env.go()
        entry = await env.store.get(env.scope, env.key())

        async def check(anchors):
            return await revalidate(
                env.scope,
                entry,
                cold.response,
                now=T0,
                expected_index_version_id=env.version.id,
                anchors=anchors,
                gate=env.gate,
            )

        class Missing:
            async def resolve(self, scope, anchors):
                raise NotFoundError()

        class TooMany:
            async def resolve(self, scope, anchors):
                raise AnchorError("too_many_views")

        return await check(MismatchAnchors()), await check(env.anchors), await check(Missing()), await check(TooMany())

    bad, good, missing, many = run(scenario())
    assert bad == "anchor_unresolved" and good is None
    assert missing == "anchor_unresolved" and many == "anchor_unresolved"


def test_anchor_failure_inside_a_run_is_a_miss(built):
    env = Env(built)
    bad_anchors = MismatchAnchors()

    async def scenario():
        await env.go()
        return await env.cache().run(
            env.scope,
            QUERY_EN,
            env.inputs(),
            env.answer_fn(),
            expected_index_version_id=env.version.id,
            anchors=bad_anchors,
            gate=env.gate,
            clock=env.clock,
            run_id_factory=env.run_id,
        )

    result = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "anchor_unresolved"
    assert len(env.chats) == 2


def test_revalidate_reports_scope_first(built):
    env = Env(built)

    async def scenario():
        cold = await env.go()
        entry = await env.store.get(env.scope, env.key())
        # scope, expiry and index version all fail; scope is reported first, then expiry, then the version.
        kwargs = {"anchors": env.anchors, "gate": env.gate}
        first = await revalidate(
            make_scope(),
            entry,
            cold.response,
            now=T0 + timedelta(days=9),
            expected_index_version_id="x",
            **kwargs,
        )
        second = await revalidate(
            env.scope, entry, cold.response, now=T0 + timedelta(days=9), expected_index_version_id="x", **kwargs
        )
        third = await revalidate(env.scope, entry, cold.response, now=T0, expected_index_version_id="x", **kwargs)
        return first, second, third

    assert run(scenario()) == ("scope_mismatch", "expired", "index_version_changed")


def test_store_errors_give_fresh_answers(built):
    env = Env(built)

    class BadGet(InMemoryResponseCacheStore):
        async def get(self, scope, key):
            raise RuntimeError("boom")

    class BadPut(InMemoryResponseCacheStore):
        async def put(self, scope, entry):
            raise RuntimeError("boom")

    class Exists(InMemoryResponseCacheStore):
        async def put(self, scope, entry):
            raise CacheError("cache_exists")

    async def scenario():
        return [await env.go(cache=env.cache(store=s)) for s in (BadGet(), BadPut(), Exists())]

    get_err, put_err, exists = run(scenario())
    for result in (get_err, put_err, exists):
        assert result.trace.outcome == "miss" and result.response.status == "answered"
        assert FACT_EN in result.response.answer
    assert get_err.trace.reason == "cache_error" and get_err.trace.store_reason is None
    assert put_err.trace.reason == "not_found" and put_err.trace.store_reason == "cache_error"
    assert exists.trace.reason == "not_found" and exists.trace.store_reason == "cache_exists"


def test_abstention_is_not_stored(built):
    env = Env(built, responder=cite_where("no such text anywhere", "x"))

    async def scenario():
        return await env.go(), await env.go()

    first, second = run(scenario())
    assert first.response.status == "abstained"
    assert first.trace.store_reason == "not_cacheable_status" and second.trace.outcome == "miss"
    assert len(env.chats) == 2


def test_pins_mismatch_is_not_stored(built):
    env = Env(built)

    async def scenario():
        first = await env.go(fn=env.answer_fn(k=3))  # pins say k=3, the key inputs say k=5
        return first, await env.go()

    first, second = run(scenario())
    assert first.response.status == "answered"
    assert first.trace.store_reason == "pins_mismatch" and first.trace.reason == "not_found"
    assert second.trace.outcome == "miss" and second.trace.reason == "not_found"


def test_unbuildable_entry_is_not_stored_and_not_raised(built):
    env = Env(built)

    async def scenario():
        fresh = await env.answer_fn()()
        big = fresh.model_copy(update={"answer": "x" * 250_000})  # response_json over its cap

        async def fn():
            return big

        result = await env.go(fn=fn)
        return result, await env.store.get(env.scope, env.key())

    result, stored = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.store_reason == "not_cacheable_status"
    assert result.response.answer.startswith("xxx")
    assert stored is None


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"freshness": "current"}, "freshness_current"),
        ({"mode": "enriched"}, "enriched_mode"),
        ({"tools": ("web",)}, "tools"),
        ({"is_action_run": True}, "action_run"),
    ],
)
def test_bypass_never_touches_the_store(built, changes, reason):
    env = Env(built)

    class Forbidden(InMemoryResponseCacheStore):
        async def get(self, scope, key):
            raise AssertionError("store accessed")

        async def put(self, scope, entry):
            raise AssertionError("store accessed")

        async def delete(self, scope, key):
            raise AssertionError("store accessed")

    result = run(env.go(inputs=env.inputs(**changes), cache=env.cache(store=Forbidden())))
    assert result.trace.outcome == "bypass" and result.trace.reason == reason
    assert result.response.status == "answered" and len(env.chats) == 1


def test_no_question_text_in_trace_or_entry(built):
    env = Env(built)

    async def scenario():
        cold, warm = await env.go(), await env.go()
        return cold, warm, await env.store.get(env.scope, env.key())

    cold, warm, entry = run(scenario())
    for text in (cold.trace.model_dump_json(), warm.trace.model_dump_json(), entry.model_dump_json()):
        assert QUERY_EN not in text and "Port Sallow observatory record" not in text
    assert entry.features_json is None and entry.question_vector is None


def test_foreign_entry_is_rejected_before_parsing(built):
    env = Env(built)
    other = make_scope()

    async def scenario():
        await env.go()
        entry = await env.store.get(env.scope, env.key())
        foreign = entry.model_copy(update={"key": env.key(other), "response_json": "{"})
        colliding = CollidingStore(foreign)
        return await env.go(scope=other, cache=env.cache(store=colliding)), colliding

    result, colliding = run(scenario())
    assert result.trace.outcome == "miss" and result.trace.reason == "scope_mismatch"
    assert colliding.deleted == [] and colliding._entries == {}


def test_hit_reports_whether_revocation_was_checked(built):
    env = Env(built)

    class Real:
        checks_revocation = True

        async def denied(self, scope, version_ids):
            return frozenset()

    async def scenario():
        await env.go()
        return await env.go(), await env.go(gate=Real())

    allow_all, real = run(scenario())
    assert allow_all.trace.outcome == "hit" and allow_all.trace.revocation_checked is False
    assert real.trace.outcome == "hit" and real.trace.revocation_checked is True


def test_lookup_reason_and_store_reason_are_both_visible(built):
    env = Env(built)

    class Deny:
        checks_revocation = True

        async def denied(self, scope, version_ids):
            return frozenset(version_ids)

    class Failing(InMemoryResponseCacheStore):
        async def put(self, scope, entry):
            raise RuntimeError("boom")

    async def scenario():
        store = Failing()
        await InMemoryResponseCacheStore.put(store, env.scope, (await _seed(env)))
        return await env.go(cache=env.cache(store=store), gate=Deny())

    result = run(scenario())
    assert result.trace.reason == "source_denied" and result.trace.store_reason == "cache_error"


async def _seed(env):
    await env.go()
    return await env.store.get(env.scope, env.key())
