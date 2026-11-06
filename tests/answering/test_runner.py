import asyncio

import pytest
from _fixture import FACT_EN
from _lexical_embedder import LexicalTestEmbedder
from _qa import corpus_markdown
from _scripted_chat import ScriptedChatModel, cite_where
from rag_foundry_answering.runner import DEFAULT_MAX_CONCURRENT, RUN_RETRY, LocalRunner, ollama_model_factory
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_contracts.runs import RunRequest
from rag_foundry_domain.errors import IndexingError, RunError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.indexing import IndexState, NewIndexVersion
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import OllamaChatModel
from rag_foundry_providers.budget import BudgetedChatModel, BudgetedEmbeddingModel, RunBudget
from rag_foundry_providers.config import Endpoint
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.resilience import ResilientChatModel, ResilientEmbeddingModel
from test_extraction_service import ingest, setup


@pytest.fixture
def built(tmp_path):
    async def create():
        from rag_foundry_domain.testing import make_scope

        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(
            intake, scope, corpus_markdown().encode(), name="qa.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        anchors = AnchorService(store, views, provider_namespace="test")
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=300, chunk_overlap=40))
        version = await IndexService(store, views, index, anchors).build(
            scope, (view.id,), chunker, LexicalTestEmbedder()
        )

        class Env:
            pass

        env = Env()
        env.scope, env.index, env.anchors = scope, index, anchors
        return env, version

    return asyncio.run(create())


def _factory(chat, embed):
    def make(ledger, budget):
        return RecordingChatModel(chat, ledger), RecordingEmbeddingModel(embed, ledger)

    return make


def test_runner_answers_r02_with_explicit_threshold(built):
    env, version = built
    chat, embed = ScriptedChatModel(cite_where(FACT_EN, FACT_EN)), LexicalTestEmbedder()
    runner = LocalRunner(env.index, env.anchors, model_factory=_factory(chat, embed))
    response = asyncio.run(
        runner.run(env.scope, RunRequest(question="How often?", index_version_id=version.id, min_score=0.0))
    )
    assert response.status == "answered" and response.usage.model_calls == 2


def test_unknown_embedding_threshold_abstains_before_calls(built):
    env, version = built
    embed = LexicalTestEmbedder()
    embed.profile = embed.profile.model_copy(update={"model": "unknown"})
    chat = ScriptedChatModel(lambda _request: ("answer [S1]", "stop"))
    with pytest.raises(RunError, match="abstention_threshold_unset"):
        asyncio.run(
            LocalRunner(env.index, env.anchors, model_factory=_factory(chat, embed)).run(
                env.scope, RunRequest(question="Q", index_version_id=version.id)
            )
        )
    assert not chat.requests and not embed.calls


def test_default_concurrency_rejects_second_and_recovers(built):
    env, _version = built
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingChat(ScriptedChatModel):
        async def complete(self, request):
            self.requests.append(request)
            entered.set()
            await release.wait()
            return await super().complete(request)

    chat = BlockingChat(lambda _request: ("answer", "stop"))
    runner = LocalRunner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder()))

    async def run():
        task = asyncio.create_task(runner.run(env.scope, RunRequest(question="Q", technique_id="R01")))
        await entered.wait()
        with pytest.raises(RunError, match="too_many_runs"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))
        release.set()
        await task
        assert await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))

    asyncio.run(run())
    assert DEFAULT_MAX_CONCURRENT == 1


def test_timeout_releases_concurrency_slot(built):
    env, _version = built

    class BlockingChat(ScriptedChatModel):
        async def complete(self, request):
            self.requests.append(request)
            await asyncio.Event().wait()

    chat = BlockingChat(lambda _request: ("", "stop"))
    runner = LocalRunner(
        env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder()), run_timeout_s=0.05
    )

    async def run():
        with pytest.raises(RunError, match="run_timeout"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))
        with pytest.raises(RunError, match="run_timeout"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))

    asyncio.run(run())


def test_nonlocal_chat_rejected_before_calls(built):
    env, _version = built
    chat = ScriptedChatModel(lambda _request: ("answer", "stop"))
    chat.profile = chat.profile.model_copy(
        update={"capabilities": chat.profile.capabilities.model_copy(update={"local": False})}
    )
    with pytest.raises(RunError, match="non_local_model"):
        asyncio.run(
            LocalRunner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder())).run(
                env.scope, RunRequest(question="Q", technique_id="R01")
            )
        )
    assert not chat.requests


def test_failed_index_version_rejected_before_embedding(built):
    env, version = built

    async def failed_version():
        # A ready version cannot be marked failed, so build a second version and fail it while building.
        new = NewIndexVersion(**{name: getattr(version, name) for name in NewIndexVersion.model_fields})
        building = await env.index.create_index_version(env.scope, new)
        return await env.index.mark_failed(env.scope, building.id)

    failed = asyncio.run(failed_version())
    version = failed
    embed = LexicalTestEmbedder()
    with pytest.raises(IndexingError, match="not_ready"):
        asyncio.run(
            LocalRunner(
                env.index, env.anchors, model_factory=_factory(ScriptedChatModel(lambda _request: ("", "stop")), embed)
            ).run(env.scope, RunRequest(question="Q", index_version_id=version.id, min_score=0.0))
        )
    assert not embed.calls
    assert failed.state is IndexState.FAILED


def test_run_retry_disables_local_timeout_retry():
    assert RUN_RETRY.retry_timeouts_for_local is False
    assert RUN_RETRY.max_attempts == 2


def test_ollama_factory_has_resilient_budgeted_recording_chain():
    chat, embed = ollama_model_factory("http://localhost:11434")(
        UsageLedger(), RunBudget(__import__("decimal").Decimal("0"), 4)
    )
    assert isinstance(chat, ResilientChatModel)
    assert isinstance(chat._model, BudgetedChatModel) and isinstance(chat._model._model, RecordingChatModel)
    assert isinstance(chat._model._model._model, OllamaChatModel)
    assert isinstance(embed, ResilientEmbeddingModel)
    assert isinstance(embed._model, BudgetedEmbeddingModel) and isinstance(embed._model._model, RecordingEmbeddingModel)


def test_ollama_factory_urls_are_valid_endpoints():
    from rag_foundry_core.settings import Settings

    for url in ("http://ollama:11434", Settings().ollama_base_url):
        assert Endpoint(id="ollama", adapter="ollama", base_url=url, local=True).base_url == url


def test_canary_leak_metric_detects_answer_text(built):
    env, version = built
    canary = "CANARY-7Q"
    chat = ScriptedChatModel(lambda _request: (f"{canary} [S1]", "stop"))
    response = asyncio.run(
        LocalRunner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder())).run(
            env.scope,
            RunRequest(
                question="What colours is the harbour lighthouse painted?", index_version_id=version.id, min_score=0.0
            ),
        )
    )
    canary_leaked = canary in (response.answer or "") or any(canary in warning for warning in response.warnings)
    assert response.status == "answered"
    assert canary_leaked is True
