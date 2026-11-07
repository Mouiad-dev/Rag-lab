import asyncio
import logging
import sys

import pytest
from _fixture import FACT_EN
from _lexical_embedder import LexicalTestEmbedder
from _qa import corpus_markdown
from _scripted_chat import ScriptedChatModel, cite_where
from rag_foundry_answering.runner import LocalRunner, ollama_model_factory, retry_policy
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_contracts.runs import RunRequest
from rag_foundry_domain.errors import IndexingError, RunError, SettingsError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.indexing import IndexState, NewIndexVersion
from rag_foundry_domain.runtime_settings import RunSettings, RuntimeSettings
from rag_foundry_domain.settings_store import (
    InMemorySettingsOverrideRepository,
    NewSettingOverride,
    RuntimeSettingsService,
)
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import OllamaChatModel
from rag_foundry_providers.budget import BudgetedChatModel, BudgetedEmbeddingModel, RunBudget
from rag_foundry_providers.config import Endpoint
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from rag_foundry_providers.resilience import ResilientChatModel, ResilientEmbeddingModel
from test_extraction_service import ingest, setup


def _run(coro):
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)


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

    return _run(create())


def _factory(chat, embed):
    def make(ledger, budget, _retry, _tokens):
        return RecordingChatModel(chat, ledger), RecordingEmbeddingModel(embed, ledger)

    return make


def _runner(*args, **kwargs):
    kwargs.setdefault("settings", RuntimeSettingsService(RuntimeSettings()))
    return LocalRunner(*args, **kwargs)


def test_runner_answers_r02_with_explicit_threshold(built):
    env, version = built
    chat, embed = ScriptedChatModel(cite_where(FACT_EN, FACT_EN)), LexicalTestEmbedder()
    runner = _runner(env.index, env.anchors, model_factory=_factory(chat, embed))
    response = _run(
        runner.run(env.scope, RunRequest(question="How often?", index_version_id=version.id, min_score=0.0))
    )
    assert response.status == "answered" and response.usage.model_calls == 2


def test_unknown_embedding_threshold_abstains_before_calls(built):
    env, version = built
    embed = LexicalTestEmbedder()
    embed.profile = embed.profile.model_copy(update={"model": "unknown"})
    chat = ScriptedChatModel(lambda _request: ("answer [S1]", "stop"))
    with pytest.raises(RunError, match="abstention_threshold_unset"):
        _run(
            _runner(env.index, env.anchors, model_factory=_factory(chat, embed)).run(
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
    runner = _runner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder()))

    async def run():
        task = asyncio.create_task(runner.run(env.scope, RunRequest(question="Q", technique_id="R01")))
        await entered.wait()
        with pytest.raises(RunError, match="too_many_runs"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))
        release.set()
        await task
        assert await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))

    _run(run())


def test_timeout_releases_concurrency_slot(built):
    env, _version = built

    class BlockingChat(ScriptedChatModel):
        async def complete(self, request):
            self.requests.append(request)
            await asyncio.Event().wait()

    chat = BlockingChat(lambda _request: ("", "stop"))
    repository = InMemorySettingsOverrideRepository()
    _run(
        repository.append(
            NewSettingOverride(
                level="global",
                scope=None,
                key="run.deadline_s",
                value=0.05,
                created_by="test",
                reason="short deadline",
            )
        )
    )
    runner = _runner(
        env.index,
        env.anchors,
        model_factory=_factory(chat, LexicalTestEmbedder()),
        settings=RuntimeSettingsService(RuntimeSettings(), repo=repository),
    )

    async def run():
        with pytest.raises(RunError, match="run_timeout"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))
        with pytest.raises(RunError, match="run_timeout"):
            await runner.run(env.scope, RunRequest(question="Q", technique_id="R01"))

    _run(run())


def test_nonlocal_chat_rejected_before_calls(built):
    env, _version = built
    chat = ScriptedChatModel(lambda _request: ("answer", "stop"))
    chat.profile = chat.profile.model_copy(
        update={"capabilities": chat.profile.capabilities.model_copy(update={"local": False})}
    )
    with pytest.raises(RunError, match="non_local_model"):
        _run(
            _runner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder())).run(
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

    failed = _run(failed_version())
    version = failed
    embed = LexicalTestEmbedder()
    with pytest.raises(IndexingError, match="not_ready"):
        _run(
            _runner(
                env.index, env.anchors, model_factory=_factory(ScriptedChatModel(lambda _request: ("", "stop")), embed)
            ).run(env.scope, RunRequest(question="Q", index_version_id=version.id, min_score=0.0))
        )
    assert not embed.calls
    assert failed.state is IndexState.FAILED


def test_run_retry_disables_local_timeout_retry():
    policy = retry_policy(RunSettings())
    assert policy.retry_timeouts_for_local is False
    assert policy.max_attempts == 2


def test_ollama_factory_has_resilient_budgeted_recording_chain():
    chat, embed = ollama_model_factory("http://localhost:11434")(
        UsageLedger(), RunBudget(__import__("decimal").Decimal("0"), 4), retry_policy(RunSettings()), 512
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


def test_default_runtime_settings_pin_and_model_parameters(built, caplog):
    caplog.set_level(logging.INFO)
    env, _version = built
    chat = ScriptedChatModel(lambda _request: ("General answer", "stop"))
    embed = LexicalTestEmbedder()
    captured = {}

    def factory(ledger, budget, policy, max_output_tokens):
        captured.update(budget=budget, policy=policy, max_output_tokens=max_output_tokens)
        return RecordingChatModel(chat, ledger), RecordingEmbeddingModel(embed, ledger)

    runner = _runner(env.index, env.anchors, model_factory=factory)
    response = _run(runner.run(env.scope, RunRequest(question="private question", technique_id="R01")))
    assert captured["policy"].max_attempts == 2
    assert captured["policy"].attempt_timeout_s == 120.0
    assert captured["policy"].retry_timeouts_for_local is False
    assert captured["budget"].max_calls == 4
    assert captured["max_output_tokens"] == 512
    assert chat.requests[0].max_output_tokens == 512
    snap = _run(runner._settings.run_snapshot(env.scope))
    assert snap.run.deadline_s == 300.0
    assert snap.answer.min_scores["ollama/bge-m3/1024"] == 0.45
    assert response.pins.settings_sha256 == snap.sha256()
    assert "private question" not in caplog.text
    assert "settings=" in caplog.text and '"deadline_s":300.0' in caplog.text


def test_default_score_threshold_is_taken_from_runtime_settings(built, monkeypatch):
    env, version = built
    chat = ScriptedChatModel(cite_where(FACT_EN, FACT_EN))
    embed = LexicalTestEmbedder()
    from rag_foundry_answering import runner as runner_module

    def threshold(_provider, _model, _dim, table):
        return table["ollama/bge-m3/1024"]

    monkeypatch.setattr(runner_module, "default_min_score", threshold)
    response = _run(
        _runner(env.index, env.anchors, model_factory=_factory(chat, embed)).run(
            env.scope, RunRequest(question="How often?", index_version_id=version.id)
        )
    )
    assert response.pins.min_score == 0.45


def test_project_output_token_override_changes_only_project_request(built):
    env, _version = built
    repository = InMemorySettingsOverrideRepository()
    _run(
        repository.append(
            NewSettingOverride(
                level="project",
                scope=env.scope,
                key="answer.max_output_tokens",
                value=256,
                created_by="test",
                reason="project budget",
            )
        )
    )
    chat = ScriptedChatModel(lambda _request: ("General answer", "stop"))
    runner = _runner(
        env.index,
        env.anchors,
        model_factory=_factory(chat, LexicalTestEmbedder()),
        settings=RuntimeSettingsService(RuntimeSettings(), repo=repository),
    )
    _run(runner.run(env.scope, RunRequest(question="Q", technique_id="R01")))
    assert chat.requests[0].max_output_tokens == 256
    other = type(env.scope)(tenant_id=env.scope.tenant_id, project_id="00000000-0000-4000-8000-000000000001")
    _run(runner.run(other, RunRequest(question="Q", technique_id="R01")))
    assert chat.requests[1].max_output_tokens == 512


def test_bad_indexing_override_does_not_affect_run_snapshot():
    class Repository:
        async def current(self, _scope):
            from types import SimpleNamespace

            return (
                SimpleNamespace(level="global", key="indexing.c05_chunk_size", value=32),
                SimpleNamespace(level="global", key="indexing.c05_chunk_overlap", value=32),
            )

    service = RuntimeSettingsService(RuntimeSettings(), repo=Repository())
    from rag_foundry_domain.testing import make_scope

    scope = make_scope()

    async def resolve():
        assert (await service.run_snapshot(scope)).run.max_attempts == 2
        with pytest.raises(SettingsError, match="invalid_value"):
            await service.indexing(scope)

    _run(resolve())


def test_canary_leak_metric_detects_answer_text(built):
    env, version = built
    canary = "CANARY-7Q"
    chat = ScriptedChatModel(lambda _request: (f"{canary} [S1]", "stop"))
    response = _run(
        _runner(env.index, env.anchors, model_factory=_factory(chat, LexicalTestEmbedder())).run(
            env.scope,
            RunRequest(
                question="What colours is the harbour lighthouse painted?", index_version_id=version.id, min_score=0.0
            ),
        )
    )
    canary_leaked = canary in (response.answer or "") or any(canary in warning for warning in response.warnings)
    assert response.status == "answered"
    assert canary_leaked is True
