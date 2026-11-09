import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from _fixture import DISTRACTORS, FACT_EN, MARKDOWN, QUERY_EN
from _inmemory_index import ready_index
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError, QueryTransformError
from rag_foundry_indexing import HydeConfig, HydeRetriever
from rag_foundry_providers.budget import BudgetedEmbeddingModel, BudgetExceededError, RunBudget
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.messages import ChatResult, EmbeddingResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage
from rag_foundry_techniques import PluginKind, TechniqueRegistry, load_catalogue


class ScriptedChat:
    def __init__(self, text):
        self.profile = fake_chat_profile("scripted-hyde")
        self.text = text
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ChatResult(text=self.text, finish_reason="stop", usage=usage(self.profile, Operation.CHAT))


def usage(profile, operation):
    return Usage(
        profile_id=profile.id,
        provider=profile.provider,
        model=profile.model,
        operation=operation,
        input_tokens=3,
        output_tokens=2,
        token_source=TokenSource.ESTIMATED,
        latency_ms=1,
        estimated_cost_usd=0,
    )


class ScriptedEmbedder(LexicalTestEmbedder):
    def __init__(self, vectors):
        super().__init__()
        self.fixed = vectors

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((tuple(texts), input_type))
        if self.fixed is None:
            return await super().embed(texts, input_type=input_type)
        vectors = self.fixed[input_type]
        if len(vectors) == 1 and len(texts) > 1:
            vectors = vectors * len(texts)
        return EmbeddingResult(vectors=tuple(vectors), usage=usage(self.profile, Operation.EMBED))


def spans():
    paragraphs = MARKDOWN.split("\n\n")
    pos = 0
    result = []
    for paragraph in paragraphs:
        start = MARKDOWN.index(paragraph, pos)
        end = start + len(paragraph)
        pos = end
        if paragraph != "# Foundry field notes":
            result.append((start, end, paragraph))
    return result


def chat_for(text):
    return ScriptedChat(text)


def test_hyde_success_and_synthetic_text_is_not_a_hit():
    async def run():
        embedder = LexicalTestEmbedder()
        owner, _, index, version = await ready_index(spans(), embedder)
        embedder.calls.clear()
        passage = "The tidal observatory records sea levels every few minutes."
        chat = chat_for('{"passages":["' + passage + '"]}')
        result = await HydeRetriever(
            HydeConfig(include_question=False), scope=owner, index_version=version, index=index, embedder=embedder
        ).retrieve(QUERY_EN, chat=chat)
        assert result.result.hits[0].text == FACT_EN
        assert embedder.calls == [((passage,), "document")]
        assert passage not in result.result.model_dump_json()
        assert all(hit.text in {item[2] for item in spans()} for hit in result.result.hits)
        assert result.trace.retrieval_queries == ()
        assert [item.operation.value for item in result.trace.usage] == ["chat", "embed"]

    asyncio.run(run())


def test_hyde_mean_vector_is_normalized_mean():
    async def run():
        embedder = ScriptedEmbedder({"document": [(3.0, 4.0) + (0.0,) * 254], "query": [(0.0, 2.0) + (0.0,) * 254]})
        owner, _, index, version = await ready_index(spans(), LexicalTestEmbedder())
        observed = {}
        original = index.search

        async def spy(scope, version_id, vector, k):
            observed["vector"] = vector
            return await original(scope, version_id, vector, k)

        index.search = spy
        chat = chat_for('{"passages":["A distinct reference passage about tides and records."]}')
        await HydeRetriever(HydeConfig(), scope=owner, index_version=version, index=index, embedder=embedder).retrieve(
            QUERY_EN, chat=chat
        )
        assert observed["vector"] == pytest.approx((0.3, 0.9) + (0.0,) * 254)

    asyncio.run(run())


def test_hyde_can_rank_the_bakery_first():
    async def run():
        embedder = LexicalTestEmbedder()
        owner, _, index, version = await ready_index(spans(), embedder)
        retriever = HydeRetriever(
            HydeConfig(include_question=False), scope=owner, index_version=version, index=index, embedder=embedder
        )
        result = await retriever.retrieve(
            QUERY_EN, chat=chat_for('{"passages":["A bakery on Mill Lane bakes rye bread before dawn."]}')
        )
        assert result.result.hits[0].text == DISTRACTORS[0]

    asyncio.run(run())


def test_hyde_invalid_and_pre_call_paths():
    async def run():
        embedder = LexicalTestEmbedder()
        owner, stranger, index, version = await ready_index(spans(), embedder, ready=False)
        chat = chat_for('{"passages":["A sufficiently long hypothetical passage for testing."]}')
        retriever = HydeRetriever(HydeConfig(), scope=owner, index_version=version, index=index, embedder=embedder)
        with pytest.raises(IndexingError, match="not_ready"):
            await retriever.retrieve(QUERY_EN, chat=chat)
        assert chat.requests == []
        ready_owner, ready_stranger, ready_index_repo, ready = await ready_index(spans(), embedder)
        with pytest.raises(NotFoundError) as stranger_error:
            await HydeRetriever(
                HydeConfig(), scope=ready_stranger, index_version=ready, index=ready_index_repo, embedder=embedder
            ).retrieve(QUERY_EN, chat=chat_for('{"passages":["A sufficiently long hypothetical passage."]}'))
        assert stranger_error.value.chat_result is not None
        assert [u.operation.value for u in stranger_error.value.usage] == ["chat", "embed", "embed"]
        wrong = ScriptedEmbedder(None)
        wrong.profile = wrong.profile.model_copy(update={"model": "other-model"})
        chat = chat_for('{"passages":[]}')
        with pytest.raises(IndexingError, match="incompatible_embedding"):
            await HydeRetriever(
                HydeConfig(), scope=ready_owner, index_version=ready, index=ready_index_repo, embedder=wrong
            ).retrieve(QUERY_EN, chat=chat)
        assert chat.requests == []
        for response, code, config in (
            ("not json", "invalid_model_output", HydeConfig()),
            ('{"passages":[""]}', "invalid_model_output", HydeConfig()),
            (
                '{"passages":["one sufficiently long passage.","another sufficiently long passage."]}',
                "too_many_queries",
                HydeConfig(),
            ),
        ):
            model = chat_for(response)
            with pytest.raises(QueryTransformError) as exc:
                await HydeRetriever(
                    config, scope=ready_owner, index_version=ready, index=ready_index_repo, embedder=embedder
                ).retrieve(QUERY_EN, chat=model)
            assert exc.value.code == code

    asyncio.run(run())


def test_hyde_embedding_budget_failure_preserves_chat_result():
    async def run():
        embedder = ScriptedEmbedder(None)
        owner, _, index, version = await ready_index(spans(), embedder)
        budgeted = BudgetedEmbeddingModel(embedder, RunBudget(Decimal("0"), max_calls=0))
        chat = chat_for('{"passages":["A sufficiently long hypothetical passage."]}')
        with pytest.raises(BudgetExceededError) as exc:
            await HydeRetriever(
                HydeConfig(), scope=owner, index_version=version, index=index, embedder=budgeted
            ).retrieve(QUERY_EN, chat=chat)
        assert len(chat.requests) == 1
        assert exc.value.chat_result is not None
        assert exc.value.usage == (exc.value.chat_result.usage,)

    asyncio.run(run())


def test_hyde_embedding_errors_preserve_usage():
    class BrokenEmbedder(ScriptedEmbedder):
        def __init__(self, vectors):
            super().__init__(vectors)

        async def embed(self, texts, *, input_type="document"):
            self.calls.append((tuple(texts), input_type))
            if self.fixed == "mismatch":
                return EmbeddingResult(vectors=(), usage=usage(self.profile, Operation.EMBED))
            return EmbeddingResult(vectors=((0.0,) * 256,), usage=usage(self.profile, Operation.EMBED))

    async def run():
        for mode, error_type, code in (
            ("mismatch", IndexingError, "embedding_count_mismatch"),
            ("zero", QueryTransformError, "invalid_model_output"),
        ):
            embedder = BrokenEmbedder(mode)
            owner, _, index, version = await ready_index(spans(), LexicalTestEmbedder())
            chat = chat_for('{"passages":["A sufficiently long hypothetical passage."]}')
            with pytest.raises(error_type) as exc:
                await HydeRetriever(
                    HydeConfig(include_question=False),
                    scope=owner,
                    index_version=version,
                    index=index,
                    embedder=embedder,
                ).retrieve(QUERY_EN, chat=chat)
            assert getattr(exc.value, "code", None) == code
            assert exc.value.chat_result is not None
            assert [item.operation for item in exc.value.usage] == [Operation.CHAT, Operation.EMBED]

    asyncio.run(run())


def test_hyde_registry_registration():
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    from rag_foundry_indexing import register_builtin_plugins

    register_builtin_plugins(registry)
    assert registry.validate_config("R17", {}) == HydeConfig()
    assert HydeRetriever.manifest.kind == PluginKind.QUERY_TRANSFORM


def test_hyde_config_defaults():
    assert HydeConfig() == HydeConfig(
        n_passages=1,
        include_question=True,
        k=5,
        max_output_tokens=256,
        temperature=0.7,
        max_passage_chars=1200,
    )
