import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import (
    DenseConfig,
    DenseRetriever,
    MetadataFilterConfig,
    MetadataFilteredRetriever,
    SelfQuery,
    SelfQueryConfig,
    register_builtin_plugins,
    retrieve_with_self_query,
)
from rag_foundry_indexing.metadata_filter import ViewMetadata
from rag_foundry_indexing.self_query import FILTER_SCHEMA
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._multi_doc_index import multi_doc_index

NULLS = {"document_refs": None, "media_types": None, "created_from": None, "created_to": None}


def _chat(payload, *, profile=None):
    return ScriptedChat(lambda _request: (json.dumps(payload), "stop"), profile=profile)


@pytest.fixture
def corpus():
    embedder = LexicalTestEmbedder()
    data = asyncio.run(multi_doc_index(embedder))
    inner = DenseRetriever(
        DenseConfig(k=50), scope=data.owner, index_version=data.owner_version, index=data.index, embedder=embedder
    )
    retriever = MetadataFilteredRetriever(
        MetadataFilterConfig(k=50),
        scope=data.owner,
        index_version=data.owner_version,
        index=data.index,
        extractions=data.extractions,
        documents=data.documents,
        inner=inner,
    )
    return data, embedder, retriever


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_refs", "20000000-0000-4000-8000-000000000099"),
        ("media_types", ["20000000-0000-4000-8000-000000000099"]),
        ("created_from", "20000000-0000-4000-8000-000000000099"),
        ("created_to", "20000000-0000-4000-8000-000000000099"),
        ("document_refs", ["d9"]),
    ],
)
def test_scope_widening_always_raises(field, value):
    async def check():
        payload = dict(NULLS)
        payload[field] = value
        with pytest.raises(QueryTransformError, match="filter_scope_widening"):
            await SelfQuery(SelfQueryConfig(on_invalid="no_filter")).parse("q", choices=(), chat=_chat(payload))

    asyncio.run(check())


@pytest.mark.parametrize("field", ("document_refs", "media_types", "created_from", "created_to"))
def test_scalar_uuid_in_any_allowed_field_widens(field):
    async def check():
        payload = dict(NULLS)
        payload[field] = "20000000-0000-4000-8000-000000000099"
        with pytest.raises(QueryTransformError, match="filter_scope_widening"):
            await SelfQuery(SelfQueryConfig(on_invalid="no_filter")).parse("q", choices=(), chat=_chat(payload))

    asyncio.run(check())


def test_multi_document_applied_result_and_request_has_no_uuids(corpus):
    async def check():
        data, _, retriever = corpus
        chat = _chat({**NULLS, "document_refs": ["d2"]})
        parsed, result = await retrieve_with_self_query(
            "quarterly report", self_query=SelfQuery(), chat=chat, retriever=retriever
        )
        assert parsed.status == "applied"
        assert parsed.filter.document_ids == (data.document_ids["report-2025"],)
        choices = await retriever.choices()
        allowed_views = {v.view_id for v in choices if v.document_id == data.document_ids["report-2025"]}
        assert result.hits
        assert all(hit.extraction_view_id in allowed_views for hit in result.hits)
        assert len(parsed.usage) == 1
        content = " ".join(message.content for request in chat.requests for message in request.messages)
        owner_views = await retriever.choices()
        stranger_views = [
            await data.extractions.get_view(data.stranger, view_id)
            for view_id in data.stranger_version.extraction_view_ids
        ]
        forbidden = [
            data.owner.tenant_id,
            data.owner.project_id,
            data.stranger.tenant_id,
            data.stranger.project_id,
            data.stranger_version.id,
            *(value for view in owner_views for value in (view.document_id, view.view_id, view.document_version_id)),
            *(value for view in stranger_views for value in (view.document_id, view.id, view.document_version_id)),
        ]
        assert all(str(value) not in content for value in forbidden)

    asyncio.run(check())


@pytest.mark.parametrize("ref_kind", ("stranger_uuid", "unknown_ref"))
def test_retrieve_scope_widening_skips_embedding(corpus, ref_kind):
    async def check():
        data, embedder, retriever = corpus
        ref = str(data.document_ids["stranger"]) if ref_kind == "stranger_uuid" else "d9"
        chat = _chat({**NULLS, "document_refs": [ref]})
        with pytest.raises(QueryTransformError, match="filter_scope_widening"):
            await retrieve_with_self_query(
                "quarterly report",
                self_query=SelfQuery(SelfQueryConfig(on_invalid="no_filter")),
                chat=chat,
                retriever=retriever,
            )
        assert not any(kind == "query" for _, kind in embedder.calls)

    asyncio.run(check())


def test_unknown_tenant_field_no_filter_retrieves_unfiltered(corpus):
    async def check():
        data, _, retriever = corpus
        baseline = await retriever.retrieve("quarterly report")
        chat = _chat({"tenant_id": str(data.owner.tenant_id)})
        parsed, filtered = await retrieve_with_self_query(
            "quarterly report",
            self_query=SelfQuery(SelfQueryConfig(on_invalid="no_filter")),
            chat=chat,
            retriever=retriever,
        )
        assert parsed.status == "rejected"
        assert {hit.chunk_id for hit in filtered.hits} == {hit.chunk_id for hit in baseline.hits}
        assert len(filtered.hits) <= len(baseline.hits)

    asyncio.run(check())


def test_invalid_values_and_dates():
    async def check():
        choices = ()
        for payload in (
            {**NULLS, "document_refs": []},
            {**NULLS, "media_types": []},
            {**NULLS, "created_from": "2025-03-02", "created_to": "2025-03-01"},
            {**NULLS, "created_from": "\u0662\u0660\u0662\u0665-01-01"},
        ):
            with pytest.raises(QueryTransformError, match="invalid_filter_value"):
                await SelfQuery().parse("q", choices=choices, chat=_chat(payload))
        parsed = await SelfQuery().parse("q", choices=choices, chat=_chat({**NULLS, "created_to": "2025-03-01"}))
        assert parsed.filter.created_to == datetime(2025, 3, 2, 0, 0, tzinfo=UTC)

    asyncio.run(check())


def test_schema_depends_on_capability():
    async def check():
        for enabled, expected in ((True, FILTER_SCHEMA), (False, None)):
            profile = fake_chat_profile()
            capabilities = profile.capabilities.model_copy(update={"structured_output": enabled})
            chat = _chat(NULLS, profile=profile.model_copy(update={"capabilities": capabilities}))
            await SelfQuery().parse("q", choices=(), chat=chat)
            assert chat.requests[0].response_schema == expected

    asyncio.run(check())


def test_default_config():
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R12", {}) == SelfQueryConfig()


def test_preflight_rejections_do_not_call_chat():
    async def check():
        cases = (("", "empty_query"), ("q" * 2001, "query_too_long"))
        for question, code in cases:
            chat = _chat(NULLS)
            with pytest.raises(QueryTransformError, match=code):
                await SelfQuery().parse(question, choices=(), chat=chat)
            assert chat.requests == []
        chat = _chat(NULLS)
        choices = tuple(
            ViewMetadata(
                view_id=f"10000000-0000-4000-8000-{i:012d}",
                document_id=f"20000000-0000-4000-8000-{i:012d}",
                document_version_id=f"30000000-0000-4000-8000-{i:012d}",
                display_name="doc",
                media_type="text/plain",
                version_created_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            for i in range(1, 52)
        )
        with pytest.raises(QueryTransformError, match="too_many_choices"):
            await SelfQuery().parse("q", choices=choices, chat=chat)
        assert chat.requests == []
        duplicate_views = (
            choices[0],
            choices[0].model_copy(update={"view_id": "10000000-0000-4000-8000-000000000099"}),
        )
        duplicate_chat = _chat(NULLS)
        await SelfQuery(SelfQueryConfig(max_choices=1)).parse("q", choices=duplicate_views, chat=duplicate_chat)
        assert len(duplicate_chat.requests) == 1
        scripted = _chat(NULLS)
        budgeted = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        with pytest.raises(BudgetExceededError):
            await SelfQuery().parse("q", choices=(), chat=budgeted)
        assert scripted.requests == []

    asyncio.run(check())


def test_bad_json_and_bad_media_type_reject():
    async def check():
        bad_json = ScriptedChat(lambda _request: ("not json", "stop"))
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await SelfQuery().parse("q", choices=(), chat=bad_json)
        bad_media = _chat({**NULLS, "media_types": ["image/png"]})
        with pytest.raises(QueryTransformError, match="invalid_filter_value"):
            await SelfQuery().parse("q", choices=(), chat=bad_media)
        truncated = ScriptedChat(lambda _request: ('{"document_refs":', "length"))
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await SelfQuery().parse("q", choices=(), chat=truncated)

    asyncio.run(check())


def test_missing_keys_are_invalid_output():
    async def check():
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await SelfQuery().parse("q", choices=(), chat=_chat({"document_refs": None}))

    asyncio.run(check())


def test_missing_model_and_wrong_model_kind_make_no_call():
    from rag_foundry_providers.fake import fake_embedding_profile

    async def check():
        with pytest.raises(QueryTransformError, match="missing_model"):
            await SelfQuery().parse("report", choices=(), chat=None)
        chat = _chat(NULLS, profile=fake_embedding_profile())
        with pytest.raises(QueryTransformError, match="wrong_model_kind"):
            await SelfQuery().parse("report", choices=(), chat=chat)
        assert chat.requests == []

    asyncio.run(check())


def test_arabic_question_is_sent_unescaped(corpus):
    question = (
        "\u0645\u0627 \u0633\u0639\u0631 \u0627\u0644\u0642\u0637\u0639\u0629"
        " \u0641\u064a \u062a\u0642\u0631\u064a\u0631 2025\u061f"
    )

    async def check():
        _, _, retriever = corpus
        chat = _chat(NULLS)
        parsed = await SelfQuery().parse(question, choices=await retriever.choices(), chat=chat)
        assert parsed.status == "no_filter"
        user_text = chat.requests[0].messages[-1].content
        assert question in user_text
        assert json.loads(user_text)["question"] == question

    asyncio.run(check())
