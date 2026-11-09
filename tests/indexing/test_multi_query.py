import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import (
    MultiQueryConfig,
    register_builtin_plugins,
)
from rag_foundry_indexing import (
    MultiQueryExpansion as R14,
)
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(queries):
    return ScriptedChat(lambda _request: (json.dumps({"queries": queries}), "stop"))


def test_expansion_order_metering_and_union_component():
    async def check():
        q = "code AB-1234"
        variants = [q + " search", q + " find", q + " lookup"]
        chat = _chat(variants)
        retriever = RecordingRetriever(
            {q: ("common",), variants[0]: ("variant-only",), variants[1]: (), variants[2]: ()}
        )
        out = await R14().run(q, chat=chat, retriever=retriever)
        assert retriever.queries == [q, *variants]
        assert len(chat.requests) == 1
        assert out.trace.original_question == q
        assert out.trace.retrieval_queries == (q, *variants)
        only = next(hit for hit in out.result.hits if hit.chunk_id == "variant-only")
        assert only.score == 1.0
        assert [(c.name, c.value, c.source, c.source_rank) for c in only.score_components] == [
            ("q1_rank_score", 1.0, "R02", 1)
        ]

    asyncio.run(check())


def test_duplicate_loss_and_too_many():
    async def check():
        q = "code AB-1234"
        chat = _chat(["CODE ab-1234", "code AB-1234 useful", "code AB-1234 more"])
        out = await R14().run(
            q, chat=chat, retriever=RecordingRetriever({q: (), "code AB-1234 useful": (), "code AB-1234 more": ()})
        )
        assert out.trace.duplicates_dropped == 1
        assert out.trace.retrieval_queries == (q, "code AB-1234 useful", "code AB-1234 more")
        lossy = await R14().run(q, chat=_chat(["search only"]), retriever=RecordingRetriever({q: ()}))
        assert lossy.trace.generated[1].lost_constraints == ("ab-1234",)
        assert lossy.trace.lost_constraints == ("ab-1234",)
        assert lossy.trace.retrieval_queries == (q,)
        with pytest.raises(QueryTransformError, match="too_many_queries"):
            await R14(MultiQueryConfig(n_variants=5)).run(
                q, chat=_chat(["a", "b", "c", "d", "e", "f"]), retriever=RecordingRetriever({})
            )

    asyncio.run(check())


def test_nfc_duplicate_and_invalid_output_usage():
    async def check():
        q = "caf\u00e9 AB-1234"
        decomposed = "cafe\u0301 AB-1234"
        dup = await R14().run(
            q,
            chat=_chat([decomposed]),
            retriever=RecordingRetriever({q: ()}),
        )
        assert dup.trace.duplicates_dropped == 1
        assert dup.trace.retrieval_queries == (q,)

        rejected_retriever = RecordingRetriever({q: ()})
        rejected = await R14(MultiQueryConfig(on_invalid="original")).run(
            q,
            chat=_chat(["ok AB-1234", ""]),
            retriever=rejected_retriever,
        )
        assert rejected_retriever.queries == [q]
        assert rejected.trace.status == "rejected"
        assert rejected.trace.reason == "invalid_model_output"
        assert rejected.per_query[0].technique_id == "R02"
        assert len(rejected.trace.usage) == 1

        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await R14().run(q, chat=_chat(["ok AB-1234", ""]), retriever=RecordingRetriever({}))

    asyncio.run(check())


def test_budget_and_config_registration():
    async def check():
        scripted = _chat(["x"])
        chat = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        retriever = RecordingRetriever({})
        with pytest.raises(BudgetExceededError):
            await R14().run("q", chat=chat, retriever=retriever)
        assert retriever.queries == []
        registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
        register_builtin_plugins(registry)
        assert registry.validate_config("R14", {}) == MultiQueryConfig()

    asyncio.run(check())
