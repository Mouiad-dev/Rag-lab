import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import StepBackConfig, StepBackRetrieval, register_builtin_plugins
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(text):
    return ScriptedChat(lambda _request: (json.dumps({"step_back_question": text}), "stop"))


def test_step_back_union_loss_and_duplicate():
    async def check():
        q, broader = "Find AB-1234 failures", "What causes failures in general?"
        chat = _chat(broader)
        retriever = RecordingRetriever({q: ("original",), broader: ("general",)})
        out = await StepBackRetrieval().run(q, chat=chat, retriever=retriever)
        assert retriever.queries == [q, broader]
        assert out.trace.retrieval_queries == (q, broader)
        assert out.trace.status == "applied"
        assert out.trace.lost_constraints == ("ab-1234",)
        assert [h.chunk_id for h in out.result.hits] == ["original", "general"]
        general = next(h for h in out.result.hits if h.chunk_id == "general")
        assert [(c.name, c.value, c.source_rank) for c in general.score_components] == [("q1_rank_score", 1.0, 1)]
        assert len(chat.requests) == 1
        duplicate = await StepBackRetrieval().run(
            "Caf\u00e9", chat=_chat("cafe\u0301"), retriever=RecordingRetriever({})
        )
        assert duplicate.trace.duplicates_dropped == 1
        assert duplicate.trace.retrieval_queries == ("Caf\u00e9",)
        assert len(duplicate.per_query) == 1

    asyncio.run(check())


def test_step_back_invalid_budget_pre_call_and_registration():
    async def check():
        bad_chat = ScriptedChat(lambda _r: ("not-json", "stop"))
        rejected = await StepBackRetrieval(StepBackConfig(on_invalid="original")).run(
            "q", chat=bad_chat, retriever=RecordingRetriever({})
        )
        assert rejected.trace.reason == "invalid_model_output" and len(rejected.trace.usage) == 1
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await StepBackRetrieval().run(
                "q", chat=ScriptedChat(lambda _r: ("not-json", "stop")), retriever=RecordingRetriever({})
            )
        scripted = _chat("general q")
        budgeted = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        no_call_retriever = RecordingRetriever({})
        with pytest.raises(BudgetExceededError):
            await StepBackRetrieval().run("q", chat=budgeted, retriever=no_call_retriever)
        assert no_call_retriever.queries == []

        class Wrong:
            profile = type("P", (), {"kind": "embedding"})()

        for model in (None, Wrong()):
            before = RecordingRetriever({})
            with pytest.raises(QueryTransformError):
                await StepBackRetrieval().run("q", chat=model, retriever=before)
            assert before.queries == []
        registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
        register_builtin_plugins(registry)
        assert registry.validate_config("R16", {}) == StepBackConfig()

    asyncio.run(check())
