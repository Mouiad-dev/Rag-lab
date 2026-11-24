import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import DecomposeConfig, QuestionDecomposition, register_builtin_plugins
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(items):
    return ScriptedChat(lambda _request: (json.dumps({"subquestions": items}), "stop"))


def _sq(ident, text, depends=()):
    return {"id": ident, "text": text, "depends_on": list(depends)}


def test_dependencies_order_and_applied_retrieval():
    async def check():
        q = "Compare 2025 revenue and explain the cause"
        items = [_sq("s1", "What was 2025 revenue?"), _sq("s2", "What caused that revenue?", ("s1",))]
        chat = _chat(items)
        retriever = RecordingRetriever({q: ("base",), items[0]["text"]: ("one",), items[1]["text"]: ("two",)})
        out = await QuestionDecomposition().run(q, chat=chat, retriever=retriever)
        assert retriever.queries == [q, items[0]["text"], items[1]["text"]]
        assert out.trace.status == "applied"
        assert out.trace.generated[2].depends_on == ("s1",)
        assert [h.chunk_id for h in out.result.hits] == ["base", "one", "two"]
        assert len(chat.requests) == 1
        for bad in (
            [_sq("s1", "part", ("s2",)), _sq("s2", "2025 part")],
            [_sq("s1", "part", ("s1",))],
            [_sq("s1", "part", ("s9",))],
        ):
            with pytest.raises(QueryTransformError, match="invalid_dependency"):
                await QuestionDecomposition().run(q, chat=_chat(bad), retriever=RecordingRetriever({}))
        not_a_list = [{"id": "s1", "text": "2025 part", "depends_on": "s0"}]
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await QuestionDecomposition().run(q, chat=_chat(not_a_list), retriever=RecordingRetriever({}))
        duplicate = [_sq("s1", "2025 first"), _sq("s1", "2025 second")]
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await QuestionDecomposition().run(q, chat=_chat(duplicate), retriever=RecordingRetriever({}))
        with pytest.raises(QueryTransformError, match="too_many_queries"):
            await QuestionDecomposition().run(
                q,
                chat=_chat([_sq(f"s{i}", f"2025 part {i}") for i in range(1, 6)]),
                retriever=RecordingRetriever({}),
            )

    asyncio.run(check())


def test_constraint_loss_arabic_invalid_usage_and_pre_call_checks():
    async def check():
        q = "Find reports from 2025 about AB-1234"
        items = [_sq("s1", "Find reports about AB-1234")]
        retriever = RecordingRetriever({q: ("orig",)})
        out = await QuestionDecomposition().run(q, chat=_chat(items), retriever=retriever)
        assert out.trace.status == "constraint_loss"
        assert out.trace.lost_constraints == ("2025",)
        assert retriever.queries == [q]
        with pytest.raises(QueryTransformError, match="constraint_loss"):
            await QuestionDecomposition(DecomposeConfig(on_constraint_loss="fail")).run(
                q, chat=_chat(items), retriever=RecordingRetriever({})
            )
        arabic = "\u0645\u0627 \u062d\u062f\u062b \u0641\u064a 2025 \u0648\u0645\u0627 \u0633\u0628\u0628\u0647\u061f"
        ar_items = [
            _sq("s1", "\u0645\u0627 \u062d\u062f\u062b \u0641\u064a 2025\u061f"),
            _sq("s2", "\u0645\u0627 \u0633\u0628\u0628\u0647\u061f"),
        ]
        ar = await QuestionDecomposition().run(arabic, chat=_chat(ar_items), retriever=RecordingRetriever({}))
        assert ar.trace.status == "applied"
        bad_chat = ScriptedChat(lambda _r: ("not-json", "stop"))
        bad = await QuestionDecomposition(DecomposeConfig(on_invalid="original")).run(
            "q", chat=bad_chat, retriever=RecordingRetriever({})
        )
        assert bad.trace.reason == "invalid_model_output" and len(bad.trace.usage) == 1
        scripted = _chat([_sq("s1", "2025 part")])
        budgeted = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        no_call_retriever = RecordingRetriever({})
        with pytest.raises(BudgetExceededError):
            await QuestionDecomposition().run("q", chat=budgeted, retriever=no_call_retriever)
        assert no_call_retriever.queries == []

        class Wrong:
            profile = type("P", (), {"kind": "embedding"})()

        for model in (None, Wrong()):
            before = RecordingRetriever({})
            with pytest.raises(QueryTransformError):
                await QuestionDecomposition().run("q", chat=model, retriever=before)
            assert before.queries == []
        registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
        register_builtin_plugins(registry)
        assert registry.validate_config("R15", {}) == DecomposeConfig()

    asyncio.run(check())
