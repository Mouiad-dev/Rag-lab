import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import QuestionRewriter, RewriteConfig, retrieve_with_rewrite
from rag_foundry_indexing.query_transform import hard_constraints, missing_constraints
from rag_foundry_indexing.rewrite import REWRITE_PROMPT_SHA256
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_providers.fake import fake_embedding_profile
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(payload, reason="stop"):
    return ScriptedChat(lambda _request: (json.dumps(payload), reason))


def test_applied_rewrite_is_metered_and_retrieves_rewrite():
    async def check():
        question = "\u0645\u0627 \u0633\u0639\u0631 2025"
        rewrite = question + " \u0627\u0644\u0645\u062a\u0648\u0633\u0637"
        chat = _chat({"query": rewrite})
        retriever = RecordingRetriever({rewrite: ("c1",)})
        output = await retrieve_with_rewrite(question, rewriter=QuestionRewriter(), chat=chat, retriever=retriever)
        assert output.trace.status == "applied"
        assert retriever.queries == [rewrite]
        assert output.trace.original_question == question
        assert output.trace.generated[0].role == "original"
        assert output.trace.generated[0].synthetic is False
        assert output.trace.generated[0].text == question
        assert output.trace.generated[1].synthetic
        assert output.trace.generated[1].text == rewrite
        assert len(output.trace.usage) == 1
        assert output.trace.prompt_sha256 == REWRITE_PROMPT_SHA256
        assert "untrusted" in chat.requests[0].messages[0].content
        assert json.loads(chat.requests[0].messages[1].content)["question"] == question
        assert question in chat.requests[0].messages[1].content

    asyncio.run(check())


@pytest.mark.parametrize(
    "question,rewrite",
    [
        ("report 2025", "report"),
        ("code AB-1234", "code"),
        ("not cats", "cats"),
        ("isn't cats", "cats"),
        ("\u0644\u064a\u0633 \u0642\u0637\u0629", "\u0642\u0637\u0629"),
        ("\u0648\u0644\u0627 \u0642\u0637\u0629", "\u0642\u0637\u0629"),
    ],
)
def test_constraint_loss_uses_original_and_fail_mode_raises(question, rewrite):
    async def check():
        chat = _chat({"query": rewrite})
        trace = await QuestionRewriter().rewrite(question, chat=chat)
        assert trace.status == "constraint_loss"
        assert trace.retrieval_queries == (question,)
        retriever = RecordingRetriever({question: ("c1",)})
        await retrieve_with_rewrite(
            question, rewriter=QuestionRewriter(), chat=_chat({"query": rewrite}), retriever=retriever
        )
        assert retriever.queries == [question]
        with pytest.raises(QueryTransformError, match="constraint_loss") as error:
            await QuestionRewriter(RewriteConfig(on_constraint_loss="fail")).rewrite(
                question, chat=_chat({"query": rewrite})
            )
        assert error.value.chat_result is not None

    asyncio.run(check())


def test_negation_equivalence_and_arabic_digits():
    assert missing_constraints("doesn't work", ("does not work",)) == ()
    assert hard_constraints("2025") == hard_constraints("\u0662\u0660\u0662\u0665")


def test_doesnt_to_does_not_applies_through_rewrite_path():
    async def check():
        question = "doesn't work"
        rewrite = "does not work"
        retriever = RecordingRetriever({rewrite: ("c1",)})
        output = await retrieve_with_rewrite(
            question, rewriter=QuestionRewriter(), chat=_chat({"query": rewrite}), retriever=retriever
        )
        assert output.trace.status == "applied"
        assert retriever.queries == [rewrite]

    asyncio.run(check())


@pytest.mark.parametrize(
    "response,reason",
    [("not json", "stop"), ('{"query":', "length"), ('{"query":"x","extra":1}', "stop"), ('{"query":""}', "stop")],
)
def test_invalid_model_output(response, reason):
    async def check():
        chat = ScriptedChat(lambda _request: (response, reason))
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await QuestionRewriter().rewrite("q", chat=chat)
        rejected = await QuestionRewriter(RewriteConfig(on_invalid="original")).rewrite("q", chat=chat)
        assert rejected.status == "rejected"
        assert len(rejected.usage) == 1
        retriever = RecordingRetriever({"q": ("c1",)})
        rejected_result = await retrieve_with_rewrite(
            "q",
            rewriter=QuestionRewriter(RewriteConfig(on_invalid="original")),
            chat=ScriptedChat(lambda _request: (response, reason)),
            retriever=retriever,
        )
        assert rejected_result.trace.status == "rejected"
        assert retriever.queries == ["q"]

    asyncio.run(check())


def test_preflight_and_budget_do_not_retrieve():
    async def check():
        scripted = _chat({"query": "q"})
        budgeted = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        retriever = RecordingRetriever({})
        with pytest.raises(BudgetExceededError):
            await retrieve_with_rewrite("q", rewriter=QuestionRewriter(), chat=budgeted, retriever=retriever)
        assert scripted.requests == []
        assert retriever.queries == []
        for question, code in (("", "empty_query"), ("q" * 2001, "query_too_long")):
            chat = _chat({"query": "x"})
            with pytest.raises(QueryTransformError, match=code):
                await QuestionRewriter().rewrite(question, chat=chat)
            assert chat.requests == []
        with pytest.raises(QueryTransformError, match="missing_model"):
            await QuestionRewriter().rewrite("q", chat=None)
        wrong_kind = ScriptedChat(
            lambda _request: (json.dumps({"query": "x"}), "stop"), profile=fake_embedding_profile()
        )
        with pytest.raises(QueryTransformError, match="wrong_model_kind"):
            await QuestionRewriter().rewrite("q", chat=wrong_kind)
        assert wrong_kind.requests == []

    asyncio.run(check())


def test_default_config():
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    from rag_foundry_indexing import register_builtin_plugins

    register_builtin_plugins(registry)
    assert registry.validate_config("R13", {}) == RewriteConfig()
