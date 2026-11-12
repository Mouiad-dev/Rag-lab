import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_indexing import (
    ConversationalRewriter,
    ConversationConfig,
    ConversationTurn,
    retrieve_with_conversation,
)
from rag_foundry_indexing.conversation import CONVERSATION_PROMPT_SHA256
from rag_foundry_providers.budget import BudgetedChatModel, BudgetExceededError, RunBudget
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(payload, reason="stop"):
    return ScriptedChat(lambda _request: (json.dumps(payload), reason))


def _turn(index, text, role="user", conversation_id="c1"):
    return ConversationTurn(conversation_id=conversation_id, turn_index=index, role=role, text=text)


S0 = _turn(0, "Tell me about the reading room schedule.")
OBS0 = _turn(0, "Tell me about the Port Sallow observatory.")
OBS1 = _turn(1, "It records sea levels every 17 minutes.", role="assistant")
OBS_Q = "How often does it record them?"


async def _rewrite(question, history, reply, config=None, conversation_id="c1"):
    chat = _chat({"query": reply})
    trace = await ConversationalRewriter(config).rewrite(
        question, history=history, conversation_id=conversation_id, chat=chat
    )
    return trace, chat


def test_en_applied():
    async def check():
        reply = "How often does the Port Sallow observatory record sea levels?"
        trace, chat = await _rewrite(OBS_Q, (OBS0, OBS1), reply)
        assert trace.transform.status == "applied"
        assert trace.history_used == (OBS0,)
        assert trace.assistant_excluded == 1
        assert trace.history_dropped == 0
        assert trace.transform.retrieval_queries == (reply,)
        assert trace.transform.generated[0].synthetic is False
        assert trace.transform.generated[1].synthetic is True
        request = chat.requests[0]
        payload = json.loads(request.messages[1].content)
        assert len(payload["history"]) == 1
        assert payload["question"] == OBS_Q
        assert "17" not in request.messages[0].content + request.messages[1].content
        assert len(trace.transform.usage) == 1
        assert trace.transform.prompt_sha256 == CONVERSATION_PROMPT_SHA256
        assert "untrusted" in request.messages[0].content

    asyncio.run(check())


def test_unsupported_memory_fact():
    async def check():
        reply = "How often does the Port Sallow observatory record sea levels every 17 minutes?"
        trace, _ = await _rewrite(OBS_Q, (OBS0, OBS1), reply)
        assert trace.transform.status == "unsupported_fact"
        assert trace.transform.reason == "unsupported_memory_fact"
        assert trace.introduced_constraints == ("17",)
        assert trace.transform.retrieval_queries == (OBS_Q,)
        with_assistant, _ = await _rewrite(OBS_Q, (OBS0, OBS1), reply, ConversationConfig(include_assistant=True))
        assert with_assistant.transform.status == "applied"
        assert with_assistant.assistant_excluded == 0
        with pytest.raises(QueryTransformError, match="unsupported_memory_fact") as error:
            await _rewrite(OBS_Q, (OBS0, OBS1), reply, ConversationConfig(on_unsupported_fact="fail"))
        assert error.value.chat_result is not None

    asyncio.run(check())


def test_constraint_loss():
    async def check():
        question = "What about 2024, not 2025?"
        trace, _ = await _rewrite(question, (S0,), "What about 2024?")
        assert trace.transform.status == "constraint_loss"
        assert trace.transform.lost_constraints == ("2025", "not")
        assert trace.transform.retrieval_queries == (question,)
        with pytest.raises(QueryTransformError, match="constraint_loss") as error:
            await _rewrite(question, (S0,), "What about 2024?", ConversationConfig(on_constraint_loss="fail"))
        assert error.value.chat_result is not None

    asyncio.run(check())


def test_constraint_loss_wins_over_unsupported_fact():
    async def check():
        question = "Show the report for 2025."
        trace, _ = await _rewrite(question, (S0,), "Show the report for 2026.")
        assert trace.transform.status == "constraint_loss"
        assert trace.transform.lost_constraints == ("2025",)
        assert trace.transform.retrieval_queries == (question,)

    asyncio.run(check())


def test_arabic_follow_up():
    async def check():
        t0 = _turn(
            0,
            (
                "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629 "
                "\u062a\u062d\u0641\u0638 \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629\u061f"
            ),
        )
        question = "\u0648\u0623\u064a\u0646 \u062a\u062d\u0641\u0638\u0647\u0627\u061f"
        reply = (
            "\u0623\u064a\u0646 \u062a\u062d\u0641\u0638 "
            "\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 "
            "\u0627\u0644\u0645\u062e\u0637\u0648\u0637\u0627\u062a\u061f"
        )
        trace, _ = await _rewrite(question, (t0,), reply)
        assert trace.transform.status == "applied"
        assert trace.transform.retrieval_queries == (reply,)
        negated = "\u0623\u064a\u0646 \u0644\u0627 " + reply[4:]
        trace, _ = await _rewrite(question, (t0,), negated)
        assert trace.transform.status == "unsupported_fact"
        assert trace.introduced_constraints == ("\u0644\u0627",)
        assert trace.transform.retrieval_queries == (question,)

    asyncio.run(check())


def test_mixed_language_and_digit_normalisation():
    async def check():
        question = (
            "Is \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 open at \u0660\u0668:\u0663\u0660?"
        )
        trace, _ = await _rewrite(question, (S0,), "Is the Oasis Library open at 08:30?")
        assert trace.transform.status == "applied"
        trace, _ = await _rewrite(question, (S0,), "Is the Oasis Library open?")
        assert trace.transform.status == "constraint_loss"
        assert trace.transform.lost_constraints == ("08", "30")

    asyncio.run(check())


def test_topic_switch_keeps_history_and_marks_synthetic():
    async def check():
        question = "Which club meets on Tuesday evenings?"
        retriever = RecordingRetriever({question: ("c1",)})
        out = await retrieve_with_conversation(
            question,
            history=(OBS0,),
            conversation_id="c1",
            rewriter=ConversationalRewriter(),
            chat=_chat({"query": question}),
            retriever=retriever,
        )
        assert out.trace.transform.status == "applied"
        assert retriever.queries == [question]
        assert out.trace.transform.generated[1].synthetic is True
        assert out.trace.history_used == (OBS0,)
        assert len(out.result.hits) == 1

    asyncio.run(check())


def test_window():
    async def check():
        turns = tuple(_turn(i, chr(97 + i) * 100) for i in range(8))
        trace, _ = await _rewrite("q", turns, "q", ConversationConfig(max_turns=3))
        assert trace.history_used == turns[-3:]
        assert trace.history_dropped == 5
        trace, _ = await _rewrite("q", turns, "q", ConversationConfig(max_turns=6, max_history_chars=250))
        assert trace.history_used == turns[-2:]
        assert trace.history_dropped == 6

    asyncio.run(check())


def test_scope_order_and_no_history_make_no_requests():
    async def check():
        chat = _chat({"query": "q"})
        rewriter = ConversationalRewriter()
        with pytest.raises(QueryTransformError, match="history_scope_mismatch"):
            await rewriter.rewrite(
                "q", history=(S0, _turn(1, "x", conversation_id="c2")), conversation_id="c1", chat=chat
            )
        for indices in ([0, 2, 1], [0, 0]):
            history = tuple(_turn(i, "x") for i in indices)
            with pytest.raises(QueryTransformError, match="history_order_invalid"):
                await rewriter.rewrite("q", history=history, conversation_id="c1", chat=chat)
        trace = await rewriter.rewrite("q", history=(), conversation_id="c1", chat=chat)
        assert trace.transform.status == "no_history"
        assert trace.transform.usage == ()
        assert trace.transform.retrieval_queries == ("q",)
        assert len(trace.transform.generated) == 1
        assert chat.requests == []

    asyncio.run(check())


@pytest.mark.parametrize(
    "response,reason",
    [("not json", "stop"), ('{"query":', "length"), ('{"query":"x","extra":1}', "stop"), ('{"query":""}', "stop")],
)
def test_invalid_model_output(response, reason):
    async def check():
        chat = ScriptedChat(lambda _request: (response, reason))
        with pytest.raises(QueryTransformError, match="invalid_model_output"):
            await ConversationalRewriter().rewrite("q", history=(S0,), conversation_id="c1", chat=chat)
        trace = await ConversationalRewriter(ConversationConfig(on_invalid="original")).rewrite(
            "q", history=(S0,), conversation_id="c1", chat=chat
        )
        assert trace.transform.status == "rejected"
        assert len(trace.transform.usage) == 1
        assert trace.transform.retrieval_queries == ("q",)

    asyncio.run(check())


def test_budget_raises_before_retrieval_and_registry():
    async def check():
        scripted = _chat({"query": "q"})
        budgeted = BudgetedChatModel(scripted, RunBudget(Decimal("0"), max_calls=0), default_max_output_tokens=None)
        retriever = RecordingRetriever({})
        with pytest.raises(BudgetExceededError):
            await retrieve_with_conversation(
                "q",
                history=(S0,),
                conversation_id="c1",
                rewriter=ConversationalRewriter(),
                chat=budgeted,
                retriever=retriever,
            )
        assert scripted.requests == []
        assert retriever.queries == []

    asyncio.run(check())
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    from rag_foundry_indexing import register_builtin_plugins

    register_builtin_plugins(registry)
    assert registry.validate_config("R28", {}) == ConversationConfig()
