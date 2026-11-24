"""Opt-in live smoke for R28 conversational rewrite on local Ollama (gemma4:e4b). One run, not an evaluation."""

import asyncio
import os

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_indexing import ConversationalRewriter, ConversationConfig, ConversationTurn
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

ALLOWED = {"applied", "rejected", "constraint_loss", "unsupported_fact"}


def _turn(index, text, role="user"):
    return ConversationTurn(conversation_id="c1", turn_index=index, role=role, text=text)


S0 = _turn(0, "Tell me about the reading room schedule.")
OBS0 = _turn(0, "Tell me about the Port Sallow observatory.")
OBS1 = _turn(1, "It records sea levels every 17 minutes.", role="assistant")
AR_T0 = _turn(
    0,
    "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629 "
    "\u062a\u062d\u0641\u0638 \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629\u061f",
)
CASES = (
    ("EN", "How often does it record them?", (OBS0, OBS1)),
    ("AR", "\u0648\u0623\u064a\u0646 \u062a\u062d\u0641\u0638\u0647\u0627\u061f", (AR_T0,)),
    (
        "mixed",
        "Is \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 open at \u0660\u0668:\u0663\u0660?",
        (S0,),
    ),
)


def test_conversation_live_smoke():
    async def run():
        settings = load_settings()
        ledger = UsageLedger()
        chat = RecordingChatModel(
            OllamaChatModel(ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300),
            ledger,
        )
        rewriter = ConversationalRewriter(ConversationConfig(on_invalid="original"))
        for name, question, history in CASES:
            trace = await rewriter.rewrite(question, history=history, conversation_id="c1", chat=chat)
            transform = trace.transform
            assert transform.status in ALLOWED
            assert transform.original_question == question
            assert transform.generated[0].synthetic is False
            assert all(q.synthetic for q in transform.generated if q.role == "rewrite")
            rewrite = [q.text for q in transform.generated if q.role == "rewrite"]
            print(f"{name} status={transform.status}")
            print(f"standalone={rewrite[0] if rewrite else '-'}")
            print(f"introduced_constraints={list(trace.introduced_constraints)}")
            print(f"lost_constraints={list(transform.lost_constraints)}")
            for item in transform.usage:
                print(
                    "usage",
                    f"model={item.model}",
                    f"input_tokens={item.input_tokens}",
                    f"output_tokens={item.output_tokens}",
                    f"latency_ms={item.latency_ms}",
                )
        assert ledger.totals().calls == 3

    asyncio.run(run())
