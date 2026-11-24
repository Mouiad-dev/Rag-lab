# ruff: noqa: UP031
"""P6.4 lesson demo: R28 conversational standalone rewrite, guards, history window, usage.

Offline. The chat is a scripted responder (a fixed JSON reply per case); it is not a model. Every status, flag,
constraint list, query and count printed comes from calling the package code on those inputs, so the output shows
plumbing and guards, not resolution quality. The output has no timing values and is byte-identical between runs.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_indexing import ConversationalRewriter, ConversationConfig, ConversationTurn  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat  # noqa: E402

OUT = []
USAGE = []  # (case, chat calls from the trace, requests the scripted chat received)


def emit(line=""):
    OUT.append(line)


def turn(index, text, role="user"):
    return ConversationTurn(conversation_id="c1", turn_index=index, role=role, text=text)


S0 = turn(0, "Tell me about the reading room schedule.")
OBS0 = turn(0, "Tell me about the Port Sallow observatory.")
OBS1 = turn(1, "It records sea levels every 17 minutes.", role="assistant")
OBS_Q = "How often does it record them?"
AR_T0 = turn(
    0,
    "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629 "
    "\u062a\u062d\u0641\u0638 \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629\u061f",
)
AR_Q = "\u0648\u0623\u064a\u0646 \u062a\u062d\u0641\u0638\u0647\u0627\u061f"
AR_REPLY = (
    "\u0623\u064a\u0646 \u062a\u062d\u0641\u0638 "
    "\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 "
    "\u0627\u0644\u0645\u062e\u0637\u0648\u0637\u0627\u062a\u061f"
)
MIXED_Q = "Is \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 open at \u0660\u0668:\u0663\u0660?"


async def run(name, question, history, reply, config=None):
    chat = ScriptedChat(lambda _request: (json.dumps({"query": reply}), "stop"))
    trace = await ConversationalRewriter(config).rewrite(question, history=history, conversation_id="c1", chat=chat)
    USAGE.append((name, len(trace.transform.usage), len(chat.requests)))
    return trace


def show(title, question, history, reply, trace):
    transform = trace.transform
    emit("## %s" % title)
    emit()
    emit("| field | value |")
    emit("|---|---|")
    emit("| history given | %s |" % " / ".join("%s: %s" % (t.role, t.text) for t in history))
    emit("| history used (user turns) | %d of %d given |" % (len(trace.history_used), len(history)))
    emit("| assistant turns excluded | %d |" % trace.assistant_excluded)
    emit("| follow-up question | %s |" % question)
    emit("| scripted reply (stands in for a model) | %s |" % reply)
    emit("| status | %s |" % transform.status)
    emit("| reason | %s |" % (transform.reason or "-"))
    emit("| introduced_constraints | %s |" % (", ".join(trace.introduced_constraints) or "-"))
    emit("| lost_constraints | %s |" % (", ".join(transform.lost_constraints) or "-"))
    emit(
        "| generated (role, synthetic) | %s |"
        % ", ".join("%s synthetic=%s" % (q.role, q.synthetic) for q in transform.generated)
    )
    emit("| retrieval query | %s |" % ", ".join(transform.retrieval_queries))
    emit("| chat calls | %d |" % len(transform.usage))
    emit()


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P6.4 R28 conversational rewrite: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p64_conversation_demo.py`. Date: 2026-11-12. "
        "Offline; scripted chat; no model; shows plumbing and guards, not resolution quality."
    )
    emit()

    reply = "How often does the Port Sallow observatory record sea levels?"
    trace = await run("EN follow-up", OBS_Q, (OBS0, OBS1), reply)
    show("1. EN follow-up (applied)", OBS_Q, (OBS0, OBS1), reply, trace)
    emit(
        'The assistant turn is not sent to the rewriter by default, so the fact "17" from the assistant answer is '
        "not in the prompt. The standalone question is marked synthetic; the original question is not."
    )
    emit()

    trace = await run("AR follow-up", AR_Q, (AR_T0,), AR_REPLY)
    show("2. AR follow-up (applied)", AR_Q, (AR_T0,), AR_REPLY, trace)

    reply = "Is the Oasis Library open at 08:30?"
    trace = await run("Mixed AR/EN", MIXED_Q, (S0,), reply)
    show("3. Mixed language with Arabic-Indic digits (applied)", MIXED_Q, (S0,), reply, trace)
    emit("The Arabic-Indic digits are normalised before the constraint check, so `08:30` in the reply counts as kept.")
    emit()

    reply = "How often does the Port Sallow observatory record sea levels every 17 minutes?"
    trace = await run("Losing (a) unsupported_fact", OBS_Q, (OBS0, OBS1), reply)
    show("4. Losing case (a): `unsupported_fact`", OBS_Q, (OBS0, OBS1), reply, trace)
    emit(
        'The reply adds "17", which only the assistant turn contains and the user turns do not support. The guard '
        'blocks it, so retrieval falls back to the raw follow-up "%s", a weak query with no topic. This is the '
        "cost of the guard: it prevents an unsupported fact from steering retrieval, and it also loses the "
        'resolution of "it" and "them". With `include_assistant=True` the same reply is accepted.' % OBS_Q
    )
    emit()

    question = "What about 2024, not 2025?"
    reply = "What about 2024?"
    trace = await run("Losing (b) constraint_loss", question, (S0,), reply)
    show("5. Losing case (b): `constraint_loss`", question, (S0,), reply, trace)
    emit(
        'The reply drops "2025" and "not", so the meaning changed. The code rejects the rewrite and retrieves '
        "with the original question."
    )
    emit()

    turns = tuple(turn(i, chr(97 + i) * 100) for i in range(8))
    trace = await run("Window", "q", turns, "q", ConversationConfig(max_turns=3))
    emit("## 6. History window")
    emit()
    emit("| field | value |")
    emit("|---|---|")
    emit("| turns given | %d |" % len(turns))
    emit("| max_turns | 3 |")
    emit("| turns used (turn_index) | %s |" % ", ".join(str(t.turn_index) for t in trace.history_used))
    emit("| history_dropped | %d |" % trace.history_dropped)
    emit("| status | %s |" % trace.transform.status)
    emit()
    emit("The oldest turns are dropped first; the most recent `max_turns` user turns are kept.")
    emit()

    emit("## 7. Usage (scripted chat; no timing values)")
    emit()
    emit("| case | chat calls (trace) | requests the scripted chat received |")
    emit("|---|---|---|")
    for row in USAGE:
        emit("| %s | %d | %d |" % row)
    empty = ScriptedChat(lambda _request: (json.dumps({"query": "q"}), "stop"))
    none = await ConversationalRewriter().rewrite("q", history=(), conversation_id="c1", chat=empty)
    emit("| no_history (empty history) | %d | %d |" % (len(none.transform.usage), len(empty.requests)))
    emit()
    emit("A turn with no usable history makes no chat call (status `%s`)." % none.transform.status)
    emit()
    emit("## Live run (orchestrator)")
    emit()
    emit("- Command: `<orchestrator>`")
    emit("- Date: `<orchestrator>`")
    emit("- Model: `<orchestrator>`")
    emit("- Result: `<orchestrator>`")
    emit("- Printed output: `<orchestrator>`")
    emit()
    print("\n".join(OUT))


asyncio.run(main())
