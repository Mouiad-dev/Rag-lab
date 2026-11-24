# ruff: noqa: UP031
"""P6.5 demo: judge plumbing with scripted replies, per-stage cost table, agreement not yet measured.

Offline. The chat is a scripted responder, not a model: it gives the same reply for every item except one
malformed reply. The output shows plumbing only, contains no timing values and is byte-identical between runs.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_evaluation import render_cost_latency  # noqa: E402
from rag_foundry_evaluation.human_labels import agreement, placeholder_labels, render_agreement  # noqa: E402
from rag_foundry_evaluation.judge import RUBRIC_ANSWER_SUPPORT_V1, JudgeConfig, LlmJudge  # noqa: E402
from rag_foundry_evaluation.runner import StageSinks  # noqa: E402
from rag_foundry_providers.ledger import RecordingChatModel  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat  # noqa: E402
from tests.evaluation._judge_items import judge_items_v1  # noqa: E402

COMMAND = "uv run python docs/techniques/lessons/demos/p65_judge_cost_demo.py"
DATE = "2026-11-12"
MALFORMED_ID = "j-05"
OK_REPLY = json.dumps({"label": "supported", "reason": "scripted"}, separators=(",", ":"))
ITEMS = judge_items_v1()
ID_BY_ANSWER = {item.answer: item.item_id for item in ITEMS.items}


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def scripted_reply(request):
    payload = json.loads(request.messages[-1].content)
    item_id = ID_BY_ANSWER[payload["answer"]]
    return ("not json" if item_id == MALFORMED_ID else OK_REPLY, "stop")


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    items = ITEMS
    rubric = RUBRIC_ANSWER_SUPPORT_V1
    sinks = StageSinks()
    chat = RecordingChatModel(ScriptedChat(scripted_reply), sinks.sink("judge"))
    verdicts = await LlmJudge(rubric, JudgeConfig(mode="complete")).judge_all(items.items, chat=chat)

    out = []
    emit = out.append
    emit("# P6.5 judge and cost: demo output")
    emit("")
    emit("`%s` | %s | offline; scripted chat; no model; not a benchmark" % (COMMAND, DATE))
    emit("")
    emit("## Items")
    emit("")
    emit("| item | question | answer | evidence marker |")
    emit("|---|---|---|---|")
    for item in items.items:
        markers = ", ".join(e.marker for e in item.evidence)
        emit("| %s | %s | %s | %s |" % (item.item_id, cell(item.question), cell(item.answer), markers))
    emit("")
    emit("## Scripted judge")
    emit("")
    emit("**Scripted replies, identical for every item; not a judgement; shows plumbing only.**")
    emit("")
    emit(
        "Every item gets `%s`, except `%s`, which gets the malformed reply `not json` and is recorded as "
        "`invalid_output`." % (OK_REPLY, MALFORMED_ID)
    )
    emit("")
    emit("| item | status | label | prompt_sha256[:12] | rubric |")
    emit("|---|---|---|---|---|")
    for v in verdicts:
        emit(
            "| %s | %s | %s | %s | %s v%s |"
            % (v.item_id, v.status, v.label or "-", v.prompt_sha256[:12], v.rubric_id, v.rubric_version)
        )
    emit("")
    emit("## Agreement")
    emit("")
    placeholder = placeholder_labels(items, rubric)
    emit(render_agreement(agreement(verdicts, placeholder.labels, rubric=rubric)).rstrip("\n"))
    emit("")
    emit("## Cost and latency")
    emit("")
    usage = {stage: totals.model_dump(mode="json") for stage, totals in sinks.totals().items()}
    emit(
        render_cost_latency(
            title="Judge cost per stage (scripted chat, fake usage)",
            usage=usage,
            timings=(),
            notes=("Offline: no call was timed, so latency is not_measured. Token counts are fake usage.",),
        ).rstrip("\n")
    )
    emit("")
    emit("## Live run (orchestrator)")
    emit("")
    emit("<orchestrator>")
    print("\n".join(out))


asyncio.run(main())
