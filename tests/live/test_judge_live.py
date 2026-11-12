"""Opt-in live run of the answer-support judge on local Ollama (gemma4:e4b), complete and stream mode.

Whether Ollama already had the model loaded is not controlled; cold is the first call in this process, not a
guaranteed model load. The assertions check invariants only; they say nothing about judge accuracy.
"""

import asyncio
import os
import time

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_evaluation import TimingRecorder, render_cost_latency
from rag_foundry_evaluation.human_labels import agreement, placeholder_labels, render_agreement
from rag_foundry_evaluation.judge import RUBRIC_ANSWER_SUPPORT_V1, JudgeConfig, LlmJudge
from rag_foundry_evaluation.runner import StageSinks
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile
from rag_foundry_providers.ledger import RecordingChatModel

from tests.evaluation._judge_items import judge_items_v1

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]


def test_judge_live_complete_and_stream():
    async def run():
        settings = load_settings()
        ollama = OllamaChatModel(
            ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300
        )
        items = judge_items_v1()
        rubric = RUBRIC_ANSWER_SUPPORT_V1
        recorder = TimingRecorder(time.perf_counter)
        sinks = StageSinks()
        verdicts = []
        for mode in ("complete", "stream"):
            stage = "judge_" + mode
            chat = recorder.wrap(RecordingChatModel(ollama, sinks.sink(stage)), stage)
            verdicts.extend(await LlmJudge(rubric, JudgeConfig(mode=mode)).judge_all(items.items, chat=chat))
        return items, rubric, recorder, sinks, verdicts

    items, rubric, recorder, sinks, verdicts = asyncio.run(run())
    records = recorder.records()
    totals = sinks.totals()
    assert len(records) == 12
    assert all(r.ttft_ms is None for r in records if r.mode == "complete")
    assert all(0 <= r.ttft_ms <= r.total_ms for r in records if r.mode == "stream" and r.ok and r.ttft_ms is not None)
    assert sum(t.calls for t in totals.values()) == 12
    assert all(v.source == "llm_judge" and v.prompt_sha256 == rubric.prompt_sha256() for v in verdicts)
    reports = agreement(verdicts, placeholder_labels(items, rubric).labels, rubric=rubric)
    assert all(r.status == "not_yet_measured" for r in reports)

    print()
    print("| item | mode | status | label | reason |")
    print("|---|---|---|---|---|")
    for v in verdicts:
        print(f"| {v.item_id} | {v.mode} | {v.status} | {v.label or '-'} | {v.reason[:80]!r} |")
    print()
    print(
        render_cost_latency(
            title="Judge cost and latency (live, gemma4:e4b)",
            usage={stage: t.model_dump(mode="json") for stage, t in totals.items()},
            timings=records,
            notes=("one run, n=6 items per mode; not an evaluation",),
        )
    )
    print(render_agreement(reports))
