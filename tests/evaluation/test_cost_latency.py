import asyncio

import pytest
from rag_foundry_evaluation import (
    CallTiming,
    EvaluationError,
    TimingRecorder,
    render_cost_latency,
    summarize_runner_timings,
    summarize_timings,
)
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.messages import ChatMessage, ChatRequest

from tests.evaluation._stream_chat import StreamScriptedChat


class ListClock:
    def __init__(self, values):
        self._values = list(values)

    def __call__(self):
        assert self._values, "clock exhausted"
        return self._values.pop(0)


def _request():
    return ChatRequest(messages=(ChatMessage(role="user", content="hi"),))


def test_complete_timing():
    async def run():
        clock = ListClock([10.0, 10.25, 11.0, 11.1])
        rec = TimingRecorder(clock)
        model = rec.wrap(StreamScriptedChat(["a"]), "answer")
        await model.complete(_request())
        await model.complete(_request())
        return rec.records()

    first, second = asyncio.run(run())
    assert (first.total_ms, first.ttft_ms, first.cold, first.ok) == (250.0, None, True, True)
    assert second.cold is False


def test_stream_timing():
    async def run():
        rec = TimingRecorder(ListClock([1.0, 1.125, 1.5, 2.0, 2.5]))
        model = rec.wrap(StreamScriptedChat(["", "Hel", "lo"]), "answer")
        async for _ in model.stream(_request()):
            pass
        empty = rec.wrap(StreamScriptedChat([""]), "other")
        async for _ in empty.stream(_request()):
            pass
        return rec.records()

    first, second = asyncio.run(run())
    assert (first.ttft_ms, first.total_ms, first.ok) == (125.0, 500.0, True)
    assert (second.ttft_ms, second.total_ms) == (None, 500.0)


def test_early_stop_and_error():
    async def early():
        rec = TimingRecorder(ListClock([3.0, 3.2]))
        model = rec.wrap(StreamScriptedChat(["", "x"]), "answer")
        agen = model.stream(_request())
        await agen.__anext__()
        await agen.aclose()
        return rec.records()

    (record,) = asyncio.run(early())
    assert (record.total_ms, record.ttft_ms, record.ok) == (200.0, None, False)

    class Boom:
        profile = fake_chat_profile()

        async def complete(self, request):
            raise RuntimeError("boom")

    async def failing():
        rec = TimingRecorder(ListClock([0.0, 0.5]))
        with pytest.raises(RuntimeError):
            await rec.wrap(Boom(), "answer").complete(_request())
        return rec.records()

    (failed,) = asyncio.run(failing())
    assert (failed.total_ms, failed.ok) == (500.0, False)


def test_stages_and_profiles():
    rec = TimingRecorder(ListClock([]))
    with pytest.raises(EvaluationError) as info:
        rec.wrap(StreamScriptedChat([]), "Bad")
    assert info.value.code == "invalid_stage"

    async def run():
        rec = TimingRecorder(ListClock([float(i) for i in range(8)]))
        a = rec.wrap(StreamScriptedChat(["x"], fake_chat_profile("p-a")), "answer")
        b = rec.wrap(StreamScriptedChat(["x"], fake_chat_profile("p-b")), "answer")
        for m in (a, b, a, b):
            await m.complete(_request())
        return rec.records()

    assert [r.cold for r in asyncio.run(run())] == [True, True, False, False]


def _t(stage, mode, total, ttft=None, cold=False, ok=True):
    return CallTiming(stage=stage, profile_id="p", mode=mode, total_ms=total, ttft_ms=ttft, cold=cold, ok=ok)


_TIMINGS = [
    _t("judge", "complete", 250.0, cold=True),
    _t("judge", "complete", 100.0),
    _t("judge", "complete", 300.0),
    _t("judge", "complete", 50.0, ok=False),
    _t("answer", "stream", 500.0, 125.0, cold=True),
    _t("answer", "stream", 400.0, 75.0),
]


def test_summary():
    rows = {r.stage: r for r in summarize_timings(_TIMINGS)}
    judge, answer = rows["judge"], rows["answer"]
    assert (judge.calls, judge.failed, judge.cold_ms, judge.warm_median_ms) == (4, 1, 250.0, 200.0)
    assert (judge.total_median_ms, judge.total_max_ms, judge.ttft_status) == (250.0, 300.0, "not_applicable")
    assert (answer.cold_ms, answer.warm_median_ms, answer.total_median_ms, answer.total_max_ms) == (
        500.0,
        400.0,
        450.0,
        500.0,
    )
    assert (answer.ttft_median_ms, answer.ttft_status) == (100.0, "measured")
    assert list(rows) == ["answer", "judge"]


def test_render():
    usage = {
        "answer": {"calls": 2, "input_tokens": 10, "output_tokens": 4, "cost_usd": "0", "estimated_token_calls": 0},
        "retrieval": {"calls": 3, "input_tokens": 5, "output_tokens": 0, "cost_usd": None, "estimated_token_calls": 1},
    }
    out = render_cost_latency(title="T", usage=usage, timings=_TIMINGS[4:], notes=["n1"])
    assert "| answer | 2 | 0 | 10 | 4 | 0 | 0 | 500.000 | 400.000 | 450.000 | 500.000 | 100.000 | measured |" in out
    assert "| retrieval | 3 | - | 5 | 0 | unknown | 1 | - | - | - | - | - | not_measured |" in out
    assert "not reproducible" in out
    assert "## Retrieval timings" not in out


def test_runner_timings():
    timings = {
        "R02": {
            "a": {"total_ms": 9.0, "ttft": "not_applicable"},
            "b": {"total_ms": 3.0, "ttft": "not_applicable"},
            "c": {"total_ms": 5.0, "ttft": "not_applicable"},
        }
    }
    summary = summarize_runner_timings(timings)["R02"]
    assert (summary["n"], summary["cold_ms"], summary["warm_median_ms"], summary["max_ms"]) == (3, 9.0, 4.0, 9.0)
    out = render_cost_latency(title="T", usage={}, timings=[], runner_timings=timings)
    assert "## Retrieval timings" in out
    assert "| R02 | 3 | 9.000 | 4.000 | 9.000 | not_applicable |" in out
