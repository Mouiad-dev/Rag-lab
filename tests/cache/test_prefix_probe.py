import asyncio
import itertools

import pytest
from rag_foundry_cache import PrefixCacheProbe, PrefixProbeConfig
from rag_foundry_domain.errors import CacheError
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.messages import ChatChunk, ChatResult
from rag_foundry_providers.usage import Operation, ProviderTimings, TokenSource, Usage

PREFIX = "SECRET-PREFIX-TEXT " * 5
QUESTIONS = ("first secret question?", "second secret question?")


class Clock:
    """Advances 1 ms (0.001 s) on every read."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


class FakeChat:
    def __init__(
        self, *, cache_read=0, reported=False, timings=None, deltas=("", "a", "b"), usage_on_stream=True, clock=None
    ):
        self.profile = fake_chat_profile()
        self.cache_read, self.reported, self.timings = cache_read, reported, timings
        self.deltas, self.usage_on_stream, self.clock = deltas, usage_on_stream, clock
        self.requests = []

    def _usage(self, op):
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=op,
            input_tokens=10,
            output_tokens=2,
            cache_read_input_tokens=self.cache_read,
            cache_read_reported=self.reported,
            token_source=TokenSource.PROVIDER,
            latency_ms=1.0,
            timings=self.timings,
        )

    async def complete(self, request):
        self.requests.append(request)
        return ChatResult(text="ok", finish_reason="stop", usage=self._usage(Operation.CHAT))

    async def stream(self, request):
        self.requests.append(request)
        for delta in self.deltas:
            if self.clock is not None:
                self.clock.now += 0.010  # time passes before every chunk, empty or not
            yield ChatChunk(delta=delta)
        if self.usage_on_stream:
            yield ChatChunk(delta="", usage=self._usage(Operation.CHAT_STREAM))


def _run(chat, clock=None, **overrides):
    config = PrefixProbeConfig(prefix_text=PREFIX, questions=QUESTIONS, **overrides)
    counter = itertools.count()
    return asyncio.run(
        PrefixCacheProbe(config).run(chat, clock=clock or Clock(), salt_factory=lambda: f"s{next(counter)}")
    )


def _system(chat, i):
    return chat.requests[i].messages[0].content


def test_arm_order_and_counts():
    chat = FakeChat()
    report = _run(chat)
    counts = {}
    for record in report.records:
        counts[record.arm] = counts.get(record.arm, 0) + 1
    assert counts == {"cold": 2, "warm": 3, "changed": 1, "eviction_attempt": 2}
    assert len(chat.requests) == 8
    assert [r.arm for r in report.records] == ["cold"] * 2 + ["warm"] * 3 + ["changed"] + ["eviction_attempt"] * 2
    assert report.records[2].label == "prime"
    assert [a.arm for a in report.arms] == ["cold", "warm", "changed", "eviction_attempt"]


def test_arms_subset_honoured():
    chat = FakeChat()
    report = _run(chat, arms=("warm", "changed"))
    assert {r.arm for r in report.records} == {"warm", "changed"}
    assert len(chat.requests) == 4


def test_cold_prefixes_differ_and_warm_is_byte_identical():
    chat = FakeChat()
    report = _run(chat)
    cold = [_system(chat, i) for i in (0, 1)]
    warm = [_system(chat, i) for i in (2, 3, 4)]
    assert cold[0] != cold[1]
    assert all(c.endswith(PREFIX) and c != PREFIX for c in cold)
    assert warm == [PREFIX] * 3
    changed, other, original = _system(chat, 5), _system(chat, 6), _system(chat, 7)
    assert changed != PREFIX and changed[1:] == PREFIX[1:]
    assert other != PREFIX and len(other) == len(PREFIX)
    assert original == PREFIX
    assert report.records[2].prefix_sha256 == report.records[7].prefix_sha256
    assert report.records[0].prefix_sha256 != report.records[1].prefix_sha256


def test_stream_ttft_at_first_non_empty_delta():
    clock = Clock()
    report = _run(FakeChat(deltas=("", "", "a", "b"), clock=clock), clock=clock, mode="stream")
    # Two empty chunks then "a": 3 x 10 ms of stream time plus 1 ms for the probe's own clock read.
    # A TTFT taken at the first chunk of any kind would be about 11 ms.
    assert report.records[0].ttft_ms == pytest.approx(31.0)
    assert report.records[0].total_ms == pytest.approx(42.0)  # 4 chunks of 10 ms, plus 2 clock reads
    assert report.records[0].input_tokens == 10


def test_stream_without_usage_keeps_fields_missing():
    report = _run(FakeChat(usage_on_stream=False), mode="stream")
    record = report.records[0]
    assert record.input_tokens is None and record.cache_read_input_tokens is None
    assert record.cache_read_reported is False
    assert report.cache_metric == "unavailable"


def test_complete_mode_ttft_is_none():
    report = _run(FakeChat())
    assert all(r.ttft_ms is None for r in report.records)
    assert all(a.ttft_ms_median is None for a in report.arms)


def test_cache_read_tokens_make_metric_reported():
    report = _run(FakeChat(cache_read=7, reported=True))
    assert report.cache_metric == "reported"
    assert report.arms[1].cache_read_tokens_median == 7


def test_reported_zero_is_reported_with_median_zero():
    report = _run(FakeChat(cache_read=0, reported=True))
    assert report.cache_metric == "reported"
    assert all(a.cache_read_tokens_median == 0 for a in report.arms)


def test_tokens_without_flag_count_as_reported():
    assert _run(FakeChat(cache_read=3, reported=False)).cache_metric == "reported"


def test_no_flag_and_zero_tokens_is_unavailable_with_none_median():
    report = _run(FakeChat(cache_read=0, reported=False))
    assert report.cache_metric == "unavailable"
    assert all(a.cache_read_tokens_median is None for a in report.arms)


def test_missing_timings_give_none_medians_never_zero():
    report = _run(FakeChat())
    for arm in report.arms:
        assert arm.prefill_ms_median is None and arm.decode_ms_median is None
    partial = _run(FakeChat(timings=ProviderTimings(prefill_ms=5.0)))
    assert partial.arms[0].prefill_ms_median == 5.0
    assert partial.arms[0].decode_ms_median is None


def test_timings_medians_when_present():
    report = _run(FakeChat(timings=ProviderTimings(prefill_ms=4.0, decode_ms=2.0)))
    assert report.arms[1].prefill_ms_median == 4.0
    assert report.arms[1].decode_ms_median == 2.0
    assert "proxy" in report.note


def test_report_has_no_prompt_text():
    dumped = _run(FakeChat()).model_dump_json()
    assert "SECRET" not in dumped and "secret question" not in dumped


@pytest.mark.parametrize(
    "bad",
    [
        {"prefix_text": ""},
        {"questions": ("only one",)},
        {"questions": ("ok", "  ")},
        {"max_output_tokens": 0},
        {"max_output_tokens": 513},
        {"arms": ("cold", "cold")},
        {"arms": ("nope",)},
        {"extra": 1},
    ],
)
def test_invalid_config_raises_before_any_call(bad):
    chat = FakeChat()
    config = {"prefix_text": PREFIX, "questions": QUESTIONS} | bad
    with pytest.raises(CacheError) as info:
        PrefixCacheProbe(config)
    assert info.value.code == "prefix_probe_invalid"
    assert chat.requests == []


def test_provider_error_propagates_unchanged():
    class Boom(FakeChat):
        async def complete(self, request):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        _run(Boom())
