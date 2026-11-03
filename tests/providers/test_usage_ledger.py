import asyncio
from decimal import Decimal

import pytest
from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.fake import FakeChatModel, fake_chat_profile
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger, UsageSink, UsageTotals
from rag_foundry_providers.messages import ChatMessage, ChatRequest
from rag_foundry_providers.usage import Operation, TokenSource, Usage, estimate_cost


def priced(**over) -> ModelProfile:
    base = dict(
        id="c1",
        provider="p",
        model="m",
        kind="chat",
        cost_per_mtok_in=Decimal("3"),
        cost_per_mtok_out=Decimal("15"),
        cost_per_mtok_cache_read=Decimal("0.3"),
        cost_per_mtok_cache_write=Decimal("3.75"),
    )
    return ModelProfile(**{**base, **over})


def usage(**over) -> Usage:
    base = dict(
        profile_id="c1",
        provider="p",
        model="m",
        operation=Operation.CHAT,
        input_tokens=10,
        output_tokens=5,
        token_source=TokenSource.PROVIDER,
        latency_ms=1.0,
        estimated_cost_usd=Decimal("1"),
    )
    return Usage(**{**base, **over})


def req() -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content="hello"),))


def test_cache_fields_default_zero_and_validate():
    u = usage()
    assert (u.cache_read_input_tokens, u.cache_write_input_tokens) == (0, 0)
    with pytest.raises(ValueError):
        usage(cache_read_input_tokens=-1)
    with pytest.raises(ValueError):
        priced(cost_per_mtok_cache_read=Decimal("-1"))
    assert priced(cost_per_mtok_cache_read=None).cost_per_mtok_cache_read is None


def test_cost_with_cache_tokens_exact():
    p = priced()
    # 1M*3 + 1M*15 + 2M*0.3 + 1M*3.75 = 22.35
    assert estimate_cost(p, 1_000_000, 1_000_000, cache_read_tokens=2_000_000, cache_write_tokens=1_000_000) == Decimal(
        "22.35"
    )
    assert estimate_cost(p, 0, 0, cache_read_tokens=1000) == Decimal("0.0003")
    assert estimate_cost(p, 0, 0, cache_write_tokens=1000) == Decimal("0.00375")


def test_unknown_cache_rate():
    p = priced(cost_per_mtok_cache_read=None, cost_per_mtok_cache_write=None)
    assert estimate_cost(p, 10, 10, cache_read_tokens=1) is None
    assert estimate_cost(p, 10, 10, cache_write_tokens=1) is None
    assert estimate_cost(p, 1_000_000, 0, cache_read_tokens=0, cache_write_tokens=0) == Decimal("3")
    with pytest.raises(ValueError):
        estimate_cost(p, 0, 0, cache_read_tokens=-1)


def test_old_call_sites_unchanged():
    p = priced(cost_per_mtok_cache_read=None, cost_per_mtok_cache_write=None)
    assert estimate_cost(p, 2_000_000, 1_000_000) == Decimal("21")
    assert estimate_cost(p, 1000, 0) == Decimal("0.003")


def test_ledger_filters_order_and_totals():
    led = UsageLedger()
    a = usage(estimated_cost_usd=Decimal("1.5"), cache_read_input_tokens=4)
    b = usage(profile_id="c2", operation=Operation.EMBED, estimated_cost_usd=Decimal("2"), cache_write_input_tokens=6)
    c = usage(token_source=TokenSource.ESTIMATED, estimated_cost_usd=Decimal("0.5"))
    for u in (a, b, c):
        led.append(u)
    assert UsageSink in UsageLedger.__mro__
    assert led.records() == (a, b, c)
    assert led.records(profile_id="c1") == (a, c)
    assert led.records(operation=Operation.EMBED) == (b,)
    assert led.records(profile_id="c1", operation="embed") == ()
    t = led.totals()
    assert t == UsageTotals(
        calls=3,
        input_tokens=30,
        output_tokens=15,
        cache_read_input_tokens=4,
        cache_write_input_tokens=6,
        cost_usd=Decimal("4"),
        unknown_cost_calls=0,
        estimated_token_calls=1,
    )
    assert led.totals(profile_id="c1").cost_usd == Decimal("2.0")


def test_ledger_empty_and_none_propagation():
    led = UsageLedger()
    t = led.totals()
    assert (t.calls, t.cost_usd, t.unknown_cost_calls) == (0, Decimal(0), 0)
    led.append(usage())
    led.append(usage(estimated_cost_usd=None))
    t = led.totals()
    assert (t.calls, t.cost_usd, t.unknown_cost_calls) == (2, None, 1)


def test_ledger_append_only_and_snapshot():
    led = UsageLedger()
    assert not any(hasattr(led, n) for n in ("remove", "clear", "pop", "delete", "extend", "__setitem__"))
    led.append(usage())
    snap = led.records()
    led.append(usage())
    assert len(snap) == 1 and len(led.records()) == 2
    with pytest.raises(AttributeError):
        snap.append(usage())  # type: ignore[attr-defined]


def test_recording_complete_and_stream():
    async def run():
        led = UsageLedger()
        model = RecordingChatModel(FakeChatModel(fake_chat_profile()), led)
        assert model.profile.id == "fake-chat"
        res = await model.complete(req())
        assert led.records() == (res.usage,)
        chunks = [c async for c in model.stream(req())]
        assert chunks[-1].usage is not None
        assert len(led.records()) == 2
        assert led.records()[-1] == chunks[-1].usage

    asyncio.run(run())


def test_recording_early_break_and_error_record_nothing():
    async def run():
        led = UsageLedger()
        model = RecordingChatModel(FakeChatModel(fake_chat_profile()), led)
        stream = model.stream(req())
        async for _ in stream:
            break
        await stream.aclose()
        assert led.records() == ()

        class Boom:
            profile = fake_chat_profile()

            async def complete(self, request):
                raise RuntimeError("boom")

            async def stream(self, request):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        bad = RecordingChatModel(Boom(), led)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="boom"):
            await bad.complete(req())
        with pytest.raises(RuntimeError, match="boom"):
            _ = [c async for c in bad.stream(req())]
        assert led.records() == ()

    asyncio.run(run())
