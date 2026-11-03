"""AnthropicChatModel against httpx.MockTransport with docs-shaped JSON and SSE (no network, fake key)."""

import asyncio
import json
import logging
import traceback
import uuid
from contextlib import aclosing
from decimal import Decimal

import httpx
import pytest
from rag_foundry_providers import TokenSource
from rag_foundry_providers.adapters.anthropic import (
    AnthropicChatModel,
    UnsupportedRequestError,
    anthropic_chat_profile,
    anthropic_haiku_profile,
    anthropic_sonnet_profile,
)
from rag_foundry_providers.contract_checks import CHAT_CHECKS, DETERMINISTIC_CHAT_CHECKS
from rag_foundry_providers.errors import (
    MissingCapabilityError,
    MissingCredentialError,
    ProviderCallError,
    ProviderError,
)
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatMessage, ChatRequest, ToolSpec
from rag_foundry_providers.usage import Operation

SENTINEL = f"sk-ant-TEST-SENTINEL-{uuid.uuid4()}"
CANARY = f"CANARY-MESSAGE-{uuid.uuid4()}"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)


def _req(text: str = "hello", **kw) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), **kw)


def _model(handler, profile=None, **kw) -> AnthropicChatModel:
    return AnthropicChatModel(profile or anthropic_haiku_profile(), transport=httpx.MockTransport(handler), **kw)


def _usage(**extra) -> dict:
    u = {"input_tokens": 25, "output_tokens": 6}
    u.update(extra)
    return u


def _message(text="Hello world", stop="end_turn", usage=None, content=None, model="claude-haiku-5-5") -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage if usage is not None else _usage(),
    }


def _sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {n}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for n, d in events).encode()


def _stream_events(texts=("Hello", " world"), stop="end_turn", start_usage=None, delta_usage=None) -> list:
    start_msg = _message(content=[], stop=None, usage=start_usage or _usage(output_tokens=1))
    events: list = [
        ("message_start", {"type": "message_start", "message": start_msg}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        ("ping", {"type": "ping"}),
    ]
    for t in texts:
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}},
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": delta_usage or {"output_tokens": 6},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


def _ok(log: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=_sse(*_stream_events()))
        return httpx.Response(200, json=_message())

    return handler


def run(coro):
    return asyncio.run(coro)


async def _drain(model, request):
    return [c async for c in model.stream(request)]


# --- contract ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check", CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS, ids=[c.__name__ for c in CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS]
)
def test_chat_contract(check) -> None:
    run(check(_model(_ok())))


# --- profiles and construction -----------------------------------------------------------------


def test_profiles() -> None:
    h, s = anthropic_haiku_profile(), anthropic_sonnet_profile()
    assert (h.model, s.model) == ("claude-haiku-5-5", "claude-sonnet-5-5")
    for p in (h, s):
        c = p.capabilities
        assert p.provider == "anthropic" and c.local is False and c.streaming is True
        assert (c.tool_calling, c.structured_output, c.vision) == (False, False, False)
        assert c.context_window == 1_000_000 and c.max_output_tokens == 128_000
    assert (h.cost_per_mtok_in, h.cost_per_mtok_out) == (Decimal("0.10"), Decimal("0.50"))
    assert (h.cost_per_mtok_cache_read, h.cost_per_mtok_cache_write) == (Decimal("0.01"), Decimal("0.125"))
    assert (s.cost_per_mtok_in, s.cost_per_mtok_out) == (Decimal("2"), Decimal("10"))
    assert (s.cost_per_mtok_cache_read, s.cost_per_mtok_cache_write) == (Decimal("0.10"), Decimal("2.50"))
    unknown = anthropic_chat_profile("x-model", "claude-unknown-1")
    assert unknown.cost_per_mtok_in is None and unknown.cost_per_mtok_cache_write is None
    assert unknown.capabilities.context_window is None


def test_constructor_guards(monkeypatch) -> None:
    caps = anthropic_haiku_profile().capabilities.model_copy(update={"tool_calling": True})
    with pytest.raises(ProviderError):
        AnthropicChatModel(anthropic_haiku_profile().model_copy(update={"capabilities": caps}))
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(MissingCredentialError):
        AnthropicChatModel(anthropic_haiku_profile())
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    assert repr(AnthropicChatModel(anthropic_haiku_profile())) == "AnthropicChatModel(profile='anthropic-haiku')"


# --- payload -----------------------------------------------------------------------------------


def _sent(model_profile, request, *, stream=False) -> dict:
    log: list[httpx.Request] = []
    model = _model(_ok(log), model_profile)
    if stream:
        run(_drain(model, request))
    else:
        run(model.complete(request))
    assert len(log) == 1
    return json.loads(log[0].content)


def test_haiku_payload() -> None:
    body = _sent(anthropic_haiku_profile(), _req())
    assert body == {
        "model": "claude-haiku-5-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "disabled"},
    }


def test_sonnet_payload_default_and_explicit_max_tokens() -> None:
    body = _sent(anthropic_sonnet_profile(), _req())
    assert body["max_tokens"] == 16384 and body["thinking"] == {"type": "between_tools"}
    assert "disabled" not in json.dumps(body)
    assert _sent(anthropic_sonnet_profile(), _req(max_output_tokens=77))["max_tokens"] == 77
    assert _sent(anthropic_haiku_profile(), _req(max_output_tokens=5), stream=True)["max_tokens"] == 5


def test_stream_payload_sets_stream_flag_and_never_sends_forbidden_fields() -> None:
    body = _sent(anthropic_haiku_profile(), _req(), stream=True)
    assert body["stream"] is True
    for name in ("tools", "tool_choice", "cache_control", "temperature", "top_p", "top_k"):
        assert name not in body


def test_system_messages_joined_to_top_level() -> None:
    request = ChatRequest(
        messages=(
            ChatMessage(role="system", content="rule one"),
            ChatMessage(role="system", content="rule two"),
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
            ChatMessage(role="user", content="q2"),
        )
    )
    body = _sent(anthropic_haiku_profile(), request)
    assert body["system"] == "rule one\n\nrule two"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "request_",
    [
        _req(temperature=0.0),
        _req(temperature=0.7),
        ChatRequest(messages=(ChatMessage(role="user", content="q"), ChatMessage(role="system", content="late"))),
        ChatRequest(messages=(ChatMessage(role="user", content="q"), ChatMessage(role="tool", content="t"))),
        ChatRequest(messages=(ChatMessage(role="system", content="only"),)),
    ],
    ids=["temp0", "temp", "late-system", "tool-role", "no-messages"],
)
def test_refused_before_any_call(request_, stream) -> None:
    log: list = []
    model = _model(_ok(log))
    with pytest.raises(UnsupportedRequestError):
        run(_drain(model, request_) if stream else model.complete(request_))
    assert log == []


@pytest.mark.parametrize("stream", [False, True])
def test_tools_and_schema_refused_before_any_call(stream) -> None:
    log: list = []
    model = _model(_ok(log))
    tool = ToolSpec(name="t", description="d", parameters_schema={"type": "object"})
    for request in (_req(tools=(tool,)), _req(response_schema={"type": "object"})):
        with pytest.raises(MissingCapabilityError):
            run(_drain(model, request) if stream else model.complete(request))
    assert log == []


# --- complete: stop reasons, blocks, usage -----------------------------------------------------


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
    ],
)
def test_stop_reasons_complete_and_stream(stop, finish) -> None:
    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=_sse(*_stream_events(stop=stop)))
        return httpx.Response(200, json=_message(stop=stop))

    model = _model(handler)
    assert run(model.complete(_req())).finish_reason == finish


@pytest.mark.parametrize("stop", ["refusal"])
def test_refusal_is_typed_error(stop) -> None:
    model = _model(lambda r: httpx.Response(200, json=_message(stop=stop)))
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "refused"
    model = _model(lambda r: httpx.Response(200, content=_sse(*_stream_events(stop=stop))))
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()))
    assert ei.value.reason == "refused"


@pytest.mark.parametrize("stop", ["tool_use", "pause_turn", None, "mystery"])
def test_unexpected_stop_reasons_are_bad_response(stop) -> None:
    model = _model(lambda r: httpx.Response(200, json=_message(stop=stop)))
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "bad_response"
    model = _model(lambda r: httpx.Response(200, content=_sse(*_stream_events(stop=stop))))
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()))
    assert ei.value.reason == "bad_response"


def test_thinking_blocks_skipped_and_tool_use_rejected() -> None:
    content = [
        {"type": "thinking", "thinking": "SECRET-THOUGHT", "signature": "sig"},
        {"type": "redacted_thinking", "data": "xx"},
        {"type": "text", "text": "Hel"},
        {"type": "text", "text": "lo"},
    ]
    usage = _usage(output_tokens=500, output_tokens_details={"thinking_tokens": 480})
    result = run(_model(lambda r: httpx.Response(200, json=_message(content=content, usage=usage))).complete(_req()))
    assert result.text == "Hello" and "SECRET-THOUGHT" not in result.text
    assert result.usage.output_tokens == 500  # thinking tokens stay inside the billed output count
    bad = _model(
        lambda r: httpx.Response(
            200, json=_message(content=[{"type": "tool_use", "id": "t", "name": "n", "input": {}}])
        )
    )
    with pytest.raises(ProviderCallError) as ei:
        run(bad.complete(_req()))
    assert ei.value.reason == "bad_response"


def test_usage_provider_source_no_cache_exact_cost() -> None:
    usage = run(
        _model(
            lambda r: httpx.Response(200, json=_message(usage=_usage(input_tokens=1000, output_tokens=200)))
        ).complete(_req())
    ).usage
    assert usage.token_source is TokenSource.PROVIDER and usage.operation is Operation.CHAT
    assert (usage.input_tokens, usage.output_tokens) == (1000, 200)
    assert (usage.cache_read_input_tokens, usage.cache_write_input_tokens) == (0, 0)
    assert usage.estimated_cost_usd == Decimal("0.0002")


def _cached(profile, **u):
    full = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 2000,
        "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 0},
    }
    full.update(u)
    return run(_model(lambda r: httpx.Response(200, json=_message(usage=full)), profile).complete(_req())).usage


def test_cache_tokens_and_exact_cost() -> None:
    h = _cached(anthropic_haiku_profile())
    assert (h.input_tokens, h.cache_read_input_tokens, h.cache_write_input_tokens) == (1000, 5000, 2000)
    # 1000*0.10 + 200*0.50 + 5000*0.01 + 2000*0.125 = 100 + 100 + 50 + 250 = 500 per 1e6
    assert h.estimated_cost_usd == Decimal("0.0005")
    # Sonnet: 1000*2 + 200*10 + 5000*0.10 + 2000*2.50 = 2000 + 2000 + 500 + 5000 = 9500 per 1e6
    assert _cached(anthropic_sonnet_profile()).estimated_cost_usd == Decimal("0.0095")


def test_haiku_long_prompt_tier_cost_is_none_never_cheap_rate() -> None:
    edge = _cached(anthropic_haiku_profile(), input_tokens=90_000, cache_read_input_tokens=8_000)  # 100,000 total
    assert edge.estimated_cost_usd is not None
    over = _cached(anthropic_haiku_profile(), input_tokens=90_000, cache_read_input_tokens=8_001)
    assert over.estimated_cost_usd is None
    only_in = _cached(
        anthropic_haiku_profile(),
        input_tokens=100_001,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation={},
    )
    assert only_in.estimated_cost_usd is None
    # Sonnet has no long-context premium.
    assert _cached(anthropic_sonnet_profile(), input_tokens=500_000).estimated_cost_usd is not None


def test_one_hour_cache_write_cost_is_none() -> None:
    u = _cached(
        anthropic_sonnet_profile(),
        cache_creation={"ephemeral_5m_input_tokens": 1500, "ephemeral_1h_input_tokens": 500},
    )
    assert u.cache_write_input_tokens == 2000 and u.estimated_cost_usd is None


def test_unknown_model_cost_none() -> None:
    profile = anthropic_chat_profile("anthropic-unknown", "claude-unknown-1")
    u = run(_model(lambda r: httpx.Response(200, json=_message()), profile).complete(_req())).usage
    assert u.token_source is TokenSource.PROVIDER and u.estimated_cost_usd is None
    body = _sent(profile, _req())
    assert "thinking" not in body and body["max_tokens"] == 1024


@pytest.mark.parametrize(
    "usage",
    [
        {"output_tokens": 5},
        {"input_tokens": 5},
        {"input_tokens": None, "output_tokens": 5},
        {"input_tokens": "5", "output_tokens": 5},
        {"input_tokens": -1, "output_tokens": 5},
        {"input_tokens": True, "output_tokens": 5},
        {"input_tokens": 5, "output_tokens": 5, "cache_read_input_tokens": -2},
        {
            "input_tokens": 5,
            "output_tokens": 5,
            "cache_creation": {"ephemeral_1h_input_tokens": 9},
            "cache_creation_input_tokens": 3,
        },
    ],
)
def test_missing_or_invalid_usage_is_bad_response_never_estimated(usage) -> None:
    model = _model(lambda r: httpx.Response(200, json=_message(usage=usage)))
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "bad_response"


def test_usage_missing_entirely_and_bad_shapes() -> None:
    for body in (
        {**_message(), "usage": None},
        {**_message(), "content": "text"},
        {k: v for k, v in _message().items() if k != "usage"},
    ):
        with pytest.raises(ProviderCallError) as ei:
            run(_model(lambda r, b=body: httpx.Response(200, json=b)).complete(_req()))
        assert ei.value.reason == "bad_response"


def test_http_status_errors_map() -> None:
    body = {"type": "error", "error": {"type": "overloaded_error", "message": CANARY}, "request_id": "req_9"}
    with pytest.raises(ProviderCallError) as ei:
        run(_model(lambda r: httpx.Response(529, json=body)).complete(_req()))
    assert ei.value.reason == "overloaded" and CANARY not in str(ei.value)
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(401, json=body)), _req()))
    assert ei.value.reason == "auth" and ei.value.status == 401 and CANARY not in str(ei.value)


# --- streaming ---------------------------------------------------------------------------------


def test_stream_text_and_usage_merge() -> None:
    start = _usage(
        input_tokens=1000,
        output_tokens=1,
        cache_read_input_tokens=5000,
        cache_creation_input_tokens=2000,
        cache_creation={"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 0},
    )
    events = _stream_events(
        start_usage=start,
        delta_usage={"output_tokens": 200, "input_tokens": None},  # cumulative output; null input keeps start
    )
    chunks = run(_drain(_model(lambda r: httpx.Response(200, content=_sse(*events))), _req()))
    assert [c.delta for c in chunks] == ["Hello", " world", ""]
    u = chunks[-1].usage
    assert u is not None and u.operation is Operation.CHAT_STREAM and u.token_source is TokenSource.PROVIDER
    assert (u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_write_input_tokens) == (
        1000,
        200,
        5000,
        2000,
    )
    assert u.estimated_cost_usd == Decimal("0.0005")
    assert all(c.usage is None for c in chunks[:-1])


def test_stream_ignores_thinking_ping_unknown_and_handles_comments() -> None:
    events = _stream_events(texts=("ok",))
    thinking = [
        (
            "content_block_start",
            {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": "HIDDEN"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "s"}},
        ),
        ("future_event", {"type": "future_event", "x": 1}),
    ]
    events = events[:1] + thinking + events[1:]
    raw = b": keepalive comment\n\n" + _sse(*events)
    chunks = run(_drain(_model(lambda r: httpx.Response(200, content=raw)), _req()))
    assert "".join(c.delta for c in chunks) == "ok"
    assert "HIDDEN" not in "".join(c.delta for c in chunks)


def test_cache_write_without_provable_5m_split_has_no_cost() -> None:
    for split in (None, {}, {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0}):
        u = _cached(anthropic_haiku_profile(), cache_creation=split)
        assert u.cache_write_input_tokens == 2000 and u.estimated_cost_usd is None, split
    # only the 1h count is given and it equals the total: still None (1h rate unknown)
    only_1h = _cached(anthropic_haiku_profile(), cache_creation={"ephemeral_1h_input_tokens": 2000})
    assert only_1h.estimated_cost_usd is None
    # no writes at all needs no split
    none = _cached(anthropic_haiku_profile(), cache_creation_input_tokens=0, cache_creation=None)
    assert none.estimated_cost_usd == Decimal("0.00025")


def test_total_missing_split_sums_to_total() -> None:
    raw = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 0},
    }
    u = run(_model(lambda r: httpx.Response(200, json=_message(usage=raw))).complete(_req())).usage
    assert u.cache_write_input_tokens == 2000 and u.estimated_cost_usd == Decimal("0.00045")


def test_stream_final_event_without_trailing_blank_line() -> None:
    raw = _sse(*_stream_events()).rstrip(b"\n")
    assert not raw.endswith(b"\n\n")
    chunks = run(_drain(_model(lambda r: httpx.Response(200, content=raw)), _req()))
    assert "".join(c.delta for c in chunks) == "Hello world" and chunks[-1].usage is not None
    # an unterminated event that is not message_stop still ends as truncated
    cut = _sse(*_stream_events()[:-1]).rstrip(b"\n")
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=cut)), _req()))
    assert ei.value.reason == "bad_response"


_BILLED = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 5000}


@pytest.mark.parametrize(
    ("stop", "reason"),
    [("refusal", "refused"), ("pause_turn", "bad_response"), ("tool_use", "bad_response"), (None, "bad_response")],
)
@pytest.mark.parametrize("stream", [False, True])
def test_billed_failure_is_recorded_once_with_exact_cost(stop, reason, stream) -> None:
    usage = {**_BILLED, "cache_creation_input_tokens": 2000}
    usage["cache_creation"] = {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 0}

    def handler(request):
        if json.loads(request.content).get("stream"):
            events = _stream_events(
                stop=stop, start_usage={**usage, "output_tokens": 1}, delta_usage={"output_tokens": 200}
            )
            return httpx.Response(200, content=_sse(*events))
        return httpx.Response(200, json=_message(stop=stop, text=CANARY, usage=usage))

    ledger = UsageLedger()
    model = RecordingChatModel(_model(handler), ledger)
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()) if stream else model.complete(_req()))
    err = ei.value
    assert err.reason == reason and err.usage is not None
    records = ledger.records()
    assert len(records) == 1 and records[0] == err.usage
    assert records[0].estimated_cost_usd == Decimal("0.0005")
    assert records[0].operation is (Operation.CHAT_STREAM if stream else Operation.CHAT)
    totals = ledger.totals()
    assert totals.calls == 1 and totals.cost_usd == Decimal("0.0005")
    for text in (str(err), repr(err), repr(err.args), "".join(traceback.format_exception(err))):
        assert SENTINEL not in text and CANARY not in text and "estimated_cost" not in text


def test_failure_without_valid_usage_records_nothing() -> None:
    ledger = UsageLedger()
    model = RecordingChatModel(_model(lambda r: httpx.Response(200, json=_message(stop="refusal", usage={}))), ledger)
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "refused" and ei.value.usage is None and ledger.records() == ()


def test_http_error_has_no_usage_and_records_nothing() -> None:
    ledger = UsageLedger()
    model = RecordingChatModel(_model(lambda r: httpx.Response(529, json={})), ledger)
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.usage is None and ledger.records() == ()


def test_stream_crlf_lines_and_no_space_after_colon() -> None:
    raw = _sse(*_stream_events()).replace(b"\n", b"\r\n").replace(b"data: ", b"data:")
    chunks = run(_drain(_model(lambda r: httpx.Response(200, content=raw)), _req()))
    assert "".join(c.delta for c in chunks) == "Hello world"


def test_stream_error_event_mid_stream() -> None:
    events = _stream_events()[:5]
    events.append(("error", {"type": "error", "error": {"type": "overloaded_error", "message": CANARY}}))
    model = _model(lambda r: httpx.Response(200, content=_sse(*events)))
    got: list[str] = []

    async def go():
        async for c in model.stream(_req()):
            got.append(c.delta)

    with pytest.raises(ProviderCallError) as ei:
        run(go())
    assert ei.value.reason == "overloaded" and CANARY not in str(ei.value) and got == ["Hello", " world"][: len(got)]
    unknown = _stream_events()[:3] + [("error", {"type": "error", "error": {"type": "novel_error", "message": CANARY}})]
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=_sse(*unknown))), _req()))
    assert ei.value.reason == "bad_response"


def test_stream_without_message_stop_and_invalid_json_and_missing_usage() -> None:
    truncated = _sse(*_stream_events()[:-1])
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=truncated)), _req()))
    assert ei.value.reason == "bad_response"
    bad_json = b"event: message_start\ndata: {not json\n\n"
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=bad_json)), _req()))
    assert ei.value.reason == "bad_response"
    events = _stream_events()
    events[0] = (
        "message_start",
        {"type": "message_start", "message": _message(content=[], stop=None, usage={"output_tokens": 1})},
    )
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=_sse(*events))), _req()))
    assert ei.value.reason == "bad_response"


def test_stream_tool_use_block_is_bad_response() -> None:
    events = _stream_events()
    events[1] = (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}},
        },
    )
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=_sse(*events))), _req()))
    assert ei.value.reason == "bad_response"


class _Body(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    async def __aiter__(self):
        for line in self._data.splitlines(keepends=True):
            yield line

    async def aclose(self) -> None:
        self.closed = True


def test_early_break_closes_response() -> None:
    body = _Body(_sse(*_stream_events()))
    model = _model(lambda r: httpx.Response(200, stream=body))

    async def go():
        async with aclosing(model.stream(_req())) as it:
            async for chunk in it:
                assert chunk.delta == "Hello"
                break

    run(go())
    assert body.closed is True


def test_full_stream_and_error_also_close_response() -> None:
    body = _Body(_sse(*_stream_events()))
    run(_drain(_model(lambda r: httpx.Response(200, stream=body)), _req()))
    assert body.closed is True
    bad = _Body(_sse(*_stream_events()[:-1]))
    with pytest.raises(ProviderCallError):
        run(_drain(_model(lambda r: httpx.Response(200, stream=bad)), _req()))
    assert bad.closed is True


def test_connection_error_maps() -> None:
    def boom(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(boom), _req()))
    assert ei.value.reason == "connection"
    with pytest.raises(ProviderCallError) as ei:
        run(_model(boom).complete(_req()))
    assert ei.value.reason == "connection"


# --- leak test (extends P2.3-03) ---------------------------------------------------------------


def test_leak_check_adapter_ledger_and_repr(caplog) -> None:
    seen: list[httpx.Request] = []
    err_body = {"type": "error", "error": {"type": "authentication_error", "message": CANARY}, "request_id": "req_1"}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if body["messages"][0]["content"] == "fail":
            return httpx.Response(401, json=err_body)
        if body.get("stream"):
            return httpx.Response(200, content=_sse(*_stream_events()))
        return httpx.Response(200, json=_message())

    inner = _model(handler)
    ledger = UsageLedger()
    model = RecordingChatModel(inner, ledger)
    places: list[str] = [repr(inner), str(inner), repr(inner.profile), inner.profile.model_dump_json()]
    errors: list[BaseException] = []
    with caplog.at_level(logging.DEBUG):
        for name in ("httpx", "httpcore", "h11"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        result = run(model.complete(_req()))
        chunks = run(_drain(model, _req()))
        for request in (_req("fail"),):
            for call in (model.complete(request), _drain(model, request)):
                try:
                    run(call)
                except ProviderCallError as exc:
                    errors.append(exc)
    assert len(errors) == 2 and len(ledger.records()) == 2
    assert seen[0].headers["authorization"] == f"Bearer {SENTINEL}"  # the key was really sent, so the test bites
    places += [result.model_dump_json(), result.usage.model_dump_json(), repr(result.usage)]
    places += [c.model_dump_json() for c in chunks]
    places += [ledger.totals().model_dump_json(), repr(ledger.totals()), repr(ledger.records())]
    places += [u.model_dump_json() for u in ledger.records()]
    places += [repr(model), repr(model.profile)]
    for exc in errors:
        places += [str(exc), repr(exc), repr(exc.args), "".join(traceback.format_exception(exc))]
        cause = exc.__cause__ or exc.__context__
        while cause is not None:
            places += [str(cause), repr(cause)]
            cause = cause.__cause__ or cause.__context__
    places += [r.getMessage() for r in caplog.records] + [caplog.text]
    for text in places:
        assert SENTINEL not in text
        assert CANARY not in text
    for request in seen:  # key only in the Authorization header
        assert SENTINEL.encode() not in request.content and SENTINEL not in str(request.url)
        assert SENTINEL not in json.dumps({k: v for k, v in request.headers.items() if k != "authorization"})
