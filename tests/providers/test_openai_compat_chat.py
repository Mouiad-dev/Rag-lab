"""OpenAI-compatible chat adapter against httpx.MockTransport (no network, fake keys only)."""

import asyncio
import json
import logging
import traceback
import uuid
from contextlib import aclosing
from decimal import Decimal

import httpx
import pytest
from rag_foundry_providers import Operation, TokenSource
from rag_foundry_providers.adapters._ollama_http import estimate_tokens
from rag_foundry_providers.adapters.anthropic import UnsupportedRequestError
from rag_foundry_providers.adapters.openai_compat import OpenAICompatibleChatModel
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.contract_checks import CHAT_CHECKS, DETERMINISTIC_CHAT_CHECKS
from rag_foundry_providers.errors import (
    MissingCapabilityError,
    MissingCredentialError,
    ProviderCallError,
    ProviderError,
)
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatMessage, ChatRequest, ToolSpec

KEY_ENV = "MY_CUSTOM_CHAT_KEY"
SENTINEL = f"sk-TEST-SENTINEL-{uuid.uuid4()}"
CANARY = f"CANARY-MESSAGE-{uuid.uuid4()}"
LOCAL = "http://localhost:1234/v1"
REMOTE = "https://api.example.test/v1"
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
ALL_CHECKS = (*CHAT_CHECKS, *DETERMINISTIC_CHAT_CHECKS)


def run(coro):
    return asyncio.run(coro)


def _profile(*, local: bool = True, streaming: bool | None = True, **caps) -> ModelProfile:
    return ModelProfile(
        id="oa-chat",
        provider="openai_compatible",
        model="chat-model",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(
            local=local, streaming=streaming, tool_calling=False, structured_output=False, vision=False, **caps
        ),
        cost_per_mtok_in=Decimal("1"),
        cost_per_mtok_out=Decimal("2"),
    )


def _model(
    handler, *, base: str = LOCAL, env: str | None = None, local: bool = True, **kw
) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        kw.pop("profile", None) or _profile(local=local),
        api_base=base,
        api_key_env=env,
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _paid(handler, **kw) -> OpenAICompatibleChatModel:
    return _model(handler, base=REMOTE, env=KEY_ENV, local=False, **kw)


def _req(text: str = "hello world", **kw) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), **kw)


async def _drain(model, request):
    return [c async for c in model.stream(request)]


def _completion(
    text: str | None = "Hello world",
    finish: str | None = "stop",
    usage: dict | None | bool = True,
    message: dict | None = None,
) -> dict:
    body: dict = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message if message is not None else {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
    }
    if usage is True:
        body["usage"] = dict(USAGE)
    elif usage:
        body["usage"] = usage
    return body


def _chunk(content=None, finish=None, delta: dict | None = None) -> dict:
    d = delta if delta is not None else ({} if content is None else {"content": content})
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
        "usage": None,
    }


def _events(finish: str | None = "stop", usage: dict | None | bool = True) -> list[dict]:
    events = [
        _chunk(delta={"role": "assistant", "content": ""}),
        _chunk("Hello"),
        _chunk(" world"),
        _chunk(finish=finish),
    ]
    if usage is True:
        events.append({"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": dict(USAGE)})
    elif usage:
        events.append({"choices": [], "usage": usage})
    return events


def _sse(events, *, done: bool = True) -> bytes:
    text = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


def _ok(*, usage: dict | None | bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=_sse(_events(usage=usage)))
        return httpx.Response(200, json=_completion(usage=usage))

    return handler


class _Body(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    async def __aiter__(self):
        for line in self._data.splitlines(keepends=True):
            yield line

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SENTINEL)


# --- contract ----------------------------------------------------------------------------------


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_contract_local_keyless(check) -> None:
    run(check(_model(_ok())))


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_contract_paid_keyed(check) -> None:
    run(check(_paid(_ok())))


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_contract_local_without_usage(check) -> None:
    run(check(_model(_ok(usage=False))))


# --- construction and refusals before any call -------------------------------------------------


@pytest.mark.parametrize("cap", ["tool_calling", "structured_output", "vision"])
def test_declared_unsupported_capability_refused_at_construction(cap) -> None:
    profile = _profile()
    caps = profile.capabilities.model_copy(update={cap: True})
    with pytest.raises(ProviderError):
        OpenAICompatibleChatModel(profile.model_copy(update={"capabilities": caps}), api_base=LOCAL)


def test_construction_validation(monkeypatch) -> None:
    emb = ModelProfile(
        id="e",
        provider="openai_compatible",
        model="m",
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(embedding_dim=4),
    )
    with pytest.raises(ProviderError):
        OpenAICompatibleChatModel(emb, api_base=LOCAL)
    with pytest.raises(ValueError):
        OpenAICompatibleChatModel(_profile(), api_base=LOCAL, default_max_tokens=0)
    with pytest.raises(ValueError):
        OpenAICompatibleChatModel(_profile(), api_base=LOCAL, max_tokens_field="max_new_tokens")  # type: ignore[arg-type]
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(MissingCredentialError) as ei:
        OpenAICompatibleChatModel(_profile(), api_base=LOCAL, api_key_env=KEY_ENV)
    assert KEY_ENV in str(ei.value)


_TOOL = ToolSpec(name="lookup", description="d", parameters_schema={"type": "object"})


@pytest.mark.parametrize("stream", [False, True])
def test_refusals_happen_before_any_call(stream) -> None:
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return _ok()(request)

    model = _model(handler, supports_temperature=False)
    tool_msg = ChatRequest(messages=(ChatMessage(role="tool", content="x"),))
    cases = [
        (MissingCapabilityError, _req(tools=(_TOOL,))),
        (MissingCapabilityError, _req(response_schema={"type": "object"})),
        (UnsupportedRequestError, _req(temperature=0.2)),
        (ProviderError, tool_msg),
    ]
    for exc_type, request in cases:
        with pytest.raises(exc_type):
            run(_drain(model, request) if stream else model.complete(request))
    assert calls == []


def test_non_streaming_profile_refuses_stream_before_call() -> None:
    calls: list[httpx.Request] = []
    model = _model(lambda r: calls.append(r) or _ok()(r), profile=_profile(streaming=False))
    with pytest.raises(MissingCapabilityError):
        run(_drain(model, _req()))
    assert calls == []


def test_key_refused_over_remote_http_before_any_request() -> None:
    calls: list[httpx.Request] = []
    model = _model(lambda r: calls.append(r) or _ok()(r), base="http://api.example.test/v1", env=KEY_ENV, local=False)
    for call in (model.complete(_req()), _drain(model, _req())):
        with pytest.raises(ProviderCallError) as ei:
            run(call)
        assert ei.value.reason == "invalid_request" and SENTINEL not in str(ei.value) + repr(ei.value)
    assert calls == []


# --- payload -----------------------------------------------------------------------------------


def _capture(handler=None):
    seen: list[httpx.Request] = []

    def h(request):
        seen.append(request)
        return (handler or _ok())(request)

    return seen, h


def test_complete_payload_defaults_and_auth() -> None:
    seen, h = _capture()
    msgs = (
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="yo"),
        ChatMessage(role="user", content="again"),
    )
    run(_paid(h).complete(ChatRequest(messages=msgs)))
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/chat/completions"
    assert json.loads(req.content) == {
        "model": "chat-model",
        "messages": [{"role": m.role, "content": m.content} for m in msgs],
        "stream": False,
    }
    assert req.headers["authorization"] == f"Bearer {SENTINEL}"
    assert SENTINEL.encode() not in req.content


def test_keyless_local_sends_no_authorization() -> None:
    seen, h = _capture()
    run(_model(h).complete(_req()))
    assert "authorization" not in seen[0].headers


def test_max_tokens_field_name_is_per_profile() -> None:
    for kwargs, field in (({}, "max_completion_tokens"), ({"max_tokens_field": "max_tokens"}, "max_tokens")):
        seen, h = _capture()
        run(_model(h, **kwargs).complete(_req(max_output_tokens=33)))
        body = json.loads(seen[0].content)
        assert body[field] == 33
        assert ({"max_tokens", "max_completion_tokens"} - {field}).isdisjoint(body)
    seen, h = _capture()
    run(_model(h, default_max_tokens=77).complete(_req()))
    assert json.loads(seen[0].content)["max_completion_tokens"] == 77
    seen, h = _capture()
    run(_model(h, default_max_tokens=77).complete(_req(max_output_tokens=5)))
    assert json.loads(seen[0].content)["max_completion_tokens"] == 5
    seen, h = _capture()
    run(_model(h).complete(_req()))
    assert "max_tokens" not in json.loads(seen[0].content) and "max_completion_tokens" not in json.loads(
        seen[0].content
    )


def test_temperature_sent_only_when_set_and_supported() -> None:
    seen, h = _capture()
    run(_model(h).complete(_req(temperature=0.0)))
    run(_model(h).complete(_req()))
    assert json.loads(seen[0].content)["temperature"] == 0.0
    assert "temperature" not in json.loads(seen[1].content)
    # A profile that refuses temperature still serves requests that leave it unset.
    run(_model(h, supports_temperature=False).complete(_req()))
    assert "temperature" not in json.loads(seen[2].content)


def test_stream_payload_and_never_sends_tool_fields() -> None:
    seen, h = _capture()
    run(_drain(_model(h), _req()))
    run(_model(h).complete(_req()))
    stream_body, complete_body = (json.loads(r.content) for r in seen)
    assert stream_body["stream"] is True and stream_body["stream_options"] == {"include_usage": True}
    assert "stream_options" not in complete_body
    for body in (stream_body, complete_body):
        assert {"tools", "tool_choice", "response_format"}.isdisjoint(body)


# --- finish reasons, billed failures -----------------------------------------------------------


def test_finish_reasons_map_complete() -> None:
    for finish, expected in (("stop", "stop"), ("length", "length")):
        result = run(_model(lambda r, f=finish: httpx.Response(200, json=_completion(finish=f))).complete(_req()))
        assert result.finish_reason == expected and result.text == "Hello world"
    for finish in ("stop", "length"):
        chunks = run(_drain(_model(lambda r, f=finish: httpx.Response(200, content=_sse(_events(f)))), _req()))
        assert "".join(c.delta for c in chunks) == "Hello world" and chunks[-1].usage is not None


_FAILS = [
    ("content_filter", "refused"),
    ("tool_calls", "bad_response"),
    ("function_call", "bad_response"),
    (None, "bad_response"),
    ("weird", "bad_response"),
]


@pytest.mark.parametrize(("finish", "reason"), _FAILS)
@pytest.mark.parametrize("stream", [False, True])
def test_bad_finish_is_billed_failure_recorded_once(finish, reason, stream) -> None:
    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=_sse(_events(finish)))
        return httpx.Response(200, json=_completion(text=CANARY, finish=finish))

    ledger = UsageLedger()
    model = RecordingChatModel(_paid(handler), ledger)
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()) if stream else model.complete(_req()))
    err = ei.value
    assert err.reason == reason and err.usage is not None
    assert err.usage.token_source is TokenSource.PROVIDER
    records = ledger.records()
    assert len(records) == 1 and records[0] == err.usage
    assert records[0].estimated_cost_usd == Decimal("0.002")  # 1000*1 + 500*2 per MTok
    assert records[0].operation is (Operation.CHAT_STREAM if stream else Operation.CHAT)
    for text in (str(err), repr(err), repr(err.args), "".join(traceback.format_exception(err))):
        assert SENTINEL not in text and CANARY not in text and "estimated_cost" not in text


def test_tool_calls_in_message_or_delta_is_billed_bad_response() -> None:
    call = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    body = _completion(message={"role": "assistant", "content": None, "tool_calls": call})
    with pytest.raises(ProviderCallError) as ei:
        run(_model(lambda r: httpx.Response(200, json=body)).complete(_req()))
    assert ei.value.reason == "bad_response" and ei.value.usage is not None
    events = [_chunk(delta={"role": "assistant", "tool_calls": call}), _chunk(finish="stop"), *_events()[-1:]]
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_model(lambda r: httpx.Response(200, content=_sse(events))), _req()))
    assert ei.value.reason == "bad_response" and ei.value.usage is not None


def test_failure_without_valid_usage_records_nothing() -> None:
    ledger = UsageLedger()
    model = RecordingChatModel(
        _model(lambda r: httpx.Response(200, json=_completion(finish="content_filter", usage=False))), ledger
    )
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "refused" and ei.value.usage is None and ledger.records() == ()


def test_http_error_has_no_usage_and_records_nothing() -> None:
    ledger = UsageLedger()
    model = RecordingChatModel(_model(lambda r: httpx.Response(503, json={})), ledger)
    with pytest.raises(ProviderCallError) as ei:
        run(model.complete(_req()))
    assert ei.value.reason == "overloaded" and ei.value.usage is None and ledger.records() == ()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": "x"},
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop"}]},
        {"choices": [{"message": {"content": 5}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1}},
    ],
)
def test_invalid_complete_bodies_are_bad_response(body) -> None:
    with pytest.raises(ProviderCallError) as ei:
        run(_paid(lambda r: httpx.Response(200, json=body)).complete(_req()))
    assert ei.value.reason == "bad_response"


def test_reasoning_content_is_not_returned_as_text() -> None:
    message = {"role": "assistant", "content": "answer", "reasoning_content": CANARY}
    result = run(_model(lambda r: httpx.Response(200, json=_completion(message=message))).complete(_req()))
    assert result.text == "answer" and CANARY not in result.model_dump_json()
    events = [_chunk(delta={"reasoning_content": CANARY}), _chunk("answer"), _chunk(finish="stop"), *_events()[-1:]]
    chunks = run(_drain(_model(lambda r: httpx.Response(200, content=_sse(events))), _req()))
    assert "".join(c.delta for c in chunks) == "answer"


# --- usage -------------------------------------------------------------------------------------


def test_usage_provider_and_cost() -> None:
    usage = run(_paid(_ok()).complete(_req())).usage
    assert usage.token_source is TokenSource.PROVIDER and (usage.input_tokens, usage.output_tokens) == (1000, 500)
    assert usage.estimated_cost_usd == Decimal("0.002") and usage.operation is Operation.CHAT


def test_cached_tokens_make_cost_none() -> None:
    for extra in ({"cached_tokens": 4}, {"cache_write_tokens": 4}):
        usage_body = {**USAGE, "prompt_tokens_details": extra}
        for stream in (False, True):

            def handler(r, u=usage_body):
                if json.loads(r.content).get("stream"):
                    return httpx.Response(200, content=_sse(_events(usage=u)))
                return httpx.Response(200, json=_completion(usage=u))

            model = _paid(handler)
            usage = run(_drain(model, _req()))[-1].usage if stream else run(model.complete(_req())).usage
            assert usage.token_source is TokenSource.PROVIDER and usage.estimated_cost_usd is None
            assert usage.input_tokens == 1000
    zero = {**USAGE, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}
    assert run(
        _paid(lambda r: httpx.Response(200, json=_completion(usage=zero))).complete(_req())
    ).usage.estimated_cost_usd


def test_missing_usage_local_is_estimated() -> None:
    expected_in, expected_out = estimate_tokens("hello world"), estimate_tokens("Hello world")
    complete = run(_model(_ok(usage=False)).complete(_req())).usage
    streamed = run(_drain(_model(_ok(usage=False)), _req()))[-1].usage
    for usage in (complete, streamed):
        assert usage.token_source is TokenSource.ESTIMATED
        assert (usage.input_tokens, usage.output_tokens) == (expected_in, expected_out)
    assert streamed.operation is Operation.CHAT_STREAM


@pytest.mark.parametrize("stream", [False, True])
def test_missing_usage_paid_is_bad_response(stream) -> None:
    model = _paid(_ok(usage=False))
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()) if stream else model.complete(_req()))
    assert ei.value.reason == "bad_response" and ei.value.usage is None


@pytest.mark.parametrize("stream", [False, True])
def test_missing_usage_local_profile_on_remote_host_is_bad_response(stream) -> None:
    model = _model(_ok(usage=False), base="http://203.0.113.5:8000/v1", local=True)
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(model, _req()) if stream else model.complete(_req()))
    assert ei.value.reason == "bad_response"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 1},
        {"prompt_tokens": "7", "completion_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1},
        {"completion_tokens": 1},
        [1],
    ],
)
def test_invalid_usage_paid_is_bad_response(usage) -> None:
    for stream in (False, True):
        model = _paid(_ok(usage=usage))
        with pytest.raises(ProviderCallError) as ei:
            run(_drain(model, _req()) if stream else model.complete(_req()))
        assert ei.value.reason == "bad_response"


# --- streaming ---------------------------------------------------------------------------------


def test_stream_with_usage_chunk_yields_text_then_one_usage() -> None:
    chunks = run(_drain(_paid(_ok()), _req()))
    assert [c.delta for c in chunks] == ["Hello", " world", ""]
    assert [c.usage is not None for c in chunks] == [False, False, True]
    assert chunks[-1].usage.operation is Operation.CHAT_STREAM and chunks[-1].usage.estimated_cost_usd == Decimal(
        "0.002"
    )


def test_stream_tolerates_comments_blank_lines_crlf_and_no_space() -> None:
    raw = b": keep-alive\n\n" + _sse(_events()).replace(b"\n", b"\r\n").replace(b"data: ", b"data:")
    chunks = run(_drain(_paid(lambda r: httpx.Response(200, content=raw)), _req()))
    assert "".join(c.delta for c in chunks) == "Hello world"


def test_stream_error_object_mid_stream() -> None:
    events = [*_events()[:2], {"error": {"message": CANARY, "type": "server_error", "code": None}}, *_events()[2:]]
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_paid(lambda r: httpx.Response(200, content=_sse(events))), _req()))
    err = ei.value
    assert err.reason == "bad_response" and CANARY not in str(err) + repr(err) + "".join(
        traceback.format_exception(err)
    )


def test_stream_invalid_json_and_bad_shapes() -> None:
    raw = b'data: {"choices": [' + CANARY.encode() + b"\n\ndata: [DONE]\n\n"
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_paid(lambda r: httpx.Response(200, content=raw)), _req()))
    err = ei.value
    assert err.reason == "bad_response" and err.__context__ is None
    assert CANARY not in str(err) + repr(err) + "".join(traceback.format_exception(err))
    for bad in (
        '"x"',
        "[1]",
        '{"choices": "x"}',
        '{"choices": [1]}',
        '{"choices": [{"delta": 5}]}',
        '{"choices": [{"delta": {"content": 5}}]}',
    ):
        with pytest.raises(ProviderCallError) as ei:
            run(
                _drain(
                    _paid(lambda r, b=bad: httpx.Response(200, content=f"data: {b}\n\ndata: [DONE]\n\n".encode())),
                    _req(),
                )
            )
        assert ei.value.reason == "bad_response"


def test_stream_deeply_nested_json_is_bad_response() -> None:
    raw = b"data: " + b"[" * 200_000 + b"\n\ndata: [DONE]\n\n"
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_paid(lambda r: httpx.Response(200, content=raw)), _req()))
    assert ei.value.reason == "bad_response"


def test_stream_without_done_is_bad_response_and_closes() -> None:
    body = _Body(_sse(_events(), done=False))
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_paid(lambda r: httpx.Response(200, stream=body)), _req()))
    assert ei.value.reason == "bad_response" and body.closed is True


def test_stream_ends_at_done_and_ignores_later_lines() -> None:
    raw = _sse(_events()) + b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'
    chunks = run(_drain(_paid(lambda r: httpx.Response(200, content=raw)), _req()))
    assert "late" not in "".join(c.delta for c in chunks)


def test_early_break_closes_response() -> None:
    body = _Body(_sse(_events()))
    model = _paid(lambda r: httpx.Response(200, stream=body))

    async def go():
        async with aclosing(model.stream(_req())) as it:
            async for chunk in it:
                assert chunk.delta == "Hello"
                break

    run(go())
    assert body.closed is True


def test_full_stream_and_http_error_also_close_response() -> None:
    body = _Body(_sse(_events()))
    run(_drain(_paid(lambda r: httpx.Response(200, stream=body)), _req()))
    assert body.closed is True
    err_body = _Body(json.dumps({"error": {"message": CANARY}}).encode())
    with pytest.raises(ProviderCallError) as ei:
        run(_drain(_paid(lambda r: httpx.Response(429, stream=err_body, headers={"x-request-id": "req_1"})), _req()))
    assert ei.value.reason == "rate_limited" and ei.value.request_id == "req_1" and err_body.closed is True
    assert CANARY not in str(ei.value) + repr(ei.value)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (400, "invalid_request"),
        (401, "auth"),
        (404, "model_not_found"),
        (429, "rate_limited"),
        (500, "server_error"),
        (503, "overloaded"),
    ],
)
def test_status_mapping_both_paths(status, reason) -> None:
    model = _paid(lambda r: httpx.Response(status, json={"error": {"message": CANARY, "type": "t"}}))
    for call in (model.complete(_req()), _drain(model, _req())):
        with pytest.raises(ProviderCallError) as ei:
            run(call)
        assert ei.value.reason == reason and ei.value.status == status and CANARY not in str(ei.value)


def test_connection_and_timeout_errors_map() -> None:
    def boom(request):
        raise httpx.ConnectError("boom " + request.headers["authorization"], request=request)

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    for handler, reason in ((boom, "connection"), (slow, "timeout")):
        model = _paid(handler)
        for call in (model.complete(_req()), _drain(model, _req())):
            with pytest.raises(ProviderCallError) as ei:
                run(call)
            err = ei.value
            assert err.reason == reason and err.__cause__ is None and err.__context__ is None
            assert SENTINEL not in "".join(traceback.format_exception(err))


def test_missing_httpx_reports_extra(monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "httpx":
            raise ImportError("no httpx")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from rag_foundry_providers.errors import AdapterDependencyError

    model = _model(_ok())
    for call in (model.complete(_req()), _drain(model, _req())):
        with pytest.raises(AdapterDependencyError):
            run(call)


# --- leak test (extends P2.3-03 and P2.4-03) ---------------------------------------------------


def test_repr_shows_class_and_profile_only() -> None:
    model = _paid(_ok())
    assert repr(model) == "OpenAICompatibleChatModel(profile='oa-chat')"
    assert SENTINEL not in repr(model) and SENTINEL not in repr(vars(model))


def test_leak_check_adapter_ledger_and_repr(caplog) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        if text == "fail":
            err = {"error": {"message": CANARY, "type": "invalid_authentication_error", "code": None, "param": None}}
            return httpx.Response(401, json=err)
        if text == "filtered":
            if body.get("stream"):
                return httpx.Response(200, content=_sse(_events("content_filter")))
            return httpx.Response(200, json=_completion(text=CANARY, finish="content_filter"))
        if body.get("stream"):
            return httpx.Response(200, content=_sse(_events()))
        return httpx.Response(200, json=_completion())

    inner = _paid(handler)
    ledger = UsageLedger()
    model = RecordingChatModel(inner, ledger)
    places: list[str] = [repr(inner), str(inner), repr(inner.profile), inner.profile.model_dump_json()]
    errors: list[BaseException] = []
    with caplog.at_level(logging.DEBUG):
        for name in ("httpx", "httpcore", "h11"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        result = run(model.complete(_req()))
        chunks = run(_drain(model, _req()))
        for text in ("fail", "filtered"):
            for call in (model.complete(_req(text)), _drain(model, _req(text))):
                try:
                    run(call)
                except ProviderCallError as exc:
                    errors.append(exc)
    assert len(errors) == 4 and len(ledger.records()) == 4  # 2 successes + 2 billed refusals
    assert [e.reason for e in errors] == ["auth", "auth", "refused", "refused"]
    assert seen[0].headers["authorization"] == f"Bearer {SENTINEL}"  # the key was really sent, so the test bites
    places += [result.model_dump_json(), result.usage.model_dump_json(), repr(result.usage)]
    places += [c.model_dump_json() for c in chunks]
    places += [ledger.totals().model_dump_json(), repr(ledger.totals()), repr(ledger.records())]
    places += [u.model_dump_json() for u in ledger.records()]
    places += [repr(model), repr(model.profile)]
    for exc in errors:
        places += [str(exc), repr(exc), repr(exc.args), "".join(traceback.format_exception(exc))]
        assert isinstance(exc, ProviderCallError)
        if exc.usage is not None:
            places += [exc.usage.model_dump_json(), repr(exc.usage)]
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
