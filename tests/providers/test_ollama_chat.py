"""OllamaChatModel against httpx.MockTransport with api.md-shaped JSON and NDJSON (no network)."""

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest
from rag_foundry_providers import TokenSource
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile
from rag_foundry_providers.contract_checks import CHAT_CHECKS, DETERMINISTIC_CHAT_CHECKS
from rag_foundry_providers.errors import MissingCapabilityError, ProviderCallError, ProviderError
from rag_foundry_providers.messages import ChatMessage, ChatRequest, ToolSpec
from rag_foundry_providers.usage import Operation

ARABIC = "مرحبا بالعالم"


def _req(text: str = "hello", **kw) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), **kw)


def _model(handler, **kw) -> OllamaChatModel:
    return OllamaChatModel(
        ollama_chat_profile(), api_base="http://localhost:11434", transport=httpx.MockTransport(handler), **kw
    )


def _final(extra: dict | None = None, content: str = "") -> dict:
    body = {
        "model": "gemma4:e4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 26,
        "eval_count": 3,
    }
    body.update(extra or {})
    return body


def _nd(*objs: dict) -> bytes:
    return ("\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n").encode()


def _chunk(text: str) -> dict:
    return {"model": "gemma4:e4b", "message": {"role": "assistant", "content": text}, "done": False}


def _ok(handler_log: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if handler_log is not None:
            handler_log.append(request)
        body = json.loads(request.content)
        if body["stream"]:
            return httpx.Response(200, content=_nd(_chunk("Hello"), _chunk(" world"), _final()))
        return httpx.Response(200, json=_final(content="Hello world"))

    return handler


@pytest.mark.parametrize(
    "check", CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS, ids=[c.__name__ for c in CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS]
)
def test_chat_contract(check) -> None:
    asyncio.run(check(_model(_ok())))


def test_profile_defaults() -> None:
    caps = ollama_chat_profile().capabilities
    assert caps.context_window == 8192 and caps.streaming is True and caps.local is True
    assert (caps.tool_calling, caps.structured_output, caps.vision) == (False, False, False)


@pytest.mark.parametrize("stream", [False, True])
def test_request_body(stream: bool) -> None:
    seen: list[httpx.Request] = []
    model = _model(_ok(seen), timeout_s=12.5)
    if stream:

        async def go() -> None:
            async for _ in model.stream(_req()):
                pass

        asyncio.run(go())
    else:
        asyncio.run(model.complete(_req()))
    request = seen[0]
    assert request.method == "POST" and request.url.path == "/api/chat"
    assert json.loads(request.content) == {
        "model": "gemma4:e4b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "think": False,
        "options": {"num_ctx": 8192},
    }
    assert request.extensions["timeout"] == {"connect": 12.5, "read": 12.5, "write": 12.5, "pool": 12.5}


def test_options_only_when_set() -> None:
    seen: list[httpx.Request] = []
    asyncio.run(_model(_ok(seen)).complete(_req(max_output_tokens=50, temperature=0.0)))
    body = json.loads(seen[0].content)
    assert body["options"] == {"num_ctx": 8192, "num_predict": 50, "temperature": 0.0}
    assert "tools" not in body and "format" not in body


def test_usage_from_provider() -> None:
    result = asyncio.run(_model(_ok()).complete(_req()))
    u = result.usage
    assert u.token_source is TokenSource.PROVIDER and (u.input_tokens, u.output_tokens) == (26, 3)
    assert u.operation is Operation.CHAT and u.estimated_cost_usd == 0 and result.finish_reason == "stop"


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt_eval_count": None},
        {"eval_count": None},
        {"prompt_eval_count": -1},
        {"eval_count": True},
        {"eval_count": 1.5},
        {"prompt_eval_count": "26"},
    ],
    ids=["no-prompt", "no-eval", "negative", "bool", "float", "string"],
)
def test_usage_estimated_when_a_count_is_missing_or_invalid(extra) -> None:
    body = _final(extra, content="Hello world")
    for key, value in extra.items():
        if value is None:
            del body[key]
    usage = asyncio.run(_model(lambda r: httpx.Response(200, json=body)).complete(_req(ARABIC))).usage
    assert usage.token_source is TokenSource.ESTIMATED
    assert usage.input_tokens > 0 and usage.output_tokens > 0


def test_stream_usage_and_operation() -> None:
    async def go():
        return [c async for c in _model(_ok()).stream(_req())]

    chunks = asyncio.run(go())
    assert [c.delta for c in chunks] == ["Hello", " world", ""]
    assert chunks[-1].usage is not None and chunks[-1].usage.operation is Operation.CHAT_STREAM
    assert chunks[-1].usage.token_source is TokenSource.PROVIDER
    assert all(c.usage is None for c in chunks[:-1])


def test_stream_usage_estimated_without_counts() -> None:
    final = _final()
    del final["prompt_eval_count"]
    content = _nd(_chunk("abcdefgh"), final)

    async def go():
        return [c async for c in _model(lambda r: httpx.Response(200, content=content)).stream(_req(ARABIC))]

    usage = asyncio.run(go())[-1].usage
    assert usage.token_source is TokenSource.ESTIMATED and usage.input_tokens > 0 and usage.output_tokens == 2


def test_length_finish_reason() -> None:
    body = _final({"done_reason": "length"}, content="cut")
    result = asyncio.run(_model(lambda r: httpx.Response(200, json=body)).complete(_req()))
    assert result.finish_reason == "length"


@pytest.mark.parametrize("reason", [None, "load", "unload", "weird"])
def test_bad_done_reason(reason) -> None:
    body = _final({"done_reason": reason}, content="x")
    if reason is None:
        del body["done_reason"]
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(lambda r: httpx.Response(200, json=body)).complete(_req()))
    assert exc.value.reason == "bad_response"

    async def go():
        return [c async for c in _model(lambda r: httpx.Response(200, content=_nd(_chunk("x"), body))).stream(_req())]

    with pytest.raises(ProviderCallError) as exc2:
        asyncio.run(go())
    assert exc2.value.reason == "bad_response"


def _stream_error(content: bytes) -> ProviderCallError:
    async def go():
        return [c async for c in _model(lambda r: httpx.Response(200, content=content)).stream(_req())]

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(go())
    return exc.value


def test_error_line_mid_stream() -> None:
    err = _stream_error(_nd(_chunk("a"), {"error": "secret text"}))
    assert err.reason == "bad_response" and "secret" not in str(err)


def test_invalid_json_line() -> None:
    assert _stream_error(_nd(_chunk("a")) + b"{not json\n").reason == "bad_response"


def test_stream_ends_without_done() -> None:
    assert _stream_error(_nd(_chunk("a"), _chunk("b"))).reason == "bad_response"


def test_non_object_line() -> None:
    assert _stream_error(b"[1, 2]\n").reason == "bad_response"


def test_complete_error_and_status() -> None:
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(lambda r: httpx.Response(200, json={"error": "x"})).complete(_req()))
    assert exc.value.reason == "bad_response"
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(lambda r: httpx.Response(404, json={"error": "nope"})).complete(_req()))
    assert exc.value.reason == "model_not_found"


def test_stream_status_codes() -> None:
    async def go(status: int):
        return [c async for c in _model(lambda r: httpx.Response(status, json={"error": "x"})).stream(_req())]

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(go(404))
    assert exc.value.reason == "model_not_found" and exc.value.status == 404
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(go(500))
    assert exc.value.reason == "bad_response" and exc.value.status == 500


def test_stream_transport_error_maps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def go():
        return [c async for c in _model(handler).stream(_req())]

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(go())
    assert exc.value.reason == "connection"


class _ClosableStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield _nd(_chunk("one"))
        yield _nd(_chunk("two"))
        yield _nd(_final())

    async def aclose(self) -> None:
        self.closed = True


def test_early_break_closes_response() -> None:
    stream = _ClosableStream()

    async def go() -> list[str]:
        model = _model(lambda r: httpx.Response(200, stream=stream))
        got: list[str] = []
        async with aclosing(model.stream(_req())) as gen:
            async for chunk in gen:
                got.append(chunk.delta)
                break
        return got

    assert asyncio.run(go()) == ["one"]
    assert stream.closed


def test_full_stream_closes_response() -> None:
    stream = _ClosableStream()

    async def go() -> None:
        async for _ in _model(lambda r: httpx.Response(200, stream=stream)).stream(_req()):
            pass

    asyncio.run(go())
    assert stream.closed


def test_constructor_refuses_unsupported_capabilities() -> None:
    base = ollama_chat_profile()
    for field in ("tool_calling", "structured_output", "vision"):
        caps = base.capabilities.model_copy(update={field: True})
        profile = base.model_copy(update={"capabilities": caps})
        with pytest.raises(ProviderError, match=field):
            OllamaChatModel(profile, api_base="http://localhost:11434")


def test_tools_and_schema_refused_without_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    tool = ToolSpec(name="lookup", description="d", parameters_schema={"type": "object"})
    with pytest.raises(MissingCapabilityError):
        asyncio.run(_model(handler).complete(_req(tools=(tool,))))
    with pytest.raises(MissingCapabilityError):
        asyncio.run(_model(handler).complete(_req(response_schema={"type": "object"})))


def test_stream_with_malformed_api_base_maps_to_bad_response():
    model = OllamaChatModel(
        ollama_chat_profile(), api_base="http://[::1", transport=httpx.MockTransport(lambda r: None)
    )

    async def consume():
        async for _ in model.stream(_req()):
            pass

    with pytest.raises(ProviderCallError) as info:
        asyncio.run(consume())
    assert info.value.reason == "bad_response"
    assert "[::1" not in str(info.value)
