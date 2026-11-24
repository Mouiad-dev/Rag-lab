"""OpenAI-compatible embedding adapter against httpx.MockTransport (no network, fake keys only)."""

import asyncio
import json
import logging
import traceback
import uuid
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr
from rag_foundry_providers import TokenSource
from rag_foundry_providers.adapters import _anthropic_http
from rag_foundry_providers.adapters._keys import read_api_key
from rag_foundry_providers.adapters._openai_http import build_headers, openai_status_error
from rag_foundry_providers.adapters.openai_compat import OpenAICompatibleEmbeddingModel
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.contract_checks import EMBEDDING_CHECKS
from rag_foundry_providers.errors import AdapterDependencyError, MissingCredentialError, ProviderCallError

DIM = 8
KEY_ENV = "MY_CUSTOM_EMBED_KEY"
SENTINEL = f"sk-TEST-SENTINEL-{uuid.uuid4()}"
CANARY = f"CANARY-MESSAGE-{uuid.uuid4()}"
LOCAL = "http://localhost:1234/v1"
REMOTE = "https://api.example.test/v1"


def _profile(*, local: bool = True, cost: Decimal | None = Decimal("0.5")) -> ModelProfile:
    return ModelProfile(
        id="oa-embed",
        provider="openai_compatible",
        model="embed-model",
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(local=local, embedding_dim=DIM),
        cost_per_mtok_in=cost,
        cost_per_mtok_out=cost,
    )


def _model(handler, *, base: str = LOCAL, env: str | None = None, local: bool = True, **kw):
    return OpenAICompatibleEmbeddingModel(
        _profile(local=local),
        api_base=base,
        api_key_env=env,
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _vec(seed: float) -> list[float]:
    return [seed] * DIM


def _ok(usage: dict | None = None, *, reverse: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        data = [{"object": "embedding", "index": i, "embedding": _vec(0.1 * (i + 1))} for i in range(n)]
        if reverse:
            data.reverse()
        body = {"object": "list", "model": "embed-model", "data": data}
        body["usage"] = {"prompt_tokens": 7, "total_tokens": 7} if usage is None else usage
        return httpx.Response(200, json=body)

    return handler


def _run(model, texts=("a", "b", "c")):
    return asyncio.run(model.embed(list(texts)))


@pytest.fixture
def key_env(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SENTINEL)


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=[c.__name__ for c in EMBEDDING_CHECKS])
def test_embedding_contract(check) -> None:
    asyncio.run(check(_model(_ok())))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=[c.__name__ for c in EMBEDDING_CHECKS])
def test_embedding_contract_with_key(check, key_env) -> None:
    asyncio.run(check(_model(_ok(), env=KEY_ENV)))


def test_request_shape_and_auth(key_env) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()(request)

    _run(_model(handler, env=KEY_ENV), ("x", "y"))
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/embeddings"
    assert json.loads(req.content) == {"model": "embed-model", "input": ["x", "y"], "encoding_format": "float"}
    assert req.headers["authorization"] == f"Bearer {SENTINEL}"  # the key really was sent
    assert req.headers["content-type"] == "application/json"
    assert SENTINEL.encode() not in req.content and SENTINEL not in str(req.url)


def test_keyless_local_sends_no_authorization() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()(request)

    _run(_model(handler))
    assert "authorization" not in seen[0].headers


def test_index_reordering() -> None:
    result = _run(_model(_ok(reverse=True)))
    assert [v[0] for v in result.vectors] == pytest.approx([0.1, 0.2, 0.3])


def test_key_refused_over_remote_http_before_any_request(key_env) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok()(request)

    model = _model(handler, base="http://api.example.test/v1", env=KEY_ENV, local=False)
    with pytest.raises(ProviderCallError) as ei:
        _run(model)
    assert calls == []
    assert ei.value.reason == "invalid_request"
    assert SENTINEL not in str(ei.value) + repr(ei.value)


def test_key_allowed_over_https_and_loopback_http(key_env) -> None:
    assert _run(_model(_ok(), base=REMOTE, env=KEY_ENV, local=False)).usage.token_source is TokenSource.PROVIDER
    assert _run(_model(_ok(), base="http://127.0.0.1:8000/v1", env=KEY_ENV)).usage.token_source is TokenSource.PROVIDER


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (400, "invalid_request"),
        (401, "auth"),
        (402, "billing"),
        (403, "auth"),
        (404, "model_not_found"),
        (408, "timeout"),
        (413, "invalid_request"),
        (422, "invalid_request"),
        (429, "rate_limited"),
        (500, "server_error"),
        (502, "server_error"),
        (503, "overloaded"),
        (504, "timeout"),
        (418, "bad_response"),
    ],
)
def test_status_mapping_no_message_leak(status, reason) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"error": {"code": "x", "message": CANARY, "param": None, "type": "t"}}
        return httpx.Response(status, json=body, headers={"x-request-id": "req_abc123"})

    with pytest.raises(ProviderCallError) as ei:
        _run(_model(handler))
    err = ei.value
    assert err.reason == reason and err.status == status and err.request_id == "req_abc123"
    assert CANARY not in str(err) + repr(err) + repr(err.args)
    assert err.__cause__ is None and err.__context__ is None


def test_status_error_helpers() -> None:
    assert openai_status_error("p", 200) is None
    err = openai_status_error("p", 500, b"<html>" + CANARY.encode(), {"x-request-id": "evil id\n"})
    assert err is not None and err.request_id is None and err.reason == "server_error"
    assert CANARY not in str(err)


def test_usage_provider_cost() -> None:
    usage = _run(_model(_ok({"prompt_tokens": 1000, "total_tokens": 1000}))).usage
    assert usage.token_source is TokenSource.PROVIDER and usage.input_tokens == 1000
    assert usage.estimated_cost_usd == Decimal("0.0005")


def test_cached_tokens_make_cost_none() -> None:
    usage = _run(
        _model(_ok({"prompt_tokens": 10, "total_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}))
    ).usage
    assert usage.token_source is TokenSource.PROVIDER and usage.estimated_cost_usd is None
    zero = _run(_model(_ok({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}))).usage
    assert zero.estimated_cost_usd is not None


def _no_usage(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["input"])
    data = [{"object": "embedding", "index": i, "embedding": _vec(0.1)} for i in range(n)]
    return httpx.Response(200, json={"object": "list", "data": data})


def test_missing_usage_local_is_estimated() -> None:
    usage = _run(_model(_no_usage)).usage
    assert usage.token_source is TokenSource.ESTIMATED and usage.input_tokens == 3


def test_missing_usage_paid_is_bad_response(key_env) -> None:
    with pytest.raises(ProviderCallError) as ei:
        _run(_model(_no_usage, base=REMOTE, env=KEY_ENV, local=False))
    assert ei.value.reason == "bad_response"


def test_missing_usage_local_profile_on_remote_host_is_bad_response() -> None:
    with pytest.raises(ProviderCallError) as ei:
        _run(_model(_no_usage, base="http://203.0.113.5:8000/v1", local=True))
    assert ei.value.reason == "bad_response"


@pytest.mark.parametrize("usage", [{"prompt_tokens": -1}, {"prompt_tokens": "7"}, {"prompt_tokens": True}, [1], "x"])
def test_invalid_usage_paid_is_bad_response(usage, key_env) -> None:
    with pytest.raises(ProviderCallError) as ei:
        _run(_model(_ok(usage), base=REMOTE, env=KEY_ENV, local=False))
    assert ei.value.reason == "bad_response"


def _raw(content: bytes):
    return lambda request: httpx.Response(200, content=content)


def _emb(index=0, vec=None):
    return {"object": "embedding", "index": index, "embedding": _vec(0.1) if vec is None else vec}


@pytest.mark.parametrize(
    "body",
    [
        {"data": [_emb(0)], "usage": {"prompt_tokens": 1}},  # too few
        {"data": [_emb(0), _emb(1), _emb(2), _emb(3)], "usage": {"prompt_tokens": 1}},  # too many
        {"data": [_emb(0), _emb(0), _emb(1)], "usage": {"prompt_tokens": 1}},  # duplicate index
        {"data": [_emb(0), _emb(1), _emb(5)], "usage": {"prompt_tokens": 1}},  # index gap
        {"data": [_emb(0), _emb(1), _emb(True)], "usage": {"prompt_tokens": 1}},
        {"data": [_emb(0), _emb(1), _emb(2, [0.1] * (DIM - 1))], "usage": {"prompt_tokens": 1}},  # dims
        {"data": [_emb(0), _emb(1), _emb(2, ["a"] * DIM)], "usage": {"prompt_tokens": 1}},
        {"data": [_emb(0), _emb(1), _emb(2, [True] * DIM)], "usage": {"prompt_tokens": 1}},
        {"data": [_emb(0), _emb(1), "x"], "usage": {"prompt_tokens": 1}},
        {"data": "nope"},
        {},
    ],
)
def test_invalid_bodies_are_bad_response(body) -> None:
    with pytest.raises(ProviderCallError) as ei:
        _run(_model(lambda r: httpx.Response(200, json=body)))
    assert ei.value.reason == "bad_response"


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1" + "0" * 400])
def test_non_finite_values_rejected(token) -> None:
    vec = ",".join([token] + ["0.1"] * (DIM - 1))
    items = ",".join(f'{{"index":{i},"embedding":[{vec}]}}' for i in range(3))
    content = f'{{"data":[{items}],"usage":{{"prompt_tokens":1}}}}'.encode()
    with pytest.raises(ProviderCallError) as ei:
        _run(_model(_raw(content)))
    assert ei.value.reason == "bad_response" and ei.value.__context__ is None


def test_missing_credential_at_construction(monkeypatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(MissingCredentialError) as ei:
        _model(_ok(), env=KEY_ENV)
    assert KEY_ENV in str(ei.value)


def test_repr_shows_class_and_profile_only(key_env) -> None:
    model = _model(_ok(), env=KEY_ENV)
    assert repr(model) == "OpenAICompatibleEmbeddingModel(profile='oa-embed')"
    assert SENTINEL not in repr(model) and SENTINEL not in repr(vars(model))


def test_shared_key_module_keeps_old_import_path() -> None:
    assert _anthropic_http.read_api_key is read_api_key


@pytest.mark.parametrize("value", ["sk-éabc", "sk-a\nb", "sk-a\rb", "sk a", "sk-a\tb"])
def test_malformed_key_rejected_without_echo(monkeypatch, value) -> None:
    monkeypatch.setenv(KEY_ENV, value)
    with pytest.raises(MissingCredentialError) as ei:
        _model(_ok(), env=KEY_ENV)
    assert KEY_ENV in str(ei.value) and value not in str(ei.value) + repr(ei.value) + repr(ei.value.args)


def test_build_headers_rechecks_key_format() -> None:
    assert build_headers(None) == {"content-type": "application/json"}
    assert build_headers(SecretStr("k"))["Authorization"] == "Bearer k"
    bad = "sk-é-sentinel"
    with pytest.raises(MissingCredentialError) as ei:
        build_headers(SecretStr(bad), KEY_ENV)
    assert bad not in str(ei.value) + repr(ei.value) and KEY_ENV in str(ei.value)


def test_missing_httpx_reports_extra(monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "httpx":
            raise ImportError("no httpx")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(AdapterDependencyError) as ei:
        _run(_model(_ok()))
    assert "[openai_compatible]" in str(ei.value)


def test_leak_check(key_env, caplog) -> None:
    def status_handler(code: int):
        def h(request: httpx.Request) -> httpx.Response:
            body = {"error": {"message": CANARY, "type": "t", "code": None, "param": None}}
            return httpx.Response(code, json=body)

        return h

    def conn(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom " + request.headers["authorization"], request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    sent: list[str] = []

    def ok(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers["authorization"])
        return _ok()(request)

    for name in (None, "httpx", "httpcore", "h11"):
        caplog.set_level(logging.DEBUG, logger=name)

    places: list[str] = []
    result = _run(_model(ok, env=KEY_ENV))
    assert sent == [f"Bearer {SENTINEL}"]  # the key really was sent
    places += [repr(result), str(result)]

    errors: list[BaseException] = []
    for handler in (status_handler(401), status_handler(429), conn, slow, _raw(b"nope")):
        # Remote https base so the key path is exercised without the loopback exemption.
        try:
            _run(_model(handler, base=REMOTE, env=KEY_ENV, local=False))
        except ProviderCallError as exc:
            errors.append(exc)
    assert len(errors) == 5
    assert [e.reason for e in errors] == ["auth", "rate_limited", "connection", "timeout", "bad_response"]
    for exc in errors:
        assert exc.__cause__ is None and exc.__context__ is None
        places += [str(exc), repr(exc), repr(exc.args), "".join(traceback.format_exception(exc))]
    places += [r.getMessage() for r in caplog.records] + [caplog.text]
    places += [repr(read_api_key(KEY_ENV)), str(read_api_key(KEY_ENV))]
    for text in places:
        assert SENTINEL not in text
        assert CANARY not in text


def test_deeply_nested_json_body_is_bad_response():
    model = _model(lambda request: httpx.Response(200, content=b"[" * 100_000))
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(model.embed(["hello"]))
    assert info.value.reason == "bad_response"
    assert info.value.__context__ is None
