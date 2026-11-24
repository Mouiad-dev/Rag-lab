"""OllamaEmbeddingModel against httpx.MockTransport with api.md-shaped JSON (no network)."""

import asyncio
import json
import sys

import httpx
import pytest
from rag_foundry_providers import TokenSource
from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile
from rag_foundry_providers.contract_checks import EMBEDDING_CHECKS
from rag_foundry_providers.errors import AdapterDependencyError, ProviderCallError

DIM = 1024


def _vec(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


def _model(handler, **kw) -> OllamaEmbeddingModel:
    return OllamaEmbeddingModel(
        ollama_embedding_profile(), api_base="http://localhost:11434", transport=httpx.MockTransport(handler), **kw
    )


def _ok(extra: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        body = {"model": "bge-m3", "embeddings": [_vec(0.1 * (i + 1)) for i in range(n)], "prompt_eval_count": 7}
        body.update(extra or {})
        return httpx.Response(200, json=body)

    return handler


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=[c.__name__ for c in EMBEDDING_CHECKS])
def test_embedding_contract(check) -> None:
    asyncio.run(check(_model(_ok())))


def test_request_path_and_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()(request)

    asyncio.run(_model(handler).embed(["a", "b"], input_type="query"))
    assert len(seen) == 1
    assert seen[0].method == "POST" and seen[0].url.path == "/api/embed"
    assert json.loads(seen[0].content) == {"model": "bge-m3", "input": ["a", "b"], "truncate": False}


def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    result = asyncio.run(_model(handler).embed([]))
    assert result.vectors == () and result.usage.input_tokens == 0


def test_timeout_reaches_request() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return _ok()(request)

    asyncio.run(_model(handler, timeout_s=12.5).embed(["a"]))
    assert seen[0] == {"connect": 12.5, "read": 12.5, "write": 12.5, "pool": 12.5}


def test_usage_from_provider() -> None:
    usage = asyncio.run(_model(_ok()).embed(["hello"])).usage
    assert usage.token_source is TokenSource.PROVIDER and usage.input_tokens == 7 and usage.output_tokens == 0
    assert usage.estimated_cost_usd == 0


@pytest.mark.parametrize("count", [None, -1, "7", True, 1.5])
def test_usage_estimated_when_count_missing_or_invalid(count) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"embeddings": [_vec()]}
        if count is not None:
            body["prompt_eval_count"] = count
        return httpx.Response(200, json=body)

    usage = asyncio.run(_model(handler).embed(["a"])).usage
    assert usage.token_source is TokenSource.ESTIMATED and usage.input_tokens == 1


def _resp(body, status=200):
    return lambda request: httpx.Response(status, json=body)


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": []},
        {"embeddings": [_vec(), _vec()]},
        {"embeddings": [[0.1] * 10]},
        {"embeddings": [[float("nan")] + [0.1] * (DIM - 1)]},
        {"embeddings": ["x"]},
        {"embeddings": [["a"] * DIM]},
        {"nope": 1},
    ],
    ids=["zero", "too-many", "wrong-dim", "nan", "not-list", "non-number", "missing"],
)
def test_bad_response_bodies(body) -> None:
    raw = json.dumps(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw.encode(), headers={"content-type": "application/json"})

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(handler).embed(["a"]))
    assert exc.value.reason == "bad_response"


def test_huge_json_integer_is_bad_response() -> None:
    huge = "1" + "0" * 400  # float() and isfinite() raise OverflowError on this
    raw = ('{"embeddings": [[' + huge + ", 0.1" + ", 0.1" * (DIM - 2) + "]]}").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw)

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(handler).embed(["a"]))
    assert exc.value.reason == "bad_response"


def test_non_json_and_server_error_are_bad_response() -> None:
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(lambda r: httpx.Response(200, content=b"<html>")).embed(["a"]))
    assert exc.value.reason == "bad_response"
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(_resp({"error": "secret text"}, 400)).embed(["a"]))
    assert exc.value.reason == "bad_response" and exc.value.status == 400
    assert "secret" not in str(exc.value)


def test_404_is_model_not_found() -> None:
    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(_resp({"error": "model not found"}, 404)).embed(["a"]))
    assert exc.value.reason == "model_not_found" and exc.value.status == 404


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ConnectError("refused"), "connection"),
        (httpx.ConnectTimeout("slow connect"), "connection"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.RemoteProtocolError("bad framing"), "bad_response"),
        (httpx.DecodingError("bad gzip"), "bad_response"),
        (httpx.InvalidURL("bad url"), "bad_response"),
    ],
)
def test_transport_errors(error, reason) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ProviderCallError) as exc:
        asyncio.run(_model(handler).embed(["a"]))
    assert exc.value.reason == reason and exc.value.__cause__ is error


def test_missing_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(AdapterDependencyError, match=r"rag-foundry-providers\[ollama\]"):
        asyncio.run(_model(_ok()).embed(["a"]))


def test_network_guard_blocks_literal_public_ip() -> None:
    async def go() -> None:
        await asyncio.get_running_loop().create_connection(asyncio.Protocol, "93.184.215.14", 80)

    with pytest.raises(RuntimeError, match="network blocked"):
        asyncio.run(go())
