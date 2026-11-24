"""Raw llama.cpp completion tests using an in-memory HTTP transport."""

import asyncio
import json

import httpx
import pytest
from rag_foundry_providers.adapters.llamacpp import LlamaCppRawCompletionModel, llamacpp_raw_completion_profile
from rag_foundry_providers.adapters.ollama import ollama_chat_profile
from rag_foundry_providers.errors import MissingCapabilityError, ProviderCallError
from rag_foundry_providers.messages import RawCompletionRequest
from rag_foundry_providers.usage import TokenSource


def _body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "content": "ok",
        "stop_type": "eos",
        "tokens_evaluated": 3,
        "tokens_predicted": 1,
        "completion_probabilities": [
            {
                "id": 4,
                "token": "ok",
                "logprob": -0.5,
                "top_logprobs": [
                    {"id": 4, "token": "ok", "logprob": -0.5},
                    {"id": 5, "token": " okay", "logprob": -1.5},
                ],
            }
        ],
    }
    body.update(changes)
    return body


def _model(handler: object, **kwargs: object) -> LlamaCppRawCompletionModel:
    return LlamaCppRawCompletionModel(
        llamacpp_raw_completion_profile(), transport=httpx.MockTransport(handler), **{"vocab_size": 16, **kwargs}
    )


def _call(model: LlamaCppRawCompletionModel, top_logprobs: int = 3):
    request = RawCompletionRequest(
        prompt="hello",
        max_output_tokens=4,
        temperature=0.25,
        top_p=0.9,
        top_k=10,
        top_logprobs=top_logprobs,
        stop=("!",),
    )
    return asyncio.run(model.complete_raw(request))


def test_raw_completion_sends_request_and_parses_ids() -> None:
    seen: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen.update(json.loads(await request.aread()))
        return httpx.Response(200, json=_body())

    result = _call(_model(respond))

    assert seen == {
        "path": "/completion",
        "prompt": "hello",
        "n_predict": 4,
        "n_probs": 3,
        "temperature": 0.25,
        "top_k": 10,
        "top_p": 0.9,
        "min_p": 0,
        "stream": False,
        "cache_prompt": False,
        "post_sampling_probs": False,
        "repeat_penalty": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "dry_multiplier": 0,
        "xtc_probability": 0,
        "typical_p": 1.0,
        "stop": ["!"],
    }
    assert "messages" not in seen and "system" not in seen
    assert result.text == "ok"
    assert result.finish_reason == "stop"
    assert result.tokens[0].token_id == 4
    assert result.tokens[0].top_ids == (4, 5)
    assert result.tokens[0].top == (("ok", -0.5), (" okay", -1.5))
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 1
    assert result.usage.token_source is TokenSource.PROVIDER


@pytest.mark.parametrize(("stop_type", "expected"), [("word", "stop"), ("limit", "length")])
def test_finish_reason_maps_documented_stop_types(stop_type: str, expected: str) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(stop_type=stop_type))

    assert _call(_model(respond)).finish_reason == expected


def test_missing_usage_counts_are_estimated() -> None:
    body = _body()
    del body["tokens_evaluated"], body["tokens_predicted"]

    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    result = _call(_model(respond))
    assert result.usage.token_source is TokenSource.ESTIMATED
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


def _entry(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {"id": 4, "token": "ok", "logprob": -0.5, "top_logprobs": []}
    entry.update(changes)
    return entry


_BAD_BODIES = [
    ("probabilities_missing", {k: v for k, v in _body().items() if k != "completion_probabilities"}),
    ("probabilities_not_list", _body(completion_probabilities={})),
    ("entry_wrong_type", _body(completion_probabilities=["x"])),
    ("missing_id", _body(completion_probabilities=[_entry(id=None)])),
    ("id_wrong_type", _body(completion_probabilities=[_entry(id=True)])),
    ("id_out_of_vocab", _body(completion_probabilities=[_entry(id=16)])),
    ("token_wrong_type", _body(completion_probabilities=[_entry(token=1)])),
    ("logprob_nonfinite", _body(completion_probabilities=[_entry(logprob=float("nan"))])),
    ("logprob_positive", _body(completion_probabilities=[_entry(logprob=0.2)])),
    ("logprob_wrong_type", _body(completion_probabilities=[_entry(logprob="-0.5")])),
    ("top_logprobs_missing", _body(completion_probabilities=[{"id": 4, "token": "ok", "logprob": -0.5}])),
    ("top_logprobs_wrong_type", _body(completion_probabilities=[_entry(top_logprobs={})])),
    (
        "alternative_id_out_of_vocab",
        _body(completion_probabilities=[_entry(top_logprobs=[{"id": 16, "token": "x", "logprob": -1}])]),
    ),
    (
        "alternative_logprob_positive",
        _body(completion_probabilities=[_entry(top_logprobs=[{"id": 5, "token": "x", "logprob": 0.1}])]),
    ),
    (
        "alternative_duplicate_id",
        _body(
            completion_probabilities=[
                _entry(top_logprobs=[{"id": 5, "token": "x", "logprob": -1}, {"id": 5, "token": "x", "logprob": -2}])
            ]
        ),
    ),
    ("wrong_probability_count", _body(completion_probabilities=[])),
    ("content_missing", {k: v for k, v in _body().items() if k != "content"}),
    ("content_wrong_type", _body(content=1)),
    ("stop_type_missing", {k: v for k, v in _body().items() if k != "stop_type"}),
    ("stop_type_unknown", _body(stop_type="none")),
]


@pytest.mark.parametrize(("name", "body"), _BAD_BODIES, ids=[name for name, _ in _BAD_BODIES])
def test_malformed_response_is_bad_response(name: str, body: dict[str, object]) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        # Raw text, so NaN reaches the adapter parser (httpx json= refuses non-finite floats).
        return httpx.Response(200, content=json.dumps(body).encode(), headers={"content-type": "application/json"})

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond))
    assert caught.value.reason == "bad_response"


def test_response_over_byte_cap_is_rejected() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{" + b" " * 40)

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond, max_response_bytes=16))
    assert caught.value.reason == "bad_response"


def test_server_error_keeps_status() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond))
    assert caught.value.reason == "server_error"
    assert caught.value.status == 500


def test_connection_error_is_mapped() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond))
    assert caught.value.reason == "connection"


@pytest.mark.parametrize("failure", ["capability", "vocab"])
def test_preflight_rejections_send_no_requests(failure: str) -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    if failure == "capability":
        model = LlamaCppRawCompletionModel(ollama_chat_profile(), transport=httpx.MockTransport(respond))
        error = MissingCapabilityError
    else:
        model = _model(respond, vocab_size=2)
        error = ProviderCallError
    request = RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=3)

    with pytest.raises(error) as caught:
        asyncio.run(model.complete_raw(request))
    if failure == "vocab":
        assert caught.value.reason == "invalid_request"
    assert calls == 0


def test_raw_completion_contract_accepts_mock_model() -> None:
    from rag_foundry_providers.contract_checks import check_raw_completion

    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body())

    asyncio.run(check_raw_completion(_model(respond)))
