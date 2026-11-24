"""Anthropic key, headers, error mapping and leak checks against httpx.MockTransport (no network)."""

import asyncio
import json
import logging
import traceback
import uuid

import httpx
import pytest
from pydantic import SecretStr
from rag_foundry_core.settings import load_settings
from rag_foundry_providers.adapters._anthropic_http import (
    ANTHROPIC_VERSION,
    anthropic_status_error,
    build_headers,
    post_messages,
    read_api_key,
)
from rag_foundry_providers.errors import AdapterDependencyError, MissingCredentialError, ProviderCallError

SENTINEL = f"sk-ant-TEST-SENTINEL-{uuid.uuid4()}"
CANARY = f"CANARY-MESSAGE-{uuid.uuid4()}"
BASE = "https://api.anthropic.com"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)


def _call(handler, key: SecretStr | None = None):
    return asyncio.run(
        post_messages(
            BASE,
            {"model": "m", "max_tokens": 1, "messages": []},
            key=key or read_api_key(),
            timeout_s=5,
            transport=httpx.MockTransport(handler),
            profile_id="p",
        )
    )


def _err_body(etype: str) -> dict:
    return {"type": "error", "error": {"type": etype, "message": CANARY}, "request_id": "req_abc123"}


def test_headers_and_key_only_in_authorization() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    assert _call(handler) == {"ok": True}
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/messages"
    assert req.headers["authorization"] == f"Bearer {SENTINEL}"
    assert "x-api-key" not in req.headers
    assert req.headers["anthropic-version"] == ANTHROPIC_VERSION == "2026-10-21"
    assert req.headers["content-type"] == "application/json"
    assert SENTINEL not in req.url.path and SENTINEL.encode() not in req.content
    others = {k: v for k, v in req.headers.items() if k != "authorization"}
    assert SENTINEL not in json.dumps(others)


@pytest.mark.parametrize("value", ["sk-ant-éabc", "sk-ant-a\nb", "sk-ant-a\rb", "sk-ant a", "sk-ant-a\tb"])
def test_malformed_key_rejected(monkeypatch, value) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(MissingCredentialError) as ei:
        read_api_key()
    err = ei.value
    for text in (str(err), repr(err), repr(err.args)):
        assert "ANTHROPIC_API_KEY" in text or text == repr(err.args)
        assert value not in text and value.strip() not in text
    assert err.__cause__ is None and err.__context__ is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY")
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(MissingCredentialError) as ei:
        read_api_key()
    assert "ANTHROPIC_API_KEY" in str(ei.value) and SENTINEL not in str(ei.value)


def test_secret_masked_in_repr() -> None:
    key = read_api_key()
    assert SENTINEL not in repr(key) and SENTINEL not in str(key)
    assert key.get_secret_value() == SENTINEL


@pytest.mark.parametrize(
    ("status", "etype", "reason"),
    [
        (400, "invalid_request_error", "invalid_request"),
        (401, "authentication_error", "auth"),
        (402, "billing_error", "billing"),
        (403, "permission_error", "auth"),
        (404, "not_found_error", "model_not_found"),
        (409, "conflict_error", "invalid_request"),
        (413, "request_too_large", "invalid_request"),
        (429, "rate_limit_error", "rate_limited"),
        (500, "api_error", "server_error"),
        (504, "timeout_error", "timeout"),
        (529, "overloaded_error", "overloaded"),
        (418, "weird_error", "bad_response"),
        (503, "overloaded_error", "overloaded"),
    ],
)
def test_status_mapping_no_message_leak(status, etype, reason) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=_err_body(etype), headers={"request-id": "req_abc123"})

    with pytest.raises(ProviderCallError) as ei:
        _call(handler)
    err = ei.value
    assert err.reason == reason and err.status == status and err.request_id == "req_abc123"
    assert CANARY not in str(err) and CANARY not in repr(err) and CANARY not in repr(err.args)


def test_non_json_error_body_and_bad_request_id() -> None:
    err = anthropic_status_error("p", 500, b"<html>" + CANARY.encode(), {"request-id": "evil id\n"})
    assert err is not None and err.reason == "server_error" and err.request_id is None
    assert CANARY not in str(err)
    assert anthropic_status_error("p", 200, b"{}") is None


def test_bad_success_body() -> None:
    with pytest.raises(ProviderCallError) as ei:
        _call(lambda r: httpx.Response(200, content=b"not json"))
    assert ei.value.reason == "bad_response"
    with pytest.raises(ProviderCallError):
        _call(lambda r: httpx.Response(200, json=[1]))


def test_leak_check_logs_errors_objects(caplog) -> None:
    key = read_api_key()
    secret_places: list[str] = [repr(key), str(key)]

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    def status_handler(code: int, etype: str):
        def h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(code, json=_err_body(etype))

        return h

    def conn(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    handlers = [ok, status_handler(401, "authentication_error"), status_handler(429, "rate_limit_error")]

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"nope")

    handlers += [status_handler(529, "overloaded_error"), conn, slow, bad_json]
    errors: list[BaseException] = []
    for name in (None, "httpx", "httpcore", "h11"):
        caplog.set_level(logging.DEBUG, logger=name)
    for h in handlers:
        try:
            _call(h, key)
        except ProviderCallError as exc:
            errors.append(exc)
    assert len(errors) == 6
    for exc in errors:
        assert exc.__cause__ is None and exc.__context__ is None
        secret_places += [str(exc), repr(exc), repr(exc.args), "".join(traceback.format_exception(exc))]
        cause = exc.__cause__ or exc.__context__
        while cause is not None:
            secret_places += [str(cause), repr(cause), repr(getattr(cause, "args", ()))]
            cause = cause.__cause__ or cause.__context__
    secret_places += [r.getMessage() for r in caplog.records] + [caplog.text]
    for text in secret_places:
        assert SENTINEL not in text
        assert CANARY not in text


def test_settings_do_not_hold_key(monkeypatch) -> None:
    settings = load_settings()
    assert SENTINEL not in repr(settings) and SENTINEL not in json.dumps(settings.model_dump(mode="json"))
    monkeypatch.setenv("RAG_FOUNDRY_ANTHROPIC_API_KEY", SENTINEL)
    with pytest.raises(ValueError) as ei:
        load_settings()
    assert "RAG_FOUNDRY_ANTHROPIC_API_KEY" in str(ei.value) and SENTINEL not in str(ei.value)


def test_missing_httpx_reports_anthropic_extra(monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "httpx":
            raise ImportError("no httpx")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(AdapterDependencyError) as ei:
        _call(lambda r: httpx.Response(200, json={}))
    assert "[anthropic]" in str(ei.value)


def test_build_headers_returns_plain_strings() -> None:
    assert set(build_headers(SecretStr("k"))) == {"Authorization", "anthropic-version", "content-type"}


def test_build_headers_rejects_malformed_key_without_echoing_it():
    from pydantic import SecretStr
    from rag_foundry_providers.adapters._anthropic_http import build_headers
    from rag_foundry_providers.errors import MissingCredentialError

    bad = "sk-ant-é-sentinel"
    with pytest.raises(MissingCredentialError) as info:
        build_headers(SecretStr(bad))
    assert bad not in str(info.value) and bad not in repr(info.value)
