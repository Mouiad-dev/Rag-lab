"""Retry-After parsing and 429 quota classification (no network)."""

import json

import pytest
from rag_foundry_providers.adapters._anthropic_http import anthropic_status_error
from rag_foundry_providers.adapters._openai_http import openai_status_error
from rag_foundry_providers.errors import RETRYABLE_REASONS, parse_retry_after

SENTINEL = "SENTINEL-MESSAGE-must-not-leak"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("7", 7.0),
        ("120", 120.0),
        ("1234567", None),
        ("-1", None),
        ("1.5", None),
        (" 7", None),
        ("7 ", None),
        ("", None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        (None, None),
        (7, None),
    ],
)
def test_parse_retry_after(value, expected):
    assert parse_retry_after(value) == expected


@pytest.mark.parametrize("helper", [anthropic_status_error, openai_status_error])
def test_429_retry_after_header(helper):
    with_header = helper("p", 429, b"", {"retry-after": "7"})
    assert with_header is not None
    assert with_header.reason == "rate_limited"
    assert with_header.retry_after_s == 7.0
    assert "7" not in str(with_header).replace("429", "")
    without = helper("p", 429, b"", {})
    assert without is not None
    assert without.retry_after_s is None
    assert helper("p", 429, b"").retry_after_s is None


def _body(code=None, type_=None, message=SENTINEL):
    return json.dumps({"error": {"code": code, "type": type_, "message": message}}).encode()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (_body(code="insufficient_quota"), "billing"),
        (_body(type_="insufficient_quota"), "billing"),
        (_body(type_="exceeded_current_quota_error"), "billing"),
        (_body(type_="engine_overloaded_error"), "overloaded"),
        (_body(code="rate_limit_exceeded"), "rate_limited"),
        (b"not json", "rate_limited"),
        (b"[1]", "rate_limited"),
    ],
)
def test_openai_429_classification(body, reason):
    err = openai_status_error("p", 429, body, {})
    assert err is not None
    assert err.reason == reason
    assert err.status == 429


def test_body_text_never_in_error():
    err = openai_status_error("p", 429, _body(code="insufficient_quota"), {"retry-after": "3"})
    assert err is not None
    for text in (str(err), repr(err), repr(err.args)):
        assert SENTINEL not in text
    assert err.__cause__ is None
    assert err.__context__ is None


def test_retryable_reasons():
    assert RETRYABLE_REASONS == {"connection", "rate_limited", "overloaded", "server_error"}
