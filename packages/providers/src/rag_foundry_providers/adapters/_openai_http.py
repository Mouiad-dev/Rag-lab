"""Key-aware headers, error mapping and JSON POST for OpenAI-compatible endpoints.

Verified 2026-11-02 (ADR-0001, P2.4-01): Bearer auth, error statuses and `x-request-id`.
The key (optional for local servers) is sent only in the Authorization header. httpx is
imported lazily.
"""

import json
import re
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr

from rag_foundry_providers._urls import is_loopback
from rag_foundry_providers.adapters._keys import _KEY_CHARS
from rag_foundry_providers.adapters._ollama_http import map_http_error
from rag_foundry_providers.errors import (
    AdapterDependencyError,
    MissingCredentialError,
    ProviderCallError,
    ProviderCallReason,
    parse_retry_after,
)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Documented statuses: 400, 401, 403, 429, 500, 503 (OpenAI error-codes page). The others are
# standard HTTP statuses mapped defensively; anything else is bad_response.
_STATUS_REASON: dict[int, ProviderCallReason] = {
    400: "invalid_request",
    401: "auth",
    402: "billing",
    403: "auth",
    404: "model_not_found",
    408: "timeout",
    413: "invalid_request",
    422: "invalid_request",
    429: "rate_limited",
    500: "server_error",
    502: "server_error",
    503: "overloaded",
    504: "timeout",
}


def _load_httpx() -> ModuleType:
    try:
        import httpx
    except ImportError as exc:
        raise AdapterDependencyError("openai_compatible") from exc
    return httpx


def build_headers(key: SecretStr | None, env_var: str = "the configured API key variable") -> dict[str, str]:
    """Content type, plus `Authorization: Bearer` only when a key is given.

    The key format is re-checked here (same rule as read_api_key) so a SecretStr built by a
    caller cannot make httpx raise an error that echoes the header value.
    """
    headers = {"content-type": "application/json"}
    if key is not None:
        value = key.get_secret_value()
        if _KEY_CHARS.fullmatch(value) is None:
            raise MissingCredentialError(env_var)
        headers["Authorization"] = f"Bearer {value}"
    return headers


_QUOTA_CODES = frozenset({"insufficient_quota", "exceeded_current_quota_error"})
_OVERLOAD_CODES = frozenset({"engine_overloaded_error"})


def _rate_limit_reason(body_bytes: bytes) -> ProviderCallReason:
    """Classify a 429 from fixed error.code / error.type strings only; no body text is copied."""
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (ValueError, RecursionError):
        return "rate_limited"
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return "rate_limited"
    values = {v for v in (error.get("code"), error.get("type")) if isinstance(v, str)}
    if values & _QUOTA_CODES:
        return "billing"
    if values & _OVERLOAD_CODES:
        return "overloaded"
    return "rate_limited"


def openai_status_error(
    profile_id: str, status: int, body_bytes: bytes = b"", headers: Any = None
) -> ProviderCallError | None:
    """Error for a non-2xx status, or None for 2xx. Never copies error.message or the body."""
    if 200 <= status < 300:
        return None
    reason: ProviderCallReason = _STATUS_REASON.get(status, "bad_response")
    if status == 429:
        reason = _rate_limit_reason(body_bytes)
    request_id = None
    retry_after = None
    if headers is not None:
        candidate = headers.get("x-request-id")
        if isinstance(candidate, str) and _REQUEST_ID.match(candidate):
            request_id = candidate
        retry_after = parse_retry_after(headers.get("retry-after"))
    return ProviderCallError(profile_id, reason, status=status, request_id=request_id, retry_after_s=retry_after)


async def post_json(
    api_base: str,
    path: str,
    payload: dict[str, Any],
    *,
    key: SecretStr | None,
    timeout_s: float,
    transport: Any,
    profile_id: str,
    key_env: str = "the configured API key variable",
) -> dict[str, Any]:
    """POST `payload` to `api_base + path` and return the decoded JSON object. No retries.

    Ignores environment proxies and .netrc, follows no redirects, refuses to send a key over
    plain http to a non-loopback host (before any I/O).
    """
    httpx = _load_httpx()
    headers = build_headers(key, key_env)
    if key is not None:
        try:
            scheme = urlsplit(api_base).scheme
        except (ValueError, RecursionError):
            scheme = ""
        if scheme != "https" and not is_loopback(api_base):
            raise ProviderCallError(profile_id, "invalid_request")
    failure: ProviderCallError | None = None
    response = None
    try:
        async with httpx.AsyncClient(
            base_url=api_base, transport=transport, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.post(path, json=payload, headers=headers, timeout=httpx.Timeout(timeout_s))
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        failure = map_http_error(httpx, exc, profile_id)
    # Raised outside the except block so no __cause__/__context__ (request headers hold the key).
    if failure is not None or response is None:
        raise failure or ProviderCallError(profile_id, "connection")
    status = response.status_code
    failure = openai_status_error(profile_id, status, response.content, response.headers)
    body: Any = None
    if failure is None:
        try:
            body = json.loads(response.content)
        except (ValueError, RecursionError):
            body = None
        if not isinstance(body, dict):
            failure = ProviderCallError(profile_id, "bad_response", status=status)
    if failure is not None:
        raise failure
    return body
