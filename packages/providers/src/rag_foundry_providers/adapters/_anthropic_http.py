"""Key handling, headers, error mapping and JSON POST for the Anthropic Messages API.

The key is read only from the environment, held as SecretStr, and sent only in the
Authorization header. httpx is imported lazily.
"""

import json
import re
from types import ModuleType
from typing import Any

from pydantic import SecretStr

from rag_foundry_providers.adapters._keys import _KEY_CHARS
from rag_foundry_providers.adapters._keys import read_api_key as read_api_key  # re-export, old import path
from rag_foundry_providers.adapters._ollama_http import map_http_error
from rag_foundry_providers.errors import (
    AdapterDependencyError,
    MissingCredentialError,
    ProviderCallError,
    ProviderCallReason,
    parse_retry_after,
)

# Verified 2026-11-02 (ADR-0001, P2.3-01): https://platform.claude.com/docs/en/api/versioning
ANTHROPIC_VERSION = "2026-10-21"
MESSAGES_PATH = "/v1/messages"

_REQUEST_ID = re.compile(r"^req_[A-Za-z0-9_]{1,128}$")

# Source: https://platform.claude.com/docs/en/api/errors (retrieved 2026-11-02).
_STATUS_REASON: dict[int, ProviderCallReason] = {
    400: "invalid_request",
    401: "auth",
    402: "billing",
    403: "auth",
    404: "model_not_found",
    409: "invalid_request",
    413: "invalid_request",
    429: "rate_limited",
    500: "server_error",
    504: "timeout",
    529: "overloaded",
}


def _load_httpx() -> ModuleType:
    try:
        import httpx
    except ImportError as exc:
        raise AdapterDependencyError("anthropic") from exc
    return httpx


def build_headers(key: SecretStr) -> dict[str, str]:
    """Bearer auth (the documented primary method; x-api-key is not sent), version and content type."""
    value = key.get_secret_value()
    if _KEY_CHARS.fullmatch(value) is None:
        # Same rule as read_api_key, for callers that build a SecretStr themselves:
        # httpx would otherwise raise errors that echo the header value.
        raise MissingCredentialError("ANTHROPIC_API_KEY")
    return {
        "Authorization": f"Bearer {value}",
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _request_id(headers: Any, body: Any) -> str | None:
    candidates = [headers.get("request-id") if headers is not None else None]
    if isinstance(body, dict):
        candidates.append(body.get("request_id"))
    for c in candidates:
        if isinstance(c, str) and _REQUEST_ID.match(c):
            return c
    return None


def anthropic_status_error(
    profile_id: str, status: int, body_bytes: bytes = b"", headers: Any = None
) -> ProviderCallError | None:
    """Error for a non-2xx status, or None for 2xx. Never copies error.message."""
    if 200 <= status < 300:
        return None
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (ValueError, RecursionError):
        body = None
    reason: ProviderCallReason = _STATUS_REASON.get(status, "bad_response")
    err_type = (
        body.get("error", {}).get("type") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
    )
    if status not in _STATUS_REASON and err_type == "overloaded_error":
        reason = "overloaded"
    retry_after = parse_retry_after(headers.get("retry-after")) if headers is not None else None
    return ProviderCallError(
        profile_id, reason, status=status, request_id=_request_id(headers, body), retry_after_s=retry_after
    )


async def post_messages(
    api_base: str,
    payload: dict[str, Any],
    *,
    key: SecretStr,
    timeout_s: float,
    transport: Any,
    profile_id: str,
) -> dict[str, Any]:
    """POST `payload` to /v1/messages and return the decoded JSON object. No retries."""
    httpx = _load_httpx()
    failure: ProviderCallError | None = None
    response = None
    try:
        async with httpx.AsyncClient(
            base_url=api_base, transport=transport, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.post(
                MESSAGES_PATH, json=payload, headers=build_headers(key), timeout=httpx.Timeout(timeout_s)
            )
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        failure = map_http_error(httpx, exc, profile_id)
    # Raised outside the except block so no __context__ (whose request headers hold the key) is attached.
    if failure is not None or response is None:
        raise failure or ProviderCallError(profile_id, "connection")
    status = response.status_code
    failure = anthropic_status_error(profile_id, status, response.content, response.headers)
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
