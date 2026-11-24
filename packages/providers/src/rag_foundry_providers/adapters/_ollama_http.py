"""Minimal JSON POST helper for the Ollama HTTP API. httpx is imported lazily."""

import json
import math
from types import ModuleType
from typing import Any

from rag_foundry_providers.errors import AdapterDependencyError, ProviderCallError


def load_httpx() -> ModuleType:
    try:
        import httpx
    except ImportError as exc:
        raise AdapterDependencyError() from exc
    return httpx


def estimate_tokens(text: str) -> int:
    """Approximate token count: ceil(chars / 4). Never 0 for non-empty text; 0 for empty text.

    A rough estimate for usage when Ollama does not report counts; not a tokenizer.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def map_http_error(httpx: ModuleType, exc: Exception, profile_id: str) -> ProviderCallError:
    """Map an httpx exception to ProviderCallError without echoing any content.

    ConnectError and ConnectTimeout -> connection; other timeouts -> timeout; invalid URLs,
    protocol and decoding errors, and redirect loops -> bad_response; any other
    httpx.HTTPError -> connection. The caller attaches the cause with `raise ... from exc`.
    """
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return ProviderCallError(profile_id, "connection")
    if isinstance(exc, httpx.TimeoutException):
        return ProviderCallError(profile_id, "timeout")
    if isinstance(
        exc, httpx.InvalidURL | httpx.ProtocolError | httpx.DecodingError | httpx.TooManyRedirects | httpx.StreamError
    ):
        return ProviderCallError(profile_id, "bad_response")
    return ProviderCallError(profile_id, "connection")


def status_error(profile_id: str, status: int) -> ProviderCallError | None:
    """Error for a non-2xx HTTP status, or None for 2xx."""
    if status == 404:
        return ProviderCallError(profile_id, "model_not_found", status=status)
    if not 200 <= status < 300:
        return ProviderCallError(profile_id, "bad_response", status=status)
    return None


async def post_json(
    api_base: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    transport: Any,
    profile_id: str,
) -> dict[str, Any]:
    """POST `payload` as JSON and return the decoded JSON object. No retries.

    The client ignores environment proxies and .netrc (trust_env=False) and does not
    follow redirects. Errors map to ProviderCallError without echoing any content.
    """
    httpx = load_httpx()
    try:
        async with httpx.AsyncClient(
            base_url=api_base, transport=transport, trust_env=False, follow_redirects=False
        ) as client:
            response = await client.post(path, json=payload, timeout=httpx.Timeout(timeout_s))
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise map_http_error(httpx, exc, profile_id) from exc
    status = response.status_code
    error = status_error(profile_id, status)
    if error is not None:
        raise error
    try:
        body = json.loads(response.content)
    except (ValueError, RecursionError) as exc:
        raise ProviderCallError(profile_id, "bad_response", status=status) from exc
    if not isinstance(body, dict):
        raise ProviderCallError(profile_id, "bad_response", status=status)
    return body
