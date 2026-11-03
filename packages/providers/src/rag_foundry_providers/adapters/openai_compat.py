"""OpenAI-compatible adapters over direct httpx (extra `rag-foundry-providers[openai_compatible]`).

Embeddings: `POST {api_base}/embeddings` as documented in the OpenAI reference (verified
2026-11-02, ADR-0001 P2.4-01). `api_base` includes the version prefix, for example
`http://localhost:1234/v1`. Chat: `POST {api_base}/chat/completions`, see
`OpenAICompatibleChatModel`.
"""

import json
import math
import time
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import SecretStr

from rag_foundry_providers._urls import is_loopback
from rag_foundry_providers.adapters._keys import read_api_key
from rag_foundry_providers.adapters._ollama_http import estimate_tokens, load_httpx, map_http_error
from rag_foundry_providers.adapters._openai_http import build_headers, openai_status_error, post_json
from rag_foundry_providers.adapters.anthropic import UnsupportedRequestError
from rag_foundry_providers.capabilities import ModelKind, ModelProfile
from rag_foundry_providers.errors import ProviderCallError, ProviderError
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult, EmbeddingResult
from rag_foundry_providers.ports import check_chat_request
from rag_foundry_providers.usage import (
    Clock,
    Operation,
    TokenSource,
    Usage,
    estimate_cost,
    measure_latency_ms,
)


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _to_float(value: int | float) -> float | None:
    try:
        return float(value)
    except OverflowError:  # huge JSON integer
        return None


class OpenAICompatibleEmbeddingModel:
    """EmbeddingModel on `POST {api_base}/embeddings`. Dense float vectors only, no retries.

    `input_type` is accepted and ignored (the API has no such field). The key, when
    `api_key_env` is set, is read from that environment variable at construction.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_base: str,
        api_key_env: str | None = None,
        timeout_s: float = 60.0,
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        if profile.kind is not ModelKind.EMBEDDING:
            raise ProviderError(f"OpenAICompatibleEmbeddingModel needs an embedding profile (got '{profile.id}')")
        self.profile = profile
        self._api_base = api_base
        self._api_key_env = api_key_env
        self._key: SecretStr | None = read_api_key(api_key_env) if api_key_env else None
        self._timeout_s = timeout_s
        self._transport = transport
        self._clock = clock
        # Estimates are allowed only for a local endpoint: declared local AND a loopback URL.
        self._local = bool(profile.capabilities.local) and is_loopback(api_base)

    def __repr__(self) -> str:
        return f"OpenAICompatibleEmbeddingModel(profile={self.profile.id!r})"

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        start = self._clock()
        if not texts:
            return EmbeddingResult(vectors=(), usage=self._usage(0, TokenSource.ESTIMATED, start, None))
        body = await post_json(
            self._api_base,
            "/embeddings",
            {"model": self.profile.model, "input": list(texts), "encoding_format": "float"},
            key=self._key,
            timeout_s=self._timeout_s,
            transport=self._transport,
            profile_id=self.profile.id,
            key_env=self._api_key_env or "the configured API key variable",
        )
        vectors = self._validate(body, len(texts))
        usage = body.get("usage")
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        tokens = _count(usage.get("prompt_tokens")) if isinstance(usage, dict) else None
        if tokens is not None:
            source = TokenSource.PROVIDER
        elif self._local:
            tokens, source = sum(estimate_tokens(t) for t in texts), TokenSource.ESTIMATED
        else:
            # Never estimate a paid call.
            raise ProviderCallError(self.profile.id, "bad_response")
        return EmbeddingResult(vectors=vectors, usage=self._usage(tokens, source, start, details))

    def _usage(self, tokens: int, source: TokenSource, start: float, details: Any) -> Usage:
        cost: Decimal | None = estimate_cost(self.profile, tokens, 0)
        if isinstance(details, dict) and details.get("cached_tokens") not in (None, 0):
            # Whether prompt_tokens includes cached tokens is unconfirmed for OpenAI (ADR-0001): no cost.
            cost = None
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=Operation.EMBED,
            input_tokens=tokens,
            output_tokens=0,
            token_source=source,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=cost,
        )

    def _bad(self) -> ProviderCallError:
        return ProviderCallError(self.profile.id, "bad_response")

    def _validate(self, body: dict[str, Any], expected: int) -> tuple[tuple[float, ...], ...]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise self._bad()
        indexed: list[tuple[int, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                raise self._bad()
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise self._bad()
            indexed.append((index, item.get("embedding")))
        indexed.sort(key=lambda pair: pair[0])
        if [i for i, _ in indexed] != list(range(expected)):
            raise self._bad()
        dim = self.profile.capabilities.embedding_dim
        out: list[tuple[float, ...]] = []
        for _, vec in indexed:
            if not isinstance(vec, list) or len(vec) != dim:
                raise self._bad()
            floats: list[float] = []
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, int | float):
                    raise self._bad()
                f = _to_float(v)
                if f is None or not math.isfinite(f):
                    raise self._bad()
                floats.append(f)
            out.append(tuple(floats))
        return tuple(out)


_CHAT_PATH = "/chat/completions"
_MAX_TOKENS_FIELDS = ("max_completion_tokens", "max_tokens")


class OpenAICompatibleChatModel:
    """ChatModel on `POST {api_base}/chat/completions` (complete and SSE stream). Single attempt, no retries.

    Serves LM Studio, vLLM and Moonshot (ADR-0001, P2.4-01 and owner decisions 2026-11-02).
    Tools, structured output and vision success paths are deferred (tools to P8.3, structured
    output to P4.4 or P5.4): a profile declaring any of them True is refused at construction, and
    a request that needs one raises MissingCapabilityError before any I/O. Never sends `tools`,
    `tool_choice` or `response_format`.

    Per-profile settings:
    - `max_tokens_field`: name of the output cap field. Default `"max_completion_tokens"`
      (OpenAI and Moonshot; `max_tokens` is deprecated there). LM Studio documents `max_tokens`,
      so local profiles for it should pass `"max_tokens"`.
    - `supports_temperature`: default True (LM Studio, vLLM, OpenAI). Moonshot K3 fixes
      temperature and returns an error for any other value, so its profile passes False and a
      request that sets `temperature` raises UnsupportedRequestError before any call.
    Moonshot K3 thinking is always on and its reasoning tokens are billed inside
    `completion_tokens`; they are counted but `reasoning_content` is never returned as text.
    The live budget for that is not part of P2.4.

    Usage: PROVIDER when the response carries int `prompt_tokens` and `completion_tokens`
    (streaming sends `stream_options.include_usage`; the usage chunk has empty `choices`).
    `input_tokens` is `prompt_tokens` as reported. Whether that already contains cached tokens
    is unconfirmed for OpenAI, so when `cached_tokens` or `cache_write_tokens` is above 0 the
    cost is None and the cache fields stay 0. Missing usage is ESTIMATED only for a profile that
    is `local` AND has a loopback base; otherwise it is `bad_response`.
    A 200 reply that then fails (content_filter, an unexpected finish) carries its
    provider-reported usage on `ProviderCallError.usage`.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_base: str,
        api_key_env: str | None = None,
        timeout_s: float = 60.0,
        default_max_tokens: int | None = None,
        max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
        supports_temperature: bool = True,
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        if profile.kind is not ModelKind.CHAT:
            raise ProviderError(f"OpenAICompatibleChatModel needs a chat profile (got '{profile.id}')")
        unsupported = [
            name
            for name in ("tool_calling", "structured_output", "vision")
            if getattr(profile.capabilities, name) is True
        ]
        if unsupported:
            raise ProviderError(
                f"OpenAICompatibleChatModel does not implement {unsupported} (profile '{profile.id}'); "
                "tool calling, structured output and vision success paths are deferred"
            )
        if default_max_tokens is not None and default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be > 0")
        if max_tokens_field not in _MAX_TOKENS_FIELDS:
            raise ValueError(f"max_tokens_field must be one of {_MAX_TOKENS_FIELDS}")
        self.profile = profile
        self._api_base = api_base
        self._api_key_env = api_key_env
        self._key: SecretStr | None = read_api_key(api_key_env) if api_key_env else None
        self._timeout_s = timeout_s
        self._default_max_tokens = default_max_tokens
        self._max_tokens_field = max_tokens_field
        self._supports_temperature = supports_temperature
        self._transport = transport
        self._clock = clock
        # Estimates are allowed only for a local endpoint: declared local AND a loopback URL.
        self._local = bool(profile.capabilities.local) and is_loopback(api_base)

    def __repr__(self) -> str:
        return f"OpenAICompatibleChatModel(profile={self.profile.id!r})"

    def _bad(self) -> ProviderCallError:
        return ProviderCallError(self.profile.id, "bad_response")

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        if request.temperature is not None and not self._supports_temperature:
            raise UnsupportedRequestError(
                f"profile '{self.profile.id}': temperature is not accepted by this endpoint; omit it"
            )
        messages: list[dict[str, str]] = []
        for m in request.messages:
            if m.role == "tool":
                raise UnsupportedRequestError("tool messages are not supported (tool calling is deferred)")
            messages.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {"model": self.profile.model, "messages": messages, "stream": stream}
        cap = request.max_output_tokens if request.max_output_tokens is not None else self._default_max_tokens
        if cap is not None:
            payload[self._max_tokens_field] = cap
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _check_key_transport(self) -> None:
        """Refuse to send a key over plain http to a non-loopback host, before any I/O."""
        if self._key is None:
            return
        try:
            scheme = urlsplit(self._api_base).scheme
        except (ValueError, RecursionError):
            scheme = ""
        if scheme != "https" and not is_loopback(self._api_base):
            raise ProviderCallError(self.profile.id, "invalid_request")

    def _finish(self, finish_reason: Any) -> Literal["stop", "length"]:
        if finish_reason == "stop":
            return "stop"
        if finish_reason == "length":
            return "length"
        if finish_reason == "content_filter":
            raise ProviderCallError(self.profile.id, "refused")
        # null, tool_calls, function_call or unknown: tools are refused, so none of these may appear.
        raise self._bad()

    def _provider_usage(self, raw: Any, op: Operation, start: float) -> Usage | None:
        """Usage from provider counts, or None when prompt/completion tokens are not valid ints."""
        if not isinstance(raw, dict):
            return None
        tokens_in = _count(raw.get("prompt_tokens"))
        tokens_out = _count(raw.get("completion_tokens"))
        if tokens_in is None or tokens_out is None:
            return None
        details = raw.get("prompt_tokens_details")
        cost: Decimal | None = estimate_cost(self.profile, tokens_in, tokens_out)
        if isinstance(details, dict) and any(
            details.get(name) not in (None, 0) for name in ("cached_tokens", "cache_write_tokens")
        ):
            # Whether prompt_tokens includes cached tokens is unconfirmed (ADR-0001): no cost.
            cost = None
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=op,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            token_source=TokenSource.PROVIDER,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=cost,
        )

    def _final_usage(self, billed: Usage | None, request: ChatRequest, text: str, op: Operation, start: float) -> Usage:
        if billed is not None:
            return billed
        if not self._local:
            raise self._bad()  # never estimate a paid call
        tokens_in = sum(estimate_tokens(m.content) for m in request.messages)
        tokens_out = estimate_tokens(text)
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=op,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            token_source=TokenSource.ESTIMATED,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=estimate_cost(self.profile, tokens_in, tokens_out),
        )

    async def complete(self, request: ChatRequest) -> ChatResult:
        check_chat_request(self.profile, request, streaming=False)
        payload = self._payload(request, stream=False)
        start = self._clock()
        body = await post_json(
            self._api_base,
            _CHAT_PATH,
            payload,
            key=self._key,
            timeout_s=self._timeout_s,
            transport=self._transport,
            profile_id=self.profile.id,
            key_env=self._api_key_env or "the configured API key variable",
        )
        # Build usage first: a billed 200 that fails below (content_filter, tool_calls, ...) carries it.
        billed = self._provider_usage(body.get("usage"), Operation.CHAT, start)
        try:
            choices = body.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                raise self._bad()
            if message.get("tool_calls") or message.get("function_call"):
                raise self._bad()
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise self._bad()
            finish = self._finish(choice.get("finish_reason"))
        except ProviderCallError as exc:
            exc.usage = billed
            raise
        text = content or ""
        usage = self._final_usage(billed, request, text, Operation.CHAT, start)
        return ChatResult(text=text, finish_reason=finish, usage=usage)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        check_chat_request(self.profile, request, streaming=True)
        payload = self._payload(request, stream=True)
        httpx = load_httpx()
        self._check_key_transport()
        start = self._clock()
        pid = self.profile.id
        client = None
        response = None
        parts: list[str] = []
        usage_raw: Any = None
        finish_raw: Any = None
        forbidden = False  # tool_calls/function_call seen: keep reading so the billed usage is captured
        done = False

        def handle(obj: Any) -> str:
            """Apply one decoded chunk; return the answer text it carries ("" for none)."""
            nonlocal usage_raw, finish_raw, forbidden
            if not isinstance(obj, dict) or obj.get("error") is not None:
                raise self._bad()
            raw = obj.get("usage")
            if isinstance(raw, dict):
                usage_raw = raw
            choices = obj.get("choices")
            if not isinstance(choices, list):
                raise self._bad()
            if not choices:  # the include_usage chunk
                return ""
            choice = choices[0]
            if not isinstance(choice, dict):
                raise self._bad()
            delta = choice.get("delta")
            if delta is None:
                delta = {}
            if not isinstance(delta, dict):
                raise self._bad()
            if delta.get("tool_calls") or delta.get("function_call"):
                forbidden = True
            if choice.get("finish_reason") is not None:
                finish_raw = choice["finish_reason"]
            content = delta.get("content")  # reasoning_content is never returned as text
            if content is None:
                return ""
            if not isinstance(content, str):
                raise self._bad()
            return content

        try:
            failure: ProviderCallError | None = None
            try:
                client = httpx.AsyncClient(
                    base_url=self._api_base, transport=self._transport, trust_env=False, follow_redirects=False
                )
                http_request = client.build_request(
                    "POST",
                    _CHAT_PATH,
                    json=payload,
                    headers=build_headers(self._key, self._api_key_env or "the configured API key variable"),
                    timeout=httpx.Timeout(self._timeout_s),
                )
                response = await client.send(http_request, stream=True)
                if not 200 <= response.status_code < 300:
                    error_bytes = await response.aread()
                    failure = openai_status_error(pid, response.status_code, error_bytes, response.headers)
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue  # blank separators, comments and other SSE fields
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            obj = json.loads(data)
                            decoded = True
                        except (ValueError, RecursionError):
                            obj, decoded = None, False
                        if not decoded:
                            failure = self._bad()
                            break
                        delta_text = handle(obj)
                        if delta_text:
                            parts.append(delta_text)
                            yield ChatChunk(delta=delta_text)
            except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as exc:
                failure = map_http_error(httpx, exc, pid)
            # Raised outside the except block so no __context__ (it could hold request headers).
            if failure is not None:
                raise failure
            if not done:
                raise self._bad()  # interrupted stream
            billed = self._provider_usage(usage_raw, Operation.CHAT_STREAM, start)
            try:
                if forbidden:
                    raise self._bad()
                self._finish(finish_raw)
            except ProviderCallError as exc:
                exc.usage = billed  # a billed 200 that still failed
                raise
            usage = self._final_usage(billed, request, "".join(parts), Operation.CHAT_STREAM, start)
        finally:
            if response is not None:
                await response.aclose()
            if client is not None:
                await client.aclose()
        yield ChatChunk(delta="", usage=usage)
