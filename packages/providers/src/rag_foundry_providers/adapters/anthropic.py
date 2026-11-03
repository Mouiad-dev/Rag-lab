"""Direct Anthropic chat adapter over HTTP (httpx, optional extra `rag-foundry-providers[anthropic]`).

Anthropic has no embeddings API, so this module provides a chat adapter only: no embedding
adapter exists or is planned. Fields follow https://platform.claude.com/docs/en/api/messages and
https://platform.claude.com/docs/en/build-with-claude/streaming (ADR-0001, verified 2026-11-02).

Design notes (owner decisions 2026-11-02):
- `temperature` is refused with `UnsupportedRequestError` before any HTTP call (the API returns 400).
- Thinking is set explicitly per model: Haiku 5.5 sends `{"type": "disabled"}`; Sonnet 5.5 cannot
  disable it, so it sends `{"type": "between_tools"}` as documented. Thinking blocks are never
  returned as text; their tokens are inside `output_tokens` and so are billed and counted.
- Default `max_tokens` when the request sets none: 1024 for Haiku (thinking off) and unknown
  models, 16384 for Sonnet (leaves room for billed thinking tokens; at USD 10 per MTok that
  caps a default call at about USD 0.16).
- Usage is PROVIDER-sourced. `input_tokens` is uncached input only; cache reads and writes map
  to their own fields. A missing or invalid input/output count raises `bad_response`.
- Cost tiering: Haiku is priced per request by prompt length (input + cache read + cache write
  tokens), with a dearer rate above 100,000 tokens that the profile does not carry. Above that
  threshold the cost is None, never the cheaper rate. 1-hour cache writes are also None.
- `model_context_window_exceeded` maps to finish_reason "length" (the reply was cut short).
  `refusal` raises ProviderCallError(reason "refused"); its billed usage, when parsed, is on `.usage`.
- Tools, structured output and vision are not implemented (tool calling P8.3, structured output P4.4 or
  P5.4); profiles declare them False.
"""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import SecretStr

from rag_foundry_providers.adapters._anthropic_http import (
    MESSAGES_PATH,
    anthropic_status_error,
    build_headers,
    post_messages,
    read_api_key,
)
from rag_foundry_providers.adapters._ollama_http import load_httpx, map_http_error
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import ProviderCallError, ProviderCallReason, ProviderError
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult
from rag_foundry_providers.ports import check_chat_request
from rag_foundry_providers.usage import (
    Clock,
    Operation,
    TokenSource,
    Usage,
    estimate_cost,
    measure_latency_ms,
)

HAIKU_MODEL = "claude-haiku-5-5"
SONNET_MODEL = "claude-sonnet-5-5"

DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class _ModelInfo:
    context_window: int
    max_output_tokens: int
    cost_in: Decimal
    cost_out: Decimal
    cost_cache_read: Decimal
    cost_cache_write_5m: Decimal
    thinking: dict[str, str]
    default_max_tokens: int
    long_prompt_threshold: int | None  # prompts above this are priced at a rate not in the profile


# USD per MTok. Source: https://platform.claude.com/docs/en/about-claude/pricing and
# https://platform.claude.com/docs/en/models/haiku-5-5/overview, retrieved 2026-11-02 (ADR-0001, P2.3-01 item 8).
# Haiku rates are for prompts up to 100,000 tokens. Context 1M and max output 128K per
# https://platform.claude.com/docs/en/models/overview (item 7; read as 1,000,000 and 128,000).
_PRICES: dict[str, _ModelInfo] = {
    HAIKU_MODEL: _ModelInfo(
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost_in=Decimal("0.10"),
        cost_out=Decimal("0.50"),
        cost_cache_read=Decimal("0.01"),
        cost_cache_write_5m=Decimal("0.125"),
        thinking={"type": "disabled"},
        default_max_tokens=DEFAULT_MAX_TOKENS,
        long_prompt_threshold=100_000,
    ),
    SONNET_MODEL: _ModelInfo(
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost_in=Decimal("2"),
        cost_out=Decimal("10"),
        cost_cache_read=Decimal("0.10"),
        cost_cache_write_5m=Decimal("2.50"),
        thinking={"type": "between_tools"},
        default_max_tokens=16384,
        long_prompt_threshold=None,
    ),
}


def anthropic_chat_profile(id: str, model: str) -> ModelProfile:
    """Chat profile for an Anthropic model.

    tool_calling, structured_output and vision are declared False because this adapter does not
    implement them in P2.3 (P2.4). A model missing from `_PRICES` gets None rates and limits, so
    its cost is None rather than invented.
    """
    info = _PRICES.get(model)
    return ModelProfile(
        id=id,
        provider="anthropic",
        model=model,
        kind=ModelKind.CHAT,
        capabilities=Capabilities(
            streaming=True,
            tool_calling=False,
            structured_output=False,
            vision=False,
            local=False,
            context_window=info.context_window if info else None,
            max_output_tokens=info.max_output_tokens if info else None,
        ),
        cost_per_mtok_in=info.cost_in if info else None,
        cost_per_mtok_out=info.cost_out if info else None,
        cost_per_mtok_cache_read=info.cost_cache_read if info else None,
        cost_per_mtok_cache_write=info.cost_cache_write_5m if info else None,
    )


def anthropic_haiku_profile(id: str = "anthropic-haiku") -> ModelProfile:
    return anthropic_chat_profile(id, HAIKU_MODEL)


def anthropic_sonnet_profile(id: str = "anthropic-sonnet") -> ModelProfile:
    return anthropic_chat_profile(id, SONNET_MODEL)


class UnsupportedRequestError(ProviderError):
    """The request cannot be sent to this model as asked. Raised before any HTTP call."""


# Error event / body types from https://platform.claude.com/docs/en/api/errors (retrieved 2026-11-02).
_ERROR_TYPE_REASON: dict[str, ProviderCallReason] = {
    "invalid_request_error": "invalid_request",
    "authentication_error": "auth",
    "billing_error": "billing",
    "permission_error": "auth",
    "not_found_error": "model_not_found",
    "conflict_error": "invalid_request",
    "request_too_large": "invalid_request",
    "rate_limit_error": "rate_limited",
    "api_error": "server_error",
    "timeout_error": "timeout",
    "overloaded_error": "overloaded",
}


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("bad count")
    return value


class AnthropicChatModel:
    """ChatModel on Anthropic `POST /v1/messages` (complete and SSE stream). Single attempt, no retries.

    The key is read from ANTHROPIC_API_KEY at construction and sent only as a Bearer header.
    Never sends `tools`, `tool_choice`, `temperature` or `cache_control`.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_base: str = "https://api.anthropic.com",
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_s: float = 60.0,
        default_max_tokens: int | None = None,
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        unsupported = [
            name
            for name in ("tool_calling", "structured_output", "vision")
            if getattr(profile.capabilities, name) is True
        ]
        if unsupported:
            raise ProviderError(f"AnthropicChatModel does not implement {unsupported} (profile '{profile.id}')")
        if default_max_tokens is not None and default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be > 0")
        self.profile = profile
        self._key: SecretStr = read_api_key(api_key_env)
        self._api_base = api_base
        self._timeout_s = timeout_s
        self._info = _PRICES.get(profile.model)
        if default_max_tokens is not None:
            self._default_max_tokens = default_max_tokens
        else:
            self._default_max_tokens = self._info.default_max_tokens if self._info else DEFAULT_MAX_TOKENS
        self._transport = transport
        self._clock = clock

    @property
    def default_max_tokens(self) -> int | None:
        """Output cap used when a request gives none (read-only; used for budget worst-case estimates)."""
        return self._default_max_tokens

    def __repr__(self) -> str:
        return f"AnthropicChatModel(profile={self.profile.id!r})"

    def _bad(self) -> ProviderCallError:
        return ProviderCallError(self.profile.id, "bad_response")

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        if request.temperature is not None:
            raise UnsupportedRequestError(
                f"profile '{self.profile.id}': temperature is not accepted by this API version; omit it"
            )
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for m in request.messages:
            if m.role == "system":
                if messages:
                    raise UnsupportedRequestError("system messages are only allowed before the first user message")
                system_parts.append(m.content)
            elif m.role == "tool":
                raise UnsupportedRequestError("tool messages are not supported (tools arrive in P2.4)")
            else:
                messages.append({"role": m.role, "content": m.content})
        if not messages:
            raise UnsupportedRequestError("request needs at least one user or assistant message")
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": request.max_output_tokens
            if request.max_output_tokens is not None
            else self._default_max_tokens,
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self._info is not None:
            payload["thinking"] = dict(self._info.thinking)
        if stream:
            payload["stream"] = True
        return payload

    def _finish(self, stop_reason: Any) -> Literal["stop", "length"]:
        if stop_reason in ("end_turn", "stop_sequence"):
            return "stop"
        if stop_reason in ("max_tokens", "model_context_window_exceeded"):
            return "length"
        if stop_reason == "refusal":
            raise ProviderCallError(self.profile.id, "refused")
        # null, tool_use, pause_turn or unknown: tools are off, so none of these may appear.
        raise self._bad()

    def _usage(self, raw: Any, op: Operation, start: float) -> Usage:
        if not isinstance(raw, dict):
            raise self._bad()
        try:
            tokens_in = _count(raw.get("input_tokens"))
            tokens_out = _count(raw.get("output_tokens"))
            cache_read = _count(raw.get("cache_read_input_tokens")) or 0
            total_write = _count(raw.get("cache_creation_input_tokens"))
            five = one = 0
            detail = raw.get("cache_creation")
            if detail is not None:
                if not isinstance(detail, dict):
                    raise ValueError("bad cache_creation")
                five = _count(detail.get("ephemeral_5m_input_tokens")) or 0
                one = _count(detail.get("ephemeral_1h_input_tokens")) or 0
        except (ValueError, RecursionError):
            raise self._bad() from None
        if tokens_in is None or tokens_out is None:
            raise self._bad()  # never estimate a paid call
        split_ok = isinstance(detail, dict) and (total_write is None or five + one == total_write)
        if total_write is None:
            total_write = five + one
        if one > total_write:
            raise self._bad()
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=op,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cache_read_input_tokens=cache_read,
            cache_write_input_tokens=total_write,
            token_source=TokenSource.PROVIDER,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=self._cost(tokens_in, tokens_out, cache_read, total_write, one, split_ok),
        )

    def _try_usage(self, raw: Any, op: Operation, start: float) -> Usage | None:
        """Usage, or None when the counts are missing or invalid (the caller then raises bad_response)."""
        try:
            return self._usage(raw, op, start)
        except ProviderCallError:
            return None

    def _cost(
        self, tokens_in: int, tokens_out: int, cache_read: int, cache_write: int, write_1h: int, split_ok: bool
    ) -> Decimal | None:
        if write_1h > 0:
            return None  # the 1-hour write rate is not in the profile
        if cache_write > 0 and not split_ok:
            return None  # cannot prove every write was 5-minute, so never use the 5-minute rate
        threshold = self._info.long_prompt_threshold if self._info else None
        if threshold is not None and tokens_in + cache_read + cache_write > threshold:
            return None  # long-prompt tier rate is not in the profile; never use the cheaper rate
        return estimate_cost(
            self.profile, tokens_in, tokens_out, cache_read_tokens=cache_read, cache_write_tokens=cache_write
        )

    async def complete(self, request: ChatRequest) -> ChatResult:
        check_chat_request(self.profile, request, streaming=False)
        payload = self._payload(request, stream=False)
        start = self._clock()
        body = await post_messages(
            self._api_base,
            payload,
            key=self._key,
            timeout_s=self._timeout_s,
            transport=self._transport,
            profile_id=self.profile.id,
        )
        # Build usage first: a billed 200 that fails below (refusal, tool_use, ...) carries it on the error.
        billed = self._try_usage(body.get("usage"), Operation.CHAT, start)
        try:
            content = body.get("content")
            if not isinstance(content, list):
                raise self._bad()
            parts: list[str] = []
            for block in content:
                kind = block.get("type") if isinstance(block, dict) else None
                if kind == "text":
                    text = block.get("text")
                    if not isinstance(text, str):
                        raise self._bad()
                    parts.append(text)
                elif kind in ("thinking", "redacted_thinking"):
                    continue
                else:  # tool_use, server tool blocks, unknown: tools are off
                    raise self._bad()
            finish = self._finish(body.get("stop_reason"))
        except ProviderCallError as exc:
            exc.usage = billed
            raise
        if billed is None:
            raise self._bad()
        return ChatResult(text="".join(parts), finish_reason=finish, usage=billed)

    def _error_event(self, obj: dict[str, Any]) -> ProviderCallError:
        err = obj.get("error")
        etype = err.get("type") if isinstance(err, dict) else None
        reason = _ERROR_TYPE_REASON.get(etype, "bad_response") if isinstance(etype, str) else "bad_response"
        return ProviderCallError(self.profile.id, reason)  # never copies error.message

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        check_chat_request(self.profile, request, streaming=True)
        payload = self._payload(request, stream=True)
        httpx = load_httpx()
        start = self._clock()
        pid = self.profile.id
        client = None
        response = None
        usage_raw: dict[str, Any] = {}
        stop_reason: Any = None
        done = False

        def handle(obj: Any, name: str | None) -> str:
            """Apply one decoded SSE event; return the answer text it carries ("" for none)."""
            nonlocal usage_raw, stop_reason, done
            if not isinstance(obj, dict):
                raise self._bad()
            etype = obj.get("type", name)
            if etype == "error":
                raise self._error_event(obj)
            delta = ""
            if etype == "message_start":
                message = obj.get("message")
                raw = message.get("usage") if isinstance(message, dict) else None
                if not isinstance(raw, dict):
                    raise self._bad()
                usage_raw = dict(raw)
            elif etype == "content_block_start":
                block = obj.get("content_block")
                kind = block.get("type") if isinstance(block, dict) else None
                if kind == "text":
                    initial = block.get("text", "")
                    if not isinstance(initial, str):
                        raise self._bad()
                    delta = initial
                elif kind not in ("thinking", "redacted_thinking"):
                    raise self._bad()  # tool_use and others: tools are off
            elif etype == "content_block_delta":
                d = obj.get("delta")
                dtype = d.get("type") if isinstance(d, dict) else None
                if dtype == "text_delta":
                    text = d.get("text")
                    if not isinstance(text, str):
                        raise self._bad()
                    delta = text
                elif dtype == "input_json_delta":
                    raise self._bad()
                # thinking_delta, signature_delta and unknown delta types are ignored
            elif etype == "message_delta":
                d = obj.get("delta")
                if isinstance(d, dict) and d.get("stop_reason") is not None:
                    stop_reason = d["stop_reason"]
                upd = obj.get("usage")
                if isinstance(upd, dict):
                    # Cumulative counts; null fields (input_tokens may be null) keep the start value.
                    usage_raw.update({k: v for k, v in upd.items() if v is not None})
            elif etype == "message_stop":
                done = True
            # ping, content_block_stop and unknown event types are ignored
            return delta

        def decode(data_lines: list[str]) -> Any:
            try:
                return json.loads("\n".join(data_lines))
            except (ValueError, RecursionError):
                raise self._bad() from None

        try:
            try:
                client = httpx.AsyncClient(
                    base_url=self._api_base, transport=self._transport, trust_env=False, follow_redirects=False
                )
                http_request = client.build_request(
                    "POST",
                    MESSAGES_PATH,
                    json=payload,
                    headers=build_headers(self._key),
                    timeout=httpx.Timeout(self._timeout_s),
                )
                response = await client.send(http_request, stream=True)
                if not 200 <= response.status_code < 300:
                    content = await response.aread()
                    error = anthropic_status_error(pid, response.status_code, content, response.headers)
                    if error is not None:
                        raise error
                event_name: str | None = None
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith(":"):
                            continue  # SSE comment
                        field, _, value = line.partition(":")
                        value = value.removeprefix(" ")
                        if field == "event":
                            event_name = value
                        elif field == "data":
                            data_lines.append(value)
                        continue
                    if not data_lines:  # blank line with nothing pending
                        event_name = None
                        continue
                    delta = handle(decode(data_lines), event_name)
                    event_name, data_lines = None, []
                    if delta:
                        yield ChatChunk(delta=delta)
                    if done:
                        break
                if data_lines and not done:  # final event without a trailing blank line
                    delta = handle(decode(data_lines), event_name)
                    if delta:
                        yield ChatChunk(delta=delta)
            except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as exc:
                raise map_http_error(httpx, exc, pid) from None
            if not done:
                raise self._bad()
            billed = self._try_usage(usage_raw, Operation.CHAT_STREAM, start)
            try:
                self._finish(stop_reason)
            except ProviderCallError as exc:
                exc.usage = billed  # a billed 200 that still failed (refusal and so on)
                raise
            if billed is None:
                raise self._bad()
            usage = billed
        finally:
            if response is not None:
                await response.aclose()
            if client is not None:
                await client.aclose()
        yield ChatChunk(delta="", usage=usage)
