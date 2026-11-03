"""Direct Ollama adapters over HTTP (httpx, optional extra `rag-foundry-providers[ollama]`).

Request and response fields follow https://github.com/ollama/ollama/blob/v0.40.13/docs/api.md
(`POST /api/embed`, `POST /api/chat`), read 2026-11-02.
"""

import json
import math
import time
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any, Literal

from rag_foundry_providers.adapters._ollama_http import (
    estimate_tokens,
    load_httpx,
    map_http_error,
    post_json,
    status_error,
)
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
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


def ollama_embedding_profile(id: str = "local-embed", model: str = "bge-m3") -> ModelProfile:
    """Profile for bge-m3 on Ollama: 1024 dims, 8192 context (docs/plan/FREE_MODELS.md)."""
    zero = Decimal("0")  # no per-token fee; hardware cost not modelled
    return ModelProfile(
        id=id,
        provider="ollama",
        model=model,
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(
            token_embeddings=False,
            multivector=False,
            local=True,
            context_window=8192,
            embedding_dim=1024,
        ),
        cost_per_mtok_in=zero,
        cost_per_mtok_out=zero,
    )


class OllamaEmbeddingModel:
    """EmbeddingModel on Ollama `POST /api/embed`. Dense vectors only, no retries.

    `input_type` is accepted and ignored: bge-m3 has no query/document prefix.
    Requests send `truncate: false`, so Ollama refuses over-long input instead of cutting it.
    api.md documents that error but not its status or body, so it surfaces as `bad_response`.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_base: str,
        timeout_s: float = 60.0,
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.profile = profile
        self._api_base = api_base
        self._timeout_s = timeout_s
        self._transport = transport
        self._clock = clock

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        start = self._clock()
        if not texts:
            return EmbeddingResult(vectors=(), usage=self._usage(0, TokenSource.ESTIMATED, start))
        body = await post_json(
            self._api_base,
            "/api/embed",
            {"model": self.profile.model, "input": list(texts), "truncate": False},
            timeout_s=self._timeout_s,
            transport=self._transport,
            profile_id=self.profile.id,
        )
        vectors = self._validate(body, len(texts))
        count = body.get("prompt_eval_count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            tokens, source = count, TokenSource.PROVIDER
        else:
            tokens, source = sum(estimate_tokens(t) for t in texts), TokenSource.ESTIMATED
        return EmbeddingResult(vectors=vectors, usage=self._usage(tokens, source, start))

    def _usage(self, tokens: int, source: TokenSource, start: float) -> Usage:
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=Operation.EMBED,
            input_tokens=tokens,
            output_tokens=0,
            token_source=source,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=estimate_cost(self.profile, tokens, 0),
        )

    def _validate(self, body: dict[str, Any], expected: int) -> tuple[tuple[float, ...], ...]:
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise ProviderCallError(self.profile.id, "bad_response")
        dim = self.profile.capabilities.embedding_dim
        out: list[tuple[float, ...]] = []
        for vec in embeddings:
            if not isinstance(vec, list) or len(vec) != dim:
                raise ProviderCallError(self.profile.id, "bad_response")
            for v in vec:
                if isinstance(v, bool) or not isinstance(v, int | float):
                    raise ProviderCallError(self.profile.id, "bad_response")
                try:
                    finite = math.isfinite(v)
                except OverflowError:  # huge JSON integer
                    finite = False
                if not finite:
                    raise ProviderCallError(self.profile.id, "bad_response")
            try:
                out.append(tuple(float(v) for v in vec))
            except OverflowError as exc:
                raise ProviderCallError(self.profile.id, "bad_response") from exc
        return tuple(out)


# gemma4:e4b supports up to 131072 tokens ("128K" = 128 * 1024; docs/plan/FREE_MODELS.md). The default
# profile declares an effective window of 8192 that fits a 6 GB GPU (RTX 3060); the adapter sends it as
# options.num_ctx so Ollama does not silently truncate to its own default (2048).
GEMMA4_E4B_MAX_CONTEXT = 131072
DEFAULT_CHAT_CONTEXT = 8192


def ollama_chat_profile(
    id: str = "local-chat", model: str = "gemma4:e4b", *, context_window: int = DEFAULT_CHAT_CONTEXT
) -> ModelProfile:
    """Profile for gemma4:e4b on Ollama chat.

    `context_window` is the effective window sent as `options.num_ctx` (default 8192; the model
    maximum is 131072). tool_calling, structured_output and vision are declared False even though
    the model card lists tools and vision: this adapter does not implement them in P2.2 (P2.4).
    """
    zero = Decimal("0")  # no per-token fee; hardware cost not modelled
    return ModelProfile(
        id=id,
        provider="ollama",
        model=model,
        kind=ModelKind.CHAT,
        capabilities=Capabilities(
            streaming=True,
            tool_calling=False,
            structured_output=False,
            vision=False,
            local=True,
            context_window=context_window,
        ),
        cost_per_mtok_in=zero,
        cost_per_mtok_out=zero,
    )


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class OllamaChatModel:
    """ChatModel on Ollama `POST /api/chat` (complete and NDJSON stream). No retries.

    Sends `think: false` (documented in api.md) so thinking text is never returned as the answer,
    and `options.num_ctx` from the profile. Never sends `tools` or `format`; the constructor refuses
    profiles that declare tool_calling, structured_output or vision.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_base: str,
        timeout_s: float = 120.0,
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        unsupported = [
            name
            for name in ("tool_calling", "structured_output", "vision")
            if getattr(profile.capabilities, name) is True
        ]
        if unsupported:
            raise ProviderError(f"OllamaChatModel does not implement {unsupported} (profile '{profile.id}')")
        self.profile = profile
        self._api_base = api_base
        self._timeout_s = timeout_s
        self._transport = transport
        self._clock = clock

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.profile.capabilities.context_window is not None:
            options["num_ctx"] = self.profile.capabilities.context_window
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
            "think": False,
        }
        if options:
            payload["options"] = options
        return payload

    def _bad(self) -> ProviderCallError:
        return ProviderCallError(self.profile.id, "bad_response")

    def _content(self, obj: dict[str, Any]) -> str:
        message = obj.get("message")
        if not isinstance(message, dict):
            raise self._bad()
        content = message.get("content")
        if not isinstance(content, str):
            raise self._bad()
        return content

    def _finish(self, obj: dict[str, Any]) -> Literal["stop", "length"]:
        reason = obj.get("done_reason")
        if reason == "stop":
            return "stop"
        if reason == "length":
            return "length"
        raise self._bad()

    def _usage(self, obj: dict[str, Any], request: ChatRequest, text: str, op: Operation, start: float) -> Usage:
        tokens_in = _count(obj.get("prompt_eval_count"))
        tokens_out = _count(obj.get("eval_count"))
        if tokens_in is not None and tokens_out is not None:
            source = TokenSource.PROVIDER
        else:
            tokens_in = sum(estimate_tokens(m.content) for m in request.messages)
            tokens_out = estimate_tokens(text)
            source = TokenSource.ESTIMATED
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=op,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            token_source=source,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=estimate_cost(self.profile, tokens_in, tokens_out),
        )

    async def complete(self, request: ChatRequest) -> ChatResult:
        check_chat_request(self.profile, request, streaming=False)
        start = self._clock()
        body = await post_json(
            self._api_base,
            "/api/chat",
            self._payload(request, stream=False),
            timeout_s=self._timeout_s,
            transport=self._transport,
            profile_id=self.profile.id,
        )
        if "error" in body or body.get("done") is not True:
            raise self._bad()
        text = self._content(body)
        finish = self._finish(body)
        return ChatResult(
            text=text,
            finish_reason=finish,
            usage=self._usage(body, request, text, Operation.CHAT, start),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        check_chat_request(self.profile, request, streaming=True)
        httpx = load_httpx()
        start = self._clock()
        pid = self.profile.id
        client = None
        response = None
        try:
            try:
                # Built inside the mapped try: a malformed api_base raises InvalidURL here.
                client = httpx.AsyncClient(
                    base_url=self._api_base, transport=self._transport, trust_env=False, follow_redirects=False
                )
                http_request = client.build_request(
                    "POST",
                    "/api/chat",
                    json=self._payload(request, stream=True),
                    timeout=httpx.Timeout(self._timeout_s),
                )
                response = await client.send(http_request, stream=True)
                error = status_error(pid, response.status_code)
                if error is not None:
                    raise error
                parts: list[str] = []
                final: dict[str, Any] | None = None
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, RecursionError) as exc:
                        raise self._bad() from exc
                    if not isinstance(obj, dict) or "error" in obj:
                        raise self._bad()
                    delta = self._content(obj)
                    done = obj.get("done")
                    if done is not True and done is not False:
                        raise self._bad()
                    if delta:
                        parts.append(delta)
                        yield ChatChunk(delta=delta)
                    if done:
                        final = obj
                        break
            except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as exc:
                raise map_http_error(httpx, exc, pid) from exc
            if final is None:
                raise self._bad()
            self._finish(final)
            usage = self._usage(final, request, "".join(parts), Operation.CHAT_STREAM, start)
        finally:
            if response is not None:
                await response.aclose()
            if client is not None:
                await client.aclose()
        yield ChatChunk(delta="", usage=usage)
