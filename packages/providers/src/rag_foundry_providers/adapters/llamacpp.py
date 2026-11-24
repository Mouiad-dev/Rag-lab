"""Raw completion adapter for llama.cpp server's POST /completion endpoint.

httpx is imported lazily from the provider's optional `llamacpp` extra. This
adapter connects to an existing server and never starts or downloads it.
"""

import json
import math
import sys
import time
from decimal import Decimal
from typing import Any, Literal

from rag_foundry_providers.adapters._ollama_http import estimate_tokens, load_httpx, map_http_error
from rag_foundry_providers.capabilities import Capabilities, Capability, ModelKind, ModelProfile
from rag_foundry_providers.errors import CapabilityProblem, MissingCapabilityError, ProviderCallError
from rag_foundry_providers.messages import RawCompletionRequest, RawCompletionResult, TokenLogprob
from rag_foundry_providers.usage import (
    Clock,
    Operation,
    TokenSource,
    Usage,
    estimate_cost,
    measure_latency_ms,
)


def llamacpp_raw_completion_profile(
    id: str = "local-llamacpp-raw-logprobs",
    model: str = "selfrag-local",
    *,
    context_window: int = 2048,
) -> ModelProfile:
    """Opt-in local profile for llama.cpp raw completion log probabilities."""
    zero = Decimal("0")
    return ModelProfile(
        id=id,
        provider="llamacpp",
        model=model,
        kind=ModelKind.CHAT,
        capabilities=Capabilities(raw_logprobs=True, local=True, context_window=context_window),
        cost_per_mtok_in=zero,
        cost_per_mtok_out=zero,
    )


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class LlamaCppRawCompletionModel:
    """RawCompletionModel over llama.cpp server's non-streaming /completion endpoint."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        vocab_size: int | None = None,
        max_response_bytes: int = 64 * 1024 * 1024,
        timeout: float = 120.0,
        base_url: str = "http://127.0.0.1:8091",
        transport: Any = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.profile = profile
        self._vocab_size = vocab_size
        self._max_response_bytes = max_response_bytes
        self._timeout = timeout
        self._base_url = base_url
        self._transport = transport
        self._clock = clock

    def _bad(self, status: int | None = None) -> ProviderCallError:
        return ProviderCallError(self.profile.id, "bad_response", status=status)

    def _number(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) and number <= 1e-6 else None

    def _token_id(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self._bad()
        if self._vocab_size is not None and value >= self._vocab_size:
            raise self._bad()
        return value

    def _tokens(self, body: dict[str, Any]) -> tuple[TokenLogprob, ...]:
        entries = body.get("completion_probabilities")
        if not isinstance(entries, list):
            raise self._bad()
        predicted = _count(body.get("tokens_predicted"))
        if predicted is not None and len(entries) != predicted:
            raise self._bad()
        parsed: list[TokenLogprob] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise self._bad()
            token = entry.get("token")
            logprob = self._number(entry.get("logprob"))
            alternatives = entry.get("top_logprobs")
            if not isinstance(token, str) or logprob is None or not isinstance(alternatives, list):
                raise self._bad()
            token_id = self._token_id(entry.get("id"))
            top: list[tuple[str, float]] = []
            top_ids: list[int] = []
            for alternative in alternatives:
                if not isinstance(alternative, dict):
                    raise self._bad()
                alt_token = alternative.get("token")
                alt_logprob = self._number(alternative.get("logprob"))
                if not isinstance(alt_token, str) or alt_logprob is None:
                    raise self._bad()
                top_ids.append(self._token_id(alternative.get("id")))
                top.append((alt_token, alt_logprob))
            if len(set(top_ids)) != len(top_ids):
                # A repeated id would count the same probability twice in callers that sum groups.
                raise self._bad()
            parsed.append(
                TokenLogprob(
                    token=token,
                    logprob=logprob,
                    top=tuple(top),
                    token_id=token_id,
                    top_ids=tuple(top_ids),
                )
            )
        return tuple(parsed)

    def _usage(self, body: dict[str, Any], request: RawCompletionRequest, text: str, start: float) -> Usage:
        tokens_in = _count(body.get("tokens_evaluated"))
        tokens_out = _count(body.get("tokens_predicted"))
        if tokens_in is not None and tokens_out is not None:
            source = TokenSource.PROVIDER
        else:
            tokens_in = estimate_tokens(request.prompt)
            tokens_out = estimate_tokens(text)
            source = TokenSource.ESTIMATED
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=Operation.CHAT,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            token_source=source,
            latency_ms=measure_latency_ms(self._clock, start),
            estimated_cost_usd=estimate_cost(self.profile, tokens_in, tokens_out),
        )

    @staticmethod
    def _finish(body: dict[str, Any]) -> Literal["stop", "length"]:
        stop_type = body.get("stop_type")
        if stop_type in ("eos", "word"):
            return "stop"
        if stop_type == "limit":
            return "length"
        raise ValueError("invalid stop_type")

    def _status_error(self, status: int) -> ProviderCallError | None:
        if 200 <= status < 300:
            return None
        if status == 404:
            reason = "model_not_found"
        elif status in (401, 403):
            reason = "auth"
        elif status == 402:
            reason = "billing"
        elif status == 429:
            reason = "rate_limited"
        elif status >= 500:
            reason = "server_error"
        else:
            reason = "bad_response"
        return ProviderCallError(self.profile.id, reason, status=status)

    async def complete_raw(self, request: RawCompletionRequest) -> RawCompletionResult:
        if self.profile.declared(Capability.RAW_LOGPROBS) is not True:
            raise MissingCapabilityError(
                (
                    CapabilityProblem(
                        stage=None,
                        profile_id=self.profile.id,
                        capability=Capability.RAW_LOGPROBS.value,
                        declared=self.profile.declared(Capability.RAW_LOGPROBS),
                    ),
                )
            )
        if self._vocab_size is not None and request.top_logprobs > self._vocab_size:
            raise ProviderCallError(self.profile.id, "invalid_request")
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "n_predict": request.max_output_tokens,
            "n_probs": request.top_logprobs,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
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
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        httpx = load_httpx()
        start = self._clock()
        client = None
        response = None
        try:
            client = httpx.AsyncClient(
                base_url=self._base_url,
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
            )
            http_request = client.build_request(
                "POST", "/completion", json=payload, timeout=httpx.Timeout(self._timeout)
            )
            response = await client.send(http_request, stream=True)
            error = self._status_error(response.status_code)
            if error is not None:
                raise error
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > self._max_response_bytes:
                    raise self._bad()
                chunks.append(chunk)
            try:
                body = json.loads(b"".join(chunks))
            except (ValueError, RecursionError, UnicodeDecodeError) as exc:
                raise self._bad(response.status_code) from exc
            if not isinstance(body, dict):
                raise self._bad(response.status_code)
            content = body.get("content")
            if not isinstance(content, str):
                raise self._bad(response.status_code)
            try:
                finish = self._finish(body)
            except ValueError as exc:
                raise self._bad(response.status_code) from exc
            tokens = self._tokens(body)
            return RawCompletionResult(
                text=content,
                tokens=tokens,
                finish_reason=finish,
                usage=self._usage(body, request, content, start),
            )
        except ProviderCallError:
            raise
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise map_http_error(httpx, exc, self.profile.id) from exc
        finally:
            # A close failure must not replace an error that is already being raised.
            in_flight = sys.exc_info()[1] is not None
            for closable in (response, client):
                if closable is None:
                    continue
                try:
                    await closable.aclose()
                except (httpx.HTTPError, httpx.InvalidURL) as exc:
                    if not in_flight:
                        raise map_http_error(httpx, exc, self.profile.id) from exc
