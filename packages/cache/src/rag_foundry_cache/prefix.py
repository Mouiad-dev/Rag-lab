"""R49 prefix-cache probe: a fixed cold/warm/changed/eviction experiment through the ChatModel port.

Timings are a proxy. Only provider-reported cached-token counts can support a cache-hit claim;
absent counts are "unavailable", never "0 hits". Reports hold hashes and numbers, no prompt text.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rag_foundry_domain.errors import CacheError
from rag_foundry_providers.messages import ChatMessage, ChatRequest
from rag_foundry_providers.ports import ChatModel
from rag_foundry_providers.usage import ProviderTimings, Usage
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest

ArmName = Literal["cold", "warm", "changed", "eviction_attempt"]
ALL_ARMS: tuple[ArmName, ...] = ("cold", "warm", "changed", "eviction_attempt")

PROBE_NOTE = "timings are a proxy; a lower warm prefill alone does not prove a cache hit"


class PrefixProbeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix_text: str = Field(min_length=1, max_length=200_000)
    questions: tuple[str, ...] = Field(min_length=2, max_length=10)
    max_output_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float | None = Field(default=0.0, ge=0)  # None omits it (Anthropic refuses temperature)
    mode: Literal["complete", "stream"] = "complete"
    arms: tuple[ArmName, ...] = Field(default=ALL_ARMS, min_length=1)
    cache_breakpoint: Literal["none", "system"] = "none"

    @field_validator("questions")
    @classmethod
    def _non_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not question.strip() for question in value):
            raise ValueError("questions must not be blank")
        return value

    @field_validator("arms")
    @classmethod
    def _unique_arms(cls, value: tuple[ArmName, ...]) -> tuple[ArmName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("arms must be unique")
        return value


class ProbeCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: ArmName
    label: str
    ttft_ms: float | None
    total_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None
    cache_read_reported: bool
    timings: ProviderTimings | None
    token_source: str | None
    prefix_sha256: str


class ArmSummary(BaseModel):
    """Medians over one arm; None when any value in the arm is missing (never 0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: ArmName
    calls: int
    ttft_ms_median: float | None
    prefill_ms_median: float | None
    decode_ms_median: float | None
    cache_read_tokens_median: float | None


class PrefixProbeReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["complete", "stream"]
    records: tuple[ProbeCallRecord, ...]
    arms: tuple[ArmSummary, ...]
    cache_metric: Literal["reported", "unavailable"]
    note: str = PROBE_NOTE


def _median(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(statistics.median(v for v in values if v is not None))


def _cache_read_median(records: list[ProbeCallRecord]) -> float | None:
    for record in records:
        if not record.cache_read_reported and not record.cache_read_input_tokens:
            return None
    return _median([record.cache_read_input_tokens for record in records])


def _summarise(arm: ArmName, records: list[ProbeCallRecord]) -> ArmSummary:
    return ArmSummary(
        arm=arm,
        calls=len(records),
        ttft_ms_median=_median([r.ttft_ms for r in records]),
        prefill_ms_median=_median([r.timings.prefill_ms if r.timings else None for r in records]),
        decode_ms_median=_median([r.timings.decode_ms if r.timings else None for r in records]),
        cache_read_tokens_median=_cache_read_median(records),
    )


def _is_reported(record: ProbeCallRecord) -> bool:
    return (
        record.cache_read_reported
        or (record.cache_read_input_tokens or 0) > 0
        or (record.cache_write_input_tokens or 0) > 0
    )


def _swap_first(text: str) -> str:
    return ("Y" if text[0] == "X" else "X") + text[1:]


def _other_prefix(text: str) -> str:
    reversed_text = text[::-1]
    return reversed_text if reversed_text != text else _swap_first(text)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrefixCacheProbe:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R49",
        kind=PluginKind.CACHE,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=PrefixProbeConfig,
    )

    def __init__(self, config: PrefixProbeConfig | Mapping[str, Any]) -> None:
        try:
            self.config = (
                config if isinstance(config, PrefixProbeConfig) else PrefixProbeConfig.model_validate(dict(config))
            )
        except (ValidationError, TypeError):
            raise CacheError("prefix_probe_invalid") from None

    async def _call(
        self,
        chat: ChatModel,
        clock: Callable[[], float],
        arm: ArmName,
        label: str,
        system: str,
        question: str,
    ) -> ProbeCallRecord:
        cfg = self.config
        request = ChatRequest(
            messages=(ChatMessage(role="system", content=system), ChatMessage(role="user", content=question)),
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            cache_breakpoint=cfg.cache_breakpoint,
        )
        start = clock()
        ttft: float | None = None
        usage: Usage | None
        if cfg.mode == "complete":
            usage = (await chat.complete(request)).usage
        else:
            usage = None
            async for chunk in chat.stream(request):
                if ttft is None and chunk.delta:
                    ttft = (clock() - start) * 1000.0
                if chunk.usage is not None:
                    usage = chunk.usage
        total_ms = (clock() - start) * 1000.0
        return ProbeCallRecord(
            arm=arm,
            label=label,
            ttft_ms=ttft,
            total_ms=total_ms,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            cache_read_input_tokens=usage.cache_read_input_tokens if usage else None,
            cache_write_input_tokens=usage.cache_write_input_tokens if usage else None,
            cache_read_reported=usage.cache_read_reported if usage else False,
            timings=usage.timings if usage else None,
            token_source=usage.token_source.value if usage else None,
            prefix_sha256=_sha(system),
        )

    async def run(
        self,
        chat: ChatModel,
        *,
        clock: Callable[[], float],
        salt_factory: Callable[[], str],
    ) -> PrefixProbeReport:
        cfg = self.config
        prefix, questions = cfg.prefix_text, cfg.questions
        records: list[ProbeCallRecord] = []

        async def call(arm: ArmName, label: str, system: str, question: str) -> None:
            records.append(await self._call(chat, clock, arm, label, system, question))

        if "cold" in cfg.arms:
            for i, question in enumerate(questions, 1):
                await call("cold", f"q{i}", f"[salt {salt_factory()}]\n{prefix}", question)
        if "warm" in cfg.arms:
            await call("warm", "prime", prefix, questions[0])
            for i, question in enumerate(questions, 1):
                await call("warm", f"q{i}", prefix, question)
        if "changed" in cfg.arms:
            await call("changed", "q1", _swap_first(prefix), questions[0])
        if "eviction_attempt" in cfg.arms:
            await call("eviction_attempt", "other_prefix", _other_prefix(prefix), questions[0])
            await call("eviction_attempt", "original_prefix", prefix, questions[0])

        summaries = tuple(_summarise(arm, [r for r in records if r.arm == arm]) for arm in ALL_ARMS if arm in cfg.arms)
        metric = "reported" if any(_is_reported(r) for r in records) else "unavailable"
        return PrefixProbeReport(mode=cfg.mode, records=tuple(records), arms=summaries, cache_metric=metric)


def register_builtin_plugins(registry) -> None:
    registry.register(PrefixCacheProbe)
