"""R50 exact-response cache: a hit is served only after every revalidation check passes (ADR-0021).

Any failed check or store error is a miss followed by a fresh run. The question text is never stored.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rag_foundry_contracts.runs import RunResponse, RunStatus, RunUsage
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import CacheEntry, ResponseCacheStore
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest

from rag_foundry_cache.keys import ResponseKeyInputs, exact_key, partition_key, source_version_ids
from rag_foundry_cache.revalidate import SourceGate, revalidate

_LOG = logging.getLogger(__name__)


class CacheTrace(BaseModel):
    """What the cache did for one request. Separate from RunResponse. Holds no question text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer: Literal["R50", "R51"]
    outcome: Literal["hit", "miss", "bypass"]
    reason: str | None = Field(default=None, max_length=64)
    key_prefix: str = Field(pattern=r"^([0-9a-f]{12})?$")
    source_run_id: str | None = None
    calls_avoided: int = Field(default=0, ge=0)
    input_tokens_avoided: int = Field(default=0, ge=0)
    output_tokens_avoided: int = Field(default=0, ge=0)
    lookup_ms: float = Field(default=0.0, ge=0)
    similarity: float | None = None
    revocation_checked: bool = False
    """True only on a hit whose SourceGate really checks revocation (AllowAllSourceGate does not)."""
    store_reason: str | None = Field(default=None, max_length=64)
    """Why the fresh answer was not stored (or the put failed); None when stored or not attempted."""


class CachedRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    response: RunResponse
    trace: CacheTrace


class ExactCacheConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ttl_s: int = Field(default=86400, ge=1, le=2_592_000)


def bypass_reason(inputs: ResponseKeyInputs) -> str | None:
    if inputs.freshness == "current":
        return "freshness_current"
    if inputs.mode == "enriched":
        return "enriched_mode"
    if inputs.tools:
        return "tools"
    if inputs.is_action_run:
        return "action_run"
    return None


def pins_match(response: RunResponse, inputs: ResponseKeyInputs) -> bool:
    p, g = response.pins, inputs.generation
    return (
        p.technique_id == inputs.technique_id
        and p.technique_version == inputs.technique_version
        and p.index_version_id == inputs.index_version_id
        and p.embedding_key == inputs.embedding_key
        and p.chat_profile_id == inputs.chat_profile_id
        and p.chat_model == inputs.chat_model
        and p.prompt_sha256 == inputs.prompt_sha256
        and p.min_score == g.min_score
        and p.k == g.k
        and p.settings_sha256 == inputs.settings_sha256
    )


def served_copy(response: RunResponse, run_id: str) -> RunResponse:
    """A hit costs nothing: new run id, zeroed usage."""
    cost = None if response.usage.estimated_cost_usd is None else Decimal(0)
    usage = RunUsage(model_calls=0, input_tokens=0, output_tokens=0, estimated_cost_usd=cost)
    return response.model_copy(update={"run_id": run_id, "usage": usage})


async def drop_entry(store: ResponseCacheStore, scope: Scope, key: str) -> None:
    try:
        await store.delete(scope, key)
    except Exception as exc:
        _LOG.warning("cache delete failed: %s", type(exc).__name__)


async def check_entry(
    store: ResponseCacheStore,
    scope: Scope,
    key: str,
    entry: CacheEntry,
    now: datetime,
    *,
    expected_index_version_id: str | None,
    anchors: AnchorService,
    gate: SourceGate,
) -> tuple[str | None, tuple[CacheEntry, RunResponse] | None]:
    """Shared by R50 and R51: scope, parse, revalidate. Returns (miss reason, None) or (None, (entry, response)).

    A foreign-scope entry is never parsed and never deleted. Any other failed check deletes the entry.
    """
    if entry.tenant_id != scope.tenant_id or entry.project_id != scope.project_id:
        return "scope_mismatch", None
    try:
        stored = RunResponse.model_validate_json(entry.response_json)
    except ValidationError:
        await drop_entry(store, scope, key)
        return "entry_invalid", None
    try:
        reason = await revalidate(
            scope,
            entry,
            stored,
            now=now,
            expected_index_version_id=expected_index_version_id,
            anchors=anchors,
            gate=gate,
        )
    except Exception as exc:
        _LOG.warning("cache revalidation failed: %s", type(exc).__name__)
        return "cache_error", None
    if reason is None:
        return None, (entry, stored)
    if reason != "scope_mismatch":
        await drop_entry(store, scope, key)
    return reason, None


class ExactResponseCache:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R50",
        kind=PluginKind.CACHE,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ExactCacheConfig,
    )

    def __init__(self, config: ExactCacheConfig | Mapping[str, Any], *, store: ResponseCacheStore) -> None:
        try:
            self.config = (
                config if isinstance(config, ExactCacheConfig) else ExactCacheConfig.model_validate(dict(config))
            )
        except (ValidationError, TypeError):
            raise CacheError("cache_policy_invalid") from None
        self._store = store

    async def invalidate_sources(self, scope: Scope, version_ids: tuple[str, ...]) -> int:
        return await self._store.invalidate_sources(scope, version_ids)

    async def run(
        self,
        scope: Scope,
        question: str,
        inputs: ResponseKeyInputs,
        answer_fn: Callable[[], Awaitable[RunResponse]],
        *,
        expected_index_version_id: str | None,
        anchors: AnchorService,
        gate: SourceGate,
        clock: Callable[[], datetime],
        run_id_factory: Callable[[], str],
    ) -> CachedRun:
        bypass = bypass_reason(inputs)
        if bypass is not None:
            return CachedRun(response=await answer_fn(), trace=_trace("bypass", bypass, ""))
        try:
            key = exact_key(scope, question, inputs)
            partition = partition_key(scope, inputs)
        except CacheError:
            return CachedRun(response=await answer_fn(), trace=_trace("miss", "cache_key_invalid", ""))

        started = clock()
        reason, hit = await self._lookup(
            scope, key, started, expected_index_version_id=expected_index_version_id, anchors=anchors, gate=gate
        )
        lookup_ms = max(0.0, (clock() - started).total_seconds() * 1000.0)
        if hit is not None:
            entry, stored = hit
            usage = stored.usage
            trace = _trace(
                "hit",
                None,
                key,
                source_run_id=entry.source_run_id,
                calls_avoided=usage.model_calls,
                input_tokens_avoided=usage.input_tokens,
                output_tokens_avoided=usage.output_tokens,
                lookup_ms=lookup_ms,
                revocation_checked=gate.checks_revocation,
            )
            return CachedRun(response=served_copy(stored, run_id_factory()), trace=trace)

        response = await answer_fn()
        if reason == "scope_mismatch":
            store_reason = "skipped_scope_mismatch"
        else:
            store_reason = await self._store_answer(scope, key, partition, inputs, response, clock())
        return CachedRun(
            response=response, trace=_trace("miss", reason, key, lookup_ms=lookup_ms, store_reason=store_reason)
        )

    async def _lookup(
        self,
        scope: Scope,
        key: str,
        now: datetime,
        *,
        expected_index_version_id: str | None,
        anchors: AnchorService,
        gate: SourceGate,
    ) -> tuple[str | None, tuple[CacheEntry, RunResponse] | None]:
        """Return (miss reason, None) or (None, (entry, stored response))."""
        try:
            entry = await self._store.get(scope, key)
        except CacheError as exc:
            if exc.code == "cache_scope_mismatch":
                return "scope_mismatch", None  # the foreign entry is never returned and never deleted
            _LOG.warning("cache lookup failed: %s", exc.code)
            return "cache_error", None
        except Exception as exc:
            _LOG.warning("cache lookup failed: %s", type(exc).__name__)
            return "cache_error", None
        if entry is None:
            return "not_found", None
        return await check_entry(
            self._store,
            scope,
            key,
            entry,
            now,
            expected_index_version_id=expected_index_version_id,
            anchors=anchors,
            gate=gate,
        )

    async def _store_answer(
        self,
        scope: Scope,
        key: str,
        partition: str,
        inputs: ResponseKeyInputs,
        response: RunResponse,
        now: datetime,
    ) -> str | None:
        if response.status is not RunStatus.ANSWERED or not response.citations:
            return "not_cacheable_status"
        if not pins_match(response, inputs):
            return "pins_mismatch"
        try:
            entry = CacheEntry(
                key=key,
                partition=partition,
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
                response_json=response.model_dump_json(),
                source_run_id=response.run_id,
                source_version_ids=source_version_ids(response),
                created_at=now,
                expires_at=now + timedelta(seconds=self.config.ttl_s),
            )
        except ValidationError:
            return "not_cacheable_status"
        try:
            await self._store.put(scope, entry)
        except CacheError as exc:
            return "cache_exists" if exc.code == "cache_exists" else "cache_error"
        except Exception as exc:
            _LOG.warning("cache put failed: %s", type(exc).__name__)
            return "cache_error"
        return None


def _trace(
    outcome: Literal["hit", "miss", "bypass"],
    reason: str | None,
    key: str,
    *,
    source_run_id: str | None = None,
    calls_avoided: int = 0,
    input_tokens_avoided: int = 0,
    output_tokens_avoided: int = 0,
    lookup_ms: float = 0.0,
    revocation_checked: bool = False,
    store_reason: str | None = None,
) -> CacheTrace:
    return CacheTrace(
        layer="R50",
        outcome=outcome,
        reason=reason,
        key_prefix=key[:12],
        source_run_id=source_run_id,
        calls_avoided=calls_avoided,
        input_tokens_avoided=input_tokens_avoided,
        output_tokens_avoided=output_tokens_avoided,
        lookup_ms=lookup_ms,
        revocation_checked=revocation_checked,
        store_reason=store_reason,
    )
