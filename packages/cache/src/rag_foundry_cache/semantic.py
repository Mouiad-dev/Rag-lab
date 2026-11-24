"""R51 semantic-response cache: reuse only inside the exact R50 partition (ADR-0021).

A hit needs similarity at or above a calibrated threshold, a margin over the runner-up, equal guarded
question features, and every R50 revalidation check. Any store or embed error is a miss plus a fresh run.
Entries hold guarded feature digests and a vector, never the question text. There is no default threshold.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rag_foundry_contracts.runs import RunResponse, RunStatus
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import CacheEntry, ResponseCacheStore, cosine
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.ports import EmbeddingModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest

from rag_foundry_cache.exact import (
    CachedRun,
    CacheTrace,
    bypass_reason,
    check_entry,
    drop_entry,
    pins_match,
    served_copy,
)
from rag_foundry_cache.features import QuestionFeatures, eligible, features
from rag_foundry_cache.keys import ResponseKeyInputs, exact_key, partition_key, source_version_ids
from rag_foundry_cache.revalidate import SourceGate

_LOG = logging.getLogger(__name__)


class SemanticCacheConfig(BaseModel):
    """`threshold` has no default on purpose: it must come from a calibration the caller ran."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(ge=0.5, le=1.0, allow_inf_nan=False)
    margin: float = Field(default=0.05, ge=0.0, le=0.5, allow_inf_nan=False)
    k: int = Field(default=3, ge=2, le=10)
    ttl_s: int = Field(default=86400, ge=1, le=2_592_000)

    @field_validator("threshold", "margin", mode="before")
    @classmethod
    def _no_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("must be a number")
        return value


def _trace(
    outcome: Literal["hit", "miss", "bypass"],
    reason: str | None,
    key: str,
    *,
    similarity: float | None = None,
    source_run_id: str | None = None,
    calls_avoided: int = 0,
    input_tokens_avoided: int = 0,
    output_tokens_avoided: int = 0,
    lookup_ms: float = 0.0,
    revocation_checked: bool = False,
    store_reason: str | None = None,
) -> CacheTrace:
    return CacheTrace(
        layer="R51",
        outcome=outcome,
        reason=None if reason is None else reason[:64],
        key_prefix=key[:12],
        source_run_id=source_run_id,
        calls_avoided=calls_avoided,
        input_tokens_avoided=input_tokens_avoided,
        output_tokens_avoided=output_tokens_avoided,
        lookup_ms=lookup_ms,
        similarity=similarity,
        revocation_checked=revocation_checked,
        store_reason=store_reason,
    )


def _vector_key(embedder: EmbeddingModel) -> str:
    return f"{embedder.profile.id}:{embedder.profile.model}"


def _own_cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    """Cosine computed here; None when it is not defined (length mismatch or zero norm)."""
    if len(a) != len(b) or not any(a) or not any(b):
        return None
    return cosine(a, b)


def _features_reason(names: tuple[str, ...]) -> str:
    """Whole names only, within the 64-character trace limit; the rest is counted as ',+N'."""
    prefix = "features_differ:"
    shown: list[str] = []
    for index, name in enumerate(names):
        rest = len(names) - index - 1
        candidate = prefix + ",".join([*shown, name])
        suffix = f",+{rest}" if rest else ""
        if len(candidate + suffix) > 64:
            break
        shown.append(name)
    else:
        return prefix + ",".join(shown)
    return prefix + ",".join(shown) + f",+{len(names) - len(shown)}"


def _answer_of(entry: CacheEntry) -> str | None:
    """The stored answer text, or None when it cannot be read (then answers count as different)."""
    try:
        return RunResponse.model_validate_json(entry.response_json).answer
    except ValidationError:
        return None


class SemanticResponseCache:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R51",
        kind=PluginKind.CACHE,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SemanticCacheConfig,
    )

    def __init__(self, config: SemanticCacheConfig | Mapping[str, Any], *, store: ResponseCacheStore) -> None:
        try:
            self.config = (
                config if isinstance(config, SemanticCacheConfig) else SemanticCacheConfig.model_validate(dict(config))
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
        embedder: EmbeddingModel,
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
            asked = features(question)
        except CacheError:
            return CachedRun(response=await answer_fn(), trace=_trace("miss", "cache_key_invalid", ""))

        started = clock()
        vector = await self._embed(embedder, question)
        if vector is None:
            response = await answer_fn()
            return CachedRun(
                response=response, trace=_trace("miss", "embed_error", key, store_reason="skipped_no_vector")
            )
        vector_key = _vector_key(embedder)
        reason, hit = await self._lookup(
            scope,
            partition,
            vector_key,
            vector,
            asked,
            started,
            expected_index_version_id=expected_index_version_id,
            anchors=anchors,
            gate=gate,
        )
        lookup_ms = max(0.0, (clock() - started).total_seconds() * 1000.0)
        if hit is not None:
            entry, stored, similarity = hit
            usage = stored.usage
            trace = _trace(
                "hit",
                None,
                key,
                similarity=similarity,
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
            store_reason = await self._store_answer(
                scope, key, partition, inputs, response, asked, vector, vector_key, clock()
            )
        return CachedRun(
            response=response, trace=_trace("miss", reason, key, lookup_ms=lookup_ms, store_reason=store_reason)
        )

    @staticmethod
    async def _embed(embedder: EmbeddingModel, question: str) -> tuple[float, ...] | None:
        try:
            result = await embedder.embed([question], input_type="query")
            if len(result.vectors) != 1:
                return None
            vector = tuple(float(x) for x in result.vectors[0])
        except Exception as exc:
            _LOG.warning("cache embed failed: %s", type(exc).__name__)
            return None
        if not vector or not all(math.isfinite(x) for x in vector):
            return None
        return vector

    async def _lookup(
        self,
        scope: Scope,
        partition: str,
        vector_key: str,
        vector: tuple[float, ...],
        asked: QuestionFeatures,
        now: datetime,
        *,
        expected_index_version_id: str | None,
        anchors: AnchorService,
        gate: SourceGate,
    ) -> tuple[str | None, tuple[CacheEntry, RunResponse, float] | None]:
        try:
            found = await self._store.similar(scope, partition, vector_key, vector, self.config.k)
        except CacheError as exc:
            if exc.code == "cache_scope_mismatch":
                return "scope_mismatch", None
            _LOG.warning("cache similar failed: %s", exc.code)
            return "cache_error", None
        except Exception as exc:
            _LOG.warning("cache similar failed: %s", type(exc).__name__)
            return "cache_error", None

        candidates: list[tuple[CacheEntry, float]] = []
        for entry, _store_score in found:
            if entry.tenant_id != scope.tenant_id or entry.project_id != scope.project_id:
                return "scope_mismatch", None  # never returned, never deleted
            if entry.partition != partition or entry.vector_key != vector_key:
                continue
            if entry.features_json is None or entry.question_vector is None:
                continue
            own = _own_cosine(vector, entry.question_vector)  # the store's score is not trusted
            if own is None:
                continue
            candidates.append((entry, own))
        candidates.sort(key=lambda pair: (-pair[1], pair[0].key))
        if not candidates:
            return "no_candidate", None
        top, top_score = candidates[0]
        if not top_score >= self.config.threshold:
            return "below_threshold", None
        if len(candidates) > 1:
            second, second_score = candidates[1]
            if second_score >= self.config.threshold and top_score - second_score <= self.config.margin:
                top_answer, second_answer = _answer_of(top), _answer_of(second)
                if top_answer is None or second_answer is None or top_answer != second_answer:
                    return "ambiguous", None
        try:
            stored_features = QuestionFeatures.from_json(top.features_json or "")
        except CacheError:
            await drop_entry(self._store, scope, top.key)
            return "entry_invalid", None
        same, names = eligible(asked, stored_features)
        if not same:
            return _features_reason(names), None
        reason, hit = await check_entry(
            self._store,
            scope,
            top.key,
            top,
            now,
            expected_index_version_id=expected_index_version_id,
            anchors=anchors,
            gate=gate,
        )
        if hit is None:
            return reason, None
        return None, (hit[0], hit[1], top_score)

    async def _store_answer(
        self,
        scope: Scope,
        key: str,
        partition: str,
        inputs: ResponseKeyInputs,
        response: RunResponse,
        asked: QuestionFeatures,
        vector: tuple[float, ...],
        vector_key: str,
        now: datetime,
    ) -> str | None:
        if response.status is not RunStatus.ANSWERED or not response.citations:
            return "not_cacheable_status"
        if not pins_match(response, inputs):
            return "pins_mismatch"
        try:
            features_json = asked.to_json()
            if QuestionFeatures.from_json(features_json).to_json() != features_json:
                return "not_cacheable_status"
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
                features_json=features_json,
                question_vector=vector,
                vector_key=vector_key,
            )
        except (ValidationError, CacheError):
            return "not_cacheable_status"
        try:
            await self._store.put(scope, entry)
        except CacheError as exc:
            return "cache_exists" if exc.code == "cache_exists" else "cache_error"
        except Exception as exc:
            _LOG.warning("cache put failed: %s", type(exc).__name__)
            return "cache_error"
        return None


class SplitCounts(BaseModel):
    """Counts for one split at one threshold. Rates are numerator over the stated denominator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paraphrase_total: int
    paraphrase_hits_unguarded: int
    paraphrase_hits_guarded: int
    adversarial_total: int
    false_hits_unguarded: int
    false_hits_guarded: int

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    @property
    def paraphrase_hit_rate_unguarded(self) -> float | None:
        return self._rate(self.paraphrase_hits_unguarded, self.paraphrase_total)

    @property
    def paraphrase_hit_rate_guarded(self) -> float | None:
        return self._rate(self.paraphrase_hits_guarded, self.paraphrase_total)

    @property
    def false_hit_rate_unguarded(self) -> float | None:
        return self._rate(self.false_hits_unguarded, self.adversarial_total)

    @property
    def false_hit_rate_guarded(self) -> float | None:
        return self._rate(self.false_hits_guarded, self.adversarial_total)


class CalibrationRow(BaseModel):
    """One threshold. Only the dev split is reported per threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float
    dev: SplitCounts


class CalibrationReport(BaseModel):
    """`chosen_threshold` is None when no threshold qualifies; the caller must then not use R51 at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[CalibrationRow, ...]
    chosen_threshold: float | None
    reason: str | None
    held_out: SplitCounts | None
    """Held-out counts at the chosen threshold only; None when nothing was chosen."""


VectorPairs = Mapping[str, tuple[Sequence[float], Sequence[float]]]


def _counts(pairs: Sequence[Mapping[str, Any]], vectors: VectorPairs, threshold: float) -> SplitCounts:
    para = para_u = para_g = adv = false_u = false_g = 0
    for pair in pairs:
        va, vb = vectors[pair["id"]]
        close = cosine(tuple(va), tuple(vb)) >= threshold
        same, _names = eligible(features(pair["a"]), features(pair["b"]))
        if pair["kind"] == "paraphrase":
            para += 1
            para_u += close
            para_g += close and same
        else:
            adv += 1
            false_u += close
            false_g += close and same
    return SplitCounts(
        paraphrase_total=para,
        paraphrase_hits_unguarded=para_u,
        paraphrase_hits_guarded=para_g,
        adversarial_total=adv,
        false_hits_unguarded=false_u,
        false_hits_guarded=false_g,
    )


def calibrate(
    pairs: Sequence[Mapping[str, Any]], vectors: VectorPairs, thresholds: Sequence[float]
) -> CalibrationReport:
    """Pure. `vectors` maps a pair id to the embeddings of its `a` and `b` texts.

    The threshold is chosen from the dev split only: the lowest one with zero guarded false hits and at
    least one guarded paraphrase hit. Held-out counts are computed only after the choice.
    """
    grid = sorted(set(thresholds))
    if not grid or any(isinstance(t, bool) or not math.isfinite(t) or not 0.5 <= t <= 1.0 for t in grid):
        raise ValueError("thresholds must be a non-empty set of numbers in 0.5..1.0")
    ids: set[str] = set()
    for pair in pairs:
        if pair["split"] not in ("dev", "held_out") or pair["kind"] not in ("paraphrase", "adversarial"):
            raise ValueError("invalid pair")
        if pair["id"] in ids or pair["id"] not in vectors:
            raise ValueError("duplicate pair id or missing vectors")
        ids.add(pair["id"])
    dev = [p for p in pairs if p["split"] == "dev"]
    held_out = [p for p in pairs if p["split"] == "held_out"]
    rows = tuple(CalibrationRow(threshold=t, dev=_counts(dev, vectors, t)) for t in grid)
    chosen = next(
        (r.threshold for r in rows if r.dev.false_hits_guarded == 0 and r.dev.paraphrase_hits_guarded >= 1), None
    )
    if chosen is None:
        return CalibrationReport(
            rows=rows, chosen_threshold=None, reason="no_threshold_without_dev_false_hits", held_out=None
        )
    return CalibrationReport(
        rows=rows, chosen_threshold=chosen, reason=None, held_out=_counts(held_out, vectors, chosen)
    )


__all__ = [
    "CalibrationReport",
    "CalibrationRow",
    "SemanticCacheConfig",
    "SemanticResponseCache",
    "SplitCounts",
    "calibrate",
]
