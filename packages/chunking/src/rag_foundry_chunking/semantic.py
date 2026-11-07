"""C15 semantic boundaries from neighbouring sentence embeddings."""

from __future__ import annotations

import json
import math
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rag_foundry_providers.capabilities import ModelKind
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.model_based import (
    ModelChunkContext,
    ModelChunkingError,
    ModelChunkReport,
    ModelChunkResult,
    build_units,
    segmentation_cache_key,
    segments_to_spans,
    unavailable_sync,
)


class SemanticConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=800, ge=32, le=8000)
    buffer_size: int = Field(default=1, ge=0, le=3)
    threshold_type: Literal["percentile", "std_dev", "absolute"] = "percentile"
    threshold_amount: float = 90.0
    embed_batch_size: int = Field(default=32, ge=1, le=256)
    max_units: int = Field(default=5000, ge=1, le=50000)

    @field_validator("threshold_amount")
    @classmethod
    def valid_threshold(cls, value: float, info) -> float:
        kind = info.data.get("threshold_type", "percentile")
        limit = {"percentile": 100, "std_dev": 10, "absolute": 2}.get(kind)
        minimum = 0
        if value < minimum or (limit is not None and value > limit) or not math.isfinite(value):
            raise ValueError("threshold_amount outside range for threshold_type")
        return value


class SemanticChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C15",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SemanticConfig,
    )

    def __init__(self, config: SemanticConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput):
        unavailable_sync("C15")

    async def chunk_with_models(self, doc: ChunkInput, ctx: ModelChunkContext) -> ModelChunkResult:
        embedder = ctx.embedder
        if embedder is None:
            raise ModelChunkingError("missing_model")
        if embedder.profile.kind is not ModelKind.EMBEDDING:
            raise ModelChunkingError("wrong_model_kind")
        text = doc.text
        units = build_units(text, self.config.chunk_size)
        n = len(units)
        if n > self.config.max_units:
            raise ModelChunkingError("too_many_units")
        key = segmentation_cache_key(
            technique_id="C15",
            plugin_version="1.0.0",
            config=self.config,
            text=text,
            profile=embedder.profile,
            prompt=None,
            sample=ctx.sample,
        )
        usage = ()
        cache_hit = False
        details: dict = {}
        distances: list[float] = []
        calls = 0
        if n > 1:
            cached = ctx.cache.get(key)
            if cached is not None:
                try:
                    candidate = json.loads(cached)
                    values = candidate.get("distances") if isinstance(candidate, dict) else None
                    valid = (
                        isinstance(values, list)
                        and len(values) == n - 1
                        and all(
                            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 2
                            for v in values
                        )
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    valid = False
                if valid:
                    distances = [float(v) for v in values]
                    cache_hit = True
                else:
                    details["cache_invalid"] = True
            if not cache_hit:
                windows = [
                    text[
                        units[max(0, i - self.config.buffer_size)][0] : units[min(n - 1, i + self.config.buffer_size)][
                            1
                        ]
                    ]
                    for i in range(n)
                ]
                vectors = []
                used = []
                for start in range(0, n, self.config.embed_batch_size):
                    batch = windows[start : start + self.config.embed_batch_size]
                    result = await embedder.embed(batch, input_type="document")
                    calls += 1
                    used.append(result.usage)
                    if len(result.vectors) != len(batch):
                        raise ModelChunkingError("embedding_count_mismatch")
                    vectors.extend(result.vectors)
                usage = tuple(used)
                for left, right in zip(vectors, vectors[1:], strict=False):
                    if len(left) != len(right):
                        raise ModelChunkingError("embedding_count_mismatch")
                    denom = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
                    cosine = sum(x * y for x, y in zip(left, right, strict=False)) / denom if denom else 0.0
                    distances.append(min(2.0, max(0.0, 1.0 - cosine)))
                ctx.cache.put(key, json.dumps({"distances": distances}, separators=(",", ":")))
        ordered = sorted(distances)
        if not ordered:
            threshold = self.config.threshold_amount
        elif self.config.threshold_type == "absolute":
            threshold = self.config.threshold_amount
        elif self.config.threshold_type == "std_dev":
            mean = sum(ordered) / len(ordered)
            std = math.sqrt(sum((v - mean) ** 2 for v in ordered) / len(ordered))
            threshold = mean + self.config.threshold_amount * std
        else:
            position = self.config.threshold_amount / 100 * (len(ordered) - 1)
            low = math.floor(position)
            high = math.ceil(position)
            threshold = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        breaks = [i for i, distance in enumerate(distances) if distance > threshold]
        spans = segments_to_spans(text, units, breaks, self.config.chunk_size)
        details.update({"threshold": threshold, "distances": distances, "break_after": breaks, "unit_count": n})
        report = ModelChunkReport(
            technique_id="C15",
            cache_key=key,
            cache_hit=cache_hit,
            model_calls=calls,
            usage=usage,
            model_profile_id=embedder.profile.id,
            prompt_id=None,
            prompt_sha256=None,
            details=details,
        )
        return ModelChunkResult(spans=spans, report=report)
