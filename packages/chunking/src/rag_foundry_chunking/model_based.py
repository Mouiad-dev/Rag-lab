"""Shared infrastructure for chunkers that use model ports."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, NoReturn, Protocol

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.prompts import PromptTemplate
from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.ports import ChatModel, EmbeddingModel
from rag_foundry_providers.usage import Usage
from rag_foundry_techniques import ChunkSpan, TechniqueUnavailableError

from rag_foundry_chunking.boundaries import DEFAULT_ABBREVIATIONS, pack, safe_cut_ranges, sentence_ranges, word_ranges


class ModelChunkingError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def truncate_detail(value: Any) -> str:
    return repr(value)[:200]


class Rejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    call_index: int = Field(ge=0)
    reason: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    detail: str = Field(max_length=200)


class ModelChunkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    technique_id: str
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_hit: bool
    model_calls: int = Field(ge=0)
    usage: tuple[Usage, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    fallback_calls: tuple[int, ...] = ()
    model_profile_id: str
    prompt_id: str | None = None
    prompt_sha256: str | None = None
    details: dict[str, Any]

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((item.estimated_cost_usd or Decimal(0) for item in self.usage), Decimal(0))


class ModelChunkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    spans: tuple[ChunkSpan, ...]
    report: ModelChunkReport


@dataclass(frozen=True)
class ModelChunkContext:
    cache: SegmentationCache = field(kw_only=True)
    chat: ChatModel | None = None
    embedder: EmbeddingModel | None = None
    prompt: PromptTemplate | None = None
    sample: int = 0

    def __post_init__(self) -> None:
        if self.sample < 0:
            raise ValueError("sample must be >= 0")


class SegmentationCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...


class InMemorySegmentationCache:
    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        if key not in self._values:
            self.misses += 1
            return None
        self.hits += 1
        self._values.move_to_end(key)
        return self._values[key]

    def put(self, key: str, value: str) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def segmentation_cache_key(
    *,
    technique_id: str,
    plugin_version: str,
    config: BaseModel,
    text: str,
    profile: ModelProfile,
    prompt: PromptTemplate | None,
    sample: int,
) -> str:
    config_hash = hashlib.sha256(_canonical(config.model_dump(mode="json")).encode("utf-8")).hexdigest()
    payload = {
        "v": 1,
        "technique_id": technique_id,
        "plugin_version": plugin_version,
        "config_sha256": config_hash,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "provider": profile.provider,
        "model": profile.model,
        "profile_id": profile.id,
        "prompt_id": prompt.id if prompt else None,
        "prompt_sha256": prompt.content_sha256 if prompt else None,
        "sample": sample,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any] | None:
    if len(text) > 100_000:
        return None
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0] not in ("```", "```json") or lines[-1] != "```":
            return None
        value = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def make_nonce(text: str, *others: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    counter = 0
    while True:
        nonce = hashlib.sha256(f"{digest}:{counter}".encode("ascii")).hexdigest()[:16]
        if all(nonce not in value for value in (text, *others)):
            return nonce
        counter += 1


def render_units(units: Sequence[tuple[int, str]], nonce: str) -> str:
    return "".join(f"<<<UNIT {number} {nonce}>>>\n{unit}\n<<<END UNIT {number} {nonce}>>>\n" for number, unit in units)


def build_units(text: str, chunk_size: int, abbreviations=DEFAULT_ABBREVIATIONS) -> list[tuple[int, int, int]]:
    units: list[tuple[int, int, int]] = []
    for start, end in sentence_ranges(text, 0, len(text), abbreviations):
        if end - start <= chunk_size:
            units.append((start, end, 0))
            continue
        for word_start, word_end in word_ranges(text, start, end):
            if word_end - word_start <= chunk_size:
                units.append((word_start, word_end, 1))
            else:
                units.extend((a, b, 2) for a, b in safe_cut_ranges(text, word_start, word_end, chunk_size))
    return units


def segments_to_spans(
    text: str, units: Sequence[tuple[int, int, int]], break_after: Iterable[int], chunk_size: int
) -> tuple[ChunkSpan, ...]:
    breaks = set(break_after)
    if any(index < 0 or index >= len(units) - 1 for index in breaks):
        raise ValueError("break index outside unit range")
    spans: list[ChunkSpan] = []
    begin = 0
    for end in sorted(breaks) + [len(units) - 1] if units else []:
        segment = list(units[begin : end + 1])
        packed = pack(text, segment, chunk_size, 0)
        for start, finish, _level in packed:
            all_sentences = all(unit[2] == 0 for unit in segment if unit[0] >= start and unit[1] <= finish)
            whole = start == segment[0][0] and finish == segment[-1][1]
            unit_levels = [unit[2] for unit in segment if unit[0] >= start and unit[1] <= finish]
            split_level = 0 if whole and all_sentences else 1 + max(unit_levels)
            spans.append(ChunkSpan(start=start, end=finish, split_level=split_level))
        begin = end + 1
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def unavailable_sync(technique_id: str) -> NoReturn:
    raise TechniqueUnavailableError(f"{technique_id} needs model ports; call chunk_with_models(doc, ctx)")
