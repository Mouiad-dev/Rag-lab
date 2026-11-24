"""C20 late chunking representation with an early-chunking baseline (same encoder, same pooling).

Fidelity: ADAPTATION (ADR-0022). Late mode encodes each extraction view once at token level (overlapping
macro windows when the view is longer than `window_tokens`) and mean-pools the contextual token vectors
whose start offset lies in each chunk's code-point span `[start, end)`. Early mode encodes every chunk text
alone and mean-pools all of its tokens. Per-chunk encoding is never labelled late chunking. A model trained
for CLS pooling used with mean pooling is an adaptation of the paper's setting.

Overlap rule: windows own disjoint absolute start ranges. For windows i and i+1 the cut is the absolute
start of window i's token at index used_i - ceil(overlap/2) (midpoint of the overlap); window i keeps its
tokens that start before the cut and window i+1 keeps those that start at or after it, so each position is
pooled from exactly one window even if a window starting mid-word tokenizes differently.

Late mode pools from the view text (via the scoped source reader) while early mode encodes `chunk.text`;
they must be the same string (`chunk.text == view_text[start:end]`). The bge-m3 live run (P7.5-02) showed
token offsets start at the word with no leading-space shift, so the `[start, end)` assignment holds.
Vectors stay in memory; nothing is persisted.
"""

import math
from bisect import bisect_left
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.indexing import ChunkRecord, IndexState, IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import TokenEncoding
from rag_foundry_providers.ports import TokenEncoder
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent
from rag_foundry_techniques.errors import TechniqueUnavailableError

from rag_foundry_indexing.context_expansion import SourceTextReader

_UNAVAILABLE = "C20 needs a token encoder adapter and approved local model files"
_PAGE = 500
_EARLY_BATCH = 32
_CHARS_PER_TOKEN_BUDGET = 12  # upper bound of text handed to the encoder per window (it truncates anyway)


class LateChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["late", "early"] = "late"
    window_tokens: int = Field(default=512, ge=1)
    window_overlap_tokens: int = Field(default=64, ge=0)
    k: int = Field(default=5, ge=1, le=50)


class ChunkVectors(BaseModel):
    """In-memory chunk vectors bound to the tenant and project of their chunk records.

    `windows_used` is the number of encoder calls for late mode (one per window) and 0 for early mode;
    `truncated` is True when the encoder cut a text (early mode chunks longer than the encoder limit)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tenant_id: str
    project_id: str
    index_version_id: str
    mode: Literal["late", "early"]
    encoder_provider: str
    encoder_model: str
    dim: int
    chunks: tuple[ChunkRecord, ...]
    vectors: dict[str, tuple[float, ...]]
    token_count: dict[str, int]
    windows_used: int
    truncated: bool


def _mean_pool(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    count = len(vectors)
    return tuple(math.fsum(v[i] for v in vectors) / count for i in range(len(vectors[0])))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(math.fsum(x * x for x in a)), math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def _span_empty(chunk: ChunkRecord) -> IndexingError:
    error = IndexingError("late_chunk_span_empty")
    error.chunk_id, error.ordinal = chunk.id, chunk.ordinal  # ids only, never content
    return error


def _check_encoding(encoding: TokenEncoding, text_len: int) -> None:
    for _start, end in encoding.offsets:
        if end > text_len:
            raise IndexingError("token_offsets_invalid")


async def _encode(encoder: TokenEncoder, texts: list[str], *, input_type: Literal["document", "query"]):
    try:
        result = await encoder.encode_tokens(texts, head="hidden", input_type=input_type)
    except AdapterDependencyError as exc:
        raise TechniqueUnavailableError(_UNAVAILABLE) from exc
    if len(result.items) != len(texts):
        raise IndexingError("token_encoding_count_mismatch")
    for item, text in zip(result.items, texts, strict=True):
        _check_encoding(item, len(text))
    return result


class LateChunkingRepresentation:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C20",
        kind=PluginKind.REPRESENTATION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=LateChunkingConfig,
    )

    def __init__(
        self,
        config: LateChunkingConfig,
        *,
        index: IndexRepository,
        source: SourceTextReader,
        encoder: TokenEncoder | None = None,
    ) -> None:
        self.config, self._index, self._source, self._encoder = config, index, source, encoder

    def searcher(self, vectors: ChunkVectors) -> "LateChunkingSearch":
        return LateChunkingSearch(vectors, self._encoder, k=self.config.k)

    async def build(self, scope: Scope, version: IndexVersion) -> ChunkVectors:
        encoder = self._encoder
        if encoder is None:
            raise TechniqueUnavailableError(_UNAVAILABLE)
        cfg = self.config
        window_bad = cfg.window_tokens > encoder.max_tokens or cfg.window_overlap_tokens >= cfg.window_tokens
        if cfg.mode == "late" and window_bad:  # windows are unused in early mode
            raise IndexingError("late_chunk_window_invalid")
        if version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        chunks = await self._load_chunks(scope, version)
        if not chunks:
            raise IndexingError("empty_index")
        if cfg.mode == "early":
            vectors, counts, windows, truncated = await self._build_early(encoder, chunks)
        else:
            vectors, counts, windows = await self._build_late(encoder, scope, chunks)
            truncated = False
        dim = len(next(iter(vectors.values()))) if vectors else 0
        profile = encoder.profile
        return ChunkVectors(
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            index_version_id=version.id,
            mode=cfg.mode,
            encoder_provider=profile.provider,
            encoder_model=profile.model,
            dim=dim,
            chunks=tuple(chunks),
            vectors=vectors,
            token_count=counts,
            windows_used=windows,
            truncated=truncated,
        )

    async def _load_chunks(self, scope: Scope, version: IndexVersion) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        offset = 0
        while True:
            page = await self._index.list_chunks(scope, version.id, offset=offset, limit=_PAGE)
            for chunk in page:
                if chunk.tenant_id != scope.tenant_id or chunk.project_id != scope.project_id:
                    raise NotFoundError()
            chunks.extend(page)
            offset += len(page)
            if len(page) < _PAGE:
                return chunks

    async def _build_early(self, encoder: TokenEncoder, chunks: list[ChunkRecord]):
        vectors: dict[str, tuple[float, ...]] = {}
        counts: dict[str, int] = {}
        truncated = False
        for i in range(0, len(chunks), _EARLY_BATCH):
            batch = chunks[i : i + _EARLY_BATCH]
            result = await _encode(encoder, [c.text for c in batch], input_type="document")
            for chunk, item in zip(batch, result.items, strict=True):
                if not item.vectors:
                    raise _span_empty(chunk)
                vectors[chunk.id] = _mean_pool(item.vectors)
                counts[chunk.id] = len(item.vectors)
                truncated = truncated or item.truncated
        return vectors, counts, 0, truncated

    async def _build_late(self, encoder: TokenEncoder, scope: Scope, chunks: list[ChunkRecord]):
        by_view: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            by_view.setdefault(chunk.extraction_view_id, []).append(chunk)
        vectors: dict[str, tuple[float, ...]] = {}
        counts: dict[str, int] = {}
        windows_total = 0
        for view_id, view_chunks in by_view.items():
            source = await self._source.view_source(scope, view_id)
            tokens, windows = await self._encode_view(encoder, source.text)
            windows_total += windows
            starts = [t[0] for t in tokens]
            for chunk in view_chunks:
                lo, hi = bisect_left(starts, chunk.start), bisect_left(starts, chunk.end)
                if hi <= lo:
                    raise _span_empty(chunk)
                vectors[chunk.id] = _mean_pool([t[1] for t in tokens[lo:hi]])
                counts[chunk.id] = hi - lo
        return vectors, counts, windows_total

    async def _encode_view(self, encoder: TokenEncoder, text: str) -> tuple[list[tuple[int, tuple[float, ...]]], int]:
        """Return tokens `(absolute_start, vector)` sorted by start, and the number of encoder calls.

        Window i owns the absolute start range [cut_(i-1), cut_i) (0 and infinity at the ends), so every
        token position is taken from exactly one window even when a window that starts mid-word
        re-tokenizes the overlap differently."""
        width, overlap = self.config.window_tokens, self.config.window_overlap_tokens
        half = -(-overlap // 2)
        tokens: list[tuple[int, tuple[float, ...]]] = []
        lower = 0
        calls = 0
        wstart = 0
        while True:
            piece = text[wstart : wstart + width * _CHARS_PER_TOKEN_BUDGET]
            encoding = (await _encode(encoder, [piece], input_type="document")).items[0]
            calls += 1
            count = len(encoding.vectors)
            used = min(count, width)
            reaches_end = wstart + len(piece) >= len(text)
            if not reaches_end and not encoding.truncated and used == count and used > 1:
                if encoding.offsets[used - 1][1] >= len(piece):
                    used -= 1  # the character cap may have cut the last word; the next window re-encodes it
            final = reaches_end and not encoding.truncated and count <= width
            if used == 0 and reaches_end:
                final = True
            if final:
                upper = math.inf
            elif used == 0:
                upper = wstart + len(piece)
            else:
                step = max(1, used - overlap)
                mid = min(used, max(step, used - half))
                cut = encoding.offsets[mid][0] if mid < count else encoding.offsets[used - 1][1]
                advance = encoding.offsets[step][0] if step < count else encoding.offsets[used - 1][1]
                if advance <= 0:
                    advance = max(encoding.offsets[used - 1][1], 1)
                upper = wstart + max(cut, advance)
            upper = max(upper, lower)
            for i in range(used):
                start = wstart + encoding.offsets[i][0]
                if lower <= start < upper:
                    tokens.append((start, encoding.vectors[i]))
            if final:
                break
            if used == 0:
                wstart += len(piece)
            else:
                wstart += advance
            lower = upper
        return tokens, calls


class LateChunkingSearch:
    """Helper (not registered): in-memory cosine search over `ChunkVectors`, bound to their scope."""

    def __init__(self, vectors: ChunkVectors, encoder: TokenEncoder | None, *, k: int = 5) -> None:
        self._vectors, self._encoder, self._k = vectors, encoder, k

    async def search(self, scope: Scope, query: str) -> RetrievalResult:
        v = self._vectors
        if scope.tenant_id != v.tenant_id or scope.project_id != v.project_id:
            raise NotFoundError()
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        encoder = self._encoder
        if encoder is None:
            raise TechniqueUnavailableError(_UNAVAILABLE)
        if encoder.profile.provider != v.encoder_provider or encoder.profile.model != v.encoder_model:
            raise IndexingError("incompatible_embedding")
        item = (await _encode(encoder, [query], input_type="query")).items[0]
        if not item.vectors:
            raise IndexingError("late_chunk_span_empty")
        qvec = _mean_pool(item.vectors)
        if len(qvec) != v.dim:
            raise IndexingError("incompatible_embedding")
        name = "late_cosine" if v.mode == "late" else "early_cosine"
        scored = sorted(
            ((_cosine(qvec, v.vectors[c.id]), c) for c in v.chunks),
            key=lambda pair: (-pair[0], pair[1].extraction_view_id, pair[1].ordinal),
        )[: self._k]
        hits = tuple(
            RetrievedChunk(
                chunk_id=chunk.id,
                extraction_view_id=chunk.extraction_view_id,
                start=chunk.start,
                end=chunk.end,
                block_ids=chunk.block_ids,
                text=chunk.text,
                distance=1.0 - score,
                score=score,
                rank=rank,
                score_components=(ScoreComponent(name=name, value=score),),
                parent_start=chunk.parent_start_cp,
                parent_end=chunk.parent_end_cp,
            )
            for rank, (score, chunk) in enumerate(scored, start=1)
        )
        return RetrievalResult(
            technique_id="C20",
            grounding="corpus",
            index_version_id=v.index_version_id,
            hits=hits,
            scoring=f"c20-{v.mode}-mean-pool",
        )
