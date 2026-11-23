"""R45 text late-interaction retrieval: ColBERT-style multivector MaxSim over an in-memory index.

Fidelity: ADAPTATION (ADR-0022). Scoring is MaxSim as in Khattab and Zaharia (2020): for every query token
the maximum dot product over the document's token vectors, summed over query tokens (all vectors are
L2-normalised, so the dot product is a cosine). Deviations: the model (bge-m3 ColBERT head) is not the
original ColBERT, the port drops special tokens (official BGE-M3 keeps the final end token), documents are
the stored chunks (not whole passages with [D] markers), and there is no PLAID or other candidate
pruning: every chunk is scored exhaustively. Vectors stay in memory; nothing is persisted or compressed.

Tenant scope: the index stores the tenant and project of its chunk records and refuses any other caller
scope with NotFound before any encoder call. Explanations carry the matched token pairs (the hit's
`score_components`); nothing here logs text.
"""

import math
from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.indexing import ChunkRecord, IndexState, IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import TokenEncoding, TokenEncodingResult
from rag_foundry_providers.ports import TokenEncoder
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent
from rag_foundry_techniques.errors import TechniqueUnavailableError

_UNAVAILABLE = "R45 needs a token encoder adapter and approved local model files"
_PAGE = 500
_BATCH = 32
_MAX_QUERY_CHARS = 2000
_TOP_COMPONENTS = 5


class MultivectorStats(BaseModel):
    """`estimated_bytes` is an estimate (tokens * dim * 4, float32 payload); Python tuples use more."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunks: int
    tokens: int
    dim: int
    estimated_bytes: int


class MultivectorIndex(BaseModel):
    """In-memory token vectors per chunk, bound to the tenant and project of their chunk records.

    `offsets` are code-point spans into each chunk's own text (the text the encoder saw)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tenant_id: str
    project_id: str
    index_version_id: str
    encoder_provider: str
    encoder_model: str
    dim: int
    chunks: tuple[ChunkRecord, ...]
    vectors: dict[str, tuple[tuple[float, ...], ...]]
    offsets: dict[str, tuple[tuple[int, int], ...]]

    def stats(self) -> MultivectorStats:
        tokens = sum(len(v) for v in self.vectors.values())
        return MultivectorStats(
            chunks=len(self.chunks), tokens=tokens, dim=self.dim, estimated_bytes=tokens * self.dim * 4
        )


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0.0:
        return tuple(float(x) for x in vector)
    return tuple(x / norm for x in vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


async def _encode(encoder: TokenEncoder, texts: list[str], *, input_type: str, dim: int | None) -> TokenEncodingResult:
    try:
        result = await encoder.encode_tokens(texts, head="colbert", input_type=input_type)  # type: ignore[arg-type]
    except AdapterDependencyError as exc:
        raise TechniqueUnavailableError(_UNAVAILABLE) from exc
    if len(result.items) != len(texts):
        raise IndexingError("token_encoding_count_mismatch")
    for item, text in zip(result.items, texts, strict=True):
        _check_item(item, len(text), dim if dim is not None else result.dim)
    return result


def _check_item(item: TokenEncoding, text_len: int, dim: int) -> None:
    for _start, end in item.offsets:
        if end > text_len:
            raise IndexingError("token_offsets_invalid")
    for vector in item.vectors:
        if len(vector) != dim:
            raise IndexingError("multivector_dim_mismatch")


async def _load_chunks(scope: Scope, version: IndexVersion, index_repo: IndexRepository) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    offset = 0
    while True:
        page = await index_repo.list_chunks(scope, version.id, offset=offset, limit=_PAGE)
        for chunk in page:
            if chunk.tenant_id != scope.tenant_id or chunk.project_id != scope.project_id:
                raise NotFoundError()
        chunks.extend(page)
        offset += len(page)
        if len(page) < _PAGE:
            return chunks


async def build_multivector_index(
    scope: Scope, version: IndexVersion, index_repo: IndexRepository, encoder: TokenEncoder
) -> MultivectorIndex:
    if version.state is not IndexState.READY:
        raise IndexingError("not_ready")
    chunks = await _load_chunks(scope, version, index_repo)
    if not chunks:
        raise IndexingError("empty_index")
    vectors: dict[str, tuple[tuple[float, ...], ...]] = {}
    offsets: dict[str, tuple[tuple[int, int], ...]] = {}
    dim: int | None = None
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i : i + _BATCH]
        result = await _encode(encoder, [c.text for c in batch], input_type="document", dim=dim)
        if dim is None:
            dim = result.dim
        elif result.dim != dim:
            raise IndexingError("multivector_dim_mismatch")
        for chunk, item in zip(batch, result.items, strict=True):
            if not item.vectors:
                error = IndexingError("late_chunk_span_empty")
                error.chunk_id, error.ordinal = chunk.id, chunk.ordinal  # ids only, never content
                raise error
            vectors[chunk.id] = tuple(_normalise(v) for v in item.vectors)
            offsets[chunk.id] = tuple((int(s), int(e)) for s, e in item.offsets)
    assert dim is not None
    profile = encoder.profile
    return MultivectorIndex(
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        index_version_id=version.id,
        encoder_provider=profile.provider,
        encoder_model=profile.model,
        dim=dim,
        chunks=tuple(chunks),
        vectors=vectors,
        offsets=offsets,
    )


class LateInteractionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)
    max_query_tokens: int = Field(default=32, ge=1, le=512)


class LateInteractionRetriever:
    """R45; exhaustive MaxSim over a `MultivectorIndex` (no PLAID), bound to one scope and index version."""

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R45",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=LateInteractionConfig,
    )

    def __init__(
        self,
        config: LateInteractionConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: MultivectorIndex | None,
        encoder: TokenEncoder | None = None,
    ) -> None:
        self.config, self._scope, self._version, self._index, self._encoder = (
            config,
            scope,
            index_version,
            index,
            encoder,
        )

    async def retrieve(self, query: str) -> RetrievalResult:
        index = self._index
        if index is None:
            raise IndexingError("multivector_index_missing")
        if self._scope.tenant_id != index.tenant_id or self._scope.project_id != index.project_id:
            raise NotFoundError()
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > _MAX_QUERY_CHARS:
            raise IndexingError("query_too_long")
        if self._version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if self._version.id != index.index_version_id:
            raise IndexingError("multivector_index_missing")
        encoder = self._encoder
        if encoder is None:
            raise TechniqueUnavailableError(_UNAVAILABLE)
        if encoder.profile.provider != index.encoder_provider or encoder.profile.model != index.encoder_model:
            raise IndexingError("incompatible_embedding")
        result = await _encode(encoder, [query], input_type="query", dim=index.dim)
        item = result.items[0]
        if not item.vectors:
            raise IndexingError("late_chunk_span_empty")
        limit = self.config.max_query_tokens
        qvecs = [_normalise(v) for v in item.vectors[:limit]]
        qoffsets = item.offsets[:limit]
        scored = []
        for chunk in index.chunks:
            dvecs = index.vectors[chunk.id]
            contributions: list[tuple[float, int, int]] = []  # (value, query token, best doc token)
            for qi, qv in enumerate(qvecs):
                best_j, best = 0, _dot(qv, dvecs[0])
                for j in range(1, len(dvecs)):
                    sim = _dot(qv, dvecs[j])
                    if sim > best:
                        best_j, best = j, sim
                contributions.append((best, qi, best_j))
            scored.append((math.fsum(c[0] for c in contributions), chunk, contributions))
        scored.sort(key=lambda row: (-row[0], row[1].id))
        hits = []
        for rank, (score, chunk, contributions) in enumerate(scored[: self.config.k], start=1):
            top = sorted(contributions, key=lambda c: (-c[0], c[1]))[:_TOP_COMPONENTS]
            doffsets = index.offsets[chunk.id]
            components = tuple(
                ScoreComponent(
                    name="maxsim_token",
                    value=value,
                    term=f"{query[slice(*qoffsets[qi])]}->{chunk.text[slice(*doffsets[dj])]}",
                )
                for value, qi, dj in top
            )
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    extraction_view_id=chunk.extraction_view_id,
                    start=chunk.start,
                    end=chunk.end,
                    block_ids=chunk.block_ids,
                    text=chunk.text,
                    distance=None,
                    score=score,
                    rank=rank,
                    score_components=components,
                    parent_start=chunk.parent_start_cp,
                    parent_end=chunk.parent_end_cp,
                )
            )
        return RetrievalResult(
            technique_id="R45",
            grounding="corpus",
            index_version_id=index.index_version_id,
            hits=tuple(hits),
            scoring="r45-maxsim",
        )
