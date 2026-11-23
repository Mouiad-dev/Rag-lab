"""R41 visual page retrieval: ColPali-style patch multivectors and MaxSim over an in-memory page index.

Fidelity: ADAPTATION (ADR-0022). Pages are rendered to PNG and encoded into patch vectors; a query is encoded into
token vectors; a page scores the sum over query tokens of the maximum dot product over its patch vectors (MaxSim
as in Faysse et al., 2024, "ColPali"). Deviations: the local colSmol-256M model with no token pooling, exhaustive
scoring (no ANN), vectors only in memory, nothing persisted.

Provenance: the caller supplies PDF bytes per extraction view. They are indexed only when sha256(bytes) equals
the scoped `DocumentVersion.sha256`; the text blocks are then re-derived from those verified bytes and must
reproduce the view's canonical hash. A page is evidence only through original text chunks on that page. Pages
without such chunks are reported as unanchored (page index and score) and never returned as hits.

Tenant scope: the index stores the tenant and project of its chunk records; a retriever bound to another scope
raises NotFound before any encoder call. Errors carry codes only; nothing here logs content.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError, IndexingError, NotFoundError
from rag_foundry_domain.extraction import ExtractionView
from rag_foundry_domain.indexing import ChunkRecord, IndexState, IndexVersion
from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository, IndexRepository
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor
from rag_foundry_ingestion.page_images import render_pdf_pages
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import MultivectorResult
from rag_foundry_providers.ports import PageImageEncoder
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent
from rag_foundry_techniques.errors import TechniqueUnavailableError

_UNAVAILABLE = "R41 needs a page-image encoder adapter and approved local model files"
_PAGE = 500
_BATCH = 4
_MAX_QUERY_CHARS = 2000


class VisualPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    extraction_view_id: str
    document_version_id: str
    page_index: int = Field(ge=0)
    vectors: tuple[tuple[float, ...], ...]
    chunk_ids: tuple[str, ...]


class VisualStats(BaseModel):
    """`estimated_bytes` is an estimate (patches * dim * 4, float32 payload); Python tuples use more."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pages: int
    patches: int
    dim: int
    estimated_bytes: int


class VisualPageIndex(BaseModel):
    """In-memory page patch vectors bound to the tenant and project of their chunk records."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tenant_id: str
    project_id: str
    index_version_id: str
    encoder_provider: str
    encoder_model: str
    dim: int
    pages: tuple[VisualPage, ...]
    chunks: dict[str, ChunkRecord]

    def stats(self) -> VisualStats:
        patches = sum(len(p.vectors) for p in self.pages)
        return VisualStats(pages=len(self.pages), patches=patches, dim=self.dim, estimated_bytes=patches * self.dim * 4)


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0.0:
        return tuple(float(x) for x in vector)
    return tuple(x / norm for x in vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _check_result(result: MultivectorResult, count: int, dim: int | None) -> int:
    if len(result.items) != count:
        raise IndexingError("multivector_count_mismatch")
    expected = result.dim if dim is None else dim
    if result.dim != expected:
        raise IndexingError("multivector_dim_mismatch")
    for item in result.items:
        for vector in item.vectors:
            if len(vector) != expected:
                raise IndexingError("multivector_dim_mismatch")
    return expected


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


async def build_visual_index(
    scope: Scope,
    version: IndexVersion,
    pdf_by_view: Mapping[str, bytes],
    index_repo: IndexRepository,
    doc_repo: DocumentRepository,
    encoder: PageImageEncoder,
    *,
    extraction_repo: ExtractionRepository,
) -> VisualPageIndex:
    """Index the PDF pages of each listed view. `extraction_repo` (scoped `get_view`) is an addition to the
    spec signature: the view and its document version are only reachable through it."""
    if version.state is not IndexState.READY:
        raise IndexingError("not_ready")
    if version.tenant_id != scope.tenant_id or version.project_id != scope.project_id:
        raise NotFoundError()
    if not pdf_by_view:
        raise IndexingError("empty_index")
    chunks = await _load_chunks(scope, version, index_repo)
    # Pass 1: validate every view (scope, version membership, media type, sha256, text identity) before any
    # rendering or encoder call, so a bad later view cannot leave earlier views encoded.
    checked: list[tuple[str, ExtractionView, bytes, dict[str, int]]] = []
    for view_id in sorted(pdf_by_view):
        if view_id not in version.extraction_view_ids:
            raise NotFoundError()
        view = await extraction_repo.get_view(scope, view_id)
        doc_version = await doc_repo.get_version(scope, view.document_id, view.document_version_id)
        pdf = bytes(pdf_by_view[view_id])
        if (
            doc_version.media_type is not MediaType.APPLICATION_PDF
            or hashlib.sha256(pdf).hexdigest() != doc_version.sha256
        ):
            raise ExtractionError("page_image_invalid", "the PDF bytes do not match the stored document version")
        extracted = get_extractor(MediaType.APPLICATION_PDF).extract(pdf, text_encoding=None, limits=ExtractionLimits())
        if hashlib.sha256(extracted.canonical_text.encode("utf-8")).hexdigest() != view.canonical_sha256:
            raise ExtractionError("integrity_mismatch", "the PDF text does not match the extraction view")
        block_page = {b.block_id: b.page_index for b in extracted.blocks if b.page_index is not None}
        checked.append((view_id, view, pdf, block_page))
    # Pass 2: render and encode.
    pages: list[VisualPage] = []
    used: dict[str, ChunkRecord] = {}
    dim: int | None = None
    for view_id, view, pdf, block_page in checked:
        view_chunks = sorted((c for c in chunks if c.extraction_view_id == view_id), key=lambda c: c.ordinal)
        images = render_pdf_pages(pdf)
        vectors: list[tuple[tuple[float, ...], ...]] = []
        for i in range(0, len(images), _BATCH):
            batch = images[i : i + _BATCH]
            try:
                result = await encoder.encode_pages([img.png for img in batch])
            except AdapterDependencyError as exc:
                raise TechniqueUnavailableError(_UNAVAILABLE) from exc
            dim = _check_result(result, len(batch), dim)
            for item in result.items:
                if not item.vectors:
                    raise IndexingError("multivector_page_empty")
                vectors.append(tuple(_normalise(v) for v in item.vectors))
        for image, page_vectors in zip(images, vectors, strict=True):
            mapped = [c for c in view_chunks if any(block_page.get(b) == image.page_index for b in c.block_ids)]
            used.update({c.id: c for c in mapped})
            pages.append(
                VisualPage(
                    extraction_view_id=view_id,
                    document_version_id=view.document_version_id,
                    page_index=image.page_index,
                    vectors=page_vectors,
                    chunk_ids=tuple(c.id for c in mapped),
                )
            )
    if dim is None:
        raise IndexingError("empty_index")
    profile = encoder.profile
    return VisualPageIndex(
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        index_version_id=version.id,
        encoder_provider=profile.provider,
        encoder_model=profile.model,
        dim=dim,
        pages=tuple(pages),
        chunks=used,
    )


class VisualConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=3, ge=1, le=50)
    max_query_tokens: int = Field(default=32, ge=1, le=512)


class UnanchoredPage(BaseModel):
    """A scored page with no text chunks: reported, never cited."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    extraction_view_id: str
    page_index: int
    score: float


class VisualSearchReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    result: RetrievalResult
    unanchored_pages: tuple[UnanchoredPage, ...]


class VisualMultivectorRetriever:
    """R41; exhaustive page MaxSim over a `VisualPageIndex`, bound to one scope and index version."""

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R41",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=VisualConfig,
    )

    def __init__(
        self,
        config: VisualConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: VisualPageIndex | None,
        encoder: PageImageEncoder | None = None,
    ) -> None:
        self.config, self._scope, self._version, self._index, self._encoder = (
            config,
            scope,
            index_version,
            index,
            encoder,
        )

    async def retrieve(self, query: str) -> RetrievalResult:
        return (await self.search_with_report(query)).result

    async def search_with_report(self, query: str) -> VisualSearchReport:
        index = self._index
        if index is None:
            raise IndexingError("visual_index_missing")
        if self._scope.tenant_id != index.tenant_id or self._scope.project_id != index.project_id:
            raise NotFoundError()
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > _MAX_QUERY_CHARS:
            raise IndexingError("query_too_long")
        if self._version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if self._version.id != index.index_version_id:
            raise IndexingError("visual_index_missing")
        encoder = self._encoder
        if encoder is None:
            raise TechniqueUnavailableError(_UNAVAILABLE)
        if encoder.profile.provider != index.encoder_provider or encoder.profile.model != index.encoder_model:
            raise IndexingError("incompatible_embedding")
        try:
            result = await encoder.encode_queries([query])
        except AdapterDependencyError as exc:
            raise TechniqueUnavailableError(_UNAVAILABLE) from exc
        _check_result(result, 1, index.dim)
        if not result.items[0].vectors:
            raise IndexingError("empty_query")
        qvecs = [_normalise(v) for v in result.items[0].vectors[: self.config.max_query_tokens]]
        scored = [(math.fsum(max(_dot(q, p) for p in page.vectors) for q in qvecs), page) for page in index.pages]
        scored.sort(key=lambda row: (-row[0], row[1].extraction_view_id, row[1].page_index))
        hits: list[RetrievedChunk] = []
        cited: set[str] = set()
        unanchored: list[UnanchoredPage] = []
        for score, page in scored:
            if not page.chunk_ids:
                unanchored.append(
                    UnanchoredPage(extraction_view_id=page.extraction_view_id, page_index=page.page_index, score=score)
                )
                continue
            if len(hits) >= self.config.k:
                continue
            fresh = [cid for cid in page.chunk_ids if cid not in cited]
            if not fresh:
                continue  # every chunk on this page is already cited from a higher-ranked page
            cited.add(fresh[0])
            chunk = index.chunks[fresh[0]]
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
                    rank=len(hits) + 1,
                    score_components=(ScoreComponent(name="page_maxsim", value=score, term=f"page:{page.page_index}"),),
                    parent_start=chunk.parent_start_cp,
                    parent_end=chunk.parent_end_cp,
                )
            )
        retrieval = RetrievalResult(
            technique_id="R41",
            grounding="corpus",
            index_version_id=index.index_version_id,
            hits=tuple(hits),
            scoring="r41-page-maxsim",
        )
        return VisualSearchReport(result=retrieval, unanchored_pages=tuple(unanchored))
