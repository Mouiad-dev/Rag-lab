"""Post-filter retrieval to a set of authorized extraction views.

Build the inner retriever with a large k (for example 50), or post-filtering can
return fewer than the requested k hits. Language and section path are not
offered because they are not stored per chunk yet.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.documents import DocumentVersion, MediaType, Scope, UtcDatetime, Uuid4
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.extraction import ExtractionView
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository, IndexRepository
from rag_foundry_techniques import (
    Fidelity,
    PluginKind,
    PluginManifest,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    ScoreComponent,
)


class MetadataFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_ids: Annotated[tuple[Uuid4, ...], Field(max_length=100)] | None = None
    document_version_ids: Annotated[tuple[Uuid4, ...], Field(max_length=100)] | None = None
    media_types: tuple[MediaType, ...] | None = None
    created_from: UtcDatetime | None = None
    created_to: UtcDatetime | None = None


def intersect(a: MetadataFilter, b: MetadataFilter | None) -> MetadataFilter:
    if b is None:
        return a
    values = {}
    for field in ("document_ids", "document_version_ids", "media_types"):
        left, right = getattr(a, field), getattr(b, field)
        if left is None:
            value = right
        elif right is None:
            value = left
        else:
            value = tuple(sorted(set(left) & set(right), key=str))
        values[field] = value
    values["created_from"] = max((v for v in (a.created_from, b.created_from) if v is not None), default=None)
    values["created_to"] = min((v for v in (a.created_to, b.created_to) if v is not None), default=None)
    return MetadataFilter(**values)


class ViewMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    view_id: Uuid4
    document_id: Uuid4
    document_version_id: Uuid4
    display_name: str
    media_type: MediaType
    version_created_at: UtcDatetime


async def resolve_views(
    filter: MetadataFilter,
    *,
    scope: Scope,
    index_version: IndexVersion,
    extractions: ExtractionRepository,
    documents: DocumentRepository,
) -> tuple[ViewMetadata, ...]:
    found = []
    for view_id in index_version.extraction_view_ids:
        view = await extractions.get_view(scope, view_id)
        version = await documents.get_version(scope, view.document_id, view.document_version_id)
        document = await documents.get_document(scope, view.document_id)
        if _matches(filter, view, version):
            found.append(
                ViewMetadata(
                    view_id=view.id,
                    document_id=view.document_id,
                    document_version_id=view.document_version_id,
                    display_name=document.display_name,
                    media_type=version.media_type,
                    version_created_at=version.created_at,
                )
            )
    return tuple(sorted(found, key=lambda v: (v.version_created_at, v.document_id, v.view_id)))


def _matches(filter: MetadataFilter, view: ExtractionView, version: DocumentVersion) -> bool:
    return not (
        (filter.document_ids is not None and view.document_id not in filter.document_ids)
        or (filter.document_version_ids is not None and view.document_version_id not in filter.document_version_ids)
        or (filter.media_types is not None and version.media_type not in filter.media_types)
        or (filter.created_from is not None and version.created_at < filter.created_from)
        or (filter.created_to is not None and version.created_at >= filter.created_to)
        or (
            filter.created_from is not None
            and filter.created_to is not None
            and filter.created_from >= filter.created_to
        )
    )


class MetadataFilterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=5, ge=1, le=50)
    filter: MetadataFilter = MetadataFilter()


class MetadataFilteredRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R11",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=MetadataFilterConfig,
    )

    def __init__(
        self,
        config: MetadataFilterConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        extractions: ExtractionRepository,
        documents: DocumentRepository,
        inner: Retriever,
    ) -> None:
        self.config, self._scope, self._version = config, scope, index_version
        self._index, self._extractions, self._documents, self._inner = index, extractions, documents, inner

    async def _ready_version(self) -> IndexVersion:
        version = await self._index.get_index_version(self._scope, self._version.id)
        if getattr(version.state, "value", version.state) != "ready":
            raise IndexingError("not_ready")
        return version

    async def choices(self) -> tuple[ViewMetadata, ...]:
        version = await self._ready_version()
        return await resolve_views(
            self.config.filter,
            scope=self._scope,
            index_version=version,
            extractions=self._extractions,
            documents=self._documents,
        )

    async def retrieve(self, query: str, *, extra_filter: MetadataFilter | None = None) -> RetrievalResult:
        version = await self._ready_version()
        effective = intersect(self.config.filter, extra_filter)
        views = await resolve_views(
            effective,
            scope=self._scope,
            index_version=version,
            extractions=self._extractions,
            documents=self._documents,
        )
        if not views:
            return RetrievalResult(technique_id="R11", grounding="corpus", index_version_id=version.id, hits=())
        result = await self._inner.retrieve(query)
        if result.grounding != "corpus" or result.index_version_id != version.id:
            raise IndexingError("selection_version_mismatch")
        allowed = {view.view_id for view in views}
        hits = []
        for hit in result.hits:
            if hit.extraction_view_id not in allowed:
                continue
            rank = len(hits) + 1
            hits.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    extraction_view_id=hit.extraction_view_id,
                    start=hit.start,
                    end=hit.end,
                    block_ids=hit.block_ids,
                    text=hit.text,
                    distance=hit.distance,
                    score=hit.score,
                    rank=rank,
                    score_components=(
                        *hit.score_components,
                        ScoreComponent(
                            name="input_rank", value=float(hit.rank), source=result.technique_id, source_rank=hit.rank
                        ),
                    ),
                    parent_start=hit.parent_start,
                    parent_end=hit.parent_end,
                )
            )
            if len(hits) == self.config.k:
                break
        return RetrievalResult(
            technique_id="R11",
            grounding="corpus",
            index_version_id=version.id,
            hits=tuple(hits),
            scoring=f"filter[{result.technique_id}]:post:{len(views)}/{len(version.extraction_view_ids)}views",
        )
