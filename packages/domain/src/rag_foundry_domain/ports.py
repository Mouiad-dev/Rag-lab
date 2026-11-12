"""Ports for object storage and document persistence. Implementations live elsewhere."""

from collections.abc import AsyncIterable, AsyncIterator
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rag_foundry_domain.documents import Document, DocumentVersion, IntakeOutcome, NewVersion, Scope, Sha256Hex
from rag_foundry_domain.extraction import ExtractionKey, ExtractionOutcome, ExtractionView, NewExtraction
from rag_foundry_domain.graphs import EntityGraph
from rag_foundry_domain.indexing import (
    ChunkRecord,
    FtsScoredChunk,
    IndexVersion,
    NewChunk,
    NewIndexVersion,
    ScoredChunk,
)
from rag_foundry_domain.representations import NewRepresentation, RepresentationSet, ScoredRepresentation


class StagedObject(BaseModel):
    """Bytes written to a temporary location, hashed, not yet visible under a content key."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    token: Annotated[str, Field(min_length=1)]
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]


@runtime_checkable
class ObjectStore(Protocol):
    async def stage(self, chunks: AsyncIterable[bytes], *, max_bytes: int) -> StagedObject:
        """Stream into temporary storage; raise UploadTooLargeError past `max_bytes`."""
        ...

    async def commit(self, staged: StagedObject, key: str) -> None:
        """Publish under `key`. No-op if the key already holds the same hash."""
        ...

    async def discard(self, staged: StagedObject) -> None: ...

    def open_read(self, key: str) -> AsyncIterator[bytes]:
        """Stream the stored bytes in chunks; ObjectStoreError if the key is invalid or missing."""
        ...

    def key_for(self, scope: Scope, sha256: str) -> str:
        """Synchronous, pure: content-addressed key for a scope and hash (D7)."""
        ...


@runtime_checkable
class DocumentRepository(Protocol):
    """Unknown and out-of-scope IDs raise the same NotFoundError. Dedup is atomic inside the repository."""

    async def project_exists(self, scope: Scope) -> bool: ...

    async def get_document(self, scope: Scope, document_id: str) -> Document: ...

    async def create_document_or_get(self, scope: Scope, new: NewVersion, *, display_name: str) -> IntakeOutcome:
        """Dedup by sha256 within the project; several matches return the earliest-created version
        (ties: lowest version ID)."""
        ...

    async def add_version_or_get(self, scope: Scope, document_id: str, new: NewVersion) -> IntakeOutcome:
        """Dedup within the document only. New version number is max + 1."""
        ...

    async def get_version(self, scope: Scope, document_id: str, version_id: str) -> DocumentVersion: ...

    async def list_versions(self, scope: Scope, document_id: str) -> list[DocumentVersion]:
        """Ascending by version number."""
        ...


@runtime_checkable
class ExtractionRepository(Protocol):
    """Unknown, malformed and out-of-scope IDs raise the same NotFoundError (find_view returns None)."""

    async def find_view(self, scope: Scope, key: ExtractionKey) -> ExtractionView | None: ...

    async def create_view_or_get(self, scope: Scope, new: NewExtraction) -> ExtractionOutcome:
        """The version must exist in scope, else NotFoundError. Dedup by identity key is atomic."""
        ...

    async def get_view(self, scope: Scope, view_id: str) -> ExtractionView: ...

    async def list_views(self, scope: Scope, document_version_id: str) -> list[ExtractionView]:
        """Ordered by created_at, then id."""
        ...


@runtime_checkable
class IndexRepository(Protocol):
    async def create_index_version(self, scope: Scope, new: NewIndexVersion) -> IndexVersion: ...

    async def add_chunks(self, scope: Scope, index_version_id: str, chunks: tuple[NewChunk, ...]) -> None: ...

    async def mark_ready(self, scope: Scope, index_version_id: str) -> IndexVersion: ...

    async def mark_failed(self, scope: Scope, index_version_id: str) -> IndexVersion: ...

    async def get_index_version(self, scope: Scope, index_version_id: str) -> IndexVersion: ...

    async def list_chunks(
        self, scope: Scope, index_version_id: str, *, offset: int = 0, limit: int = 100
    ) -> list[ChunkRecord]: ...

    async def search(
        self, scope: Scope, index_version_id: str, query: tuple[float, ...], k: int
    ) -> list[ScoredChunk]: ...

    async def add_sparse(self, scope: Scope, index_version_id: str, encoder_key: str, rows: tuple) -> None: ...

    async def sparse_search(
        self,
        scope: Scope,
        index_version_id: str,
        encoder_key: str,
        query: tuple[tuple[str, ...], tuple[float, ...]],
        k: int,
    ) -> list: ...


@runtime_checkable
class RepresentationStore(Protocol):
    """Synthetic chunk representations. Separate from IndexRepository; never evidence."""

    async def add_representations(
        self,
        scope: Scope,
        index_version_id: str,
        rep_set: RepresentationSet,
        rows: tuple[NewRepresentation, ...],
    ) -> None: ...

    async def get_representation_set(self, scope: Scope, index_version_id: str, key: str) -> RepresentationSet: ...

    async def representation_search(
        self, scope: Scope, index_version_id: str, key: str, query: tuple[float, ...], k: int
    ) -> list[ScoredRepresentation]: ...


@runtime_checkable
class EntityGraphStore(Protocol):
    """One immutable entity graph per (index version, graph key). Separate from IndexRepository; never evidence."""

    async def add_entity_graph(self, scope: Scope, index_version_id: str, graph: EntityGraph) -> None:
        """Atomic: all rows or none."""
        ...

    async def get_entity_graph(self, scope: Scope, index_version_id: str, graph_key: str) -> EntityGraph:
        """NotFoundError when missing or outside scope."""
        ...


@runtime_checkable
class ChunkVectorReader(Protocol):
    """Read stored chunk embeddings without exposing chunks outside the version scope."""

    async def chunk_embeddings(
        self, scope: Scope, index_version_id: str, chunk_ids: tuple[str, ...]
    ) -> dict[str, tuple[float, ...]]: ...


@runtime_checkable
class FullTextIndex(Protocol):
    async def fts_search(
        self,
        scope: Scope,
        index_version_id: str,
        query: str,
        k: int,
        *,
        operator: Literal["and", "or"],
    ) -> list[FtsScoredChunk]: ...
