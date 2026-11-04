"""Ports for object storage and document persistence. Implementations live elsewhere."""

from collections.abc import AsyncIterable
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rag_foundry_domain.documents import Document, DocumentVersion, IntakeOutcome, NewVersion, Scope, Sha256Hex


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
