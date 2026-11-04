"""RAG Foundry domain: document lineage types, intake errors and ports (pydantic only, no I/O)."""

from rag_foundry_domain.documents import (
    LOCAL_DEV_SCOPE,
    Document,
    DocumentVersion,
    IntakeOutcome,
    MediaType,
    NewVersion,
    Scope,
    new_id,
)
from rag_foundry_domain.errors import (
    EmptyUploadError,
    IntakeError,
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaTypeMismatchError,
    NotFoundError,
    ObjectStoreError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from rag_foundry_domain.ports import DocumentRepository, ObjectStore, StagedObject

__all__ = [
    "LOCAL_DEV_SCOPE",
    "Document",
    "DocumentRepository",
    "DocumentVersion",
    "EmptyUploadError",
    "IntakeError",
    "IntakeOutcome",
    "InvalidFilenameError",
    "InvalidTextEncodingError",
    "MediaType",
    "MediaTypeMismatchError",
    "NewVersion",
    "NotFoundError",
    "ObjectStore",
    "ObjectStoreError",
    "Scope",
    "StagedObject",
    "UnsupportedMediaTypeError",
    "UploadTooLargeError",
    "new_id",
]
