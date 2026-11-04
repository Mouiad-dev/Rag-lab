"""Types and errors of the document domain."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from rag_foundry_domain import (
    LOCAL_DEV_SCOPE,
    Document,
    DocumentVersion,
    EmptyUploadError,
    IntakeError,
    IntakeOutcome,
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaType,
    MediaTypeMismatchError,
    NewVersion,
    NotFoundError,
    ObjectStoreError,
    Scope,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
    new_id,
)

SHA = "a" * 64


def _new(**over) -> NewVersion:
    base = dict(
        sha256=SHA,
        size_bytes=3,
        media_type=MediaType.TEXT_PLAIN,
        text_encoding="utf-8",
        original_filename="a.txt",
        object_key="k",
    )
    return NewVersion(**{**base, **over})


def _doc() -> Document:
    return Document(
        id=new_id(),
        tenant_id=LOCAL_DEV_SCOPE.tenant_id,
        project_id=LOCAL_DEV_SCOPE.project_id,
        display_name="d",
        created_at=datetime.now(UTC),
    )


def _version(doc: Document) -> DocumentVersion:
    return DocumentVersion(
        **_new().model_dump(),
        id=new_id(),
        document_id=doc.id,
        tenant_id=doc.tenant_id,
        project_id=doc.project_id,
        version_number=1,
        created_at=datetime.now(UTC),
    )


def test_media_types_exact() -> None:
    assert {m.name: m.value for m in MediaType} == {
        "TEXT_PLAIN": "text/plain",
        "TEXT_MARKDOWN": "text/markdown",
        "TEXT_HTML": "text/html",
        "APPLICATION_PDF": "application/pdf",
    }


def test_local_dev_scope_is_fixed_uuid4() -> None:
    assert LOCAL_DEV_SCOPE.tenant_id == "7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f"
    assert LOCAL_DEV_SCOPE.project_id == "2d9e8f7a-6b5c-4a3d-8e2f-0c1b2a3d4e5f"


@pytest.mark.parametrize(
    "bad", ["", "not-a-uuid", "7F3C2B1E-5D4A-4C8B-9E6F-1A2B3C4D5E6F", "6ba7b810-9dad-11d1-80b4-00c04fd430c8"]
)
def test_scope_rejects_non_uuid4(bad: str) -> None:
    with pytest.raises(ValidationError):
        Scope(tenant_id=bad, project_id=new_id())


@pytest.mark.parametrize(
    "over",
    [
        {"sha256": "A" * 64},
        {"sha256": "a" * 63},
        {"size_bytes": 0},
        {"original_filename": ""},
        {"original_filename": "x" * 256},
        {"object_key": ""},
        {"text_encoding": "latin-1"},
        {"text_encoding": None},  # not allowed for text
        {"media_type": "image/png"},
        {"extra": 1},
    ],
)
def test_new_version_rejects(over) -> None:
    with pytest.raises(ValidationError):
        _new(**over)


def test_pdf_allows_no_encoding() -> None:
    assert _new(media_type=MediaType.APPLICATION_PDF, text_encoding=None).text_encoding is None
    assert _new(text_encoding="utf-8-sig").text_encoding == "utf-8-sig"


def test_created_at_must_be_utc_aware() -> None:
    doc = _doc()
    with pytest.raises(ValidationError):
        Document(**{**doc.model_dump(), "created_at": datetime.now()})
    with pytest.raises(ValidationError):
        Document(**{**doc.model_dump(), "created_at": datetime.now(timezone(timedelta(hours=3)))})


def test_version_number_minimum() -> None:
    doc = _doc()
    with pytest.raises(ValidationError):
        DocumentVersion(**{**_version(doc).model_dump(), "version_number": 0})


def test_records_are_frozen() -> None:
    doc = _doc()
    for record, field in (
        (LOCAL_DEV_SCOPE, "tenant_id"),
        (_new(), "sha256"),
        (doc, "display_name"),
        (_version(doc), "version_number"),
    ):
        with pytest.raises(ValidationError):
            setattr(record, field, getattr(record, field))


def test_outcome_flags_and_consistency() -> None:
    doc = _doc()
    ver = _version(doc)
    created = IntakeOutcome(document=doc, version=ver, created_document=True, created_version=True)
    assert not created.deduplicated
    assert IntakeOutcome(document=doc, version=ver, created_document=False, created_version=False).deduplicated
    assert not IntakeOutcome(document=doc, version=ver, created_document=False, created_version=True).deduplicated
    with pytest.raises(ValidationError):
        IntakeOutcome(document=doc, version=ver, created_document=True, created_version=False)
    with pytest.raises(ValidationError):
        IntakeOutcome(document=_doc(), version=ver, created_document=False, created_version=False)


def test_new_id_is_uuid4() -> None:
    assert new_id() != new_id()
    Scope(tenant_id=new_id(), project_id=new_id())


@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (UploadTooLargeError, "upload_too_large"),
        (EmptyUploadError, "empty_upload"),
        (UnsupportedMediaTypeError, "unsupported_media_type"),
        (MediaTypeMismatchError, "media_type_mismatch"),
        (InvalidTextEncodingError, "invalid_text_encoding"),
        (InvalidFilenameError, "invalid_filename"),
        (NotFoundError, "not_found"),
        (ObjectStoreError, "object_store_error"),
    ],
)
def test_errors(cls, code: str) -> None:
    err = cls()
    assert isinstance(err, IntakeError) and err.code == code and str(err)
    assert not isinstance(err, (KeyError, LookupError, OSError))
    assert "/" not in str(err) and "\\" not in str(err)
