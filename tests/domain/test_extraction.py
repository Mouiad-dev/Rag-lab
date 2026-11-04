"""Extraction types and errors (ADR-0010)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from rag_foundry_domain import IntakeError, new_id
from rag_foundry_domain.documents import DocumentVersion
from rag_foundry_domain.errors import ExtractionAlignmentError, ExtractionError
from rag_foundry_domain.extraction import BlockKind, ExtractionOutcome, ExtractionView, NewExtraction, SourceBlock
from rag_foundry_domain.testing import make_new_version


def _block(**over) -> SourceBlock:
    data = {"block_id": "b00000", "ordinal": 0, "kind": BlockKind.PARAGRAPH, "start": 0, "end": 5}
    data.update(over)
    return SourceBlock(**data)


def _new(**over) -> NewExtraction:
    data = {
        "document_id": new_id(),
        "document_version_id": new_id(),
        "extractor": "plain-text",
        "extractor_version": "1",
        "profile_id": "canonical-v1",
        "unicode_version": "15.1.0",
        "canonical_sha256": "a" * 64,
        "char_count": 5,
        "block_count": 1,
        "payload_sha256": "b" * 64,
        "payload_key": "k",
    }
    data.update(over)
    return NewExtraction(**data)


def test_block_ok_and_frozen() -> None:
    block = _block(section_path=("A", "B"))
    assert block.section_path == ("A", "B")
    with pytest.raises(ValidationError):
        block.end = 9  # type: ignore[misc]
    assert _block(start=3, end=3).end == 3


@pytest.mark.parametrize(
    "over",
    [
        {"block_id": "b0001"},
        {"block_id": "B00000"},
        {"block_id": "b00001"},
        {"ordinal": -1},
        {"start": -1},
        {"start": 6},
        {"heading_level": 1},
        {"page_index": -1},
        {"extra": 1},
        {"kind": "table"},
    ],
)
def test_block_rejects(over) -> None:
    with pytest.raises(ValidationError):
        _block(**over)


def test_heading_levels() -> None:
    assert _block(kind=BlockKind.HEADING, heading_level=6).heading_level == 6
    for level in (0, 7):
        with pytest.raises(ValidationError):
            _block(kind=BlockKind.HEADING, heading_level=level)


def test_heading_requires_level() -> None:
    with pytest.raises(ValidationError):
        _block(kind=BlockKind.HEADING)


def test_block_id_wide_ordinals() -> None:
    assert _block(block_id="b123456", ordinal=123456).ordinal == 123456


@pytest.mark.parametrize(
    "over",
    [
        {"extractor": "Plain Text"},
        {"extractor": ""},
        {"extractor_version": ""},
        {"extractor_version": "x" * 129},
        {"unicode_version": "15.1"},
        {"canonical_sha256": "A" * 64},
        {"char_count": 0},
        {"block_count": 0},
        {"page_count": 0},
        {"payload_schema_version": 2},
        {"warnings": ("has space",)},
        {"warnings": ("x" * 65,)},
        {"document_version_id": "not-a-uuid"},
        {"extra": 1},
    ],
)
def test_new_extraction_rejects(over) -> None:
    with pytest.raises(ValidationError):
        _new(**over)


def test_key_and_view() -> None:
    new = _new(warnings=("empty_page:3",), page_count=2)
    assert new.key.document_version_id == new.document_version_id
    assert set(new.key.model_dump()) == {
        "document_version_id",
        "extractor",
        "extractor_version",
        "profile_id",
        "unicode_version",
    }
    view = ExtractionView(
        **new.model_dump(), id=new_id(), tenant_id=new_id(), project_id=new_id(), created_at=datetime.now(UTC)
    )
    assert view.key == new.key
    with pytest.raises(ValidationError):
        view.profile_id = "x"  # type: ignore[misc]
    assert ExtractionOutcome(view=view, created=True).created
    with pytest.raises(ValidationError):
        ExtractionView(
            **new.model_dump(), id=new_id(), tenant_id=new_id(), project_id=new_id(), created_at=datetime(2026, 1, 1)
        )


def test_errors() -> None:
    for code in ExtractionError.CODES:
        err = ExtractionError(code)
        assert isinstance(err, IntakeError) and err.code == code
        assert "/" not in err.message
    with pytest.raises(ValueError):
        ExtractionError("made_up")
    align = ExtractionAlignmentError()
    assert isinstance(align, ExtractionError) and align.code == "alignment_failed"


def test_version_helper_type() -> None:
    assert isinstance(
        DocumentVersion(
            **make_new_version("x").model_dump(),
            id=new_id(),
            document_id=new_id(),
            tenant_id=new_id(),
            project_id=new_id(),
            version_number=1,
            created_at=datetime.now(UTC),
        ),
        DocumentVersion,
    )
