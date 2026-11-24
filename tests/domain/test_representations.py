import asyncio
import hashlib

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.index_testing import REPRESENTATION_STORE_CHECKS, InMemoryIndexRepository
from rag_foundry_domain.ports import IndexRepository, RepresentationStore
from rag_foundry_domain.representations import (
    NewRepresentation,
    RepresentationSet,
    representation_key_for,
)

CHUNK_ID = "7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f"


def _row(**over) -> dict:
    text = over.pop("text", "a summary")
    base = dict(
        chunk_id=CHUNK_ID,
        kind="summary",
        ordinal=0,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        embedded_input_sha256=hashlib.sha256(b"e").hexdigest(),
        embedding=(1.0, 0.0),
    )
    base.update(over)
    return base


def test_valid_representation_normalizes_embedding() -> None:
    row = NewRepresentation(**_row(kind="question", ordinal=15))
    assert row.embedding == (1.0, 0.0) and row.ordinal == 15


def test_text_hash_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NewRepresentation(**_row(text_sha256=hashlib.sha256(b"other").hexdigest()))


@pytest.mark.parametrize("kind", ["summary", "context_prefix"])
def test_single_kinds_require_ordinal_zero(kind: str) -> None:
    with pytest.raises(ValidationError):
        NewRepresentation(**_row(kind=kind, ordinal=1))
    NewRepresentation(**_row(kind=kind, ordinal=0))


def test_ordinal_and_text_bounds() -> None:
    for over in ({"kind": "question", "ordinal": 16}, {"kind": "question", "ordinal": -1}, {"text": ""}):
        with pytest.raises(ValidationError):
            NewRepresentation(**_row(**over))
    long_text = "x" * 4001
    with pytest.raises(ValidationError):
        NewRepresentation(**_row(text=long_text))


def test_bad_key_and_technique_id_are_rejected() -> None:
    good = "a" * 64
    RepresentationSet(representation_key=good, technique_id="R23", generator_json="{}")
    for key, tech in ((good[:-1], "R23"), ("A" * 64, "R23"), (good, "R2"), (good, "C05"), (good, "r23")):
        with pytest.raises(ValidationError):
            RepresentationSet(representation_key=key, technique_id=tech, generator_json="{}")
    with pytest.raises(ValidationError):
        RepresentationSet(representation_key=good, technique_id="R23", generator_json="x" * 8001)


def test_representation_key_is_canonical_sha256() -> None:
    a = representation_key_for({"b": 1, "a": "\u0627"})
    b = representation_key_for({"a": "\u0627", "b": 1})
    assert a == b
    assert a == hashlib.sha256('{"a":"\u0627","b":1}'.encode()).hexdigest()
    assert a != representation_key_for({"a": "\u0627", "b": 2})


def test_new_error_codes_are_registered() -> None:
    for code in (
        "too_many_rows",
        "representation_set_conflict",
        "duplicate_representation",
        "representations_unavailable",
    ):
        assert IndexingError(code).code == code


def test_in_memory_representation_store_checks() -> None:
    from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository

    async def factory(scopes):
        docs = InMemoryDocumentRepository(scopes)
        views = InMemoryExtractionRepository(docs)
        repo = InMemoryIndexRepository(views)
        assert isinstance(repo, IndexRepository) and isinstance(repo, RepresentationStore)
        return docs, views, repo

    for check in REPRESENTATION_STORE_CHECKS:
        asyncio.run(check(factory))
