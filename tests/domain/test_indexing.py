import asyncio
import hashlib
import json
import math

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.index_testing import INDEX_REPOSITORY_CHECKS, InMemoryIndexRepository
from rag_foundry_domain.indexing import NewChunk, check_vector, embedding_key_for
from rag_foundry_domain.ports import ExtractionRepository, IndexRepository


def test_chunk_block_ids_sort_by_numeric_ordinal() -> None:
    base = dict(
        extraction_view_id="7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f",
        ordinal=0,
        start=0,
        end=1,
        split_level=0,
        text="x",
        text_sha256=hashlib.sha256(b"x").hexdigest(),
        embedding=(1.0,),
    )
    NewChunk(**base, block_ids=("b99999", "b100000"))
    with pytest.raises(ValidationError):
        NewChunk(**base, block_ids=("b100000", "b99999"))


def test_new_chunk_parent_span_pair_validation():
    base = dict(
        extraction_view_id="7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f",
        ordinal=0,
        start=2,
        end=3,
        split_level=0,
        block_ids=("b00000",),
        text="x",
        text_sha256=hashlib.sha256(b"x").hexdigest(),
        embedding=(1.0,),
    )
    with pytest.raises(ValidationError):
        NewChunk(**base, parent_start_cp=0)
    with pytest.raises(ValidationError):
        NewChunk(**base, parent_start_cp=3, parent_end_cp=5)
    assert NewChunk(**base, parent_start_cp=2, parent_end_cp=3)


@pytest.mark.parametrize("values", [(3.5e38, 0, 0, 0), (math.inf, 0), (math.nan, 0), (0, 0, 0), (1e-30,) * 4])
def test_invalid_vectors(values) -> None:
    with pytest.raises(IndexingError) as exc:
        check_vector(values)
    assert exc.value.code == "invalid_vector"


def test_valid_vector_is_float32() -> None:
    assert check_vector((0.1, 2.0)) == (0.10000000149011612, 2.0)


def test_embedding_key_golden() -> None:
    payload = {
        "scheme": "embedding-key-v1",
        "provider": "p",
        "model": "m",
        "embedding_dim": 4,
        "distance": "cosine",
        "preprocessing": "none",
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert embedding_key_for("p", "m", 4) == expected
    assert embedding_key_for("p", "m2", 4) != expected


def test_in_memory_repository_checks() -> None:
    # Set up the fixture dependencies required by the conformance factory shape.
    from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository

    async def factory(scopes):
        docs = InMemoryDocumentRepository(scopes)
        views = InMemoryExtractionRepository(docs)
        repo = InMemoryIndexRepository(views)
        assert isinstance(repo, IndexRepository)
        assert hasattr(ExtractionRepository, "get_view") and hasattr(ExtractionRepository, "list_views")
        return docs, views, repo

    for check in INDEX_REPOSITORY_CHECKS:
        asyncio.run(check(factory))


def test_indexing_error_codes() -> None:
    with pytest.raises(ValueError):
        IndexingError("bogus")
    assert str(IndexingError("not_ready")) == "not_ready"
