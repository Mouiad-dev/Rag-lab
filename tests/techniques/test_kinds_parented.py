import pytest
from pydantic import ValidationError
from rag_foundry_chunking import ParentChildChunker, ParentChildConfig, RecursiveChunker, RecursiveConfig
from rag_foundry_techniques import ChunkSpan, ParentedSpan, ParentLinkedChunker


def test_parent_span_pair_and_containment():
    child = ChunkSpan(start=2, end=5, split_level=0)
    assert ParentedSpan(start=2, end=5, split_level=0, parent_start=2, parent_end=5)
    with pytest.raises(ValidationError):
        ParentedSpan(start=2, end=5, split_level=0, parent_start=1)
    with pytest.raises(ValidationError):
        ParentedSpan(start=2, end=5, split_level=0, parent_start=3, parent_end=7)
    assert child.start == 2


def test_parent_linked_protocol_runtime_detection():
    assert isinstance(ParentChildChunker(ParentChildConfig()), ParentLinkedChunker)
    assert not isinstance(RecursiveChunker(RecursiveConfig()), ParentLinkedChunker)
