import asyncio

import pytest
from rag_foundry_domain.errors import NotFoundError
from rag_foundry_indexing.context_expansion import PayloadSourceReader
from rag_foundry_indexing.sentence_window import SentenceWindowConfig, SentenceWindowExpander

from tests.indexing.test_index_service import _built


async def _payload_reader_expansion_has_anchors_and_enforces_scope(tmp_path):
    env, embedder, version = await _built(tmp_path)
    factory = env.service()._uow
    reader = PayloadSourceReader(env.store, factory)
    base_result = await env.index.search(
        env.scope, version.id, [1.0] + [0.0] * (embedder.profile.capabilities.embedding_dim - 1), 1
    )
    from rag_foundry_techniques import RetrievalResult, RetrievedChunk

    row = base_result[0]
    hit = RetrievedChunk(
        chunk_id=row.chunk.id,
        extraction_view_id=row.chunk.extraction_view_id,
        start=row.chunk.start,
        end=row.chunk.end,
        block_ids=row.chunk.block_ids,
        text=row.chunk.text,
        score=1.0 - row.distance,
        rank=1,
    )
    result = RetrievalResult(technique_id="R02", grounding="corpus", index_version_id=version.id, hits=(hit,))
    expander = SentenceWindowExpander(
        SentenceWindowConfig(), scope=env.scope, index_version=version, index=env.index, source=reader
    )
    expanded = await expander.expand(result)
    source = await reader.view_source(env.scope, env.view.id)
    out = expanded.result.hits[0]
    assert out.text == source.text[out.start : out.end]
    assert await env.anchors.anchors_for_span(env.scope, env.view.id, out.start, out.end)
    with pytest.raises(NotFoundError):
        await reader.view_source(make_stranger(env.scope), env.view.id)


def make_stranger(scope):
    from rag_foundry_domain.testing import make_scope

    stranger = make_scope()
    return stranger


def test_payload_reader_expansion_has_anchors_and_enforces_scope(tmp_path):
    asyncio.run(_payload_reader_expansion_has_anchors_and_enforces_scope(tmp_path))
