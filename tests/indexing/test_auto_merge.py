import asyncio

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing.auto_merge import AutoMergeConfig, AutoMergingExpander
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader
from tests.indexing.test_parent_document import INPUT, SPANS, T2, _result


class SpyIndex:
    def __init__(self, inner):
        self.inner, self.list_calls = inner, 0

    async def list_chunks(self, *args, **kwargs):
        self.list_calls += 1
        return await self.inner.list_chunks(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


async def _case(**config):
    scope, _, index, version = await ready_index(SPANS, LexicalTestEmbedder())
    view = version.extraction_view_ids[0]
    spy = SpyIndex(index)
    source = StaticSourceReader(
        {
            (scope.tenant_id, scope.project_id, view): ViewSource(
                extraction_view_id=view, text=T2, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(T2)),)
            )
        }
    )
    expander = AutoMergingExpander(
        AutoMergeConfig(**config), scope=scope, index_version=version, index=spy, source=source
    )
    result = _result(version.id, view, T2, INPUT)
    out = await expander.expand(result)
    assert all(h.text == T2[h.start : h.end] for h in out.result.hits)
    return out, spy, expander, result


async def _defaults():
    out, spy, expander, result = await _case()
    first, second = out.result.hits
    assert (first.start, first.end, first.rank) == (0, 20, 1)
    assert out.trace.contexts[0].flags == ("merged:2/2",)
    assert (second.start, second.end, second.chunk_id, second.rank) == (22, 34, "c3", 2)
    assert out.trace.contexts[1].flags == ("not_merged:1/2",)
    assert out.trace.output_chars == 32
    assert out.result.technique_id == "R25" and out.result.scoring == "auto-merge"
    assert spy.list_calls == 1
    await expander.expand(result)
    assert spy.list_calls == 1  # counts cached per version id


async def _min_children_one_merges_and_loses():
    out, _, _, _ = await _case(min_children=1)
    second = out.result.hits[1]
    assert (second.start, second.end) == (22, 46)
    assert out.trace.contexts[1].flags == ("merged:1/2",)
    assert out.trace.output_chars == 44
    # Losing case: the merged parent carries text that no seed retrieved.
    assert "Delta four." in second.text
    assert all(s.end <= 34 for s in out.trace.contexts[1].seeds)


async def _ratio_blocks_merge():
    out, _, _, _ = await _case(merge_ratio=0.75, min_children=1)
    second = out.result.hits[1]
    assert (second.start, second.end, second.chunk_id) == (22, 34, "c3")


async def _parent_too_large_blocks_merge():
    out, _, _, _ = await _case(max_parent_chars=20, min_children=1)
    assert [(h.start, h.end) for h in out.result.hits] == [(0, 20), (22, 34)]
    assert out.trace.contexts[0].flags == ("merged:2/2",)
    assert out.trace.contexts[1].flags == ("not_merged:1/2",)


async def _index_too_large():
    scope, _, index, version = await ready_index(SPANS, LexicalTestEmbedder())
    view = version.extraction_view_ids[0]
    spy = SpyIndex(index)
    source = StaticSourceReader(
        {(scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=T2, blocks=())}
    )
    expander = AutoMergingExpander(
        AutoMergeConfig(max_index_chunks=3), scope=scope, index_version=version, index=spy, source=source
    )
    with pytest.raises(IndexingError) as exc:
        await expander.expand(_result(version.id, view, T2, INPUT))
    assert exc.value.code == "auto_merge_index_too_large"
    assert spy.list_calls == 0


def test_defaults():
    asyncio.run(_defaults())


def test_min_children_one_merges_and_loses():
    asyncio.run(_min_children_one_merges_and_loses())


def test_ratio_blocks_merge():
    asyncio.run(_ratio_blocks_merge())


def test_parent_too_large_blocks_merge():
    asyncio.run(_parent_too_large_blocks_merge())


def test_index_too_large():
    asyncio.run(_index_too_large())
