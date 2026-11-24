import asyncio

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_indexing.parent_document import ParentDocumentConfig, ParentDocumentExpander
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

T2 = "Alpha one. Beta two.\n\nGamma three. Delta four."


def _cp(codes):
    return "".join(chr(int(code, 16)) for code in codes.split())


# Arabic built from code points (no non-ASCII characters in source); 25 characters in total.
TA = " ".join(
    [
        _cp("0627 0644 0623 0648 0644 0649") + ".",
        _cp("0627 0644 062b 0627 0646 064a 0629 061f"),
        _cp("0627 0644 062b 0627 0644 062b 0629") + ".",
    ]
)
SPANS = [
    (0, 10, T2[0:10], 0, 20),
    (11, 20, T2[11:20], 0, 20),
    (22, 34, T2[22:34], 22, 46),
    (35, 46, T2[35:46], 22, 46),
]
# (chunk_id, start, end, parent_start, parent_end, score, rank)
INPUT = [("c2", 11, 20, 0, 20, 0.9, 1), ("c3", 22, 34, 22, 46, 0.8, 2), ("c1", 0, 10, 0, 20, 0.7, 3)]


def _result(version, view, text, rows):
    hits = tuple(
        RetrievedChunk(
            chunk_id=cid,
            extraction_view_id=view,
            start=a,
            end=b,
            block_ids=("b00000",),
            text=text[a:b],
            score=score,
            rank=rank,
            parent_start=ps,
            parent_end=pe,
        )
        for cid, a, b, ps, pe, score, rank in rows
    )
    return RetrievalResult(technique_id="R02", grounding="corpus", index_version_id=version, hits=hits, scoring="test")


async def _case(text, spans, rows, **config):
    scope, _, index, version = await ready_index(spans, LexicalTestEmbedder())
    view = version.extraction_view_ids[0]
    result = _result(version.id, view, text, rows)
    source = StaticSourceReader(
        {
            (scope.tenant_id, scope.project_id, view): ViewSource(
                extraction_view_id=view, text=text, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(text)),)
            )
        }
    )
    expander = ParentDocumentExpander(
        ParentDocumentConfig(**config), scope=scope, index_version=version, index=index, source=source
    )
    out = await expander.expand(result)
    assert all(h.text == text[h.start : h.end] for h in out.result.hits)
    return scope, view, index, version, source, result, out


async def _parent_return_collapse_and_ranks():
    _, _, _, _, _, _, out = await _case(T2, SPANS, INPUT)
    first, second = out.result.hits
    assert (first.start, first.end, first.text, first.rank) == (0, 20, "Alpha one. Beta two.", 1)
    assert (second.start, second.end, second.text, second.rank) == (22, 46, "Gamma three. Delta four.", 2)
    assert first.score == 0.9 and second.score == 0.8
    assert tuple(s.chunk_id for s in out.trace.contexts[0].seeds) == ("c2", "c1")
    assert (out.trace.input_chars, out.trace.output_chars) == (31, 44)
    assert out.result.technique_id == "R22" and out.result.scoring == "parent-document"
    assert out.trace.dropped_for_budget == ()


async def _too_large_parent_keeps_child():
    _, _, _, _, _, _, out = await _case(T2, SPANS, INPUT, max_parent_chars=20)
    first, second = out.result.hits
    assert (first.start, first.end) == (0, 20)
    assert (second.start, second.end, second.chunk_id) == (22, 34, "c3")
    assert out.trace.contexts[1].flags == ("parent_too_large",)
    assert out.trace.output_chars == 32


async def _budget_drops_second_parent():
    _, _, _, _, _, _, out = await _case(T2, SPANS, INPUT, max_total_chars=30)
    assert len(out.result.hits) == 1
    assert out.trace.dropped_for_budget == ("c3",)


async def _missing_parent():
    rows = [("c2", 11, 20, None, None, 0.9, 1)]
    scope, view, index, version, source, result, _ = await _case(T2, SPANS, INPUT)
    bare = _result(version.id, view, T2, rows)
    strict = ParentDocumentExpander(
        ParentDocumentConfig(), scope=scope, index_version=version, index=index, source=source
    )
    with pytest.raises(IndexingError) as exc:
        await strict.expand(bare)
    assert exc.value.code == "parent_unavailable"
    lenient = ParentDocumentExpander(
        ParentDocumentConfig(on_missing_parent="keep_child"),
        scope=scope,
        index_version=version,
        index=index,
        source=source,
    )
    out = await lenient.expand(bare)
    assert [(h.chunk_id, h.start, h.end) for h in out.result.hits] == [("c2", 11, 20)]
    assert out.trace.contexts[0].flags == ("no_parent",)


async def _arabic_parent():
    spans = [(0, 7, TA[:7], 0, 25), (8, 16, TA[8:16], 0, 25), (17, 25, TA[17:25], 0, 25)]
    _, _, _, _, _, _, out = await _case(TA, spans, [("a1", 8, 16, 0, 25, 0.9, 1)])
    assert [(h.start, h.end, h.text) for h in out.result.hits] == [(0, 25, TA)]


async def _scope_and_source_mismatch():
    scope, view, index, version, source, result, _ = await _case(T2, SPANS, INPUT)
    with pytest.raises(NotFoundError):
        await ParentDocumentExpander(
            ParentDocumentConfig(), scope=make_scope(), index_version=version, index=index, source=source
        ).expand(result)
    expander = ParentDocumentExpander(
        ParentDocumentConfig(), scope=scope, index_version=version, index=index, source=source
    )
    bad = _result(version.id, view, T2, [("c4", 35, 46, 22, 47, 0.9, 1)])
    with pytest.raises(IndexingError) as exc:
        await expander.expand(bad)
    assert exc.value.code == "expansion_source_mismatch"


def test_parent_return_collapse_and_ranks():
    asyncio.run(_parent_return_collapse_and_ranks())


def test_too_large_parent_keeps_child():
    asyncio.run(_too_large_parent_keeps_child())


def test_budget_drops_second_parent():
    asyncio.run(_budget_drops_second_parent())


def test_missing_parent():
    asyncio.run(_missing_parent())


def test_arabic_parent():
    asyncio.run(_arabic_parent())


def test_scope_and_source_mismatch():
    asyncio.run(_scope_and_source_mismatch())
