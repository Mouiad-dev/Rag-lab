import asyncio

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing.context_expansion import (
    ExpandingRetriever,
    ExpansionSeed,
    SourceBlockRef,
    ViewSource,
    merge_overlapping,
)
from rag_foundry_indexing.sentence_window import SentenceWindowConfig, SentenceWindowExpander
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

T1 = "Alpha one. Beta two. Gamma three.\n\nDelta four. Eps five."
TA = (
    "\u0627\u0644\u0623\u0648\u0644\u0649. "
    "\u0627\u0644\u062b\u0627\u0646\u064a\u0629\u061f "
    "\u0627\u0644\u062b\u0627\u0644\u062b\u0629."
)
T3 = "Alphaaaaa oneeeeee twoooooo three."


def _result(version, view, text, spans, technique="R02"):
    hits = tuple(
        RetrievedChunk(
            chunk_id=f"c{i}",
            extraction_view_id=view,
            start=a,
            end=b,
            block_ids=("b00000",),
            text=text[a:b],
            score=1.0 - i / 10,
            rank=i + 1,
        )
        for i, (a, b) in enumerate(spans)
    )
    return RetrievalResult(
        technique_id=technique, grounding="corpus", index_version_id=version, hits=hits, scoring="test"
    )


async def _case(text, spans, seeds, **config):
    embedder = LexicalTestEmbedder()
    scope, _, index, version = await ready_index(spans, embedder)
    view = version.extraction_view_ids[0]
    result = _result(version.id, view, text, seeds)
    source = StaticSourceReader(
        {
            (scope.tenant_id, scope.project_id, view): ViewSource(
                extraction_view_id=view, text=text, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(text)),)
            )
        }
    )
    expander = SentenceWindowExpander(
        SentenceWindowConfig(**config), scope=scope, index_version=version, index=index, source=source
    )
    out = await expander.expand(result)
    assert all(h.text == text[h.start : h.end] for h in out.result.hits)
    return scope, view, index, version, result, out


async def _sentence_windows_and_merge_rules():
    _, _, _, _, _, out = await _case(
        T1,
        [(0, 10, T1[0:10]), (11, 20, T1[11:20]), (21, 33, T1[21:33]), (35, 46, T1[35:46]), (47, 56, T1[47:56])],
        [(11, 20)],
    )
    h = out.result.hits[0]
    assert (h.start, h.end, h.text) == (0, 33, "Alpha one. Beta two. Gamma three.")
    assert h.chunk_id.startswith("ctx_") and out.result.technique_id == "R21"
    assert out.result.grounding == "corpus" and out.result.scoring == "sentence-window"
    assert out.result.index_version_id == out.input.index_version_id
    _, _, _, _, _, out = await _case(
        T1,
        [(0, 10, T1[:10]), (11, 20, T1[11:20]), (21, 33, T1[21:33]), (35, 46, T1[35:46]), (47, 56, T1[47:])],
        [(21, 33)],
    )
    assert (out.result.hits[0].start, out.result.hits[0].end) == (11, 33)
    _, _, _, _, _, out = await _case(
        T1,
        [(0, 10, T1[:10]), (11, 20, T1[11:20]), (21, 33, T1[21:33]), (35, 46, T1[35:46]), (47, 56, T1[47:])],
        [(21, 33)],
        stay_in_paragraph=False,
    )
    assert (out.result.hits[0].start, out.result.hits[0].end) == (11, 46)


async def _overlap_budget_long_sentence_and_arabic():
    spans = [(0, 10, T1[:10]), (11, 20, T1[11:20]), (21, 33, T1[21:33]), (35, 46, T1[35:46]), (47, 56, T1[47:])]
    _, _, _, _, _, out = await _case(T1, spans, [(11, 20), (35, 46)])
    assert [(h.start, h.end) for h in out.result.hits] == [(0, 33), (35, 56)]
    assert (out.trace.input_chars, out.trace.output_chars) == (20, 54)
    _, _, _, _, _, out = await _case(T1, spans, [(11, 20), (35, 46)], max_total_chars=40)
    assert out.trace.dropped_for_budget == ("c1",)
    _, _, _, _, _, out = await _case(T1, spans, [(0, 10), (21, 33)])
    assert [(h.start, h.end) for h in out.result.hits] == [(0, 33)]
    assert tuple(x.chunk_id for x in out.trace.contexts[0].seeds) == ("c0", "c1")
    assert merge_overlapping(
        (
            ("v", 0, 10, (ExpansionSeed(chunk_id="a", start=0, end=10, rank=1),), ()),
            ("v", 8, 20, (ExpansionSeed(chunk_id="b", start=8, end=20, rank=2),), ()),
            ("v", 18, 30, (ExpansionSeed(chunk_id="c", start=18, end=30, rank=3),), ()),
        )
    )[0][1:3] == (0, 30)
    assert (
        len(
            merge_overlapping(
                (
                    ("v", 0, 10, (ExpansionSeed(chunk_id="a", start=0, end=10, rank=1),), ()),
                    ("v", 10, 20, (ExpansionSeed(chunk_id="b", start=10, end=20, rank=2),), ()),
                )
            )
        )
        == 2
    )
    _, _, _, _, _, out = await _case(T1, spans, [(11, 20)], max_window_chars=25)
    assert (out.result.hits[0].start, out.result.hits[0].end) == (11, 20)
    assert out.result.hits[0].chunk_id == "c0" and "window_reduced" in out.trace.contexts[0].flags
    _, _, _, _, _, out = await _case(T3, [(0, 9, T3[:9])], [(0, 9)], max_window_chars=20)
    assert (out.result.hits[0].start, out.result.hits[0].end) == (0, 9)
    assert out.result.hits[0].chunk_id == "c0" and "seed_only" in out.trace.contexts[0].flags
    _, _, _, _, _, out = await _case(TA, [(0, 7, TA[:7]), (8, 16, TA[8:16]), (17, 25, TA[17:25])], [(8, 16)])
    assert [(h.start, h.end, h.text) for h in out.result.hits] == [(0, 25, TA)]


async def _validation_and_expanding_retriever():
    spans = [(0, 10, T1[:10]), (11, 20, T1[11:20]), (21, 33, T1[21:33]), (35, 46, T1[35:46]), (47, 56, T1[47:])]
    scope, view, index, version, result, out = await _case(T1, spans, [(11, 20)])
    source = StaticSourceReader(
        {(scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=T1, blocks=())}
    )
    expander = SentenceWindowExpander(
        SentenceWindowConfig(), scope=scope, index_version=version, index=index, source=source
    )

    class Base:
        async def retrieve(self, q):
            return result

    wrapper = ExpandingRetriever(Base(), expander)
    got = await wrapper.retrieve("q")
    assert [(h.chunk_id, h.start, h.end) for h in got.hits] == [(h.chunk_id, h.start, h.end) for h in out.result.hits]
    assert wrapper.last is not None and wrapper.last.result == got
    assert all(h.text == T1[h.start : h.end] for h in out.result.hits)
    with pytest.raises(NotFoundError):
        await SentenceWindowExpander(
            SentenceWindowConfig(), scope=make_scope(), index_version=version, index=index, source=source
        ).expand(result)
    stranger = make_scope()
    both = StaticSourceReader(
        {
            (scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=T1, blocks=()),
            (stranger.tenant_id, stranger.project_id, view): ViewSource(extraction_view_id=view, text=T1, blocks=()),
        }
    )
    with pytest.raises(NotFoundError):  # index scope check must fire even when the reader would serve the view
        await SentenceWindowExpander(
            SentenceWindowConfig(), scope=stranger, index_version=version, index=index, source=both
        ).expand(result)
    with pytest.raises(IndexingError) as exc:
        await expander.expand(result.model_copy(update={"index_version_id": "another-version"}))
    assert exc.value.code == "expansion_version_mismatch"
    foreign = result.hits[0].model_copy(update={"extraction_view_id": "outside-view"})
    with pytest.raises(IndexingError) as exc:
        await expander.expand(result.model_copy(update={"hits": (foreign,)}))
    assert exc.value.code == "expansion_view_not_in_index"
    bad = result.hits[0].model_copy(update={"text": "Beta twx."})
    with pytest.raises(IndexingError) as exc:
        await expander.expand(result.model_copy(update={"hits": (bad,)}))
    assert exc.value.code == "expansion_source_mismatch"
    too_many = tuple(result.hits[0].model_copy(update={"chunk_id": f"extra-{i}", "rank": i + 1}) for i in range(21))
    with pytest.raises(IndexingError) as exc:
        await expander.expand(result.model_copy(update={"hits": too_many}))
    assert exc.value.code == "expansion_too_many_candidates"


def test_sentence_windows_and_merge_rules():
    asyncio.run(_sentence_windows_and_merge_rules())


def test_overlap_budget_long_sentence_and_arabic():
    asyncio.run(_overlap_budget_long_sentence_and_arabic())


def test_validation_and_expanding_retriever():
    asyncio.run(_validation_and_expanding_retriever())


def test_not_ready_index_version_is_rejected():
    async def run():
        embedder = LexicalTestEmbedder()
        scope, _, index, version = await ready_index([(0, 10, T1[:10])], embedder, ready=False)
        view = version.extraction_view_ids[0]
        result = _result(version.id, view, T1, [(0, 10)])
        source = StaticSourceReader(
            {(scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=T1, blocks=())}
        )
        expander = SentenceWindowExpander(
            SentenceWindowConfig(), scope=scope, index_version=version, index=index, source=source
        )
        with pytest.raises(IndexingError) as exc:
            await expander.expand(result)
        assert exc.value.code == "not_ready"

    asyncio.run(run())
