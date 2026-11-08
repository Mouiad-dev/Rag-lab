import asyncio
import os
from pathlib import Path

from rag_foundry_indexing import BM25Config, BM25Retriever, DenseConfig, DenseRetriever, HybridConfig, HybridRetriever
from rag_foundry_indexing.paired import PairedCase, case_metrics, choose_weight, render_markdown, run_paired

from tests.indexing._inmemory_index import ready_index
from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._paired_fixture import CASES, SPANS


def evaluation():
    embedder = LexicalTestEmbedder()
    owner, _, index, version = asyncio.run(ready_index(SPANS, embedder))
    labels = {
        "R02": DenseRetriever(DenseConfig(k=5), scope=owner, index_version=version, index=index, embedder=embedder),
        "R03": BM25Retriever(BM25Config(k=5), scope=owner, index_version=version, index=index),
        "R06+R07": HybridRetriever(
            HybridConfig(k=5, branch_k=10), scope=owner, index_version=version, index=index, embedder=embedder
        ),
    }

    def sweep(weight):
        return HybridRetriever(
            HybridConfig(k=5, branch_k=10, fusion="R08", lexical_weight=weight),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
        )

    cases = []
    for case_id, cohort, split, query, relevant in CASES:
        spans = tuple((start, end) for start, end, text in SPANS if text == relevant) if relevant else ()
        cases.append(PairedCase(case_id, cohort, split, query, spans))
    return tuple(cases), labels, sweep


def test_paired_render_is_deterministic_and_has_denominators():
    cases, labels, sweep = evaluation()
    first = render_markdown(asyncio.run(run_paired(cases, labels, sweep)))
    second = render_markdown(asyncio.run(run_paired(cases, labels, sweep)))
    assert first == second
    report = asyncio.run(run_paired(cases, labels, sweep))
    expected = {
        split: sum(case.split == split and bool(case.relevant) for case in cases) for split in ("dev", "validation")
    }
    assert all(values["n"] == expected["dev"] for values in report["summaries"]["dev"].values())
    assert all(values["n"] == expected["validation"] for values in report["summaries"]["validation"].values())
    assert f"denominators: {sum(not case.relevant for case in cases)} case(s)" in first
    assert all(m["mrr"] >= 0 for split in report["summaries"].values() for m in split.values())
    assert (
        case_metrics(next(c for c in cases if c.case_id == "id-exact"), report["results"]["id-exact"]["R03"])[
            "first_hit_k"
        ]
        == 1
    )
    assert all(
        metric["first_hit_k"] in (None, 1, 2, 3, 4, 5)
        for case in cases
        if case.relevant
        for hit in report["results"][case.case_id].values()
        for metric in [case_metrics(case, hit)]
    )
    evidence = Path(__file__).resolve().parents[2] / "docs/evidence/P5.2/paired_vs_dense.md"
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with evidence.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(first)
    else:
        assert evidence.read_text(encoding="utf-8").replace("\r\n", "\n") == first


def test_weight_selection_uses_dev_only():
    cases, labels, sweep = evaluation()
    created = []

    class RecordingRetriever:
        def __init__(self, inner):
            self.inner = inner
            self.queries = []

        async def retrieve(self, query):
            self.queries.append(query)
            return await self.inner.retrieve(query)

    def recording_sweep(weight):
        wrapped = RecordingRetriever(sweep(weight))
        created.append(wrapped)
        return wrapped

    original = asyncio.run(run_paired(cases, labels, recording_sweep))
    dev_queries = {case.query for case in cases if case.split == "dev"}
    assert len(created) == 12
    assert all(set(retriever.queries) == dev_queries for retriever in created[:11])
    assert set(created[11].queries) == {case.query for case in cases}
    changed = tuple(
        case.__class__(
            case.case_id, case.cohort, case.split, "zzz" if case.split == "validation" else case.query, case.relevant
        )
        for case in cases
    )
    altered = asyncio.run(run_paired(changed, labels, sweep))
    assert original["weight"] == altered["weight"]


def test_choose_weight_non_tied_and_tied():
    assert choose_weight({0.0: 0.7, 0.4: 0.8, 0.8: 0.9}) == 0.8
    assert choose_weight({0.2: 0.9, 0.8: 0.9, 0.7: 0.9}) == 0.7
    assert choose_weight({0.4: 1.0, 0.6: 1.0}) == 0.4  # equally far from 0.5: the smaller weight wins
