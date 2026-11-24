"""Opt-in live run of R37 (summary tree) on local Ollama: gemma4:e4b (chat) and bge-m3 (embeddings).

One local run, not an evaluation; output not bit-identical across runs. Only lineage, grounding, tracing and
call-count invariants are asserted; no summary, clustering or retrieval quality is asserted. A skipped invalid
summary is a valid outcome. Local Ollama only: no keys, no paid calls.
"""

import asyncio
import os

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.summary_tree import SummaryTree
from rag_foundry_indexing import (
    DenseConfig,
    DenseRetriever,
    RaptorBuildConfig,
    RaptorConfig,
    RaptorRetriever,
    RaptorTreeBuilder,
)
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger

from tests.indexing._graph_fixture import evidence_span_recall
from tests.indexing._raptor_fixture import (
    LABEL_OF_START,
    Q_ARABIC,
    Q_BUDGET,
    Q_DETAIL,
    Q_OVERVIEW,
    make_fixture_index,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

QUESTIONS = (("Q_BUDGET", Q_BUDGET), ("Q_OVERVIEW", Q_OVERVIEW), ("Q_DETAIL", Q_DETAIL), ("Q_ARABIC", Q_ARABIC))


def _stage(name, usage):
    print(
        f"stage={name}",
        f"calls={len(usage)}",
        f"input_tokens={sum(u.input_tokens for u in usage)}",
        f"output_tokens={sum(u.output_tokens for u in usage)}",
        f"latency_ms={round(sum(u.latency_ms for u in usage))}",
    )


def _key(span):
    return (span.chunk_id, span.start, span.end)


def test_raptor_live_smoke():
    async def run():
        settings = load_settings()
        chat_ledger, embed_ledger = UsageLedger(), UsageLedger()
        chat = RecordingChatModel(
            OllamaChatModel(ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300),
            chat_ledger,
        )
        embedder = RecordingEmbeddingModel(
            OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url), embed_ledger
        )
        scope, _stranger, index, version, source, _e, chunks, view_id = await make_fixture_index(embedder=embedder)
        _stage("index_embed", embed_ledger.records())

        all_leaf_spans = {(c.id, c.start, c.end) for c in chunks}

        async def build(label, config):
            chat_before, embed_before = chat_ledger.totals().calls, embed_ledger.totals().calls
            report = await RaptorTreeBuilder(
                config,
                scope=scope,
                index_version=version,
                index=index,
                source=source,
                chat=chat,
                embedder=embedder,
            ).build()
            chat_calls = chat_ledger.totals().calls - chat_before
            embed_calls = embed_ledger.totals().calls - embed_before
            tree = await index.get_summary_tree(scope, version.id, report.tree_key)
            # Structure: the tree validates (pydantic) and round-trips through JSON and the store.
            assert SummaryTree.from_json(tree.to_json()) == tree
            assert tree.tree_key == report.tree_key
            leaves = [n for n in tree.nodes if n.level == 0]
            summaries = [n for n in tree.nodes if n.synthetic]
            assert len(leaves) == len(chunks) and all(not n.synthetic for n in leaves)
            assert {_key(s) for root in tree.roots() for s in tree.lineage(root)} == all_leaf_spans
            # Call counts: chat calls are summary nodes plus skipped (promoted) clusters, counted on the ledger.
            assert chat_calls == len(summaries) + len(report.promoted) == report.chat_calls
            assert embed_calls == report.embed_calls
            print(
                f"{label} levels={report.levels} nodes={report.nodes} summaries={len(summaries)} leaves={len(leaves)}"
            )
            print(f"{label} promoted_clusters={len(report.promoted)} sizes={[len(s) for _, s in report.promoted]}")
            _stage(f"{label}_chat", report.usage_chat)
            _stage(f"{label}_embed", report.usage_embed)
            return tree

        tree = await build("full_tree", RaptorBuildConfig(on_invalid="skip"))
        flat_tree = await build("flat_tree", RaptorBuildConfig(max_levels=1, on_invalid="skip"))

        def raptor(tree_, **config):
            return RaptorRetriever(
                RaptorConfig(max_hits=3, **config),
                scope=scope,
                tree=tree_,
                index=index,
                index_version=version,
                embedder=embedder,
            )

        dense = DenseRetriever(DenseConfig(k=3), scope=scope, index_version=version, index=index, embedder=embedder)
        by_id = {c.id: c for c in chunks}
        query_embeds = 0
        for name, question in QUESTIONS:
            print(f"question {name}: {question.text}")
            arms = (
                ("collapsed", tree, raptor(tree, mode="collapsed")),
                ("traversal", tree, raptor(tree, mode="traversal")),
                ("flat summary index (R37 on max_levels=1)", flat_tree, raptor(flat_tree)),
            )
            for arm, tree_, retriever in arms:
                before = embed_ledger.totals().calls
                result = await retriever.retrieve(question.text)
                query_embeds += embed_ledger.totals().calls - before
                trace = retriever.last_trace
                assert trace is not None
                assert [chunk_id for chunk_id, _ in trace.hit_lineage] == [h.chunk_id for h in result.hits]
                for hit, (_, via) in zip(result.hits, trace.hit_lineage, strict=True):
                    chunk = by_id[hit.chunk_id]  # an original chunk of the index
                    assert (hit.start, hit.end, hit.text) == (chunk.start, chunk.end, chunk.text)
                    assert hit.chunk_id in {s.chunk_id for s in tree_.lineage(via)}
                labels = [LABEL_OF_START[h.start] for h in result.hits if h.extraction_view_id == view_id]
                recall = evidence_span_recall(result, question.gold, view_id)
                print(f"{name} {arm} hits={labels} recall={recall:.2f} (hits={len(result.hits)})")
            result = await dense.retrieve(question.text)
            labels = [LABEL_OF_START[h.start] for h in result.hits if h.extraction_view_id == view_id]
            recall = evidence_span_recall(result, question.gold, view_id)
            print(f"{name} R02 dense k=3 hits={labels} recall={recall:.2f} (hits={len(result.hits)})")
        print(f"query_embed_calls_raptor={query_embeds} (3 arms x {len(QUESTIONS)} questions)")
        assert query_embeds == 3 * len(QUESTIONS)
        totals = chat_ledger.totals()
        print(
            f"chat ledger calls={totals.calls} input_tokens={totals.input_tokens} output_tokens={totals.output_tokens}"
        )

    asyncio.run(run())
