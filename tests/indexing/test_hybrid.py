import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_indexing import (
    BM25Config,
    BM25Retriever,
    DenseConfig,
    DenseRetriever,
    HybridConfig,
    HybridRetriever,
    register_builtin_plugins,
)
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.indexing._inmemory_index import ready_index
from tests.indexing._lexical_embedder import LexicalTestEmbedder


def run(awaitable):
    return asyncio.run(awaitable)


def _spans():
    texts = (
        "the tidal observatory records sea levels",
        "\u0631\u0642\u0645 \u0627\u0644\u0642\u0637\u0639\u0629 AB-1234",  # "part number"
        "\u0631\u0642\u0645 \u0627\u0644\u0642\u0637\u0639\u0629 AB-1243",
        "chess club meets on tuesday",
    )
    output = []
    offset = 0
    for text in texts:
        output.append((offset, offset + len(text), text))
        offset += len(text) + 2
    return output


def test_hybrid_fuses_branch_evidence_and_respects_branch_candidates():
    async def scenario():
        embedder = LexicalTestEmbedder()
        owner, _, index, version = await ready_index(_spans(), embedder)
        hybrid = HybridRetriever(HybridConfig(), scope=owner, index_version=version, index=index, embedder=embedder)
        result = await hybrid.retrieve("AB-1234")
        assert result.technique_id == "R06"
        assert result.index_version_id == version.id
        assert result.scoring == "hybrid[R03+R02]/R07:rrf-k60"
        hit = result.hits[0]
        assert "AB-1234" in hit.text
        term = next(
            component
            for component in hit.score_components
            if component.name == "bm25_term" and component.term == "ab-1234"
        )
        assert term.term == "ab-1234" and term.source == "R03"
        assert any(
            component.name == "cosine_similarity" and component.source == "R02" for component in hit.score_components
        )
        contributions = [component.value for component in hit.score_components if component.name == "rrf_contribution"]
        assert len(contributions) == 2
        assert sum(contributions) == pytest.approx(hit.score)
        lexical = BM25Retriever(BM25Config(k=20), scope=owner, index_version=version, index=index)
        dense = DenseRetriever(DenseConfig(k=20), scope=owner, index_version=version, index=index, embedder=embedder)
        lexical_result = await lexical.retrieve("AB-1234")
        dense_result = await dense.retrieve("AB-1234")
        assert {h.chunk_id for h in result.hits} <= {h.chunk_id for h in (*lexical_result.hits, *dense_result.hits)}

    run(scenario())


@pytest.mark.parametrize("weight,branch", [(1.0, "R03"), (0.0, "R02")])
def test_weighted_fusion_extremes_follow_branch_order(weight, branch):
    async def scenario():
        embedder = LexicalTestEmbedder()
        owner, _, index, version = await ready_index(_spans(), embedder)
        lexical = await BM25Retriever(BM25Config(k=20), scope=owner, index_version=version, index=index).retrieve(
            "AB-1234"
        )
        dense = await DenseRetriever(
            DenseConfig(k=20), scope=owner, index_version=version, index=index, embedder=embedder
        ).retrieve("AB-1234")
        hybrid = HybridRetriever(
            HybridConfig(fusion="R08", lexical_weight=weight),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
        )
        result = await hybrid.retrieve("AB-1234")
        expected = lexical if branch == "R03" else dense
        shared = {hit.chunk_id for hit in lexical.hits} & {hit.chunk_id for hit in dense.hits}
        assert [hit.chunk_id for hit in result.hits if hit.chunk_id in shared] == [
            hit.chunk_id for hit in expected.hits if hit.chunk_id in shared
        ]
        assert result.scoring == f"hybrid[R03+R02]/R08:weighted-minmax:wl={weight:.2f}"
        # Both branches order the shared chunks alike here, so also pin the weights themselves.
        for hit in result.hits:
            norms = {c.source: c.value for c in hit.score_components if c.name == "min_max_score"}
            parts = {c.source: c.value for c in hit.score_components if c.name == "weighted_contribution"}
            for source, expected_weight in (("R03", weight), ("R02", 1.0 - weight)):
                if source in parts:
                    assert parts[source] == pytest.approx(expected_weight * norms[source])
            assert sum(parts.values()) == pytest.approx(hit.score)
        scores = {hit.text[-7:]: hit.score for hit in result.hits}
        if weight == 1.0:
            assert sorted(hit.score for hit in result.hits) == pytest.approx([0.0, 0.0, 0.0, 1.0])
        else:
            # the test embedder's 0.01 baseline keeps this a hair above the dense half-way point
            assert scores["AB-1243"] == pytest.approx(0.5, abs=1e-4)

    run(scenario())


def test_scope_readiness_and_lexical_failures():
    async def scenario():
        embedder = LexicalTestEmbedder()
        owner, stranger, index, version = await ready_index(_spans(), embedder)
        foreign = HybridRetriever(HybridConfig(), scope=stranger, index_version=version, index=index, embedder=embedder)
        with pytest.raises(NotFoundError):
            await foreign.retrieve("AB-1234")

        pending_embedder = LexicalTestEmbedder()
        pending_owner, _, pending_index, pending_version = await ready_index(_spans(), pending_embedder, ready=False)
        pending = HybridRetriever(
            HybridConfig(),
            scope=pending_owner,
            index_version=pending_version,
            index=pending_index,
            embedder=pending_embedder,
        )
        with pytest.raises(IndexingError) as info:
            await pending.retrieve("AB-1234")
        assert info.value.code == "not_ready"
        assert not any(call[1] == "query" for call in pending_embedder.calls)
        # Hybrid's own ready check runs before any branch: with R04 the branch would say fts_unavailable.
        pending_fts = HybridRetriever(
            HybridConfig(lexical="R04"),
            scope=pending_owner,
            index_version=pending_version,
            index=pending_index,
            embedder=pending_embedder,
            full_text=pending_index,
        )
        with pytest.raises(IndexingError) as info:
            await pending_fts.retrieve("AB-1234")
        assert info.value.code == "not_ready"

        with pytest.raises(IndexingError) as info:
            HybridRetriever(
                HybridConfig(lexical="R04"), scope=owner, index_version=version, index=index, embedder=embedder
            )
        assert info.value.code == "fts_unavailable"
        fts = HybridRetriever(
            HybridConfig(lexical="R04"),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
            full_text=index,
        )
        with pytest.raises(IndexingError) as info:
            await fts.retrieve("AB-1234")
        assert info.value.code == "fts_unavailable"

        calls_before = len(embedder.calls)
        with pytest.raises(IndexingError) as info:
            await HybridRetriever(
                HybridConfig(), scope=owner, index_version=version, index=index, embedder=embedder
            ).retrieve("?!")
        assert info.value.code == "no_query_terms"
        assert len(embedder.calls) == calls_before

    run(scenario())


def test_registry_validates_hybrid_config():
    catalogue = Path(__file__).resolve().parents[2] / "config" / "techniques.json"
    registry = TechniqueRegistry(load_catalogue(catalogue))
    register_builtin_plugins(registry)
    assert "R06" in registry.registered_ids()
    assert registry.validate_config("R06", {}) == HybridConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("R06", {"lexical": "R05"})
