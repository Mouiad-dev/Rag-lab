import asyncio
import uuid
from pathlib import Path

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import EntityGraphConfig, EntityGraphRetriever
from rag_foundry_indexing.sentence_window import SentenceWindowConfig, SentenceWindowExpander
from rag_foundry_techniques import check_plugin_contract, load_catalogue

from tests.chunking._scripted_models import ScriptedEmbedder
from tests.indexing._graph_fixture import (
    GOLD_SPANS,
    QUESTION,
    SPANS,
    build_fixture_graph,
    entity_seed,
    evidence_span_recall,
)
from tests.indexing._inmemory_index import ready_index

CATALOGUE = load_catalogue(Path(__file__).resolve().parents[2] / "config" / "techniques.json")


def _retriever(fx, **config):
    return EntityGraphRetriever(
        EntityGraphConfig(**config), scope=fx.scope, graph=fx.graph, index=fx.index, index_version=fx.version
    )


def _paragraph_of(hit):
    return next(i for i, (a, _, _) in enumerate(SPANS) if a == hit.start)


def test_r38_finds_all_three_gold_spans_of_the_three_hop_question():
    async def run():
        fx = await build_fixture_graph()
        retriever = _retriever(fx)
        result = await retriever.retrieve(QUESTION)
        assert evidence_span_recall(result, GOLD_SPANS, fx.view_id) == 1.0
        assert [_paragraph_of(h) for h in result.hits] == [0, 7, 14]
        assert [h.text for h in result.hits] == [SPANS[i][2] for i in (0, 7, 14)]
        assert all(h.score == 1.0 for h in result.hits) and result.scoring == "entity-graph"
        assert result.technique_id == "R38" and result.index_version_id == fx.version.id
        trace = retriever.last_trace
        assert (
            trace.status == "ok"
            and trace.mode == "path"
            and len(trace.path) == 3
            and len(trace.matched_entity_ids) == 2
        )
        assert [d for _, d in trace.path] == ["forward", "forward", "reverse"]
        assert [s.quote for s in trace.evidence] == [SPANS[i][2] for i in (0, 7, 14)]
        assert {c.name for c in result.hits[0].score_components} == {"graph_hop", "path_edges"}

    asyncio.run(run())


def test_r21_sentence_window_from_the_same_seeds_finds_two_of_three():
    async def run():
        fx = await build_fixture_graph()
        seed = entity_seed(QUESTION, fx.chunks, fx.version)
        assert len(seed.hits) == 2
        expander = SentenceWindowExpander(
            SentenceWindowConfig(window=5, stay_in_paragraph=False, max_window_chars=20000),
            scope=fx.scope,
            index_version=fx.version,
            index=fx.index,
            source=fx.source,
        )
        expanded = await expander.expand(seed)
        assert evidence_span_recall(expanded.result, GOLD_SPANS, fx.view_id) == pytest.approx(2 / 3)
        a, b = GOLD_SPANS[1]
        assert not any(h.start <= a and b <= h.end for h in expanded.result.hits)

    asyncio.run(run())


def test_losing_case_alias_matches_one_entity_and_recovers_one_third():
    async def run():
        fx = await build_fixture_graph()
        retriever = _retriever(fx)
        result = await retriever.retrieve("How is Dr. Haddad connected to Omar Nasser?")
        assert (
            retriever.last_trace.mode == "neighborhood"
            and len(retriever.last_trace.matched_entity_ids) == 1
            and retriever.last_trace.path == ()
        )
        assert evidence_span_recall(result, GOLD_SPANS, fx.view_id) == pytest.approx(1 / 3)
        assert [_paragraph_of(h) for h in result.hits] == [14]
        assert result.hits[0].score == 0.5

    asyncio.run(run())


def test_losing_case_no_entity_and_hop_limit_return_no_hits():
    async def run():
        fx = await build_fixture_graph()
        retriever = _retriever(fx)
        result = await retriever.retrieve("Who repaired the lighthouse?")
        assert result.hits == () and retriever.last_trace.status == "no_entity_match"
        assert retriever.last_trace.mode == "none"
        short = _retriever(fx, max_hops=2)
        result = await short.retrieve(QUESTION)
        assert result.hits == () and short.last_trace.status == "no_path_within_hops"
        assert short.last_trace.mode == "path"
        assert len(short.last_trace.matched_entity_ids) == 2
        with pytest.raises(IndexingError) as empty:
            await retriever.retrieve("  ")
        assert empty.value.code == "empty_query"

    asyncio.run(run())


def test_guards_scope_version_readiness_and_missing_evidence():
    async def run():
        fx = await build_fixture_graph()
        with pytest.raises(IndexingError) as err:
            EntityGraphRetriever(
                EntityGraphConfig(), scope=fx.stranger, graph=fx.graph, index=fx.index, index_version=fx.version
            )
        assert err.value.code == "graph_scope_mismatch"
        embedder = ScriptedEmbedder(lambda text: (1.0, 0.0, 0.0, 0.0))
        scope, _, index, pending = await ready_index(SPANS, embedder, ready=False)
        graph = fx.graph.model_copy(
            update={"tenant_id": scope.tenant_id, "project_id": scope.project_id, "index_version_id": pending.id}
        )
        retriever = EntityGraphRetriever(
            EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=pending
        )
        with pytest.raises(IndexingError) as err:
            await retriever.retrieve(QUESTION)
        assert err.value.code == "not_ready"

        relations = tuple(
            r.model_copy(
                update={"evidence": tuple(s.model_copy(update={"chunk_id": str(uuid.uuid4())}) for s in r.evidence)}
            )
            if "supplies" in r.evidence[0].quote
            else r
            for r in fx.graph.relations
        )
        broken = fx.graph.model_copy(update={"relations": relations})
        with pytest.raises(IndexingError) as err:
            await _retriever_for(fx, broken).retrieve(QUESTION)
        assert err.value.code == "graph_evidence_missing"

    asyncio.run(run())


def _retriever_for(fx, graph):
    return EntityGraphRetriever(
        EntityGraphConfig(), scope=fx.scope, graph=graph, index=fx.index, index_version=fx.version
    )


def test_graph_of_another_version_in_the_same_scope_is_rejected():
    async def run():
        fx = await build_fixture_graph()
        foreign = fx.graph.model_copy(update={"index_version_id": str(uuid.uuid4())})
        with pytest.raises(IndexingError) as err:
            _retriever_for(fx, foreign)
        assert err.value.code == "graph_scope_mismatch"

    asyncio.run(run())


def test_plugin_passes_the_contract_kit():
    manifest = check_plugin_contract(EntityGraphRetriever, CATALOGUE, sample_config={})
    assert manifest.technique_id == "R38" and manifest.kind.value == "retriever"
    assert manifest.fidelity.value == "adaptation"
