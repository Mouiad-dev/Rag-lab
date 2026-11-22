import asyncio
import json

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import (
    CommunitySummary,
    CommunitySummaryBuilder,
    CommunitySummaryConfig,
    detect_communities,
    trace_claims,
)

from tests.indexing._graph_fixture import (
    AR_PERSON,
    AR_PORT,
    SPANS,
    TEXT,
    build_fixture_graph,
    rebuild_graph_reversed,
    summary_chat,
)


def _names(graph, community):
    by_id = {e.entity_id: e.name for e in graph.entities}
    return {by_id[i] for i in community.entity_ids}


def _builder(fx, communities=None, replies=None, **config):
    communities = communities or detect_communities(fx.graph)
    chat = summary_chat(fx.graph, communities, replies)
    return CommunitySummaryBuilder(CommunitySummaryConfig(**config), graph=fx.graph, chat=chat), chat


def _reply(*claims, title="T"):
    return json.dumps({"title": title, "claims": list(claims)})


def test_detection_is_deterministic_and_independent_of_input_order():
    async def run():
        fx = await build_fixture_graph()
        first = detect_communities(fx.graph)
        assert detect_communities(fx.graph) == first
        assert all(c.converged for c in first)

        second_graph = await rebuild_graph_reversed(fx)
        assert second_graph.entities == fx.graph.entities and second_graph.relations == fx.graph.relations
        assert detect_communities(second_graph) == first

        third = await build_fixture_graph(spans=tuple(reversed(SPANS)))
        third_communities = detect_communities(third.graph)
        assert {(c.community_id, c.entity_ids) for c in third_communities} == {
            (c.community_id, c.entity_ids) for c in first
        }

    asyncio.run(run())


def test_two_clusters_and_the_isolated_entity_is_skipped():
    async def run():
        fx = await build_fixture_graph()
        communities = detect_communities(fx.graph)
        assert all(c.community_id.startswith("com_") and len(c.community_id) == 20 for c in communities)
        builder, chat = _builder(fx, communities)
        report = await builder.build()
        assert len(report.summaries) == 2 and len(chat.requests) == 2
        assert {frozenset(_names(fx.graph, c)) for c in communities if len(c.entity_ids) > 1} == {
            frozenset({"Lina Haddad", "Cedar Labs", "Harbor Authority", "Omar Nasser"}),
            frozenset({AR_PERSON, AR_PORT}),
        }
        assert [_names(fx.graph, c) for c in report.skipped_small] == [{"Mira Yassin"}]
        assert report.skipped_small[0].relation_ids == ()
        for c in communities:
            member = set(c.entity_ids)
            for r in fx.graph.relations:
                assert (r.relation_id in c.relation_ids) == (r.source_id in member and r.target_id in member)
        assert report.chat_calls == len(report.summaries) + report.rejected_summaries == len(report.usage)

    asyncio.run(run())


def test_every_kept_claim_has_spans_equal_to_the_view_slice():
    async def run():
        fx = await build_fixture_graph()
        report = await _builder(fx)[0].build()
        view = await fx.source.view_source(fx.scope, fx.view_id)
        assert view.text == TEXT
        for summary in report.summaries:
            assert summary.synthetic is True and summary.graph_key == fx.graph.graph_key
            assert summary.chat_provider and summary.chat_model and summary.prompt_sha256
            assert trace_claims(summary, fx.graph)
            for i, claim in enumerate(summary.claims):
                assert claim.claim_id == f"{summary.community_id}:{i}" and claim.sources
                assert len(set((s.chunk_id, s.start, s.end) for s in claim.sources)) == len(claim.sources)
                for span in claim.sources:
                    assert view.text[span.start : span.end] == span.quote

    asyncio.run(run())


def test_untraced_claims_are_rejected_and_counted():
    async def run():
        fx = await build_fixture_graph()
        communities = detect_communities(fx.graph)
        big = next(c for c in communities if len(c.entity_ids) == 4)
        small = next(c for c in communities if len(c.entity_ids) == 2)
        good = big.entity_ids[0]
        replies = {
            big.community_id: _reply(
                {"text": "unknown", "fact_ids": ["ent_0000000000000000"]},
                {"text": "empty", "fact_ids": []},
                {"text": "other community", "fact_ids": [small.entity_ids[0]]},
                {"text": "mixed", "fact_ids": [good, small.entity_ids[0]]},
                {"text": "ok", "fact_ids": [good]},
            ),
            small.community_id: _reply({"text": "empty", "fact_ids": []}),
        }
        report = await _builder(fx, communities, replies)[0].build()
        assert [s.community_id for s in report.summaries] == [big.community_id]
        assert [c.claim_id for c in report.summaries[0].claims] == [f"{big.community_id}:4"]
        assert sorted(report.rejected_claims) == sorted(
            [
                (big.community_id, 0, "untraced_claim"),
                (big.community_id, 1, "untraced_claim"),
                (big.community_id, 2, "untraced_claim"),
                (big.community_id, 3, "untraced_claim"),
                (small.community_id, 0, "untraced_claim"),
            ]
        )
        assert report.rejected_summaries == 1
        assert report.chat_calls == len(report.summaries) + report.rejected_summaries == len(report.usage)

    asyncio.run(run())


def test_item_limits_and_invalid_replies():
    async def run():
        fx = await build_fixture_graph()
        communities = detect_communities(fx.graph)
        big = next(c for c in communities if len(c.entity_ids) == 4)
        small = next(c for c in communities if len(c.entity_ids) == 2)
        fid = big.entity_ids[0]
        claims = [
            {"text": "x" * 21, "fact_ids": [fid]},
            {"text": "fine", "fact_ids": [fid]},
            {"text": "extra", "fact_ids": [fid]},
        ]
        replies = {big.community_id: _reply(*claims), small.community_id: "not json"}
        report = await _builder(fx, communities, replies, max_claims=2, max_claim_chars=20, on_invalid="skip")[
            0
        ].build()
        assert report.rejected_claims == (
            (big.community_id, 0, "claim_too_long"),
            (big.community_id, 2, "too_many_claims"),
        )
        assert report.rejected_summaries == 1 and len(report.summaries) == 1
        assert report.chat_calls == len(report.summaries) + report.rejected_summaries == len(report.usage)
        with pytest.raises(IndexingError) as error:
            await _builder(fx, communities, replies, max_claims=2, max_claim_chars=20)[0].build()
        assert error.value.code == "graph_invalid"
        order = [c.community_id for c in communities if len(c.entity_ids) >= 2]
        # one usage per chat call made up to and including the failing call
        assert len(error.value.usage) == order.index(small.community_id) + 1

    asyncio.run(run())


def test_trace_claims_rejects_another_graph_and_unresolved_claims():
    async def run():
        fx = await build_fixture_graph()
        other = await build_fixture_graph()
        assert other.graph.graph_key != fx.graph.graph_key
        report = await _builder(fx)[0].build()
        summary = report.summaries[0]
        with pytest.raises(IndexingError) as error:
            trace_claims(summary, other.graph)
        assert error.value.code == "graph_key_mismatch"
        bad_claim = summary.claims[0].model_copy(update={"fact_ids": ("ent_0000000000000000",)})
        bad = CommunitySummary.model_validate(summary.model_copy(update={"claims": (bad_claim,)}).model_dump())
        with pytest.raises(IndexingError) as error:
            trace_claims(bad, fx.graph)
        assert error.value.code == "untraced_claim"

    asyncio.run(run())


def test_community_cap_raises_before_any_chat_call():
    async def run():
        fx = await build_fixture_graph()
        builder, chat = _builder(fx, max_communities=1)
        with pytest.raises(IndexingError) as error:
            await builder.build()
        assert error.value.code == "community_cap_exceeded"
        assert chat.requests == []

    asyncio.run(run())


def test_config_rejects_unknown_fields_and_out_of_range_values():
    with pytest.raises(ValueError):
        CommunitySummaryConfig(unknown=1)
    with pytest.raises(ValueError):
        CommunitySummaryConfig(min_members=1)
    with pytest.raises(ValueError):
        CommunitySummaryConfig(max_communities=501)
    with pytest.raises(ValueError):
        CommunitySummaryConfig(max_claims=21)


def _tie_graph(edges, count):
    import uuid

    from rag_foundry_domain.graphs import Entity, EntityGraph, GraphSpan, Relation

    span = GraphSpan(chunk_id=str(uuid.uuid4()), extraction_view_id=str(uuid.uuid4()), start=0, end=1, quote="x")
    ids = [f"ent_{i:016x}" for i in range(1, count + 1)]
    entities = tuple(Entity(entity_id=i, name=i, type="t", mentions=(span,)) for i in ids)
    relations = tuple(
        Relation(relation_id=f"rel_{n:016x}", source_id=ids[a - 1], target_id=ids[b - 1], label="r", evidence=(span,))
        for n, (a, b) in enumerate(edges, start=1)
    )
    return EntityGraph(
        schema="rag_foundry.entity_graph/1",
        graph_key="0" * 64,
        tenant_id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        index_version_id=str(uuid.uuid4()),
        generator_json="{}",
        entities=entities,
        relations=relations,
    ), ids


def test_equal_weight_ties_go_to_the_smallest_label():
    # Path 1-2-3-4 with equal weights: nodes 2 and 3 each face a tie. The smallest label wins every tie,
    # so the labels collapse onto node 2's label in the first pass and nothing changes in the second.
    graph, ids = _tie_graph([(1, 2), (2, 3), (3, 4)], 4)
    (community,) = detect_communities(graph)
    assert community.entity_ids == tuple(ids)
    assert community.relation_ids == tuple(f"rel_{n:016x}" for n in (1, 2, 3))
    assert community.iterations == 2 and community.converged is True
