import asyncio
import json
import uuid
from pathlib import Path

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import (
    CommunityGlobalSearch,
    CommunitySummaryBuilder,
    CommunitySummaryConfig,
    GlobalSearchConfig,
    detect_communities,
)
from rag_foundry_indexing.graph_global import R39_MAP_SYSTEM_V1, R39_REDUCE_SYSTEM_V1
from rag_foundry_techniques import check_plugin_contract, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._graph_fixture import TEXT, build_fixture_graph, summary_chat

CATALOGUE = load_catalogue(Path(__file__).resolve().parents[2] / "config" / "techniques.json")
QUESTION = "Which organisations work with the Harbor Authority?"
UNRELATED = "What is the capital of Atlantis?"
# "What are the organisations that work with the harbour authority", with no token shared with the Arabic cluster.
AR_QUESTION = json.loads(
    '"'
    "\u0645\u0627 \u0647\u064a \u0627\u0644\u0645\u0624\u0633\u0633\u0627\u062a "
    "\u0627\u0644\u062a\u064a \u062a\u0639\u0645\u0644 \u0645\u0639 "
    "\u0647\u064a\u0626\u0629 \u0627\u0644\u0645\u064a\u0646\u0627\u0621"
    '"'
)


async def _setup():
    fx = await build_fixture_graph()
    communities = detect_communities(fx.graph)
    report = await CommunitySummaryBuilder(
        CommunitySummaryConfig(), graph=fx.graph, chat=summary_chat(fx.graph, communities)
    ).build()
    big = next(c for c in communities if len(c.entity_ids) == 4)
    small = next(c for c in communities if len(c.entity_ids) == 2)
    return fx, report.summaries, big, small


def _chat(scores=None, raw=None, reduce=None):
    """Scripted map and reduce chat. Every community scores 50 and cites all its claims unless overridden."""
    scores, raw = scores or {}, raw or {}

    def respond(request):
        system = request.messages[0].content
        payload = json.loads(request.messages[1].content)
        if system == R39_REDUCE_SYSTEM_V1:
            if callable(reduce):
                return reduce(payload), "stop"
            if reduce is not None:
                return reduce, "stop"
            ids = [p["point_id"] for p in payload["points"]]
            return json.dumps({"answer": [{"text": "Answer.", "point_ids": [i]} for i in ids]}), "stop"
        assert system == R39_MAP_SYSTEM_V1
        claims = payload["community"]["claims"]
        community_id = claims[0]["claim_id"].split(":")[0]
        if community_id in raw:
            return raw[community_id], "stop"
        points = [{"text": c["text"], "claim_ids": [c["claim_id"]]} for c in claims]
        return json.dumps({"helpfulness": scores.get(community_id, 50), "points": points}), "stop"

    return ScriptedChat(respond)


def _search(fx, summaries, chat, **config):
    return CommunityGlobalSearch(
        GlobalSearchConfig(**config),
        scope=fx.scope,
        graph=fx.graph,
        summaries=summaries,
        index=fx.index,
        index_version=fx.version,
        chat=chat,
    )


def test_success_every_sentence_resolves_to_original_source_spans():
    async def run():
        fx, summaries, big, small = await _setup()
        chat = _chat(scores={big.community_id: 90, small.community_id: 10})
        answer = await _search(fx, summaries, chat).answer(QUESTION)
        assert answer.status == "answered" and answer.reason is None and answer.sentences
        assert answer.map_calls == len(summaries) == 2 and answer.reduce_calls == 1
        assert len(answer.map_usage) == answer.map_calls and len(answer.reduce_usage) == 1
        assert len(chat.requests) == 3 and answer.dropped_sentences == 0 and answer.rejected_points == ()
        assert all(s.synthetic is True for s in answer.summaries_used)
        view = await fx.source.view_source(fx.scope, fx.view_id)
        assert view.text == TEXT
        for sentence in answer.sentences:
            assert sentence.point_ids and sentence.claim_ids and sentence.sources
            for span in sentence.sources:
                assert view.text[span.start : span.end] == span.quote
        # The reduce request lists the points best first: helpfulness 90 before 10.
        reduce_request = json.loads(chat.requests[-1].messages[1].content)
        assert [p["point_id"].split(":")[0] for p in reduce_request["points"]][0] == big.community_id

    asyncio.run(run())


def test_retrieve_returns_original_chunks_ordered_by_helpfulness_then_community_id():
    async def run():
        fx, summaries, big, small = await _setup()
        chunks = {c.id: c for c in fx.chunks}
        result = await _search(fx, summaries, _chat(scores={big.community_id: 90, small.community_id: 10})).retrieve(
            QUESTION
        )
        assert result.technique_id == "R39" and result.hits
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True) and scores[0] == 0.9 and scores[-1] == 0.1
        assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))
        assert all(h.text == chunks[h.chunk_id].text for h in result.hits)
        # Equal helpfulness breaks ties by community_id.
        tie = await _search(fx, tuple(reversed(summaries)), _chat()).retrieve(QUESTION)
        first, second = sorted([big.community_id, small.community_id])
        owner = {c.claim_id.split(":")[0]: {s.chunk_id for s in c.sources} for s in summaries for c in s.claims}
        assert tie.hits[0].chunk_id in owner[first] - owner[second]

    asyncio.run(run())


def test_nothing_relevant_abstains_without_a_reduce_call():
    async def run():
        fx, summaries, big, small = await _setup()
        chat = _chat(scores={big.community_id: 0, small.community_id: 0})
        answer = await _search(fx, summaries, chat).answer(UNRELATED)
        assert answer.status == "abstained" and answer.reason == "no_relevant_community"
        assert answer.reduce_calls == 0 and answer.sentences == () and answer.reduce_usage == ()
        assert answer.map_calls == 2 and len(answer.map_usage) == 2 and len(chat.requests) == 2

    asyncio.run(run())


def test_all_untraced_points_are_rejected_and_abstain():
    async def run():
        fx, summaries, big, small = await _setup()
        bad = json.dumps({"helpfulness": 90, "points": [{"text": "x", "claim_ids": ["com_0000000000000000:0"]}]})
        mixed = json.dumps({"helpfulness": 90, "points": [{"text": "y", "claim_ids": []}]})
        chat = _chat(raw={big.community_id: bad, small.community_id: mixed})
        answer = await _search(fx, summaries, chat).answer(QUESTION)
        assert answer.status == "abstained" and answer.reason == "no_relevant_community"
        assert answer.reduce_calls == 0 and len(answer.rejected_points) == 2
        assert {r[2] for r in answer.rejected_points} == {"untraced_point"}

    asyncio.run(run())


def test_point_citing_a_claim_of_another_summary_is_rejected():
    async def run():
        fx, summaries, big, small = await _setup()
        other_claim = next(c.claim_id for s in summaries if s.community_id == small.community_id for c in s.claims)
        cross = json.dumps({"helpfulness": 90, "points": [{"text": "x", "claim_ids": [other_claim]}]})
        chat = _chat(raw={big.community_id: cross}, scores={small.community_id: 0})
        answer = await _search(fx, summaries, chat).answer(QUESTION)
        assert answer.reason == "no_relevant_community"
        assert answer.rejected_points == ((big.community_id, 0, "untraced_point"),)

    asyncio.run(run())


def test_reduce_citing_only_unknown_points_is_untraced():
    async def run():
        fx, summaries, *_ = await _setup()
        reply = json.dumps({"answer": [{"text": "Made up.", "point_ids": ["com_0000000000000000:p0"]}]})
        answer = await _search(fx, summaries, _chat(reduce=reply)).answer(QUESTION)
        assert answer.status == "abstained" and answer.reason == "reduce_untraced"
        assert answer.sentences == () and answer.dropped_sentences == 1
        assert answer.reduce_calls == 1 and len(answer.reduce_usage) == 1

    asyncio.run(run())


def test_unknown_point_sentence_is_dropped_but_traced_sentences_stay():
    async def run():
        fx, summaries, *_ = await _setup()

        def reduce(payload):
            real = payload["points"][0]["point_id"]
            return json.dumps(
                {
                    "answer": [
                        {"text": "Good.", "point_ids": [real]},
                        {"text": "Mixed.", "point_ids": [real, "com_0000000000000000:p0"]},
                        {"text": "Empty.", "point_ids": []},
                    ]
                }
            )

        answer = await _search(fx, summaries, _chat(reduce=reduce)).answer(QUESTION)
        assert answer.status == "answered" and [s.text for s in answer.sentences] == ["Good."]
        assert answer.dropped_sentences == 2

    asyncio.run(run())


def test_invalid_map_output_fail_and_skip():
    async def run():
        fx, summaries, big, small = await _setup()
        order = sorted([big.community_id, small.community_id])
        with pytest.raises(IndexingError) as error:
            await _search(fx, summaries, _chat(raw={order[0]: "not json"})).answer(QUESTION)
        assert error.value.code == "graph_invalid" and len(error.value.usage) == 1

        chat = _chat(raw={order[0]: "not json"})
        answer = await _search(fx, summaries, chat, on_invalid="skip").answer(QUESTION)
        assert answer.status == "answered" and answer.invalid_map_replies == 1
        assert answer.map_calls == 2 and len(answer.map_usage) == 2 and answer.reduce_calls == 1
        assert {s.community_id for s in answer.summaries_used} == {order[1]}

        shape = json.dumps({"helpfulness": 101, "points": []})
        chat = _chat(raw={big.community_id: "nope", small.community_id: shape})
        answer = await _search(fx, summaries, chat, on_invalid="skip").answer(QUESTION)
        assert answer.status == "abstained" and answer.reason == "no_relevant_community"
        assert answer.invalid_map_replies == 2 and answer.reduce_calls == 0 and len(answer.map_usage) == 2

    asyncio.run(run())


def test_invalid_reduce_output_abstains_and_keeps_usage():
    async def run():
        fx, summaries, *_ = await _setup()
        for reply in ("not json", json.dumps({"answer": "x"})):
            answer = await _search(fx, summaries, _chat(reduce=reply)).answer(QUESTION)
            assert answer.status == "abstained" and answer.reason == "reduce_invalid"
            assert answer.reduce_calls == 1 and len(answer.reduce_usage) == 1 and answer.sentences == ()
            assert len(answer.map_usage) == answer.map_calls == 2

    asyncio.run(run())


def test_prefilter_can_lose_recall_which_the_full_map_does_not():
    async def run():
        fx, summaries, big, small = await _setup()
        scores = {big.community_id: 90, small.community_id: 0}
        chat = _chat(scores=scores)
        answer = await _search(fx, summaries, chat, prefilter_top=1).answer(AR_QUESTION)
        assert answer.status == "abstained" and answer.reason == "no_relevant_community"
        assert answer.map_calls == 0 and chat.requests == []
        full = await _search(fx, summaries, _chat(scores=scores)).answer(AR_QUESTION)
        assert full.status == "answered" and full.map_calls == 2

        # With an English question the prefilter keeps the best-overlapping community only.
        chat = _chat(scores=scores)
        hit = await _search(fx, summaries, chat, prefilter_top=1).answer(QUESTION)
        assert hit.status == "answered" and hit.map_calls == 1
        assert {s.community_id for s in hit.summaries_used} == {big.community_id}

    asyncio.run(run())


def test_guards_cap_scope_graph_and_contract():
    async def run():
        fx, summaries, *_ = await _setup()
        chat = _chat()
        with pytest.raises(IndexingError) as error:
            _ = await _search(fx, summaries, chat, max_map_calls=1).answer(QUESTION)
        assert error.value.code == "global_search_too_many_communities" and chat.requests == []

        with pytest.raises(IndexingError) as error:
            CommunityGlobalSearch(
                GlobalSearchConfig(),
                scope=fx.stranger,
                graph=fx.graph,
                summaries=summaries,
                index=fx.index,
                index_version=fx.version,
                chat=chat,
            )
        assert error.value.code == "graph_scope_mismatch"

        other = await build_fixture_graph()
        assert other.graph.graph_key != fx.graph.graph_key
        foreign = tuple(s.model_copy(update={"graph_key": other.graph.graph_key}) for s in summaries)
        with pytest.raises(IndexingError) as error:
            _search(fx, foreign, chat)
        assert error.value.code == "graph_key_mismatch"

        wrong_version = fx.version.model_copy(update={"id": str(uuid.uuid4())})
        with pytest.raises(IndexingError) as error:
            CommunityGlobalSearch(
                GlobalSearchConfig(),
                scope=fx.scope,
                graph=fx.graph,
                summaries=summaries,
                index=fx.index,
                index_version=wrong_version,
                chat=chat,
            )
        assert error.value.code == "graph_scope_mismatch"

        with pytest.raises(IndexingError) as error:
            await _search(fx, summaries, chat).answer("   ")
        assert error.value.code == "empty_query"

    asyncio.run(run())


def test_config_rejects_bad_values():
    for bad in ({"unknown": 1}, {"max_map_calls": 201}, {"min_helpfulness": 0}, {"prefilter_top": 0}):
        with pytest.raises(ValueError):
            GlobalSearchConfig(**bad)


def test_plugin_passes_the_contract_kit():
    manifest = check_plugin_contract(CommunityGlobalSearch, CATALOGUE, sample_config={})
    assert manifest.technique_id == "R39" and manifest.kind.value == "retriever"
    assert manifest.fidelity.value == "adaptation"


def test_overlong_points_are_rejected_and_max_points_cuts_the_trace():
    async def run():
        fx, summaries, big, small = await _setup()
        long_point = json.dumps(
            {"helpfulness": 90, "points": [{"text": "x" * 21, "claim_ids": [summaries[0].claims[0].claim_id]}]}
        )
        owner = summaries[0].community_id
        search = _search(fx, summaries, _chat(raw={owner: long_point}), max_point_chars=20)
        answer = await search.answer(QUESTION)
        assert (owner, 0, "point_too_long") in answer.rejected_points
        assert {r[2] for r in answer.rejected_points} == {"point_too_long"}

        search = _search(fx, summaries, _chat(scores={big.community_id: 90, small.community_id: 10}), max_points=1)
        answer = await search.answer(QUESTION)
        assert answer.status == "answered" and len(answer.sentences) == 1
        trace = search.last_trace
        assert len(trace.points) == 1 and trace.points[0].community_id == big.community_id
        listed = [i for e in trace.entries for i in e.point_ids]
        assert listed == [trace.points[0].point_id]
        assert {s.community_id for s in answer.summaries_used} == {big.community_id}

    asyncio.run(run())
