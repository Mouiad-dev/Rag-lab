# ruff: noqa: UP031
"""P7.2 lesson demo: R38 entity-relation graph retrieval and R39 community summaries with map-reduce search.

Offline. The chat is a scripted responder (fixed JSON replies); it is not a model. Every status, count, quote and
token figure printed comes from calling the package code on the shared relationship fixture, so the output shows
grounding, tracing and plumbing, not extraction or answer quality. The output has no timing values and no random
IDs, and is byte-identical between runs. R21 (neighbor expansion over adjacent sentences) is shown only as the
baseline; it is not GraphRAG.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_indexing import (  # noqa: E402
    CommunityGlobalSearch,
    CommunitySummaryBuilder,
    CommunitySummaryConfig,
    EntityGraphConfig,
    EntityGraphRetriever,
    EntityRelationExtractConfig,
    EntityRelationGraphBuilder,
    GlobalSearchConfig,
    detect_communities,
    trace_claims,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource  # noqa: E402
from rag_foundry_indexing.graph_global import R39_MAP_SYSTEM_V1, R39_REDUCE_SYSTEM_V1  # noqa: E402
from rag_foundry_indexing.sentence_window import SentenceWindowConfig, SentenceWindowExpander  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat, ScriptedEmbedder  # noqa: E402
from tests.indexing._graph_fixture import (  # noqa: E402
    GOLD_SPANS,
    P16,
    PARAGRAPHS,
    QUESTION,
    REPLIES,
    SPANS,
    TEXT,
    _ent,
    _reply,
    entity_seed,
    evidence_span_recall,
)
from tests.indexing._inmemory_index import ready_index  # noqa: E402
from tests.indexing._source_reader import StaticSourceReader  # noqa: E402

OUT = []
GLOBAL_QUESTION = "Which organisations work with the Harbor Authority?"
UNRELATED = "What is the capital of Atlantis?"
# "What are the organisations that work with the harbour authority", sharing no token with the Arabic cluster.
AR_QUESTION = " ".join(
    (
        "\u0645\u0627",
        "\u0647\u064a",
        "\u0627\u0644\u0645\u0624\u0633\u0633\u0627\u062a",
        "\u0627\u0644\u062a\u064a",
        "\u062a\u0639\u0645\u0644",
        "\u0645\u0639",
        "\u0647\u064a\u0626\u0629",
        "\u0627\u0644\u0645\u064a\u0646\u0627\u0621",
    )
)
GHOST_QUOTE = "Ghost walked the pier."


def emit(line=""):
    OUT.append(line)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for row in rows:
        emit("| " + " | ".join(str(cell) for cell in row) + " |")
    emit()


def paragraph_of(start):
    return next(i for i, (a, _, _) in enumerate(SPANS) if a == start)


def tokens(usage):
    return sum(u.input_tokens for u in usage), sum(u.output_tokens for u in usage)


async def build_graph():
    """Build the fixture graph with the public builder; one reply carries an ungrounded entity to be rejected."""
    replies = dict(REPLIES)
    replies[P16] = _reply([_ent("Mira Yassin", "person"), _ent("Ghost", "person", GHOST_QUOTE)])
    embedder = ScriptedEmbedder(lambda text: (1.0, 0.0, 0.0, 0.0))
    scope, _stranger, index, version = await ready_index(SPANS, embedder)
    chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
    view_id = chunks[0].extraction_view_id
    view = ViewSource(
        extraction_view_id=view_id, text=TEXT, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(TEXT)),)
    )
    source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})

    def respond(request):
        return replies.get(json.loads(request.messages[1].content)["chunk"], _reply()), "stop"

    builder = EntityRelationGraphBuilder(
        EntityRelationExtractConfig(),
        scope=scope,
        index_version=version,
        index=index,
        source=source,
        chat=ScriptedChat(respond),
    )
    report = await builder.build()
    graph = await index.get_entity_graph(scope, version.id, report.graph_key)
    return scope, index, version, source, view_id, chunks, report, graph


def community_chat(communities):
    """Scripted R39 summary chat: two claims per community, plus one claim citing an unknown fact for the big one."""
    by_facts = {frozenset(c.entity_ids) | frozenset(c.relation_ids): c for c in communities}

    def respond(request):
        facts = json.loads(request.messages[1].content)["facts"]
        community = by_facts[frozenset(f["fact_id"] for f in facts)]
        claims = [{"text": f["text"], "fact_ids": [f["fact_id"]]} for f in facts[-2:]]
        if len(community.entity_ids) == 4:
            claims.append({"text": "An invented fact.", "fact_ids": ["ent_0000000000000000"]})
        return json.dumps({"title": "Cluster of %d entities" % len(community.entity_ids), "claims": claims}), "stop"

    return ScriptedChat(respond)


def map_reduce_chat(scores, reduce=None):
    """Scripted map and reduce chat. Each community cites all its claims; the reduce echoes each point."""

    def respond(request):
        system = request.messages[0].content
        payload = json.loads(request.messages[1].content)
        if system == R39_REDUCE_SYSTEM_V1:
            if reduce is not None:
                return reduce, "stop"
            sentences = [{"text": p["text"], "point_ids": [p["point_id"]]} for p in payload["points"]]
            return json.dumps({"answer": sentences}), "stop"
        assert system == R39_MAP_SYSTEM_V1
        claims = payload["community"]["claims"]
        community_id = claims[0]["claim_id"].split(":")[0]
        points = [{"text": c["text"], "claim_ids": [c["claim_id"]]} for c in claims]
        return json.dumps({"helpfulness": scores[community_id], "points": points}), "stop"

    return ScriptedChat(respond)


def describe_path(graph, trace):
    names = {e.entity_id: e.name for e in graph.entities}
    relations = {r.relation_id: r for r in graph.relations}
    steps = []
    for relation_id, direction in trace.path:
        r = relations[relation_id]
        steps.append("%s: %s -[%s]-> %s" % (direction, names[r.source_id], r.label, names[r.target_id]))
    return steps


def show_hits(result):
    for hit in result.hits:
        emit("- P%d [%d, %d) score %.2f: %s" % (paragraph_of(hit.start), hit.start, hit.end, hit.score, hit.text))
    emit()


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P7.2 R38 entity graph and R39 community summaries: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p72_graphrag_demo.py`. Date: 2026-11-22. "
        "Offline; scripted chat; shows grounding, tracing and plumbing, not extraction or answer quality."
    )
    emit()

    scope, index, version, source, view_id, chunks, report, graph = await build_graph()
    names = {e.entity_id: e.name for e in graph.entities}

    emit("## 1. Fixture")
    emit()
    emit(
        "17 paragraphs, one chunk each. Evidence for the 3-hop question is P0, P7 and P14; the others are filler, "
        "an Arabic sentence (P15) and a lone entity (P16). Offsets are Unicode code points in the view text."
    )
    emit()
    gold = {(a, b) for a, b in GOLD_SPANS}
    rows = []
    for i, (a, b, text) in enumerate(SPANS):
        if i in (0, 7, 14, 15, 16) or i == 1:
            rows.append((i, "[%d, %d)" % (a, b), "gold" if (a, b) in gold else "-", text))
    table(("paragraph", "span", "role", "text"), rows)
    emit("Paragraphs P1 to P6 and P8 to P13 are filler (%d paragraphs in total)." % len(PARAGRAPHS))
    emit()
    emit("Question: %s" % QUESTION)
    emit()

    emit("## 2. Graph (R38 builder, adaptation)")
    emit()
    emit("Entities and relations are kept only when their quote occurs exactly once in the chunk.")
    emit()
    emit("**Entities** (%d)" % len(graph.entities))
    emit()
    table(
        ("entity", "type", "span", "quote"),
        [
            (e.name, e.type, "[%d, %d)" % (e.mentions[0].start, e.mentions[0].end), e.mentions[0].quote)
            for e in sorted(graph.entities, key=lambda e: e.mentions[0].start)
        ],
    )
    emit("**Relations** (%d)" % len(graph.relations))
    emit()
    table(
        ("source", "label", "target", "span", "quote"),
        [
            (
                names[r.source_id],
                r.label,
                names[r.target_id],
                "[%d, %d)" % (r.evidence[0].start, r.evidence[0].end),
                r.evidence[0].quote,
            )
            for r in sorted(graph.relations, key=lambda r: r.evidence[0].start)
        ],
    )
    emit("**Rejected items** (%d)" % len(report.rejected))
    emit()
    table(
        ("paragraph", "kind", "item index", "reason"),
        [
            (
                paragraph_of(next(c.start for c in chunks if c.id == item.chunk_id)),
                item.item_kind,
                item.item_index,
                item.reason,
            )
            for item in report.rejected
        ],
    )
    emit(
        'The rejected entity "Ghost" cites the quote "%s", which is not in P16, so it never enters the graph.'
        % GHOST_QUOTE
    )
    emit()
    again = await index.get_entity_graph(scope, version.id, report.graph_key)
    emit(
        "Graph key: %d hex characters (sha256 of the canonical generator and index version; the value differs per "
        "run because the index version ID is random, so it is not printed)." % len(graph.graph_key)
    )
    emit()
    emit("Store round trip (build, store, read again): graph equal = %s." % (again == graph))
    emit()
    all_spans = [s for e in graph.entities for s in e.mentions] + [s for r in graph.relations for s in r.evidence]
    emit("Every kept span quote equals the view slice: %s." % all(TEXT[s.start : s.end] == s.quote for s in all_spans))
    emit()

    retriever = EntityGraphRetriever(EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version)
    result = await retriever.retrieve(QUESTION)
    trace = retriever.last_trace
    emit("## 3. Relationship question: R38 path vs the R21 baseline")
    emit()
    emit("**R38 entity-graph path** (%s)" % QUESTION)
    emit()
    emit(
        "Status `%s`, mode `%s`, matched entities: %s."
        % (trace.status, trace.mode, ", ".join(sorted(names[i] for i in trace.matched_entity_ids)))
    )
    emit()
    for step in describe_path(graph, trace):
        emit("- hop %s" % step)
    emit()
    show_hits(result)
    emit("Evidence recall over the 3 gold spans: %.1f." % evidence_span_recall(result, GOLD_SPANS, view_id))
    emit()

    seed = entity_seed(QUESTION, chunks, version)
    expander = SentenceWindowExpander(
        SentenceWindowConfig(window=5, stay_in_paragraph=False, max_window_chars=20000),
        scope=scope,
        index_version=version,
        index=index,
        source=source,
    )
    expanded = (await expander.expand(seed)).result
    recall21 = evidence_span_recall(expanded, GOLD_SPANS, view_id)
    a, b = GOLD_SPANS[1]
    missing = not any(h.start <= a and b <= h.end for h in expanded.hits)
    emit(
        "**R21 neighbor expansion** (adjacent sentences, 5-sentence window; not GraphRAG) from the same %d seed chunks "
        "(chunks whose text names a question entity):" % len(seed.hits)
    )
    emit()
    emit(
        "Evidence recall: %d/3 (%.2f). P7 missing: %s. The window around P0 and P14 stops at distance 5, and P7 is "
        "7 paragraphs from P0, so only the relation path reaches it." % (round(recall21 * 3), recall21, missing)
    )
    emit()

    emit("## 4. R38 losing cases")
    emit()
    alias = EntityGraphRetriever(EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version)
    res = await alias.retrieve("How is Dr. Haddad connected to Omar Nasser?")
    emit('**Alias question** ("Dr. Haddad" is not an entity name). Exact name matching finds one entity only.')
    emit()
    emit(
        "Status `%s`, mode `%s`, matched entities: %s, path length %d."
        % (
            alias.last_trace.status,
            alias.last_trace.mode,
            ", ".join(names[i] for i in alias.last_trace.matched_entity_ids),
            len(alias.last_trace.path),
        )
    )
    emit()
    show_hits(res)
    emit(
        "Evidence recall: %.2f (1/3). The alias is not resolved; the path through Cedar Labs is lost."
        % evidence_span_recall(res, GOLD_SPANS, view_id)
    )
    emit()
    unknown = EntityGraphRetriever(EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version)
    res = await unknown.retrieve("Who repaired the lighthouse?")
    emit("**Unknown entity** (Who repaired the lighthouse?)")
    emit()
    emit("Status `%s`, mode `%s`, hits %d." % (unknown.last_trace.status, unknown.last_trace.mode, len(res.hits)))
    emit()
    short = EntityGraphRetriever(
        EntityGraphConfig(max_hops=2), scope=scope, graph=graph, index=index, index_version=version
    )
    res = await short.retrieve(QUESTION)
    emit("**Hop limit** (`max_hops=2`, the true path has 3 hops)")
    emit()
    emit(
        "Status `%s`, mode `%s`, matched entities %d, hits %d."
        % (short.last_trace.status, short.last_trace.mode, len(short.last_trace.matched_entity_ids), len(res.hits))
    )
    emit()
    emit(
        "The R38 cited answer (relationship question through the answer service, exact citations) is covered by the "
        "P7.2-06 test `tests/answering/test_graph_answer.py`; the demo does not rebuild the ingestion stack."
    )
    emit()

    communities = detect_communities(graph)
    summary_chat = community_chat(communities)
    summaries = await CommunitySummaryBuilder(CommunitySummaryConfig(), graph=graph, chat=summary_chat).build()
    emit("## 5. Communities and summaries (R39)")
    emit()
    emit("Label propagation, fixed order, no randomness. Summaries are generated text and marked `synthetic`.")
    emit()
    table(
        ("community", "members", "relations", "iterations", "converged"),
        [
            (
                c.community_id,
                ", ".join(sorted(names[i] for i in c.entity_ids)),
                len(c.relation_ids),
                c.iterations,
                c.converged,
            )
            for c in sorted(summaries.communities, key=lambda c: c.community_id)
        ],
    )
    emit(
        "Skipped (fewer than 2 members): %s."
        % ", ".join(names[i] for c in summaries.skipped_small for i in c.entity_ids)
    )
    emit()
    for summary in sorted(summaries.summaries, key=lambda s: s.community_id):
        emit("**%s** (`%s`, synthetic=%s)" % (summary.title, summary.community_id, summary.synthetic))
        emit()
        spans = trace_claims(summary, graph)
        for claim in summary.claims:
            emit("- claim `%s`: %s" % (claim.claim_id, claim.text))
            emit("  - fact IDs: %s" % ", ".join("`%s`" % f for f in claim.fact_ids))
            for s in claim.sources:
                emit('  - source [%d, %d): "%s"' % (s.start, s.end, s.quote))
        emit()
        emit("`trace_claims` resolves %d source spans for this summary." % len(spans))
        emit()
    emit("**Rejected claims** (community, claim index, reason)")
    emit()
    table(
        ("community", "claim index", "reason"),
        [(c, i, reason) for c, i, reason in sorted(summaries.rejected_claims)],
    )
    emit("The invented claim cites a fact ID that is not in its community, so it is rejected before R39 search.")
    emit()

    scores = {
        s.community_id: (90 if len(c.entity_ids) == 4 else 10)
        for s in summaries.summaries
        for c in summaries.communities
        if c.community_id == s.community_id
    }

    def search(chat, **config):
        return CommunityGlobalSearch(
            GlobalSearchConfig(**config),
            scope=scope,
            graph=graph,
            summaries=summaries.summaries,
            index=index,
            index_version=version,
            chat=chat,
        )

    searcher = search(map_reduce_chat(scores))
    answer = await searcher.answer(GLOBAL_QUESTION)
    gtrace = searcher.last_trace
    emit("## 6. R39 map-reduce global search")
    emit()
    emit("Question: %s" % GLOBAL_QUESTION)
    emit()
    emit("Status `%s`, map calls %d, reduce calls %d." % (answer.status, answer.map_calls, answer.reduce_calls))
    emit()
    table(
        ("community", "map status", "helpfulness", "points"),
        [
            (e.community_id, e.status, e.helpfulness, len(e.point_ids))
            for e in sorted(gtrace.entries, key=lambda e: e.community_id)
        ],
    )
    emit("**Answer sentences** (the scripted reduce echoes each point; the demo checks traceability, not wording)")
    emit()
    for n, sentence in enumerate(answer.sentences, 1):
        emit("%d. %s (points: %s)" % (n, sentence.text, ", ".join("`%s`" % p for p in sentence.point_ids)))
        for s in sentence.sources:
            emit('   - source [%d, %d): "%s"' % (s.start, s.end, s.quote))
    emit()
    emit(
        "Dropped sentences: %d. Rejected points: %d. Every source quote equals the view slice: %s."
        % (
            answer.dropped_sentences,
            len(answer.rejected_points),
            all(TEXT[s.start : s.end] == s.quote for sen in answer.sentences for s in sen.sources),
        )
    )
    emit()

    zero = {s.community_id: 0 for s in summaries.summaries}
    none = await search(map_reduce_chat(zero)).answer(UNRELATED)
    emit("**Losing case: nothing relevant** (%s)" % UNRELATED)
    emit()
    emit(
        "Status `%s`, reason `%s`, map calls %d, reduce calls %d."
        % (none.status, none.reason, none.map_calls, none.reduce_calls)
    )
    emit()
    reply = json.dumps({"answer": [{"text": "Made up.", "point_ids": ["com_0000000000000000:p0"]}]})
    untraced = await search(map_reduce_chat(scores, reduce=reply)).answer(GLOBAL_QUESTION)
    emit("**Losing case: untraced reduce** (the reduce cites a point that does not exist)")
    emit()
    emit(
        "Status `%s`, reason `%s`, dropped sentences %d, sentences kept %d, reduce calls %d."
        % (untraced.status, untraced.reason, untraced.dropped_sentences, len(untraced.sentences), untraced.reduce_calls)
    )
    emit()
    miss = await search(map_reduce_chat(scores), prefilter_top=1).answer(AR_QUESTION)
    full = await search(map_reduce_chat(scores)).answer(AR_QUESTION)
    emit("**Losing case: prefilter miss** (`prefilter_top=1`, Arabic question with no token shared with any summary)")
    emit()
    emit(
        "With the prefilter: status `%s`, reason `%s`, map calls %d. Without it: status `%s`, map calls %d. "
        "The cheap lexical prefilter saves map calls and can lose recall."
        % (miss.status, miss.reason, miss.map_calls, full.status, full.map_calls)
    )
    emit()

    emit("## 7. Cost (scripted chat; token counts are the fake model's estimates)")
    emit()
    ex_in, ex_out = tokens(report.usage)
    su_in, su_out = tokens(summaries.usage)
    mp_in, mp_out = tokens(answer.map_usage)
    rd_in, rd_out = tokens(answer.reduce_usage)
    table(
        ("stage", "chat calls", "input tokens", "output tokens"),
        [
            ("R38 extraction (build)", report.chat_calls, ex_in, ex_out),
            ("R39 community summaries (build)", summaries.chat_calls, su_in, su_out),
            ("R39 map (per question)", answer.map_calls, mp_in, mp_out),
            ("R39 reduce (per question)", answer.reduce_calls, rd_in, rd_out),
        ],
    )
    emit("R38 retrieval makes no model call. Update cost = full rebuild (same as build); not measured separately.")
    emit()
    # Static text about one past live run (2026-11-22); edit by hand if the live run is redone.
    emit("## Live run")
    emit()
    emit("One local live smoke run (gemma4:e4b on Ollama, 2026-11-22, cost 0, not an evaluation) is recorded")
    emit("separately in [graphrag_live_run.md](graphrag_live_run.md), because live output is not deterministic.")
    emit("In that run the structural checks passed, but R38 found no path within the hop limit on the")
    emit("model-built graph (recall 0.00), while R39 answered with one traced sentence.")
    emit()
    print("\n".join(OUT))


asyncio.run(main())
