# ruff: noqa: UP031
"""P7.3 lesson demo: R48 KAG-style schema-constrained graph, logical forms, grounding and re-planning.

Offline. Extraction and planning are scripted chat replies (fixed JSON, not a model); the embedder is the test-only
lexical vectorizer; the store is in memory. Every status, count, quote and token figure printed comes from calling the
package code on the shared KAG fixture, so the output shows schema checks, logical forms and grounding, not extraction
or planning quality. Chunk, entity, fact, graph and index IDs are never printed: paragraphs are shown by label
(K0..K8 carry facts, F0..F8 are filler). A derived answer lives in the trace only; it is not evidence and hits are
always original chunks. There are no timing values and the output is byte-identical between runs.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_domain.knowledge import KnowledgeGraph  # noqa: E402
from rag_foundry_indexing import (  # noqa: E402
    R48_EXTRACT_SYSTEM_V1_SHA256,
    R48_PLAN_SYSTEM_V1_SHA256,
    R48_REPLAN_SYSTEM_V1_SHA256,
    DenseConfig,
    DenseRetriever,
    EntityGraphConfig,
    EntityGraphRetriever,
    EntityRelationExtractConfig,
    EntityRelationGraphBuilder,
    KagConfig,
    KagRetriever,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat  # noqa: E402
from tests.indexing import test_kag_retrieve as kt  # noqa: E402  (plans and question texts shared with the tests)
from tests.indexing._graph_fixture import evidence_span_recall  # noqa: E402
from tests.indexing._inmemory_index import ready_index  # noqa: E402
from tests.indexing._kag_fixture import (  # noqa: E402
    PARAGRAPHS,
    QUESTIONS,
    R38_REPLIES,
    SPANS,
    TEXT,
    build_fixture_kg,
    span_of,
)
from tests.indexing._lexical_embedder import LexicalTestEmbedder  # noqa: E402
from tests.indexing._source_reader import StaticSourceReader  # noqa: E402

OUT = []
Q = {q["id"]: q for q in QUESTIONS}
LABEL_BY_START = {a: ("K%d" % (i // 2) if i % 2 == 0 else "F%d" % (i // 2)) for i, (a, _, _) in enumerate(SPANS)}
ORDER = {label: i for i, label in enumerate(LABEL_BY_START.values())}
PLANS = dict(kt.PLANS)
PLANS[Q["Q0"]["text"]] = json.dumps(
    {"steps": [kt.retrieve(1, None, "founded", kt.E("Cedar Labs", "Organization")), kt.output(2, "s1")]}
)
PLANS[Q["Q1"]["text"]] = json.dumps(
    {"steps": [kt.retrieve(1, kt.E("Cedar Labs", "Organization"), "founded_year", None), kt.output(2, "s1")]}
)
PLANS[Q["Q2"]["text"]] = json.dumps(
    {
        "steps": [
            kt.retrieve(1, None, "supplies", kt.E("Harbor Authority", "Organization")),
            {"op": "count", "id": "s2", "input": "s1"},
            kt.output(3, "s2"),
        ]
    }
)
COST = []  # (section, question, plan calls, input tokens, output tokens)
SECTION = ["?"]
LAST_CHAT = []


def emit(line=""):
    OUT.append(line)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for row in rows:
        emit("| " + " | ".join(str(cell) for cell in row) + " |")
    emit()


def tokens(usage):
    return sum(u.input_tokens for u in usage), sum(u.output_tokens for u in usage)


def label_at(offset):
    """Label of the paragraph that contains a code-point offset."""
    return next(LABEL_BY_START[a] for a, b, _ in SPANS if a <= offset < b)


def span_text(span):
    return "%s [%d, %d)" % (label_at(span.start), span.start, span.end)


class Naming:
    def __init__(self, graph):
        self.names = {e.entity_id: e.name for e in graph.entities}
        self.facts = {f.fact_id: f for f in graph.facts}

    def value(self, fact):
        return self.names[fact.object_id] if fact.object_id is not None else fact.literal

    def fact(self, fact):
        return "%s %s %s" % (self.names[fact.subject_id], fact.predicate, self.value(fact))

    def fact_rows(self, fact_ids):
        facts = sorted((self.facts[i] for i in set(fact_ids)), key=lambda f: (f.evidence[0].start, self.fact(f)))
        return [(self.fact(f), span_text(f.evidence[0]), f.evidence[0].quote) for f in facts]


async def ask(fx, text, plans=PLANS, replans=kt.REPLANS):
    chat = kt.plan_chat(plans, replans)
    retriever = KagRetriever(
        KagConfig(), scope=fx.scope, graph=fx.graph, index=fx.index, index_version=fx.version, chat=chat
    )
    result = await retriever.retrieve(text)
    trace = retriever.last_trace
    ci, co = tokens(trace.usage)
    LAST_CHAT[:] = [chat]
    COST.append((SECTION[0], text, len(chat.requests), ci, co))
    return result, trace


def failure_sent(n):
    """The failure the package put in the re-plan request after attempt n (attempt n+1), or "-"."""
    requests = LAST_CHAT[0].requests
    if n >= len(requests):
        return "-"
    failure = json.loads(requests[n].messages[1].content)["failure"]
    return "sent to re-plan: reason `%s`, detail: %s" % (failure["reason"], failure["detail"])


def status_text(n, attempt):
    sent = failure_sent(n)
    if sent == "-":
        return attempt.execution_status
    return "%s (an entity not found in the graph counts as a plan failure)" % attempt.execution_status


def hit_labels(result, view_id):
    return [LABEL_BY_START[h.start] for h in result.hits if h.extraction_view_id == view_id]


def answer_text(trace):
    return ", ".join(str(b.label) for b in trace.answer) if trace.answer else "none"


def show_plan(trace):
    emit("```json")
    emit(trace.attempts[-1].plan_json if trace.attempts else "(no plan recorded)")
    emit("```")
    emit()


async def r38_and_r02():
    """R38 graph built from the shared R38 replies on the KAG text, and a dense index (both lexical, in memory)."""
    embedder = LexicalTestEmbedder()
    scope, _, index, version = await ready_index(SPANS, embedder)
    chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
    view_id = chunks[0].extraction_view_id
    view = ViewSource(
        extraction_view_id=view_id, text=TEXT, blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(TEXT)),)
    )
    source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})

    def respond(request):
        reply = R38_REPLIES.get(json.loads(request.messages[1].content)["chunk"], '{"entities":[],"relations":[]}')
        return reply, "stop"

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
    r38 = EntityGraphRetriever(EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version)
    r02 = DenseRetriever(DenseConfig(k=3), scope=scope, index_version=version, index=index, embedder=embedder)
    return r38, r02, view_id


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P7.3 R48 KAG-style retrieval: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p73_kag_demo.py`. Date: 2026-11-22. "
        "Offline; scripted extraction and plans; shows schema checks, logical forms and grounding, not extraction or "
        "planning quality."
    )
    emit()
    emit(
        "Scripted chat (fixed JSON replies, not a model), test-only lexical embedder, in-memory store. R48 here is an "
        "adaptation of the KAG idea (schema-constrained extraction, logical-form planning, deterministic execution), "
        "not OpenSPG. The derived answer lives in the trace only and is never evidence; hits are original chunks. "
        "None of this is a benchmark."
    )
    emit()

    fx = await build_fixture_kg()
    graph = fx.graph
    names = Naming(graph)
    report = fx.report
    label_of_chunk = {c.id: LABEL_BY_START[c.start] for c in fx.chunks}
    chunk_of_label = {label: c.id for c in fx.chunks for label in [LABEL_BY_START[c.start]]}

    emit("## 1. Schema")
    emit()
    emit(
        "Paragraph labels: K0 to K8 carry facts, F0 to F8 are filler; %d paragraphs, one chunk each. Offsets are "
        "Unicode code points in the view text." % len(PARAGRAPHS)
    )
    emit()
    schema = graph.knowledge_schema
    emit("Schema `%s` version `%s`, sha256 `%s`." % (schema.schema_id, schema.version, schema.schema_sha256()))
    emit()
    table(
        ("entity type", "property", "value kind", "cardinality"),
        [
            (t.name, p.name, p.value_kind, p.cardinality)
            for t in schema.entity_types
            for p in (t.properties or (None,))
            if p is not None
        ]
        + [(t.name, "-", "-", "-") for t in schema.entity_types if not t.properties],
    )
    table(
        ("relation", "domain", "range", "cardinality"),
        [(r.name, r.domain, r.range, r.cardinality) for r in schema.relation_types],
    )
    emit(
        "Prompt SHAs (stable): extraction `%s`, plan `%s`, re-plan `%s`."
        % (R48_EXTRACT_SYSTEM_V1_SHA256, R48_PLAN_SYSTEM_V1_SHA256, R48_REPLAN_SYSTEM_V1_SHA256)
    )
    emit()

    emit("## 2. Graph")
    emit()
    emit(
        "%d entities and %d facts kept from %d chunks; %d items rejected; %d conflict(s). Every kept entity and fact "
        "carries source spans; nothing is kept without a quote that occurs exactly once in its chunk."
        % (report.entities, report.facts, report.chunks_processed, len(report.rejected), len(report.conflicts))
    )
    emit()
    emit("**Entities** (one row per mention)")
    emit()
    table(
        ("entity", "type", "paragraph and span", "quote"),
        [
            (e.name, e.type, span_text(m), m.quote)
            for e in sorted(graph.entities, key=lambda e: min(m.start for m in e.mentions))
            for m in sorted(e.mentions, key=lambda m: m.start)
        ],
    )
    emit("**Facts** (one row per evidence span)")
    emit()
    fact_rows = sorted(
        ((f.evidence[0].start, names.fact(f), s) for f in graph.facts for s in f.evidence), key=lambda r: r[:2]
    )
    table(
        ("fact", "paragraph and span", "quote"),
        [(text, span_text(s), s.quote) for _, text, s in fact_rows],
    )
    emit("**Rejected items** (kept out of the graph, by reason)")
    emit()
    table(
        ("paragraph", "kind", "item index", "reason"),
        sorted((label_of_chunk[i.chunk_id], i.item_kind, i.item_index, i.reason) for i in report.rejected),
    )
    emit(
        "Two rejections matter later. K6 has a scripted fact whose quote is not in the chunk (`quote_not_found`), so "
        "the graph does not know who leads Delta Works. K7 has a scripted fact whose subject is not in its quote "
        "(`name_not_in_quote`), so the graph does not know Gulf Marine supplies the Harbor Authority."
    )
    emit()
    emit("**Conflict** (reported, never resolved)")
    emit()
    rows = []
    for conflict in graph.conflicts():
        for fact_id in sorted(conflict.fact_ids, key=lambda i: names.facts[i].evidence[0].start):
            fact = names.facts[fact_id]
            rows.append(
                (names.names[conflict.subject_id], conflict.predicate, names.value(fact), span_text(fact.evidence[0]))
            )
    table(("entity", "property (cardinality one)", "value", "paragraph and span"), rows)
    emit("**Mutual index** (facts to chunks and chunks to facts) for two paragraphs")
    emit()
    rows = []
    for label in ("K0", "K1"):
        for fact in sorted(graph.facts_for_chunk(chunk_of_label[label]), key=lambda f: names.fact(f)):
            back = sorted((label_of_chunk[c] for c in graph.chunks_for_fact(fact.fact_id)), key=ORDER.get)
            rows.append((label, names.fact(fact), ", ".join(back)))
    table(("chunk", "fact (from `facts_for_chunk`)", "chunks (from `chunks_for_fact`)"), rows)
    again = await fx.index.get_knowledge_graph(fx.scope, fx.version.id, report.kg_key)
    emit(
        "Store round trip (build, store, read again): graph equal = %s. JSON round trip: graph equal = %s. "
        "Every kept span quote equals the view slice: %s."
        % (
            again == graph,
            KnowledgeGraph.from_json(graph.to_json()) == graph,
            all(TEXT[s.start : s.end] == s.quote for e in graph.entities for s in e.mentions)
            and all(TEXT[s.start : s.end] == s.quote for f in graph.facts for s in f.evidence),
        )
    )
    emit()

    SECTION[0] = "3"
    emit("## 3. Questions answered by the executor")
    emit()
    emit(
        "Each plan is the scripted chat reply, parsed and checked against the schema before it runs. Bindings are "
        "shown by name. The derived answer is computed from graph facts; the supporting facts and quotes below are "
        "the evidence behind it, and the hits are the original chunks of those facts. Hit order follows the "
        "hash-derived fact order, not relevance."
    )
    emit()
    shown = (
        ("Q0", Q["Q0"]["text"]),
        ("Q1", Q["Q1"]["text"]),
        ("Q2", Q["Q2"]["text"]),
        ("Q3 (multi-hop)", Q["Q3"]["text"]),
        ("count with filter", kt.COUNT_TEXT),
        ("Arabic", kt.AR_QUESTION),
    )
    for title, text in shown:
        result, trace = await ask(fx, text)
        emit("**%s**: %s" % (title, text))
        emit()
        show_plan(trace)
        table(
            ("step", "op", "status", "bindings", "missing"),
            [
                (
                    s.step_id,
                    s.op,
                    s.status,
                    ", ".join(str(b.label) for b in s.bindings) or "-",
                    ", ".join("%s/%s" % m for m in s.missing) or "-",
                )
                for s in trace.steps
            ],
        )
        emit(
            "Final status `%s`. Derived answer (from the trace, not evidence): %s. Hits (original chunks): %s."
            % (trace.status, answer_text(trace), ", ".join(hit_labels(result, fx.view_id)) or "none")
        )
        emit()
        fact_ids = [i for b in trace.answer for i in b.fact_ids]
        table(("supporting fact", "paragraph and span", "quote"), names.fact_rows(fact_ids))

    SECTION[0] = "4"
    emit("## 4. Re-planning")
    emit()
    emit(
        "At most one extra plan call, only when the plan itself failed (`invalid_logical_form`, "
        "`unsupported_operator`, `schema_violation`, `unknown_entity`). Data problems such as missing facts or "
        "conflicts are not re-planned. Re-plan prompt sha256 `%s`." % R48_REPLAN_SYSTEM_V1_SHA256
    )
    emit()
    typo = {
        "steps": [
            kt.retrieve(1, kt.E("Cedar Lab", "Organization"), "supplies", None),
            kt.retrieve(2, None, "leads", kt.S("s1")),
            kt.output(3, "s2"),
        ]
    }
    q3 = Q["Q3"]["text"]
    cases = (
        (
            "Success: a misspelled entity is fixed by the second plan",
            q3,
            {q3: json.dumps(typo)},
            {q3: json.dumps(kt.MULTI)},
        ),
        (
            "Failure: the second plan is also unusable",
            kt.DEDUCE_TEXT,
            {kt.DEDUCE_TEXT: json.dumps(kt.DEDUCE)},
            {kt.DEDUCE_TEXT: "not json {"},
        ),
    )
    sha_name = {R48_PLAN_SYSTEM_V1_SHA256: "plan prompt", R48_REPLAN_SYSTEM_V1_SHA256: "re-plan prompt"}
    for title, text, plans, replans in cases:
        result, trace = await ask(fx, text, plans, replans)
        emit("**%s** (%s)" % (title, text))
        emit()
        table(
            ("attempt", "prompt", "reason or execution status", "detail"),
            [
                (
                    n,
                    sha_name[a.prompt_sha256],
                    a.rejection[0] if a.rejection else status_text(n, a),
                    a.rejection[1] if a.rejection else failure_sent(n),
                )
                for n, a in enumerate(trace.attempts, 1)
            ],
        )
        emit(
            "Plan calls: %d (usage records %d). Final status `%s`. Derived answer: %s. Hits: %s."
            % (
                COST[-1][2],
                len(trace.usage),
                trace.status,
                answer_text(trace),
                ", ".join(hit_labels(result, fx.view_id)) or "none",
            )
        )
        emit()

    r38, r02, other_view = await r38_and_r02()
    SECTION[0] = "5"
    emit("## 5. R48 against R38 and R02")
    emit()
    emit(
        "Metric: span recall = gold paragraph spans fully inside one hit / number of gold spans, printed as "
        "`recall (hits=n)`. Gold counts (denominators): Q0 1, Q1 1, Q2 3, Q3 2, count with filter 2, Q4 1. R38 is "
        "built in memory from scripted entity-relation replies on the same text; R02 is dense with k=3 over the "
        "lexical test embedder. Six hand-written questions on a fixture built for the R48 tests, not a benchmark."
    )
    emit()
    base = (
        ("Q0", Q["Q0"]["text"], Q["Q0"]["gold"]),
        ("Q1", Q["Q1"]["text"], Q["Q1"]["gold"]),
        ("Q2", Q["Q2"]["text"], Q["Q2"]["gold"]),
        ("Q3", Q["Q3"]["text"], Q["Q3"]["gold"]),
        ("count with filter", kt.COUNT_TEXT, (span_of("K0"), span_of("K1"))),
        ("Q4", Q["Q4"]["text"], Q["Q4"]["gold"]),
    )
    rows = []
    for name, text, gold in base:
        result, trace = await ask(fx, text)
        r48 = evidence_span_recall(result, gold, fx.view_id)
        res38 = await r38.retrieve(text)
        res02 = await r02.retrieve(text)
        rows.append(
            (
                name,
                len(gold),
                "%.2f (hits=%d), status %s" % (r48, len(result.hits), trace.status),
                "%.2f (hits=%d)" % (evidence_span_recall(res38, gold, other_view), len(res38.hits)),
                "%.2f (hits=%d)" % (evidence_span_recall(res02, gold, other_view), len(res02.hits)),
            )
        )
    table(("question", "gold spans", "R48 recall (hits=n)", "R38 recall (hits=n)", "R02 k=3 recall (hits=n)"), rows)
    emit(
        "R38 and R02 return chunks only: they cannot count, filter or compare, so on the count questions their "
        "recall says nothing about the answer value. Per row on span recall: R48 ties R02 on Q0, Q1, Q2 and Q3; it "
        "beats both R38 and R02 on the count with filter (1.00 against 0.50 for each); it beats R38 on Q3 (1.00 "
        "against 0.50); and it loses only on Q4, the extraction miss (0.00 against 1.00 for R02). These are six "
        "scripted cases on one fixture, not a benchmark and not evidence about real corpora. R48 also returns a "
        "derived answer in its trace (section 3) and abstains explicitly."
    )
    emit()

    SECTION[0] = "6"
    emit("## 6. Losing cases")
    emit()
    losing = (
        ("Extraction miss (Q4)", Q["Q4"]["text"]),
        ("Partial graph (Q2 with the Gulf Marine fact missing)", Q["Q2"]["text"]),
        ("Conflict", kt.CONFLICT_TEXT),
        ("`unsupported_operator`", kt.DEDUCE_TEXT),
        ("`schema_violation`", kt.UNDECLARED_TEXT),
        ("Invalid plan", kt.MALFORMED_TEXT),
    )
    rows = []
    for title, text in losing:
        result, trace = await ask(fx, text)
        reasons = " then ".join(a.rejection[0] for a in trace.attempts if a.rejection) or "-"
        rows.append(
            (
                title,
                text,
                trace.status,
                answer_text(trace),
                len(result.hits),
                reasons,
                COST[-1][2],
            )
        )
    table(("case", "question", "status", "derived answer", "hits", "rejection reasons by attempt", "plan calls"), rows)
    emit(
        "The `unsupported_operator` row uses the shared scripted re-plan reply (the same `deduce` plan again), so "
        "both attempts fail alike. Section 4 asks the same question with a different scripted re-plan reply (text "
        "that is not JSON), so its second attempt fails differently (`invalid_logical_form`)."
    )
    emit()
    emit(
        "- Extraction miss: the scripted extraction dropped the Delta Works leadership fact, so R48 says `incomplete` "
        "and returns no hits. Dense R02 with k=3 still returns the K6 paragraph (recall 1.00, section 5). Abstaining "
        "is honest here but loses the evidence."
    )
    partial = Q["Q2"]
    emit(
        "- Partial graph: the true count of suppliers is %d (K1, K3, K7) but the graph holds %d, so the derived "
        "answer is wrong yet the status is `answered`, with 2 of 3 supplier spans as hits. Nothing in the trace shows "
        "it is incomplete; only the source text would." % (partial["answer"], partial["graph_answer"])
    )
    emit(
        "- Conflict: Blue Reef has two founded_year values (2012 in K4, 2014 in K5) and the property has cardinality "
        "one, so the executor reports `conflict`, returns no answer and no hits, and does not pick one."
    )
    emit(
        "- `unsupported_operator`: `deduce` is not in the grammar. `schema_violation`: the relation `owns` is not "
        "declared. Invalid plan: the reply is not JSON. Per the P7.3-08 notes, filter or sort on a `many` "
        "property is also a `schema_violation` (not run in this demo). In each case the single re-plan did not "
        "help here (2 plan calls), the status is `rejected`, and there is no fallback to chunk retrieval that "
        "could be mistaken for R48."
    )
    emit()
    emit(
        "The cited-answer path (answer service, citations of the original chunks) is covered by "
        "`tests/answering/test_kag_answer.py`; this demo does not rebuild the answering stack."
    )
    emit()

    emit("## 7. Cost")
    emit()
    emit("Scripted chat; token counts are the fake model's estimates, not provider figures.")
    emit()
    ei, eo = tokens(report.usage)
    table(
        ("stage", "chat calls", "input tokens", "output tokens"),
        [("R48 extraction (build, one call per chunk)", report.chat_calls, ei, eo)],
    )
    table(
        ("section", "question (plan calls incl. re-plans)", "calls", "input tokens", "output tokens"),
        COST,
    )
    emit(
        "Executor runs make no model call. A question run in more than one section is listed once per run. "
        "Update cost = full rebuild; not measured separately."
    )
    emit()
    emit("## Live run (orchestrator)")
    emit()
    emit("<orchestrator>")
    emit("Live run result goes here (P7.3-10). Live output is not deterministic and is recorded separately.")
    emit("</orchestrator>")
    emit()
    print("\n".join(OUT))


if __name__ == "__main__":
    asyncio.run(main())
