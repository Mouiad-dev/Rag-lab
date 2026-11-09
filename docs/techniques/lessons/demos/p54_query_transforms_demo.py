# ruff: noqa: UP031
"""P5.4 lesson demo: R13 rewriting, R14 multi-query, R15 decomposition, R16 step-back, R17 HyDE, R18 RAG fusion.

Offline. Every printed rank, status, count and token figure comes from calling the package code. The chat is
a scripted responder (fixed JSON replies) and the embedder is the test-only LexicalTestEmbedder; neither is a
model, so these rows show how the code handles given replies, not how a model behaves. The output contains no
date or timing values and is byte-identical between runs.
"""

import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_domain.errors import QueryTransformError  # noqa: E402
from rag_foundry_indexing import (  # noqa: E402
    DecomposeConfig,
    DenseConfig,
    DenseRetriever,
    HydeConfig,
    HydeRetriever,
    MultiQueryConfig,
    MultiQueryExpansion,
    QuestionDecomposition,
    QuestionRewriter,
    RagFusion,
    RagFusionConfig,
    StepBackRetrieval,
)
from rag_foundry_indexing.rewrite import retrieve_with_rewrite  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat  # noqa: E402
from tests.indexing._fixture import DISTRACTORS, FACT_AR, FACT_EN, MARKDOWN, MUSEUM  # noqa: E402
from tests.indexing._inmemory_index import ready_index  # noqa: E402
from tests.indexing._lexical_embedder import LexicalTestEmbedder  # noqa: E402
from tests.indexing._recording_retriever import RecordingRetriever  # noqa: E402

OUT = []
METER = []


def emit(line=""):
    OUT.append(line)


def spans():
    pos, result = 0, []
    for paragraph in MARKDOWN.split("\n\n"):
        start = MARKDOWN.index(paragraph, pos)
        pos = start + len(paragraph)
        if paragraph != "# Foundry field notes":
            result.append((start, pos, paragraph))
    return result


LABELS = {FACT_EN: "fact_en", FACT_AR: "fact_ar", MUSEUM: "museum"}
for _i, _text in enumerate(DISTRACTORS):
    LABELS[_text] = "dist%02d" % _i

Q_HARD = "What is the measurement schedule for the coastal station?"
Q_CLEAN = "How often does the Port Sallow observatory record sea levels?"


def scripted(payload):
    def responder(request):
        return (json.dumps(payload), "stop")

    return ScriptedChat(responder)


def labels_of(result):
    return [LABELS[hit.text] for hit in result.hits]


def rank_of(result, label):
    for hit in result.hits:
        if LABELS[hit.text] == label:
            return str(hit.rank)
    return "not in top %d" % len(result.hits)


def meter(technique, scenario, trace, retrieval_calls, embed_delta):
    chat = [u for u in trace.usage if u.operation.value == "chat"]
    embed = [u for u in trace.usage if u.operation.value == "embed"]
    tokens_in = sum(u.input_tokens for u in trace.usage)
    tokens_out = sum(u.output_tokens for u in trace.usage)
    cost = sum((Decimal(str(u.estimated_cost_usd)) for u in trace.usage), Decimal(0))
    METER.append(
        (
            technique,
            "| %s | %s | %d | %d | %d | %d | %d | %d | %s |"
            % (
                technique,
                scenario,
                len(chat),
                len(embed),
                retrieval_calls,
                embed_delta - len(embed),
                tokens_in,
                tokens_out,
                cost,
            ),
        )
    )


async def r13_section(dense, embedder):
    emit("## R13 question rewriting (scripted chat, not a model)")
    emit()
    cases = [
        (
            "applied",
            "How frequently are the water height readings taken at that coastal station?",
            "Port Sallow observatory records sea levels",
        ),
        (
            "drops 17 (detected)",
            "Does the Port Sallow observatory record sea levels every 17 minutes?",
            "How often does the Port Sallow observatory record sea levels?",
        ),
        (
            "drops a place name (not detected)",
            "What does the Port Sallow tidal observatory record?",
            "What does the tidal observatory record?",
        ),
    ]
    emit(
        "| case | status | lost_constraints | question kept in trace | retrieval queries | usage calls | "
        "fact_en rank, original question | fact_en rank, queries used |"
    )
    emit("|---|---|---|---|---|---|---|---|")
    for index, (name, question, rewrite) in enumerate(cases):
        chat = scripted({"query": rewrite})
        before = len(embedder.calls)
        out = await retrieve_with_rewrite(question, rewriter=QuestionRewriter(), chat=chat, retriever=dense)
        delta = len(embedder.calls) - before
        original = await dense.retrieve(question)
        trace = out.trace
        emit(
            "| %s | %s | %s | %s | %s | %d | %s | %s |"
            % (
                name,
                trace.status,
                ", ".join(trace.lost_constraints) or "-",
                "yes" if trace.original_question == question else "NO",
                " / ".join("`%s`" % q for q in trace.retrieval_queries),
                len(trace.usage),
                rank_of(original, "fact_en"),
                rank_of(out.result, "fact_en"),
            )
        )
        if index == 0:
            meter("R13", "applied rewrite", trace, len(trace.retrieval_queries), delta)
    emit()
    emit(
        "Heuristic limit: the hard-constraint check covers numbers, quoted phrases and negations only. A dropped "
        "place name has empty `lost_constraints` and status `applied`."
    )
    emit()


async def r14_r18_section(dense, embedder):
    emit("## R02 vs R14 vs R18: expansion and fusion (same query, same scripted variants)")
    emit()
    variants = ["Port Sallow sea levels", "tidal observatory records sea levels", Q_HARD.upper()]
    emit("Question: `%s`" % Q_HARD)
    emit()
    emit(
        "Scripted variants (same reply for R14 and R18): "
        + ", ".join("`%s`" % v for v in variants)
        + ". The third is the question in capitals, a duplicate after casefolding."
    )
    emit()
    r02 = await dense.retrieve(Q_HARD)
    before = len(embedder.calls)
    r14 = await MultiQueryExpansion(MultiQueryConfig(k=5)).run(
        Q_HARD, chat=scripted({"queries": variants}), retriever=dense
    )
    delta14 = len(embedder.calls) - before
    before = len(embedder.calls)
    r18 = await RagFusion(RagFusionConfig(k=5)).run(Q_HARD, chat=scripted({"queries": variants}), retriever=dense)
    delta18 = len(embedder.calls) - before
    meter("R14", "ablation query, 3 variants (1 duplicate)", r14.trace, len(r14.trace.retrieval_queries), delta14)
    meter("R18", "ablation query, 3 variants (1 duplicate)", r18.trace, len(r18.trace.retrieval_queries), delta18)

    emit("| method | scoring | fact_en rank | top 5 |")
    emit("|---|---|---|---|")
    emit(
        "| R02 original only | %s | %s | %s |"
        % (r02.scoring or "-", rank_of(r02, "fact_en"), ", ".join(labels_of(r02)))
    )
    emit(
        "| R14 union (best rank) | %s | %s | %s |"
        % (r14.result.scoring, rank_of(r14.result, "fact_en"), ", ".join(labels_of(r14.result)))
    )
    emit(
        "| R18 RRF | %s | %s | %s |"
        % (r18.result.scoring, rank_of(r18.result, "fact_en"), ", ".join(labels_of(r18.result)))
    )
    emit()
    emit("- Expansion (R14 vs R02): fact_en rank %s -> %s." % (rank_of(r02, "fact_en"), rank_of(r14.result, "fact_en")))
    emit(
        "- Fusion (R18 vs R14, same candidates): fact_en rank %s -> %s; top hit %s -> %s."
        % (
            rank_of(r14.result, "fact_en"),
            rank_of(r18.result, "fact_en"),
            labels_of(r14.result)[0],
            labels_of(r18.result)[0],
        )
    )
    emit()
    emit("| query | fact_en rank in this query | results | new chunks vs earlier queries |")
    emit("|---|---|---|---|")
    seen = set()
    for query, per in zip(r14.trace.retrieval_queries, r14.per_query, strict=True):
        got = labels_of(per)
        new = [label for label in got if label not in seen]
        seen.update(got)
        emit(
            "| `%s` | %s | %s | %d (%s) |"
            % (query, rank_of(per, "fact_en"), ", ".join(got), len(new), ", ".join(new) or "-")
        )
    emit()
    emit(
        "`duplicates_dropped`: R14 %d, R18 %d. Queries searched: R14 %d, R18 %d (original first). "
        "Trace status: R14 %s, R18 %s."
        % (
            r14.trace.duplicates_dropped,
            r18.trace.duplicates_dropped,
            len(r14.trace.retrieval_queries),
            len(r18.trace.retrieval_queries),
            r14.trace.status,
            r18.trace.status,
        )
    )
    emit()
    emit("| R18 hit | score | components (query: rrf) |")
    emit("|---|---|---|")
    for hit in r18.result.hits:
        parts = ", ".join("%s: %.6f" % (c.name, c.value) for c in hit.score_components)
        emit("| %s | %.6f | %s |" % (LABELS[hit.text], hit.score, parts))
    emit()
    lossy = scripted({"queries": ["Port Sallow observatory sea levels", "every minutes sea levels"]})
    question = "How often does the Port Sallow observatory record sea levels every 17 minutes?"
    out = await MultiQueryExpansion(MultiQueryConfig(k=5)).run(question, chat=lossy, retriever=dense)
    dropped = [g.text for g in out.trace.generated if g.role == "variant" and g.text not in out.trace.retrieval_queries]
    emit(
        "| losing case: variants without the number 17 | status | lost_constraints | queries searched | "
        "lossy variants dropped |"
    )
    emit("|---|---|---|---|---|")
    emit(
        "| default drop_lossy_variants | %s | %s | %d | %d |"
        % (out.trace.status, ", ".join(out.trace.lost_constraints), len(out.trace.retrieval_queries), len(dropped))
    )
    emit()


async def r15_section(dense, embedder):
    emit("## R15 question decomposition (scripted chat, not a model)")
    emit()
    question = "How often does the observatory record sea levels and who repaints the bridge every spring?"
    subs = [
        {"id": "s1", "text": Q_CLEAN, "depends_on": []},
        {"id": "s2", "text": "Who repaints the wooden bridge every spring?", "depends_on": ["s1"]},
    ]
    plain = await dense.retrieve(question)
    before = len(embedder.calls)
    out = await QuestionDecomposition().run(question, chat=scripted({"subquestions": subs}), retriever=dense)
    delta = len(embedder.calls) - before
    meter("R15", "two subquestions", out.trace, len(out.trace.retrieval_queries), delta)
    emit("Question: `%s`" % question)
    emit()
    emit("| id | depends_on | text |")
    emit("|---|---|---|")
    for item in out.trace.generated:
        if item.role == "subquestion":
            emit("| %s | %s | %s |" % (item.id, ", ".join(item.depends_on) or "-", item.text))
    emit()
    emit("| method | status | fact_en rank | dist03 rank (bridge) | top 5 |")
    emit("|---|---|---|---|---|")
    emit(
        "| R02 original question | - | %s | %s | %s |"
        % (rank_of(plain, "fact_en"), rank_of(plain, "dist03"), ", ".join(labels_of(plain)))
    )
    emit(
        "| R15 original + subquestions | %s | %s | %s | %s |"
        % (
            out.trace.status,
            rank_of(out.result, "fact_en"),
            rank_of(out.result, "dist03"),
            ", ".join(labels_of(out.result)),
        )
    )
    emit()
    emit(
        "Queries searched: %d (original first, then s1, s2). Dependencies are recorded and validated; "
        "answers are not fed forward." % len(out.trace.retrieval_queries)
    )
    emit()
    emit("| scripted reply | on_invalid | outcome | retrieval calls |")
    emit("|---|---|---|---|")
    bad = [
        {"id": "s1", "text": "first part", "depends_on": ["s2"]},
        {"id": "s2", "text": "second part", "depends_on": []},
    ]
    for mode in ("fail", "original"):
        recorder = RecordingRetriever({})
        try:
            await QuestionDecomposition(DecomposeConfig(on_invalid=mode)).run(
                question, chat=scripted({"subquestions": bad}), retriever=recorder
            )
            outcome = "no error"
        except QueryTransformError as exc:
            outcome = "raised %s" % exc.code
        emit("| s1 depends_on s2 (a later id) | %s | %s | %d |" % (mode, outcome, len(recorder.queries)))
    emit()
    lossy = [
        {"id": "s1", "text": "How often is sea level recorded?", "depends_on": []},
        {"id": "s2", "text": "Who repaints the bridge?", "depends_on": []},
    ]
    q17 = "How often is sea level recorded every 17 minutes and who repaints the bridge?"
    out17 = await QuestionDecomposition().run(q17, chat=scripted({"subquestions": lossy}), retriever=dense)
    emit("| losing case: subquestions without 17 | status | lost_constraints | queries searched |")
    emit("|---|---|---|---|")
    emit(
        "| default on_constraint_loss=original | %s | %s | %d |"
        % (out17.trace.status, ", ".join(out17.trace.lost_constraints), len(out17.trace.retrieval_queries))
    )
    emit()


async def r16_section(dense, embedder):
    emit("## R16 step-back retrieval (scripted chat, not a model)")
    emit()
    step = "What does a tidal observatory do?"
    original = await dense.retrieve(Q_HARD)
    before = len(embedder.calls)
    out = await StepBackRetrieval().run(Q_HARD, chat=scripted({"step_back_question": step}), retriever=dense)
    delta = len(embedder.calls) - before
    meter("R16", "step-back hit", out.trace, len(out.trace.retrieval_queries), delta)
    emit(
        "| case | question | step-back question | fact_en rank, original only | "
        "fact_en rank, original + step-back | lost_constraints |"
    )
    emit("|---|---|---|---|---|---|")
    emit(
        "| success | %s | %s | %s | %s | %s |"
        % (
            Q_HARD,
            step,
            rank_of(original, "fact_en"),
            rank_of(out.result, "fact_en"),
            ", ".join(out.trace.lost_constraints) or "-",
        )
    )
    ident = "What is the part number AB-1234 of the tide gauge?"
    ident_step = "What is a tide gauge?"
    out2 = await StepBackRetrieval().run(ident, chat=scripted({"step_back_question": ident_step}), retriever=dense)
    emit(
        "| losing (identifier) | %s | %s | - | - | %s |"
        % (ident, ident_step, ", ".join(out2.trace.lost_constraints) or "-")
    )
    emit()
    emit(
        "Order searched for the identifier question: %s; trace status `%s`. The step-back query lost the "
        "identifier and is still searched; the original is always searched first."
        % (" then ".join("`%s`" % q for q in out2.trace.retrieval_queries), out2.trace.status)
    )
    emit()


async def r17_section(owner, index, version, embedder, dense):
    emit("## R17 HyDE (scripted chat, not a model)")
    emit()
    good = "The tidal observatory at Port Sallow records sea levels every few minutes."
    bad = "A bakery on Mill Lane bakes rye bread before sunrise."
    corpus = {text for _, _, text in spans()}
    plain = await dense.retrieve(Q_HARD)
    emit("Question: `%s`" % Q_HARD)
    emit()
    emit(
        "| case | include_question | passage | status | fact_en rank | top 5 | "
        "passage text in result JSON | every hit text is a corpus chunk |"
    )
    emit("|---|---|---|---|---|---|---|---|")
    emit(
        "| R02 same question | - | - | - | %s | %s | - | - |" % (rank_of(plain, "fact_en"), ", ".join(labels_of(plain)))
    )
    cases = [
        ("success, defaults", True, good),
        ("success, passage only", False, good),
        ("losing, hallucinated passage, defaults", True, bad),
        ("losing, hallucinated passage, passage only", False, bad),
    ]
    for name, include, passage in cases:
        retriever = HydeRetriever(
            HydeConfig(include_question=include, k=5),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
        )
        before = len(embedder.calls)
        out = await retriever.retrieve(Q_HARD, chat=scripted({"passages": [passage]}))
        delta = len(embedder.calls) - before
        if name == "success, defaults":
            meter("R17", "success, defaults", out.trace, 1, delta)
        in_json = passage in out.result.model_dump_json()
        all_corpus = all(hit.text in corpus for hit in out.result.hits)
        emit(
            "| %s | %s | `%s` | %s | %s | %s | %s | %s |"
            % (
                name,
                "yes" if include else "no",
                passage,
                out.trace.status,
                rank_of(out.result, "fact_en"),
                ", ".join(labels_of(out.result)),
                "yes" if in_json else "no",
                "yes" if all_corpus else "no",
            )
        )
    emit()
    emit(
        "Adaptation: the passage is embedded with input type `document` and the question with `query`; the paper "
        "embeds only the generated text. The hypothetical passage is never evidence: hits are corpus chunks only."
    )
    emit()


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    embedder = LexicalTestEmbedder()
    owner, _, index, version = await ready_index(spans(), embedder)
    dense = DenseRetriever(DenseConfig(k=5), scope=owner, index_version=version, index=index, embedder=embedder)

    await r13_section(dense, embedder)
    await r14_r18_section(dense, embedder)
    await r15_section(dense, embedder)
    await r16_section(dense, embedder)
    await r17_section(owner, index, version, embedder, dense)
    body = list(OUT)
    OUT.clear()

    emit("# P5.4 query transforms: demo output")
    emit()
    emit(
        "Offline fixture. The chat is a scripted responder that returns fixed JSON, and the embedder is the "
        "test-only LexicalTestEmbedder. Neither is a model. Every table value is computed by the package code "
        "from those replies."
    )
    emit()
    emit(
        "Chunk labels: fact_en and fact_ar are the two fact paragraphs, museum is the long paragraph, "
        "dist00-dist12 are the distractors in fixture order. Dense search uses k=5."
    )
    emit()
    emit("## Metering (one row per technique, fake usage, cost 0)")
    emit()
    emit(
        "Transform calls come from `trace.usage`. Retrieval calls are `len(trace.retrieval_queries)` (R17: one "
        "index search). Retrieval embeds are `embedder.calls` deltas minus the transform embeds. Tokens are the "
        "fake usage of the transform calls only."
    )
    emit()
    emit(
        "| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | "
        "transform input tokens | transform output tokens | cost USD |"
    )
    emit("|---|---|---|---|---|---|---|---|---|")
    for _, row in sorted(METER, key=lambda item: item[0]):
        emit(row)
    emit()
    OUT.extend(body)
    print("\n".join(OUT))


asyncio.run(main())
