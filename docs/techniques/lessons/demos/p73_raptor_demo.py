# ruff: noqa: UP031
"""P7.3 lesson demo: R37 RAPTOR-style summary tree (adaptation), lineage and a labelled comparison with R02.

Offline. The chat is a scripted responder that returns hand-written summaries; the embedder is the test-only
lexical vectorizer (it hashes tokens, it is not a model); the store is in memory. Every status, count and
label printed comes from calling the package code on the shared tree fixture, so the output shows lineage and
plumbing, not summary or retrieval quality. Chunk, view, node and tree IDs are random or derived from random
IDs, so they are never printed: nodes are shown as sets of paragraph labels. There are no timing values and
the output is byte-identical between runs.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_domain.summary_tree import SummaryTree  # noqa: E402
from rag_foundry_indexing import (  # noqa: E402
    R37_SUMMARY_SYSTEM_V1_SHA256,
    ClusterConfig,
    ClusterItem,
    DenseConfig,
    DenseRetriever,
    RaptorConfig,
    RaptorRetriever,
    cluster_nodes,
)

from tests.indexing._graph_fixture import evidence_span_recall  # noqa: E402
from tests.indexing._raptor_fixture import (  # noqa: E402
    LABEL_OF_START,
    LABELS,
    PARAGRAPHS,
    Q_ARABIC,
    Q_BUDGET,
    Q_DETAIL,
    Q_OVERVIEW,
    ROOT_SUMMARY,
    SPANS,
    TOPICS,
    build_fixture_tree,
)

OUT = []
BUDGET = 3
ORDER = {label: i for i, label in enumerate(LABELS.values())}
TOPIC_OF = {label: topic for topic, labels in TOPICS.items() for label in labels}
QUESTIONS = (("Q_BUDGET", Q_BUDGET), ("Q_OVERVIEW", Q_OVERVIEW), ("Q_DETAIL", Q_DETAIL), ("Q_ARABIC", Q_ARABIC))
BUDGET_LABELS = ("H2", "H4", "O2", "C2")

# Level-1 summaries as rewritten after a first run (see the review sentence in section 4). The harbor summary
# deliberately repeats Q_DETAIL's query words so beam-1 traversal is pulled into the wrong branch; it is
# adversarial routing text, not a realistic summary. Same text as tests/indexing/test_raptor_retrieve.py.
SUMMARIES = {
    "harbor": "Harbor topic: what each harbor budget does pay for; what is kept at the harbor, at what temperature.",
    "orchard": "Orchard topic: what each orchard budget does pay for.",
    "clinic": "Clinic topic: what each clinic budget does pay for.",
}
REVIEW_SENTENCE = (
    "Fixture summaries were rewritten after a first run so the R37 advantage and the losing case appear; this "
    "illustrates how routing works and how it fails, not that R37 retrieves better than dense R02."
)


def emit(line=""):
    OUT.append(line)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for row in rows:
        emit("| " + " | ".join(str(cell) for cell in row) + " |")
    emit()


def responder(request):
    data = json.loads(request.messages[1].content)
    if data["level"] >= 2:
        return json.dumps({"summary": ROOT_SUMMARY}), "stop"
    joined = " ".join(child["text"] for child in data["children"]).casefold()
    topic = max(SUMMARIES, key=lambda name: joined.count(name))
    return json.dumps({"summary": SUMMARIES[topic]}), "stop"


def skipping_responder(request):
    """Same as `responder`, but the orchard cluster gets an empty summary, which the builder rejects."""
    data = json.loads(request.messages[1].content)
    joined = " ".join(child["text"] for child in data["children"]).casefold()
    if data["level"] == 1 and joined.count("orchard") > joined.count("harbor") + joined.count("clinic"):
        return json.dumps({"summary": ""}), "stop"
    return responder(request)


def tokens(usage):
    return sum(u.input_tokens for u in usage), sum(u.output_tokens for u in usage)


def members(tree, node_id):
    """Paragraph labels of the original leaves below a node, in paragraph order."""
    return sorted((LABEL_OF_START[s.start] for s in tree.lineage(node_id)), key=ORDER.get)


def level_nodes(tree, level):
    ids = [n.node_id for n in tree.nodes if n.level == level]
    return sorted(ids, key=lambda i: [ORDER[m] for m in members(tree, i)])


def node_text(tree, node_id):
    return next(n.text for n in tree.nodes if n.node_id == node_id)


def describe(tree, level, node_id):
    return "L%d{%s}" % (level, ",".join(members(tree, node_id)))


def hit_labels(result, fx):
    return [LABEL_OF_START[h.start] for h in result.hits if h.extraction_view_id == fx.view_id]


def topic_coverage(labels):
    wanted = {t: set(TOPICS[t]) & set(BUDGET_LABELS) for t in TOPICS}
    return sum(bool(set(labels) & mine) for mine in wanted.values()) / len(wanted)


def retriever(fx, **config):
    return RaptorRetriever(
        RaptorConfig(**config),
        scope=fx.scope,
        tree=fx.tree,
        index=fx.index,
        index_version=fx.version,
        embedder=fx.embedder,
    )


def dense(fx):
    return DenseRetriever(
        DenseConfig(k=BUDGET), scope=fx.scope, index_version=fx.version, index=fx.index, embedder=fx.embedder
    )


def cluster_level(tree, level):
    """Re-run the public clustering on the pool the builder used for `level` (all leaves, or the level below)."""
    pool = [n for n in tree.nodes if n.level == level - 1]
    items = []
    for n in pool:
        first = min(tree.lineage(n.node_id), key=lambda s: (s.start, s.end))
        items.append(ClusterItem(key=(0, first.start, first.end), node_id=n.node_id, embedding=n.embedding))
    return cluster_nodes(items, ClusterConfig())


async def run_arm(fx, name, make, question):
    """One arm: returns (hit labels, span recall, hit count, selected nodes text, embed calls used)."""
    before = len(fx.embedder.calls)
    arm = make()
    result = await arm.retrieve(question.text)
    calls = len(fx.embedder.calls) - before
    trace = getattr(arm, "last_trace", None)
    selected = "-"
    if trace is not None:
        lookup = {n.node_id: n for n in fx.tree.nodes}
        selected = "; ".join(describe(fx.tree, lookup[i].level, i) for i, _, _, _ in trace.selected_nodes)
    recall = evidence_span_recall(result, question.gold, fx.view_id)
    return hit_labels(result, fx), recall, len(result.hits), selected, calls


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P7.3 R37 summary tree: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p73_raptor_demo.py`. Date: 2026-11-22. "
        "Offline; scripted summaries; shows lineage and plumbing, not summary or retrieval quality."
    )
    emit()
    emit(
        "Scripted chat (fixed JSON replies, not a model), test-only lexical embedder (token hashing, not a model), "
        "in-memory store. Summaries are synthetic text and are never evidence; hits are always original leaf "
        "chunks. This is an adaptation of RAPTOR (cosine k-means, hard clusters), not the paper's UMAP and GMM. "
        "None of this is a benchmark."
    )
    emit()

    full = await build_fixture_tree(responder=responder)
    flat = await build_fixture_tree(responder=responder, max_levels=1)
    skip = await build_fixture_tree(responder=skipping_responder, on_invalid="skip")
    view = await full.source.view_source(full.scope, full.view_id)
    tree = full.tree

    emit("## 1. Fixture paragraphs and gold spans")
    emit()
    emit(
        "13 paragraphs, one chunk each, in three lexically disjoint topics. C5-ar is an Arabic clinic sentence that "
        "also carries the Latin word `clinic`; the lexical embedder shares no token across scripts, so that word, "
        "not meaning, groups it with the clinic topic. Offsets are Unicode code points in the view text."
    )
    emit()
    table(
        ("label", "topic", "span", "text"),
        [
            (LABELS[i], TOPIC_OF[LABELS[i]], "[%d, %d)" % (a, b), text.replace("|", "/"))
            for i, (a, b, text) in enumerate(SPANS)
        ],
    )
    label_of_span = {(a, b): LABELS[i] for i, (a, b, _) in enumerate(SPANS)}
    table(
        ("question", "text", "gold labels", "gold spans (denominator)"),
        [
            (
                name,
                q.text,
                ", ".join(label_of_span[g] for g in q.gold) if len(q.gold) < 13 else "all 13 paragraphs",
                len(q.gold),
            )
            for name, q in QUESTIONS
        ],
    )
    emit(
        "Q_BUDGET is scored as topic coverage (share of the 3 topics with at least one budget paragraph among the "
        "hits) and as span recall over its 4 gold spans, which is at most 0.75 at a 3-hit budget."
    )
    emit()

    emit("## 2. Tree")
    emit()
    emit(
        "Summary prompt `R37_SUMMARY_SYSTEM_V1`, sha256 `%s`. Nodes are listed as the set of paragraph labels "
        "below them; every summary is `synthetic=True`, every leaf `synthetic=False`." % R37_SUMMARY_SYSTEM_V1_SHA256
    )
    emit()
    emit("Levels: %d. Nodes: %d. Leaves: %d." % (full.report.levels, full.report.nodes, len(level_nodes(tree, 0))))
    emit()
    for level in range(1, full.report.levels):
        cluster = cluster_level(tree, level)
        parts = {tuple(sorted(members(tree, i), key=ORDER.get)) for i in level_nodes(tree, level)}
        again = {
            tuple(sorted((m for c in group for m in members(tree, c)), key=ORDER.get)) for group in cluster.clusters
        }
        emit(
            "**Level %d** (clustering re-run with the public `cluster_nodes`: iterations %d, converged %s, "
            "partition equals the built tree: %s)" % (level, cluster.iterations, cluster.converged, parts == again)
        )
        emit()
        table(
            ("node", "synthetic", "summary text (synthetic)"),
            [
                (describe(tree, level, i), next(n.synthetic for n in tree.nodes if n.node_id == i), node_text(tree, i))
                for i in level_nodes(tree, level)
            ],
        )
    emit(
        "Flat variant (`max_levels=1`): %d levels, %d nodes (the leaves plus the level-1 summaries, no root)."
        % (flat.report.levels, flat.report.nodes)
    )
    emit()
    promoted = skip.report.promoted
    emit(
        '**Second build with `on_invalid="skip"`** (the scripted chat returns an empty summary for the orchard '
        "cluster): %d promoted cluster(s), %d levels, %d nodes."
        % (len(promoted), skip.report.levels, skip.report.nodes)
    )
    emit()
    table(
        ("level", "promoted cluster (leaves kept in the pool without a summary)"),
        [
            (level, ", ".join(sorted((LABEL_OF_START[s.start] for s in spans), key=ORDER.get)))
            for level, spans in promoted
        ],
    )
    per_level = [len(level_nodes(skip.tree, lv)) for lv in range(skip.report.levels)]
    emit(
        "Nodes per level in that tree: %s. The promoted orchard leaves stay in the pool, so at level 2 they are "
        "clustered again together with the 2 level-1 summaries (giving %d level-2 summaries), then a root is "
        "built at level 3; the full build reaches its root a level earlier." % (per_level, per_level[2])
    )
    emit()
    again = await full.index.get_summary_tree(full.scope, full.version.id, full.report.tree_key)
    emit(
        "Store round trip (build, store, read again): tree equal = %s. JSON round trip (`to_json`, `from_json`): "
        "tree equal = %s. Tree key and node IDs differ per run (random index version ID), so they are not printed."
        % (again == tree, SummaryTree.from_json(tree.to_json()) == tree)
    )
    emit()

    emit("## 3. Lineage")
    emit()
    emit(
        "Lineage is structural: a summary stores child IDs only, and `lineage` walks down to the original leaf "
        "spans. Each quote below is compared with the slice of the view text at its offsets."
    )
    emit()
    inspected = list(tree.roots()) + level_nodes(tree, 1)
    for node_id in inspected:
        node = next(n for n in tree.nodes if n.node_id == node_id)
        spans = sorted(tree.lineage(node_id), key=lambda s: s.start)
        kind = "root" if node_id in tree.roots() else "level-1 node"
        emit("**%s %s** (synthetic=%s)" % (kind, describe(tree, node.level, node_id), node.synthetic))
        emit()
        table(
            ("leaf", "span", "quote", "equals view slice"),
            [
                (
                    LABEL_OF_START[s.start],
                    "[%d, %d)" % (s.start, s.end),
                    s.quote.replace("|", "/"),
                    view.text[s.start : s.end] == s.quote,
                )
                for s in spans
            ],
        )

    arms = (
        ("R37 collapsed", full, lambda: retriever(full, max_hits=BUDGET)),
        ("R37 traversal (beam 2)", full, lambda: retriever(full, mode="traversal", max_hits=BUDGET)),
        ("flat summary index (R37 on `max_levels=1`)", flat, lambda: retriever(flat, max_hits=BUDGET)),
        ("R02 dense (k=3)", full, lambda: dense(full)),
    )
    emit("## 4. Comparison at a 3-hit budget")
    emit()
    emit(
        "Metric: span recall = gold paragraph spans fully inside one hit / number of gold spans, printed as "
        "`recall (hits=n)`. Denominators: Q_BUDGET 4, Q_OVERVIEW 13, Q_DETAIL 1, Q_ARABIC 1. Q_BUDGET also shows "
        "topic coverage = topics (of 3) with a budget paragraph among the hits. Four hand-written questions on one "
        "13-paragraph fixture, not a benchmark."
    )
    emit()
    emit("> " + REVIEW_SENTENCE)
    emit()
    emit(
        "Traversal (beam 2) returns 2 hits on every question below, under the 3-hit budget, so its recall is capped "
        "by that. Q_OVERVIEW has 13 gold spans, so 3 hits cap recall at 3/13 and 2 hits at 2/13."
    )
    emit()
    cost_rows = []
    rows = []
    for name, question in QUESTIONS:
        for arm_name, fx, make in arms:
            labels, recall, n, selected, calls = await run_arm(fx, arm_name, make, question)
            extra = ""
            if name == "Q_BUDGET":
                extra = "; topic coverage %.2f" % topic_coverage(labels)
            rows.append((name, arm_name, ", ".join(labels), "%.2f (hits=%d)%s" % (recall, n, extra), selected))
            cost_rows.append((name, arm_name, calls))
    table(("question", "arm", "hit labels", "recall (hits=n)", "selected nodes (level, members)"), rows)
    budget = {r[1]: r[3] for r in rows if r[0] == "Q_BUDGET"}
    emit(
        "On Q_BUDGET the arms read: %s. Traversal with beam 2 is the weakest arm there; it is a second losing "
        "case for routing by summaries." % "; ".join("%s %s" % (arm, value) for arm, value in budget.items())
    )
    emit()
    emit(
        "R02 dense returns raw chunks with no tree, so it has no selected nodes. The root summary does not raise "
        "recall on Q_OVERVIEW: collapsed mode takes only the best leaf under each selected node, so at 3 hits the "
        "root changes which paragraphs come back, not how many gold ones."
    )
    emit()

    emit("## 5. Losing case: traversal with beam 1 on Q_DETAIL")
    emit()
    lbl, rec, n, sel, _ = await run_arm(
        full, "beam1", lambda: retriever(full, mode="traversal", beam=1, max_hits=BUDGET), Q_DETAIL
    )
    beam1 = (lbl, rec, n, sel)
    lbl, rec, n, sel, _ = await run_arm(full, "collapsed", lambda: retriever(full, max_hits=BUDGET), Q_DETAIL)
    collapsed = (lbl, rec, n, sel)
    lbl, rec, n, sel, _ = await run_arm(full, "dense", lambda: dense(full), Q_DETAIL)
    dense_arm = (lbl, rec, n, sel)
    table(
        ("arm", "hit labels", "recall (hits=n), 1 gold span (C3)", "selected nodes"),
        [
            ("R37 traversal, beam 1", ", ".join(beam1[0]), "%.2f (hits=%d)" % (beam1[1], beam1[2]), beam1[3]),
            ("R37 collapsed", ", ".join(collapsed[0]), "%.2f (hits=%d)" % (collapsed[1], collapsed[2]), collapsed[3]),
            ("R02 dense (k=3)", ", ".join(dense_arm[0]), "%.2f (hits=%d)" % (dense_arm[1], dense_arm[2]), "-"),
        ],
    )
    emit(
        "Why: traversal keeps only `beam` nodes per level and never looks back. The clinic summary omits the "
        "vaccine fridge detail, and the harbor summary was written to contain the question's words (what, kept, "
        "at, temperature), so the single kept branch is the harbor one and the answer paragraph C3 is never "
        "reached. Dense R02 matches the paragraph directly. This is routing text written to make the failure "
        "appear, not a realistic summary."
    )
    emit()

    emit("## 6. Cost")
    emit()
    emit("Scripted chat and test embedder; token counts are the fake models' estimates, not provider figures.")
    emit()
    cost = []
    for name, fx in (("full tree", full), ("flat tree (`max_levels=1`)", flat), ("full tree, skip build", skip)):
        ci, co = tokens(fx.report.usage_chat)
        ei, eo = tokens(fx.report.usage_embed)
        cost.append((name, fx.report.chat_calls, ci, co, fx.report.embed_calls, ei, eo))
    table(
        (
            "build",
            "chat calls",
            "chat input tokens",
            "chat output tokens",
            "embed calls",
            "embed input tokens",
            "embed output tokens",
        ),
        cost,
    )
    emit(
        "Per query, R37 and R02 make 0 chat calls and the embed calls below (counted on the embedder). The "
        "retrievers do not expose embed token usage, so tokens per query are not measured."
    )
    emit()
    table(
        ("question", "arm", "query embed calls"),
        [row for row in cost_rows],
    )
    emit("Update cost = full rebuild; not measured separately.")
    emit()
    emit("## Live run (orchestrator)")
    emit()
    emit("<orchestrator>")
    emit("Live run result goes here (P7.3-10). Live output is not deterministic and is recorded separately.")
    emit("</orchestrator>")
    emit()
    assert len(PARAGRAPHS) == 13
    print("\n".join(OUT))


if __name__ == "__main__":
    asyncio.run(main())
