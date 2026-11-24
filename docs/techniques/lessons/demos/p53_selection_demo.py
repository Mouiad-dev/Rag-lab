# ruff: noqa: UP031
"""P5.3 lesson demo: R09 MMR, R10 dedup, R11 metadata filter, R12 self-query.

Offline. Every printed number, order and status comes from calling the package code. The embedder is the
test-only LexicalTestEmbedder and the R12 chat is a scripted responder; neither is a model.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.* helpers

from rag_foundry_domain.documents import MediaType  # noqa: E402
from rag_foundry_domain.errors import QueryTransformError  # noqa: E402
from rag_foundry_indexing import (  # noqa: E402
    DedupConfig,
    DenseConfig,
    DenseRetriever,
    EvidenceDedup,
    MetadataFilter,
    MetadataFilterConfig,
    MetadataFilteredRetriever,
    MmrConfig,
    MmrSelector,
    SelfQuery,
    SelfQueryConfig,
    retrieve_with_self_query,
)
from rag_foundry_techniques import RetrievalResult, RetrievedChunk  # noqa: E402

from tests.chunking._scripted_models import ScriptedChat  # noqa: E402
from tests.indexing._lexical_embedder import LexicalTestEmbedder  # noqa: E402
from tests.indexing._multi_doc_index import multi_doc_index  # noqa: E402

BASE = (
    "The annual service plan costs {price} dollars per customer and the contract includes support during "
    "office hours for every registered site in the northern region as well as the southern region and "
    "all remote offices that were added to the account before the end of the previous financial year"
)
AUDIT = (
    "The audit report for the northern warehouse was reviewed by the finance committee during the quarterly "
    "meeting and the committee decided that the contractor payment was {verdict} without any further delay"
)


def make_result(rows, version_id):
    hits = tuple(
        RetrievedChunk(
            chunk_id="c%d" % (i + 1),
            extraction_view_id=view,
            start=start,
            end=end,
            block_ids=("b00000",),
            text=text,
            score=round(1.0 - i * 0.1, 1),
            rank=i + 1,
        )
        for i, (view, start, end, text) in enumerate(rows)
    )
    return RetrievalResult(technique_id="R03", grounding="corpus", index_version_id=version_id, hits=hits)


def hand_rows():
    base10, base12 = BASE.format(price=10), BASE.format(price=12)
    return [
        ("view-a", 0, len(base10), base10),
        ("view-b", 0, len(base10), base10),
        ("view-a", 10, 50, "support during office hours for every registered site"),
        ("view-c", 0, len(base10) + 6, base10 + " today"),
        ("view-d", 0, len(base12), base12),
        ("view-e", 0, 40, "The office closes at five and stays shut at night"),
    ]


def words_rows():
    approved, rejected = AUDIT.format(verdict="approved"), AUDIT.format(verdict="rejected")
    return [("view-a", 0, len(approved), approved), ("view-b", 0, len(rejected), rejected)]


def show_dedup(title, result, config):
    out = EvidenceDedup(config).select(result)
    n_in, n_drop = len(result.hits), len(out.dropped)
    again = EvidenceDedup(config).select(out.result)
    print("| %s | input | kept | dropped | redundancy before | redundancy after |" % title)
    print("|---|---|---|---|---|---|")
    print(
        "| counts | %d | %d | %d | %.3f | %.3f |"
        % (n_in, len(out.result.hits), n_drop, n_drop / n_in, len(again.dropped) / len(out.result.hits))
    )
    print()
    print("| dropped | rule | similarity | provenance (view, span) | merged into |")
    print("|---|---|---|---|---|")
    for d in out.dropped:
        print(
            "| %s | %s | %.3f | %s [%d:%d] | %s |"
            % (d.chunk_id, d.rule, d.similarity, d.extraction_view_id, d.start, d.end, d.kept_chunk_id)
        )
    if not out.dropped:
        print("| (none) | | | | |")
    print()
    print("| kept | view | span | text |")
    print("|---|---|---|---|")
    for h in out.result.hits:
        print("| %s | %s | [%d:%d] | %s |" % (h.chunk_id, h.extraction_view_id, h.start, h.end, h.text[:60]))
    print()
    return out


async def mmr_section(data):
    print("## R09 MMR (lexical similarity)")
    print()
    result = make_result(hand_rows(), data.owner_version.id)
    print("| k | lambda | scoring | order | distinct views | distinct texts | in top k at 1.0 but lost |")
    print("|---|---|---|---|---|---|---|")
    for k in (3, 2):
        first = None
        for lam in (1.0, 0.5, 0.0):
            selector = MmrSelector(
                MmrConfig(k=k, lambda_mult=lam, similarity="lexical"),
                scope=data.owner,
                index_version=data.owner_version,
                index=data.index,
            )
            out = await selector.select(result)
            order = [h.chunk_id for h in out.hits]
            views = len({h.extraction_view_id for h in out.hits})
            texts = len({h.text for h in out.hits})
            if first is None:
                first = set(order)
            lost = sorted(first - set(order))
            print(
                "| %d | %.1f | %s | %s | %d | %d | %s |"
                % (k, lam, out.scoring, ", ".join(order), views, texts, ", ".join(lost) or "-")
            )
    print()
    mid = await MmrSelector(
        MmrConfig(k=3, lambda_mult=0.5, similarity="lexical"),
        scope=data.owner,
        index_version=data.owner_version,
        index=data.index,
    ).select(result)
    print("| lambda 0.5 hit | relevance | max similarity | mmr score |")
    print("|---|---|---|---|")
    for h in mid.hits:
        comp = {c.name: c.value for c in h.score_components}
        print(
            "| %s | %.3f | %.3f | %.3f |"
            % (h.chunk_id, comp["mmr_relevance"], comp["mmr_max_similarity"], comp["mmr_score"])
        )
    print()
    tied = make_result([("view-a", 0, 10, "alpha beta"), ("view-b", 0, 10, "alpha beta")], data.owner_version.id)
    tied = tied.model_copy(update={"hits": tuple(h.model_copy(update={"score": 0.5}) for h in tied.hits)})
    tie_out = await MmrSelector(
        MmrConfig(k=2, lambda_mult=1.0, similarity="lexical"),
        scope=data.owner,
        index_version=data.owner_version,
        index=data.index,
    ).select(tied)
    print("| tie rule check | input scores | output order |")
    print("|---|---|---|")
    print("| equal scores | 0.5, 0.5 | %s |" % ", ".join(h.chunk_id for h in tie_out.hits))
    print()


class CountingInner:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def retrieve(self, query):
        self.calls += 1
        return await self.inner.retrieve(query)


def wrapper(data, inner, k, flt=None):
    return MetadataFilteredRetriever(
        MetadataFilterConfig(k=k, filter=flt or MetadataFilter()),
        scope=data.owner,
        index_version=data.owner_version,
        index=data.index,
        extractions=data.extractions,
        documents=data.documents,
        inner=inner,
    )


async def filter_section(data, embedder):
    print("## R11 metadata filter (post-filter)")
    print()
    query = "quarterly report"
    names = {}
    for view in await wrapper(data, None, 50).choices():
        names[view.view_id] = view.display_name
    dense50 = DenseRetriever(
        DenseConfig(k=50), scope=data.owner, index_version=data.owner_version, index=data.index, embedder=embedder
    )
    cases = [
        ("no filter", MetadataFilter()),
        ("document report-2025", MetadataFilter(document_ids=(data.document_ids["report-2025"],))),
        ("media type text/markdown", MetadataFilter(media_types=(MediaType.TEXT_MARKDOWN,))),
        ("stranger-scope document id", MetadataFilter(document_ids=(data.document_ids["stranger"],))),
    ]
    print("| filter | hits | inner retriever calls | embedder calls | scoring |")
    print("|---|---|---|---|---|")
    for label, flt in cases:
        counting = CountingInner(dense50)
        before = len(embedder.calls)
        out = await wrapper(data, counting, 50, flt).retrieve(query)
        print(
            "| %s | %d | %d | %d | %s |"
            % (label, len(out.hits), counting.calls, len(embedder.calls) - before, out.scoring or "-")
        )
    print()

    dense2 = DenseRetriever(
        DenseConfig(k=2), scope=data.owner, index_version=data.owner_version, index=data.index, embedder=embedder
    )
    top2 = await dense2.retrieve(query)
    top2_views = {h.extraction_view_id for h in top2.hits}
    target = next(v for v in await wrapper(data, None, 50).choices() if v.view_id not in top2_views)
    flt = MetadataFilter(document_ids=(target.document_id,))
    small = await wrapper(data, dense2, 5, flt).retrieve(query)
    large = await wrapper(data, dense50, 5, flt).retrieve(query)
    print("| underfill case (filter keeps one document) | inner k | hits returned |")
    print("|---|---|---|")
    print(
        "| inner top 2 holds %d hit(s) of that document | 2 | %d |"
        % (len({h.extraction_view_id for h in top2.hits} & {target.view_id}), len(small.hits))
    )
    print("| same filter, same query | 50 | %d |" % len(large.hits))
    print()
    print("| capability | status |")
    print("|---|---|")
    print("| filter fields | %s |" % ", ".join(sorted(MetadataFilter.model_fields)))
    print(
        "| language / section filter | %s |"
        % ("available" if {"language", "section"} & set(MetadataFilter.model_fields) else "not available")
    )
    print()


def scripted(payload):
    def responder(request):
        return (payload if isinstance(payload, str) else json.dumps(payload), "stop")

    return ScriptedChat(responder)


async def self_query_section(data, embedder):
    print("## R12 self-query (scripted chat, not a model)")
    print()
    dense50 = DenseRetriever(
        DenseConfig(k=50), scope=data.owner, index_version=data.owner_version, index=data.index, embedder=embedder
    )
    base = wrapper(data, dense50, 50)
    choices = await base.choices()
    refs = {}
    for view in choices:
        refs.setdefault(view.document_id, "d%d" % (len(refs) + 1))
    ref_names = {refs[v.document_id]: v.display_name for v in choices}
    print("| offered ref | document |")
    print("|---|---|")
    for ref, name in ref_names.items():
        print("| %s | %s |" % (ref, name))
    print()
    empty = dict(document_refs=None, media_types=None, created_from=None, created_to=None)
    print("| scripted response | on_invalid | status | reason | filter / result | usage (fake in/out tokens) |")
    print("|---|---|---|---|---|---|")
    chat = scripted({**empty, "document_refs": ["d1"]})
    applied = await SelfQuery().parse("what does the first report say", choices=choices, chat=chat)
    docs = [n for n, i in data.document_ids.items() if applied.filter and i in (applied.filter.document_ids or ())]
    usage = applied.usage[0]
    print(
        "| document_refs [d1] | fail | %s | %s | document_ids -> %s | %d/%d |"
        % (applied.status, applied.reason, ", ".join(docs), usage.input_tokens, usage.output_tokens)
    )
    parsed, retrieved = await retrieve_with_self_query(
        "quarterly report", self_query=SelfQuery(), chat=chat, retriever=base
    )
    print()
    print("| end to end | status | hits | views in hits |")
    print("|---|---|---|---|")
    print(
        "| retrieve_with_self_query | %s | %d | %s |"
        % (
            parsed.status,
            len(retrieved.hits),
            ", ".join(sorted({names_of(choices)[h.extraction_view_id] for h in retrieved.hits})),
        )
    )
    print()
    print("| scripted response | on_invalid | status | reason | filter / result | usage (fake in/out tokens) |")
    print("|---|---|---|---|---|---|")
    bad = scripted({**empty, "language": "ar"})
    rejected = await SelfQuery(SelfQueryConfig(on_invalid="no_filter")).parse("q", choices=choices, chat=bad)
    print(
        "| extra key language | no_filter | %s | %s | filter=%s | %d/%d |"
        % (
            rejected.status,
            rejected.reason,
            rejected.filter,
            rejected.usage[0].input_tokens,
            rejected.usage[0].output_tokens,
        )
    )
    for mode in ("no_filter", "fail"):
        widen = scripted({**empty, "document_refs": ["3f2b8a52-0c4e-4e53-9a43-6f7f6e1d9c11"]})
        try:
            await SelfQuery(SelfQueryConfig(on_invalid=mode)).parse("q", choices=choices, chat=widen)
            outcome = "no error"
        except QueryTransformError as exc:
            outcome = "raised %s" % exc.code
        print("| UUID instead of ref | %s | - | - | %s | - |" % (mode, outcome))
    print()


def names_of(choices):
    return {v.view_id: v.display_name for v in choices}


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("# P5.3 selection and filters: demo output")
    print()
    print(
        "Command: `uv run python docs/techniques/lessons/demos/p53_selection_demo.py` | Date: 2026-11-09 | "
        "offline fixture; no model; not a benchmark"
    )
    print()
    print("The embedder is the test-only LexicalTestEmbedder and the R12 chat is a scripted responder.")
    print()
    embedder = LexicalTestEmbedder()
    data = await multi_doc_index(embedder)

    print("## R10 deduplication")
    print()
    print("### Default config (protect_numbers=True)")
    print()
    hand = make_result(hand_rows(), "fixture-v1")
    show_dedup("hand list of 6", hand, DedupConfig())
    print("### Number protection off (protect_numbers=False)")
    print()
    show_dedup("hand list of 6", hand, DedupConfig(protect_numbers=False))
    print("### Losing case: a conflict stated in words, not numbers")
    print()
    print("Two hits say approved and rejected; the default near_text rule sees no number token and merges them.")
    print()
    show_dedup("approved vs rejected", make_result(words_rows(), "fixture-v1"), DedupConfig())

    await mmr_section(data)
    await filter_section(data, embedder)
    await self_query_section(data, embedder)


asyncio.run(main())
