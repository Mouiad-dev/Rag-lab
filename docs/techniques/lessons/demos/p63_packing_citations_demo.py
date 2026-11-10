"""Offline P6.3 mechanics demo. All displayed values are computed by public APIs."""

import sys

from rag_foundry_answering.claims import validate_citations
from rag_foundry_indexing import ExtractiveCompressionConfig, ExtractiveCompressor
from rag_foundry_indexing.packing import ContextPackConfig, ContextPacker
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

H = "The bridge opened on 3 March 2019. Tolls are 5 USD per car. This does not apply to bicycles. The cafe sells tea."
TA = (
    "\u0627\u0641\u062a\u062a\u062d \u0627\u0644\u062c\u0633\u0631 \u0639\u0627\u0645 "
    "\u0662\u0660\u0661\u0669. \u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631 "
    "\u0665 \u062f\u0648\u0644\u0627\u0631. \u0644\u0627 \u062a\u0646\u0637\u0628\u0642 "
    "\u0627\u0644\u0631\u0633\u0648\u0645 \u0639\u0644\u0649 \u0627\u0644\u062f\u0631\u0627\u062c\u0627\u062a. "
    "\u0627\u0644\u0645\u0642\u0647\u0649 \u064a\u0628\u064a\u0639 \u0627\u0644\u0634\u0627\u064a."
)
Q = "What are the bridge tolls?"
QTA = "\u0645\u0627 \u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631\u061f"


def result(text, start=100, rank=1, chunk_id="c1"):
    hit = RetrievedChunk(
        chunk_id=chunk_id,
        extraction_view_id="v1",
        start=start,
        end=start + len(text),
        block_ids=("b1",),
        text=text,
        score=1.0 / rank,
        rank=rank,
    )
    return RetrievalResult(technique_id="R02", grounding="corpus", index_version_id="iv1", hits=(hit,))


def main():
    out = [
        "# P6.3 demo | uv run python docs/techniques/lessons/demos/p63_packing_citations_demo.py | 2026-11-10 | "
        "offline; no model; tiny hand-made fixtures; not a benchmark",
        "",
        "## R26 success (EN)",
        "| sentence | overlap | reason | constraints |",
        "|---|---:|---|---|",
    ]
    en = ExtractiveCompressor(ExtractiveCompressionConfig()).compress(Q, result(H))
    for row in en.trace.kept + en.trace.dropped:
        text = H[row.start - 100 : row.end - 100]
        constraints = ", ".join(row.constraints) or "()"
        out.append(f"| {text} | {row.overlap:d} | {row.reason} | {constraints} |")
    out += [
        f"lost_constraints: {en.trace.lost_constraints}",
        "",
        "## R26 success (AR)",
        "| sentence | overlap | reason | constraints |",
        "|---|---:|---|---|",
    ]
    ar = ExtractiveCompressor(ExtractiveCompressionConfig()).compress(QTA, result(TA, start=0))
    for row in ar.trace.kept + ar.trace.dropped:
        text = TA[row.start : row.end]
        constraints = ", ".join(row.constraints) or "()"
        out.append(f"| {text} | {row.overlap:d} | {row.reason} | {constraints} |")
    out += [
        f"lost_constraints: {ar.trace.lost_constraints}",
        "",
        "## R26 losing",
        "| case | kept | lost_constraints |",
        "|---|---|---|",
    ]
    off = ExtractiveCompressor(ExtractiveCompressionConfig(keep_qualifiers=False)).compress(Q, result(H))
    synonym = ExtractiveCompressor(ExtractiveCompressionConfig()).compress("How much is the fee?", result(H))
    out += [
        "| keep_qualifiers=False | {} | {} |".format(
            ", ".join(h.text for h in off.result.hits), off.trace.lost_constraints
        ),
        "| synonym miss: How much is the fee? | {} | {} |".format(
            ", ".join(h.text for h in synonym.result.hits) or "no hit", synonym.trace.lost_constraints
        ),
        "",
        "## R27",
        "| order | ranks in output order |",
        "|---|---|",
    ]
    texts = [
        ("h1", "Alpha one.", 40, 50),
        ("h2", "Beta two three.", 0, 15),
        ("h3", "Gamma 12345.", 20, 32),
        ("h4", "Delta.", 60, 66),
        ("h5", "Eps five six seven.", 80, 99),
    ]
    hits = tuple(
        RetrievedChunk(chunk_id=i, extraction_view_id="view", start=s, end=e, block_ids=(), text=t, score=1.0, rank=n)
        for n, (i, t, s, e) in enumerate(texts, 1)
    )
    res = RetrievalResult(technique_id="R02", grounding="corpus", index_version_id="v1", hits=hits)
    for order in ("rank", "lost_in_middle", "source_order"):
        packed = ContextPacker(
            ContextPackConfig(order=order, prompt_overhead_tokens=0, per_item_overhead_tokens=2)
        ).pack(res, model_context_window=1000, reserve_output_tokens=10)
        out.append(f"| {order} | {tuple(i.rank for i in packed.items)} |")
    budget = ContextPacker(ContextPackConfig(prompt_overhead_tokens=30, per_item_overhead_tokens=2)).pack(
        res, model_context_window=100, reserve_output_tokens=50
    )
    out += ["", f"Budget-{budget.trace.budget_tokens:d} drops:", "| rank | reason | cost |", "|---:|---|---:|"]
    for d in budget.trace.dropped:
        out.append(f"| {d.rank:d} | {d.reason} | {d.cost_tokens:d} |")
    losing_hit = hits[-1].model_copy(update={"rank": 1})
    empty = ContextPacker(ContextPackConfig(prompt_overhead_tokens=0, per_item_overhead_tokens=2)).pack(
        RetrievalResult(technique_id="R02", grounding="corpus", index_version_id="v1", hits=(losing_hit,)),
        model_context_window=10,
        reserve_output_tokens=5,
    )
    d = empty.trace.dropped[0]
    out += [
        "",
        f"R27 losing: oversized rank-1 hit; packed items={empty.items}",
        "| rank | reason | cost |",
        "|---:|---|---:|",
        f"| {d.rank:d} | {d.reason} | {d.cost_tokens:d} |",
        "",
        "## Citation validator",
    ]
    answer = (
        "Tolls are 5 USD per car [S1]. The bridge opened in 2019 [S2]. "
        "Bicycles pay 7 USD [S1]. It has a cafe. The cafe sells tea [S3]. Hours are long [S9]."
    )
    report = validate_citations(
        answer,
        provided={
            "S1": "Tolls are 5 USD per car. This does not apply to bicycles.",
            "S2": "The bridge opened on 3 March 2019.",
        },
        known={"S1", "S2", "S3"},
    )
    out += ["| claim | status | flags | overlap |", "|---|---|---|---:|"]
    for c in report.claims:
        out.append(f"| {c.text} | {c.status} | {tuple((f.code, f.value) for f in c.flags)} | {c.overlap} |")
    fp = validate_citations(
        "Tolls apply to bicycles [S1].",
        provided={"S1": "Tolls are 5 USD per car. This does not apply to bicycles."},
        known={"S1"},
    ).claims[0]
    out += [
        f"False positive: `Tolls apply to bicycles [S1].` => `{fp.status}`, overlap `{fp.overlap}`. "
        "Lexical rule, not entailment.",
        "",
        "## Usage",
        # No chat or embedding model is constructed anywhere in this script.
        "0 model calls. Tokenizer: `lab-regex-v1` (an estimate, not a model tokenizer).",
    ]
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
