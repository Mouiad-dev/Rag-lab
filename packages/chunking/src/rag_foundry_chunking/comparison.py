"""Deterministic offline source-span comparison; lexical token overlap (not BM25)."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from rag_foundry_techniques import ChunkInput, TechniqueUnavailableError
from rag_foundry_techniques.catalogue import load_catalogue
from rag_foundry_techniques.registry import TechniqueRegistry

from rag_foundry_chunking import register_builtin_plugins
from rag_foundry_chunking.multilingual import normalize_with_map
from rag_foundry_chunking.structure import StructureParseError


class SpanCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    cohort: str
    query: str
    relevant: tuple[dict[str, Any], ...]


class SpanRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str
    cohort: str
    label: str
    status: str
    relevant_spans: int
    span_metrics: tuple[dict[str, Any], ...] = ()
    ranked: dict[str, Any] = {}
    top1_score: float = 0.0


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    fixture_version: int
    method: str
    rows: tuple[SpanRow, ...]
    cohorts: tuple[dict[str, Any], ...]


def load_span_fixture(directory: str | Path) -> tuple[dict[str, str], tuple[SpanCase, ...], tuple[dict, ...]]:
    root = Path(directory)
    case_data = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    docs = {p.name: p.read_text(encoding="utf-8") for p in root.glob("*.md")}
    cases = tuple(SpanCase.model_validate(c) for c in case_data["cases"])
    enriched_cases = []
    for case in cases:
        enriched = []
        for span in case.relevant:
            if span["doc"] not in docs:
                raise ValueError(f"case {case.case_id}: unknown doc {span['doc']}")
            text = docs[span["doc"]]
            quote = span["quote"]
            positions = [m.start() for m in re.finditer(re.escape(quote), text)]
            occurrence = span.get("occurrence", 0)
            if occurrence < 0 or occurrence >= len(positions):
                raise ValueError(f"case {case.case_id}: quote occurrence not found")
            enriched.append({**span, "start": positions[occurrence], "end": positions[occurrence] + len(quote)})
        enriched_cases.append(case.model_copy(update={"relevant": tuple(enriched)}))
    chunkers = tuple(json.loads((root / "chunkers.json").read_text(encoding="utf-8")))
    return docs, tuple(enriched_cases), chunkers


def span_metrics(start: int, end: int, chunks: list[tuple[int, int]]) -> dict[str, Any]:
    overlaps = [(max(0, min(end, b) - max(start, a)), b - a, a, b) for a, b in chunks]
    positive = [item for item in overlaps if item[0] > 0]
    if not positive:
        return {"best_coverage": 0.0, "best_precision": 0.0, "contained": False, "pieces": 0}
    best = sorted(positive, key=lambda x: (-x[0], x[1], x[2]))[0]
    return {
        "best_coverage": round(best[0] / (end - start), 4),
        "best_precision": round(best[0] / best[1], 4),
        "contained": any(a <= start and b >= end for _, _, a, b in positive),
        "pieces": len(positive),
    }


class LexicalOverlapRanker:
    """Rank by lexical token overlap (not BM25), normalizing Arabic source views."""

    @staticmethod
    def tokens(view: str) -> set[str]:
        return {t.casefold() for t in re.findall(r"\w+", view)}

    def rank(self, query: str, text: str, chunks: list[tuple[int, int]]) -> list[tuple[float, int, int]]:
        qview, _ = normalize_with_map(query)
        q = self.tokens(qview)
        scored = []
        for a, b in chunks:
            view, _ = normalize_with_map(text, a, b)
            ct = self.tokens(view)
            score = len(q & ct) / math.sqrt(len(ct)) if ct else 0.0
            scored.append((score, a, b))
        return sorted(scored, key=lambda x: (-x[0], x[1]))


def compare_chunkers(
    docs: dict[str, str], cases: tuple[SpanCase, ...], chunkers: tuple[dict, ...], registry: TechniqueRegistry
) -> ComparisonReport:
    rows = []
    ranker = LexicalOverlapRanker()
    for case in cases:
        doc_name = case.relevant[0]["doc"] if case.relevant else next(iter(docs))
        if doc_name not in docs:
            raise ValueError(f"case {case.case_id}: unknown doc {doc_name}")
        text = docs[doc_name]
        doc = ChunkInput(text=text)
        for spec in chunkers:
            label, tid = spec["label"], spec["technique_id"]
            try:
                plugin = registry.plugin(tid)
                config = registry.validate_config(tid, spec["config"])
                instance = plugin(config)
                hierarchy = instance.chunk_hierarchy(doc) if hasattr(instance, "chunk_hierarchy") else None
                chunks = [(x.start, x.end) for x in (hierarchy.children if hierarchy else instance.chunk(doc))]
                ranked = ranker.rank(case.query, text, chunks)
                relevant = [(s["start"], s["end"]) for s in case.relevant]
                rows.append(_row(case, label, chunks, ranked, relevant, "ok"))
                if hierarchy:
                    pmap = {
                        c.start: next(p for p in hierarchy.parents if p.parent_id == c.parent_id)
                        for c in hierarchy.children
                    }
                    expanded = []
                    for _, a, _ in ranked:
                        p = pmap.get(a)
                        if p and (p.start, p.end) not in expanded:
                            expanded.append((p.start, p.end))
                    erank = [(1.0, a, b) for a, b in expanded]
                    rows.append(
                        _row(
                            case,
                            label + " + parent expansion (offline span metric, not R22/R25)",
                            [(p.start, p.end) for p in hierarchy.parents],
                            erank,
                            relevant,
                            "ok",
                            top1_score=ranked[0][0] if ranked else None,
                        )
                    )
            except TechniqueUnavailableError:
                rows.append(
                    SpanRow(
                        case_id=case.case_id,
                        cohort=case.cohort,
                        label=label,
                        status="requires_models",
                        relevant_spans=len(case.relevant),
                    )
                )
            except StructureParseError:
                rows.append(
                    SpanRow(
                        case_id=case.case_id,
                        cohort=case.cohort,
                        label=label,
                        status="not_applicable",
                        relevant_spans=len(case.relevant),
                    )
                )
    cohorts = []
    for label in sorted({r.label for r in rows}):
        for cohort in sorted({c.cohort for c in cases}):
            rs = [r for r in rows if r.label == label and r.cohort == cohort and r.status == "ok"]
            all_rows = [r for r in rows if r.label == label and r.cohort == cohort]
            statuses = sorted({r.status for r in all_rows})
            unanswerable = cohort == "unanswerable"
            metrics = (
                [f"context_chars@{k}" for k in (1, 3, 5)]
                if unanswerable
                else [f"span_recall@{k}" for k in (1, 3, 5)]
                + [f"hit@{k}" for k in (1, 3, 5)]
                + ["mrr"]
                + [f"context_chars@{k}" for k in (1, 3, 5)]
            )
            cohorts.append(
                {
                    "label": label,
                    "cohort": cohort,
                    "status": statuses[0] if len(statuses) == 1 else "mixed",
                    "n": len(rs),
                    "means": {m: _mean(_exact(r.ranked, m) for r in rs) for m in metrics},
                }
            )
    return ComparisonReport(
        fixture_version=1,
        method="lexical token overlap (not BM25); source-span coverage",
        rows=tuple(rows),
        cohorts=tuple(cohorts),
    )


def _exact(ranked: dict[str, Any], metric: str):
    """Unrounded per-case value, so cohort means are rounded once (MRR is rebuilt from the first hit k)."""
    if metric == "mrr" and "mrr" in ranked:
        first = ranked.get("first_hit_k")
        return 1 / first if first else 0.0
    return ranked.get(metric)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _row(case, label, chunks, ranked, relevant, status, top1_score=None):
    metrics = tuple(span_metrics(a, b, chunks) for a, b in relevant)
    ranked_metrics = calculate_ranked_metrics(case.relevant, ranked, relevant)
    return SpanRow(
        case_id=case.case_id,
        cohort=case.cohort,
        label=label,
        status=status,
        relevant_spans=len(relevant),
        span_metrics=metrics,
        ranked=ranked_metrics,
        top1_score=round(top1_score if top1_score is not None else (ranked[0][0] if ranked else 0.0), 4),
    )


def calculate_ranked_metrics(case_spans, ranked, relevant):
    """Calculate ranked span recall, hit, MRR, and context sizes for a case."""
    ranked_metrics = {}
    first = 0
    for k in (1, 3, 5):
        top = ranked[:k]
        if case_spans:
            covered = sum(_union_coverage(a, b, [(x, y) for _, x, y in top]) >= 0.8 for a, b in relevant)
            hit = int(covered > 0)
            if hit and first == 0:
                first = k
            ranked_metrics[f"span_recall@{k}"] = round(covered / len(relevant), 4)
            ranked_metrics[f"hit@{k}"] = hit
        ranked_metrics[f"context_chars@{k}"] = sum(b - a for _, a, b in top)
    if case_spans:
        ranked_metrics["mrr"] = round(1 / first, 4) if first else 0.0
        ranked_metrics["first_hit_k"] = first or None
    return ranked_metrics


def _union_coverage(a, b, spans):
    intervals = sorted((max(a, x), min(b, y)) for x, y in spans if x < b and a < y)
    cursor = a
    total = 0
    for x, y in intervals:
        total += max(0, y - max(cursor, x))
        cursor = max(cursor, y)
    return total / (b - a)


def render_markdown(report: ComparisonReport) -> str:
    lines = [
        "# Span relevance comparison",
        "",
        f"Method: {report.method}.",
        f"Fixture version: {report.fixture_version}.",
        "",
        (
            "Metrics: best coverage and precision use the best overlapping chunk (ties choose the shorter chunk); "
            "contained means one chunk covers the span; pieces counts overlapping chunks."
        ),
        (
            "Ranked span recall uses union coverage >= 0.8 at k=1,3,5. Hit@k means any span covered; "
            "MRR is reciprocal of first successful k, zero when none. Context counts code points."
        ),
        "Unanswerable cases report context and top-1 score only; excluded from recall, hit, and MRR denominators.",
        "",
        "## Per-cohort means",
        "",
        "| Cohort | Chunker | Status | n | Means |",
        "|---|---|---|---:|---|",
    ]
    for row in report.cohorts:
        means = ", ".join(f"{key}={value}" for key, value in row["means"].items()) or "n/a"
        lines.append(f"| {row['cohort']} | {row['label']} | {row['status']} | {row['n']} | {means} |")
    lines += [
        "",
        "## Per-case",
        "",
        "| Case | Cohort | Chunker | Status | Spans | Recall@5 | Hit@5 | MRR | Top-1 score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in report.rows:
        lines.append(
            f"| {r.case_id} | {r.cohort} | {r.label} | {r.status} | {r.relevant_spans} | "
            f"{r.ranked.get('span_recall@5', '')} | {r.ranked.get('hit@5', '')} | "
            f"{r.ranked.get('mrr', '')} | {r.top1_score:.4f} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "This measures one small authored fixture with a lexical ranker and does not measure answer quality.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir")
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    docs, cases, chunkers = load_span_fixture(args.fixture_dir)
    registry = TechniqueRegistry(load_catalogue(Path(args.catalogue)))
    register_builtin_plugins(registry)
    report = compare_chunkers(docs, cases, chunkers, registry)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "span_comparison.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out / "span_comparison.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
