"""Deterministic paired retrieval evaluation and Markdown rendering."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from rag_foundry_chunking.comparison import calculate_ranked_metrics


class Hit(Protocol):
    score: float
    start: int
    end: int
    text: str


class Result(Protocol):
    hits: tuple[Hit, ...]


class Retriever(Protocol):
    async def retrieve(self, query: str) -> Result: ...


@dataclass(frozen=True)
class PairedCase:
    case_id: str
    cohort: str
    split: Literal["dev", "validation"]
    query: str
    relevant: tuple[tuple[int, int], ...]


def case_metrics(case: PairedCase, result: Result) -> dict:
    ranked = [(hit.score, hit.start, hit.end) for hit in result.hits]
    if not case.relevant:
        return {
            "top1_score": result.hits[0].score if result.hits else None,
            "context_chars@5": sum(h.end - h.start for h in result.hits[:5]),
        }
    metrics = calculate_ranked_metrics(case.relevant, ranked, case.relevant)
    first_rank = None
    for rank in range(1, 6):
        top = result.hits[:rank]
        covered = False
        for start, end in case.relevant:
            intervals = sorted(
                (max(start, hit.start), min(end, hit.end)) for hit in top if min(end, hit.end) > max(start, hit.start)
            )
            union = 0
            right = start
            for left, stop in intervals:
                if stop > right:
                    union += stop - max(left, right)
                    right = stop
            if union / (end - start) >= 0.8:
                covered = True
                break
        if covered:
            first_rank = rank
            break
    return {key: metrics[key] for key in ("hit@1", "hit@3", "hit@5", "span_recall@5")} | {
        "mrr": 1 / first_rank if first_rank is not None else 0.0,
        "first_hit_k": first_rank,
    }


def choose_weight(dev_mrr: dict[float, float]) -> float:
    return min(dev_mrr, key=lambda weight: (-dev_mrr[weight], abs(weight - 0.5), weight))


async def run_paired(cases, labels: dict[str, Retriever], sweep: Callable[[float], Retriever]):
    ordered_cases = tuple(cases)
    dev = [case for case in ordered_cases if case.split == "dev"]
    sweep_scores = {}
    for i in range(11):
        weight = round(i / 10, 1)
        retriever = sweep(weight)
        scores = []
        for case in dev:
            metrics = case_metrics(case, await retriever.retrieve(case.query))
            scores.append(metrics["mrr"])
        sweep_scores[weight] = sum(scores) / len(scores)
    best_weight = choose_weight(sweep_scores)
    label_order = tuple(labels) + (f"R06+R08(wl={best_weight:.1f})",)
    selected = dict(labels)
    selected[label_order[-1]] = sweep(best_weight)
    measured = {}
    for case in ordered_cases:
        measured[case.case_id] = {}
        for label, retriever in selected.items():
            measured[case.case_id][label] = await retriever.retrieve(case.query)
    summaries = {}
    for split in ("dev", "validation"):
        summaries[split] = {}
        split_cases = [case for case in ordered_cases if case.split == split]
        for label in label_order:
            vals = [(case, case_metrics(case, measured[case.case_id][label])) for case in split_cases if case.relevant]
            base = {
                case.case_id: case_metrics(case, measured[case.case_id]["R02"]) for case in split_cases if case.relevant
            }
            outcomes = {"win": 0, "tie": 0, "loss": 0}
            for case, metrics in vals:
                delta = round(metrics["mrr"] - base[case.case_id]["mrr"], 4)
                outcomes["win" if delta > 0 else "loss" if delta < 0 else "tie"] += 1
            n = len(vals)
            summaries[split][label] = {
                "n": n,
                "hit@1": sum(m["hit@1"] for _, m in vals) / n,
                "hit@5": sum(m["hit@5"] for _, m in vals) / n,
                "mrr": sum(m["mrr"] for _, m in vals) / n,
                **outcomes,
            }
    return {
        "cases": ordered_cases,
        "labels": label_order,
        "sweep": sweep_scores,
        "weight": best_weight,
        "results": measured,
        "summaries": summaries,
    }


def render_markdown(report) -> str:
    cases, labels = report["cases"], report["labels"]
    lines = [
        "# Paired comparison: dense and lexical retrieval",
        "",
        "Method: source-span relevance; a retrieved chunk hits when it covers at least 80% of a relevant span.",
        "Dense branch uses `LexicalTestEmbedder` (hashed bag of words, 256 buckets, test-only). "
        "It is not a semantic model, so these rows compare lexical scoring variants, not semantic recall.",
        "Hit@k uses k=1, 3, 5. Per-case RR@5 is 1/first_rank, where first_rank is the "
        "smallest r in 1..5 whose hits[:r] union covers at least 80% of a relevant span; mean MRR@5 is the "
        "mean of these reciprocal ranks.",
        "R04 absent (needs PostgreSQL).",
        (
            f"min-max has no fitted parameters; dev and validation report denominators are computed from the cases. "
            f"{_weight_selection_text(report)}"
        ),
        "",
        "## Dev sweep",
        "",
        "| w | dev MRR |",
        "|---:|---:|",
    ]
    lines += [f"| {w:.1f} | {score:.4f} |" for w, score in report["sweep"].items()]
    lines += [
        "",
        "## Per-split summary",
        "",
        "| split | label | n | mean hit@1 | mean hit@5 | mean MRR@5 | W | T | L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, labels_data in report["summaries"].items():
        for label in labels:
            d = labels_data[label]
            lines.append(
                f"| {split} | {label} | {d['n']} | {d['hit@1']:.4f} | {d['hit@5']:.4f} | "
                f"{d['mrr']:.4f} | {d['win']} | {d['tie']} | {d['loss']} |"
            )
    lines += [
        "",
        "## Per-case results",
        "",
        "| case | cohort | split | label | first relevant rank | RR@5 | delta vs R02 | outcome | "
        "top-1 chunk (first 40 chars) |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for case in cases:
        baseline = case_metrics(case, report["results"][case.case_id]["R02"]) if case.relevant else None
        for label in labels:
            result = report["results"][case.case_id][label]
            metrics = case_metrics(case, result)
            top = result.hits[0].text[:40].replace("|", "\\|") if result.hits else "-"
            if case.relevant:
                delta = round(metrics["mrr"] - baseline["mrr"], 4)
                outcome = "win" if delta > 0 else "loss" if delta < 0 else "tie"
                lines.append(
                    f"| {case.case_id} | {case.cohort} | {case.split} | {label} | "
                    f"{metrics['first_hit_k'] or '-'} | {metrics['mrr']:.4f} | {delta:.4f} | {outcome} | {top} |"
                )
            else:
                lines.append(
                    f"| {case.case_id} | {case.cohort} | {case.split} | {label} | - | - | - | - | "
                    f"top-1 score {metrics['top1_score']:.4f}; context_chars@5 {metrics['context_chars@5']} |"
                )
    lines += [
        "",
        "Unanswerable rows report top-1 score and context_chars@5; top-1 scores use different scales and are "
        "not comparable across labels; no abstention threshold is evaluated. They are excluded from hit/MRR "
        f"denominators: {sum(not case.relevant for case in cases)} case(s).",
        "",
        "## Limits",
        "",
        "Dev and validation share relevant documents (FACT_EN is used by en-fact, en-variant, and paraphrase; "
        "ID1/ID2 are near-duplicates), and a dev query is close to a validation query. OCR and table cohorts "
        "are absent. n=5 supports no significance claim. The dense branch uses a test-only lexical embedder; "
        "results describe these queries and source spans only.",
        "",
    ]
    return "\n".join(lines)


def _weight_selection_text(report) -> str:
    values = tuple(report["sweep"].values())
    if values and all(value == values[0] for value in values):
        return (
            f"dev sweep is flat (every weight gives dev MRR {values[0]:.4f}); "
            f"wl={report['weight']:.1f} chosen by tie-break, not learned"
        )
    return f"wl={report['weight']:.1f} chosen on dev"
