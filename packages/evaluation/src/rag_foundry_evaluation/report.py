"""Deterministic Markdown rendering of one evaluation run."""

from collections.abc import Sequence

from .dataset import EvalDataset

_DEFINITIONS = (
    "A relevant span counts as found at rank k when the union of the top-k hits covers at least the coverage "
    "threshold of its code points; hits from other documents are ignored. HitRate@k is 1 when at least one span is "
    "found within k, else 0. Recall@k is the share of labelled spans found within k. RR is 1 divided by the smallest "
    "rank at which any span is found (0 when none is found), and MRR is the mean RR. nDCG@k uses grade gains "
    "(2^grade - 1) and a log2(rank + 1) discount, against the ideal ordering of the labelled grades. Spans found at "
    "the same rank are credited once, at the highest grade, so when one chunk covers several labelled spans, nDCG "
    "cannot reach 1 even when Recall is 1. Unanswerable cases are excluded from every denominator and counted "
    "separately; averages are means over answerable cases only. A dash means the value is undefined (n=0 answerable)."
)


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _num(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _at(values, k) -> str:
    return "-" if values is None else _num(values.get(str(k), values.get(k)))


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(_cell(item) for item in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return lines


def _usage_cells(usage: dict) -> list[object]:
    cost = usage.get("cost_usd")
    return [
        usage["calls"],
        usage["input_tokens"],
        usage["output_tokens"],
        "-" if cost is None else cost,
    ]


def render_markdown(
    results: dict,
    *,
    title: str,
    notes: Sequence[str],
    dataset: EvalDataset | None = None,
) -> str:
    """Render parsed `results_json`. `dataset` supplies split, language and tags for the per-case table."""
    config = results["config"]
    ks = config["ks"]
    labels = [tech["label"] for tech in config["techniques"]]
    cases = {case.case_id: case for case in dataset.cases} if dataset else {}
    case_k = 5 if 5 in ks else max(ks)
    out = [f"# {title}", "", "**Tiny offline fixture. Not a benchmark; no significance claim.**", ""]
    out += [
        f"- Run id: {results['run_id']}",
        f"- Dataset: {config['dataset_id']} version {config['dataset_version']}, sha256 {config['dataset_sha256']}",
        f"- Code version: {config['code_version']}",
        f"- Index fingerprint: {config['index_fingerprint']}",
        f"- ks: {', '.join(str(k) for k in ks)}",
        f"- Coverage threshold: {config['coverage_threshold']}",
        f"- Splits run: {', '.join(config['splits'])}",
        "- Techniques: "
        + ", ".join(f"{t['label']} ({t['technique_id']} {_cell(t['config'])})" for t in config["techniques"]),
        "",
        "## Metric definitions",
        "",
        _DEFINITIONS,
        "",
        "## Summary per label",
        "",
    ]
    header = ["split", "group", "n_cases", "n_answerable", "n_unanswerable", "n_errors"]
    header += [f"HitRate@{k}" for k in ks] + [f"Recall@{k}" for k in ks] + [f"nDCG@{k}" for k in ks] + ["MRR"]
    for label in labels:
        out += [f"### {label}", ""]
        rows = []
        for item in results["summaries"][label]:
            rows.append(
                [item["split"], item["group"], item["n_cases"], item["n_answerable"], item["n_unanswerable"]]
                + [item["n_errors"]]
                + [_at(item["hit_rate"], k) for k in ks]
                + [_at(item["recall"], k) for k in ks]
                + [_at(item["ndcg"], k) for k in ks]
                + [_num(item["mrr"])]
            )
        out += _table(header, rows) + [""]
    out += ["## Per case", ""]
    rows = []
    for row in results["cases"]:
        case = cases.get(row["case_id"])
        metrics = row["metrics"]
        top = row["hits"][0] if row["hits"] else None
        top1 = "-" if top is None else f"{top['doc_key']}:{top['start']}-{top['end']}"
        found = (
            "-" if not metrics["answerable"] else ",".join("-" if r is None else str(r) for r in metrics["found_ranks"])
        )
        status = row["status"] if row["error"] is None else f"{row['status']} ({row['error']})"
        if not metrics["answerable"]:
            status += f"; returned_hits={metrics['returned_hits']}; top1_score={_num(metrics['top1_score'])}"
        rows.append(
            [
                row["case_id"],
                case.split if case else "-",
                case.language if case else "-",
                ",".join(case.tags) if case and case.tags else "-",
                row["label"],
                found,
                _num(metrics["rr"]),
                _at(metrics["recall"], case_k),
                _at(metrics["ndcg"], case_k),
                top1,
                status,
            ]
        )
    out += _table(
        [
            "case",
            "split",
            "language",
            "tags",
            "label",
            "found_ranks",
            "RR",
            f"Recall@{case_k}",
            f"nDCG@{case_k}",
            "top-1",
            "status",
        ],
        rows,
    )
    out += [
        "",
        "For unanswerable cases the table shows returned_hits and top1_score. Scores are on different scales per "
        "technique and no abstention threshold is evaluated.",
        "",
        "## Leakage check",
        "",
    ]
    leakage = results["leakage"]
    out += [f"pairs_checked: {leakage['pairs_checked']}; jaccard threshold: {leakage['jaccard_threshold']}", ""]
    if leakage["findings"]:
        out += _table(
            ["rule", "severity", "case_a", "split_a", "case_b", "split_b", "value"],
            [
                [f["rule"], f["severity"], f["case_a"], f["split_a"], f["case_b"], f["split_b"], _num(f["value"])]
                for f in leakage["findings"]
            ],
        )
    else:
        out.append("Findings: none")
    out += ["", "## Usage", ""]
    usage_rows = []
    for label in labels:
        stages = results["usage"].get(label, {})
        if not stages:
            usage_rows.append([label, "-", 0, 0, 0, "-"])
        for stage in sorted(stages):
            usage_rows.append([label, stage] + _usage_cells(stages[stage]))
    build = results.get("index_build_usage")
    usage_rows.append(
        ["index build", "document_embedding", *_usage_cells(build)]
        if build
        else ["index build", "-", "not recorded", "-", "-", "-"]
    )
    out += _table(["label", "stage", "calls", "input_tokens", "output_tokens", "cost_usd"], usage_rows)
    out += ["", "## Notes", ""]
    out += [f"- {note}" for note in notes]
    return "\n".join(out) + "\n"
