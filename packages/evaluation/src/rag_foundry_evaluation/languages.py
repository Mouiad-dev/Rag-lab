"""Query language -> document language pairs, with script-based document language detection."""

import unicodedata
from typing import Literal

from .dataset import EvalCase, EvalDataset
from .metrics import EvalHit, GroupSummary, _validate_ks, case_metrics, summarize_group
from .report import _at, _num, _table

_LETTER_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lo"})
_SPLITS = ("dev", "validation", "held_out")


def script_language(text: str) -> Literal["en", "ar", "mixed", "unknown"]:
    """Label text by the script of its letters.

    Latin script is labelled en; this is script detection, not language identification.
    """
    arabic = latin = 0
    for char in text:
        if unicodedata.category(char) not in _LETTER_CATEGORIES:
            continue
        name = unicodedata.name(char, "")
        if "ARABIC" in name:
            arabic += 1
        elif "LATIN" in name:
            latin += 1
    if arabic + latin == 0:
        return "unknown"
    share = arabic / (arabic + latin)
    if share >= 0.8:
        return "ar"
    if share <= 0.2:
        return "en"
    return "mixed"


def document_language(case: EvalCase) -> str:
    if not case.answerable:
        return "none"
    values = {script_language(span.quote) for span in case.relevant}
    return values.pop() if len(values) == 1 else "mixed"


def language_pair(case: EvalCase) -> str:
    return f"{case.language}->{document_language(case)}"


def aggregate_language_pairs(rows, ks) -> tuple[GroupSummary, ...]:
    ks = _validate_ks(ks)
    output = []
    for split in _SPLITS:
        by_pair: dict[str, list] = {}
        for row in rows:
            if row[0].split == split:
                by_pair.setdefault(language_pair(row[0]), []).append(row)
        for pair in sorted(by_pair):
            output.append(summarize_group(split, f"pair:{pair}", by_pair[pair], ks))
    return tuple(output)


def _labels(results: dict) -> list[str]:
    return [tech["label"] for tech in results["config"]["techniques"]]


def _label_rows(results: dict, dataset: EvalDataset, label: str):
    config = results["config"]
    cases = {case.case_id: case for case in dataset.cases}
    output = []
    for row in results["cases"]:
        if row["label"] != label:
            continue
        case = cases[row["case_id"]]
        hits = tuple(EvalHit.model_validate(hit) for hit in row["hits"])
        metrics = case_metrics(case, hits, ks=config["ks"], threshold=config["coverage_threshold"])
        output.append((case, metrics, row["status"] == "error"))
    return output


def language_pair_summaries(results: dict, dataset: EvalDataset) -> dict[str, tuple[GroupSummary, ...]]:
    ks = tuple(results["config"]["ks"])
    return {label: aggregate_language_pairs(_label_rows(results, dataset, label), ks) for label in _labels(results)}


def render_language_pairs(results: dict, dataset: EvalDataset) -> str:
    ks = results["config"]["ks"]
    lines = [
        "## Language pairs",
        "",
        "A pair is the question language -> the document language. The document language is the script of the "
        "relevant span quotes (>= 80% Arabic letters is ar, <= 20% is en, else mixed; any Latin script counts as "
        "en); unanswerable cases map to none.",
    ]
    header = (
        ["split", "pair", "n_cases", "n_answerable", "n_unanswerable", "n_errors"]
        + [f"HitRate@{k}" for k in ks]
        + [f"Recall@{k}" for k in ks]
        + ["MRR"]
    )
    for label, summaries in language_pair_summaries(results, dataset).items():
        body = [
            [s.split, s.group.removeprefix("pair:"), s.n_cases, s.n_answerable, s.n_unanswerable, s.n_errors]
            + [_at(s.hit_rate, k) for k in ks]
            + [_at(s.recall, k) for k in ks]
            + [_num(s.mrr)]
            for s in summaries
        ]
        lines += ["", f"### {label}", "", *_table(header, body)]
    per_case = []
    for label in _labels(results):
        for case, metrics, _ in _label_rows(results, dataset, label):
            per_case.append(
                [
                    case.case_id,
                    label,
                    language_pair(case),
                    ",".join(case.tags) or "-",
                    ",".join("-" if r is None else str(r) for r in metrics.found_ranks) or "-",
                    _num(metrics.rr),
                ]
            )
    lines += ["", "### Per case", "", *_table(["case", "label", "pair", "tags", "found_ranks", "RR"], per_case)]
    return "\n".join(lines) + "\n"
