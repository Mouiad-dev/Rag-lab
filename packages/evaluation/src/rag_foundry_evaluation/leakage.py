"""Deterministic cross-split leakage checks.

Semantic near-duplicates and document-level corpus leakage are later work.
Dropping all combining marks after NFD also removes hamza and madda from
alef forms, so U+0623, U+0625 and U+0622 compare equal to U+0627 here.
"""

import re
import unicodedata
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .dataset import EvalDataset
from .errors import EvaluationError


class LeakageFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: Literal["exact_question", "near_duplicate_question", "shared_relevant_span"]
    severity: Literal["error", "warning"]
    case_a: str
    case_b: str
    split_a: str
    split_b: str
    value: float | None


class LeakageReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jaccard_threshold: float
    pairs_checked: int
    findings: tuple[LeakageFinding, ...]

    @property
    def errors(self) -> tuple[LeakageFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn" and char != "\u0640")
    return " ".join(unicodedata.normalize("NFC", value).split())


def question_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[^\W_]+", normalize_question(text)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def check_leakage(dataset: EvalDataset, *, jaccard_threshold: float = 0.8) -> LeakageReport:
    if not 0 < jaccard_threshold <= 1:
        raise EvaluationError("invalid_threshold")

    findings: list[LeakageFinding] = []
    cross_split_pairs = [
        (left, right)
        for left, right in combinations(sorted(dataset.cases, key=lambda case: case.case_id), 2)
        if left.split != right.split
    ]
    for left, right in cross_split_pairs:
        case_a, case_b = sorted((left, right), key=lambda case: case.case_id)
        shared = {
            "case_a": case_a.case_id,
            "case_b": case_b.case_id,
            "split_a": case_a.split,
            "split_b": case_b.split,
        }
        normalized_a = normalize_question(case_a.question)
        normalized_b = normalize_question(case_b.question)
        if normalized_a == normalized_b:
            findings.append(LeakageFinding(rule="exact_question", severity="error", value=None, **shared))
        else:
            similarity = jaccard(question_tokens(case_a.question), question_tokens(case_b.question))
            if similarity >= jaccard_threshold:
                findings.append(
                    LeakageFinding(
                        rule="near_duplicate_question",
                        severity="error",
                        value=round(similarity, 6),
                        **shared,
                    )
                )
        if any(
            span_a.doc_key == span_b.doc_key and max(span_a.start, span_b.start) < min(span_a.end, span_b.end)
            for span_a in case_a.relevant
            for span_b in case_b.relevant
        ):
            findings.append(
                LeakageFinding(
                    rule="shared_relevant_span",
                    severity="warning",
                    value=None,
                    **shared,
                )
            )
    findings.sort(key=lambda item: (item.rule, item.case_a, item.case_b))
    return LeakageReport(
        jaccard_threshold=jaccard_threshold,
        pairs_checked=len(cross_split_pairs),
        findings=tuple(findings),
    )


def raise_on_errors(report: LeakageReport) -> None:
    if report.errors:
        finding = report.errors[0]
        raise EvaluationError("leakage", f"{finding.rule} {finding.case_a} {finding.case_b}")
