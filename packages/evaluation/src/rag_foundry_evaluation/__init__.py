from .dataset import (
    EvalCase,
    EvalDataset,
    EvalDocument,
    RelevantSpan,
    canonical_json,
    dataset_sha256,
    dump_dataset,
    load_dataset,
)
from .errors import EvaluationError
from .leakage import (
    LeakageFinding,
    LeakageReport,
    check_leakage,
    jaccard,
    normalize_question,
    question_tokens,
    raise_on_errors,
)
from .metrics import CaseMetrics, EvalHit, GroupSummary, aggregate, case_metrics

__all__ = [
    "EvalCase",
    "EvalDataset",
    "EvalDocument",
    "EvalHit",
    "CaseMetrics",
    "EvaluationError",
    "LeakageFinding",
    "LeakageReport",
    "RelevantSpan",
    "GroupSummary",
    "aggregate",
    "case_metrics",
    "canonical_json",
    "check_leakage",
    "dataset_sha256",
    "dump_dataset",
    "jaccard",
    "load_dataset",
    "normalize_question",
    "question_tokens",
    "raise_on_errors",
]
