from __future__ import annotations

import pytest
from rag_foundry_evaluation import EvaluationError
from rag_foundry_evaluation.dataset import EvalCase, EvalDataset, EvalDocument, RelevantSpan
from rag_foundry_evaluation.leakage import (
    check_leakage,
    jaccard,
    normalize_question,
    question_tokens,
    raise_on_errors,
)


def make_dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(
        format="rag-foundry.eval-dataset",
        format_version="1",
        dataset_id="sample",
        dataset_version="1.0",
        description="",
        documents=(EvalDocument(doc_key="d", text_sha256="a" * 64, length=20),),
        cases=cases,
    )


def make_case(case_id: str, split: str, question: str, spans: tuple[RelevantSpan, ...] = ()) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        question=question,
        language="en",
        split=split,
        answerable=bool(spans),
        relevant=spans,
    )


def test_near_duplicate_8_of_10_is_error_with_exact_value() -> None:
    report = check_leakage(
        make_dataset(
            make_case("a", "dev", "one two three four five six seven eight nine"),
            make_case("b", "validation", "one two three four five six seven eight ten"),
        )
    )
    assert [(item.rule, item.value, item.severity) for item in report.findings] == [
        ("near_duplicate_question", 0.8, "error")
    ]
    assert report.errors == report.findings


def test_same_split_pair_is_not_checked() -> None:
    report = check_leakage(make_dataset(make_case("a", "dev", "one two"), make_case("b", "dev", "one two")))
    assert report.findings == ()
    assert report.pairs_checked == 0


def test_7_of_11_is_below_threshold() -> None:
    report = check_leakage(
        make_dataset(
            make_case("a", "dev", "one two three four five six seven eight nine"),
            make_case("b", "validation", "one two three four five six seven ten eleven"),
        )
    )
    similarity = jaccard(
        question_tokens("one two three four five six seven eight nine"),
        question_tokens("one two three four five six seven ten eleven"),
    )
    assert similarity == 7 / 11
    assert round(similarity, 6) == 0.636364
    assert report.findings == ()


def test_arabic_diacritics_normalize_to_exact_question() -> None:
    report = check_leakage(
        make_dataset(
            make_case("a", "dev", "\u0627\u0644\u0645\u062e\u0632\u0646"),
            make_case("b", "held_out", "\u0627\u0644\u0645\u064e\u062e\u0652\u0632\u064e\u0646"),
        )
    )
    assert normalize_question("\u0627\u0644\u0645\u062e\u0632\u0646") == normalize_question(
        "\u0627\u0644\u0645\u064e\u062e\u0652\u0632\u064e\u0646"
    )
    assert [(item.rule, item.value, item.severity) for item in report.findings] == [("exact_question", None, "error")]


def test_punctuation_keeps_question_from_being_exact_but_tokens_match() -> None:
    report = check_leakage(make_dataset(make_case("a", "dev", "What Is X?"), make_case("b", "validation", "what is x")))
    assert [(item.rule, item.value) for item in report.findings] == [("near_duplicate_question", 1.0)]


def test_overlapping_relevant_spans_warn_but_touching_spans_do_not() -> None:
    span_a = RelevantSpan(doc_key="d", start=0, end=10, grade=1, quote="a" * 10)
    overlap = RelevantSpan(doc_key="d", start=5, end=20, grade=1, quote="b" * 15)
    touching = RelevantSpan(doc_key="d", start=10, end=20, grade=1, quote="c" * 10)
    warning = check_leakage(
        make_dataset(
            make_case("a", "dev", "first", (span_a,)),
            make_case("b", "validation", "second", (overlap,)),
        )
    )
    clear = check_leakage(
        make_dataset(
            make_case("a", "dev", "first", (span_a,)),
            make_case("b", "validation", "second", (touching,)),
        )
    )
    assert [(item.rule, item.severity, item.value) for item in warning.findings] == [
        ("shared_relevant_span", "warning", None)
    ]
    assert clear.findings == ()


def test_pairs_checked_counts_all_cross_split_pairs() -> None:
    report = check_leakage(
        make_dataset(
            make_case("a", "dev", "alpha"),
            make_case("b", "dev", "beta"),
            make_case("c", "validation", "gamma"),
            make_case("d", "held_out", "delta"),
        )
    )
    assert report.pairs_checked == 5


def test_raise_on_errors_and_warning_only_report() -> None:
    error_report = check_leakage(make_dataset(make_case("a", "dev", "same"), make_case("b", "validation", "same")))
    with pytest.raises(EvaluationError) as exc:
        raise_on_errors(error_report)
    assert exc.value.code == "leakage"
    assert exc.value.detail == "exact_question a b"

    span_a = RelevantSpan(doc_key="d", start=0, end=10, grade=1, quote="a" * 10)
    span_b = RelevantSpan(doc_key="d", start=5, end=15, grade=1, quote="b" * 10)
    warning_report = check_leakage(
        make_dataset(
            make_case("a", "dev", "first", (span_a,)),
            make_case("b", "validation", "second", (span_b,)),
        )
    )
    assert raise_on_errors(warning_report) is None


@pytest.mark.parametrize("threshold", [0, 1.5])
def test_invalid_threshold_has_specific_reason(threshold: float) -> None:
    with pytest.raises(EvaluationError) as exc:
        check_leakage(make_dataset(make_case("a", "dev", "alpha")), jaccard_threshold=threshold)
    assert exc.value.code == "invalid_threshold"
    assert exc.value.detail is None


def test_tatweel_is_removed_by_normalization() -> None:
    assert normalize_question("\u0645\u0640\u062e\u0632\u0646") == normalize_question("\u0645\u062e\u0632\u0646")
