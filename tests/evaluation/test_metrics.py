import pytest
from rag_foundry_evaluation import EvalCase, EvalHit, RelevantSpan, aggregate, case_metrics
from rag_foundry_evaluation.errors import EvaluationError


def make_case(case_id="a", relevant=None, answerable=True, tags=()):
    relevant = (
        relevant
        if relevant is not None
        else (
            RelevantSpan(doc_key="d", start=0, end=10, grade=2, quote="0123456789"),
            RelevantSpan(doc_key="d", start=20, end=30, grade=1, quote="0123456789"),
        )
    )
    return EvalCase(
        case_id=case_id,
        question="question",
        language="en",
        tags=tags,
        split="dev",
        answerable=answerable,
        relevant=relevant,
    )


def hit(rank, doc="d", start=0, end=10, score=1.0):
    return EvalHit(doc_key=doc, start=start, end=end, rank=rank, score=score, text_sha256="a" * 64)


def test_case_a_hand_computed_metrics():
    case = make_case()
    result = case_metrics(
        case,
        [
            hit(1, "other"),
            hit(2, start=0, end=5),
            hit(3, start=5, end=12),
            hit(4, start=20, end=30),
            hit(5, start=40, end=50),
        ],
    )
    assert result.found_ranks == (3, 4)
    assert result.hit_rate == {1: 0.0, 3: 1.0, 5: 1.0}
    assert result.recall == {1: 0.0, 3: 0.5, 5: 1.0}
    assert result.rr == pytest.approx(1 / 3, abs=1e-6)
    assert result.ndcg[1] == pytest.approx(0, abs=1e-6)
    assert result.ndcg[3] == pytest.approx(0.413117, abs=1e-6)
    assert result.ndcg[5] == pytest.approx(0.531731, abs=1e-6)
    assert result.hit_rate[3] == 1 and result.recall[3] == 0.5


def test_one_chunk_highest_grade_gain_and_recall():
    result = case_metrics(make_case(), [hit(1, start=0, end=30)])
    assert result.recall[1] == 1.0
    assert result.rr == 1.0
    assert result.ndcg[1] == pytest.approx(1.0, abs=1e-6)
    assert result.ndcg[3] == pytest.approx(0.826235, abs=1e-6)
    assert result.ndcg[5] == pytest.approx(0.826235, abs=1e-6)


def test_coverage_threshold_boundary():
    span = (RelevantSpan(doc_key="d", start=0, end=10, grade=1, quote="0123456789"),)
    assert case_metrics(make_case(relevant=span), [hit(1, start=0, end=8)]).found_ranks == (1,)
    assert case_metrics(make_case(relevant=span), [hit(1, start=0, end=7)]).found_ranks == (None,)


def test_overlapping_hits_are_counted_once():
    span = (RelevantSpan(doc_key="d", start=0, end=10, grade=1, quote="0123456789"),)
    # Union of [0,5) and [2,7) covers 7 of 10 code points: below 0.8, so not found.
    hits = [hit(1, start=0, end=5), hit(2, start=2, end=7)]
    assert case_metrics(make_case(relevant=span), hits).found_ranks == (None,)


def test_unanswerable_is_excluded_from_metric_denominators():
    case = make_case("u", relevant=(), answerable=False)
    result = case_metrics(case, [hit(1), hit(2, start=10, end=15, score=0.5)])
    assert result.hit_rate is result.recall is result.ndcg is result.rr is None
    assert result.returned_hits == 2 and result.top1_score == 1.0


def test_aggregate_rounding_cohorts_and_errors():
    a = make_case("a", tags=("identifier",))
    b = make_case("b")
    u = make_case("u", relevant=(), answerable=False, tags=("ocr",))
    ma = case_metrics(a, [hit(1, "other"), hit(2, start=0, end=5), hit(3, start=5, end=12), hit(4, start=20, end=30)])
    mb = case_metrics(b, [hit(1, start=0, end=30)])
    mu = case_metrics(u, [hit(1)])
    rows = [(a, ma, False), (b, mb, True), (u, mu, False)]
    summaries = aggregate(rows, (1, 3, 5))
    all_rows = next(row for row in summaries if row.group == "all")
    assert all_rows.n_answerable == 2 and all_rows.n_unanswerable == 1
    assert all_rows.n_errors == 1
    assert all_rows.mrr == pytest.approx(0.666667, abs=1e-6)
    assert all_rows.hit_rate[1] == pytest.approx(0.5, abs=1e-6)
    assert all_rows.recall[3] == pytest.approx(0.75, abs=1e-6)
    assert all_rows.ndcg[3] == pytest.approx(0.619676, abs=1e-6)
    assert all_rows.ndcg[5] == pytest.approx(0.678983, abs=1e-6)
    table = next(row for row in summaries if row.group == "tag:table")
    assert table.n_cases == 0 and table.mrr is None
    assert {row.group for row in summaries} == {
        "all",
        "language:en",
        "language:ar",
        "language:mixed",
        "tag:identifier",
        "tag:paraphrase",
        "tag:diacritics",
        "tag:table",
        "tag:ocr",
        "unanswerable",
    }


def test_non_contiguous_ranks_fail():
    with pytest.raises(EvaluationError, match="invalid_ranking"):
        case_metrics(make_case(), [hit(1), hit(3)])
