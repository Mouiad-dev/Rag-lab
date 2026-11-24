import pytest
from rag_foundry_evaluation import (
    EvalCase,
    EvalDataset,
    EvalDocument,
    EvalHit,
    RelevantSpan,
    aggregate_language_pairs,
    case_metrics,
    language_pair,
    language_pair_summaries,
    render_language_pairs,
    script_language,
)

from tests.indexing._paired_fixture import FACT_AR, FACT_EN, ID1, MIXED

B = "\u0628"
EN_QUOTE = "Port Sallow"
AR_QUOTE = "\u0645\u0643\u062a\u0628\u0629"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (FACT_EN, "en"),
        (FACT_AR, "ar"),
        (ID1, "ar"),
        (MIXED, "mixed"),
        ("12:30", "unknown"),
        ("\u0640" * 5 + "abc", "en"),
        (B * 4 + " x", "ar"),
        (B + " abcd", "en"),
        (B * 3 + " ab", "mixed"),
    ],
)
def test_script_language(text, expected):
    assert script_language(text) == expected


def span(quote, start):
    return RelevantSpan(doc_key="d", start=start, end=start + len(quote), grade=2, quote=quote)


def make_case(case_id, spans):
    return EvalCase(
        case_id=case_id,
        question="question",
        language="en",
        split="dev",
        answerable=bool(spans),
        relevant=tuple(spans),
    )


def hit(rank, start, end):
    return EvalHit(doc_key="d", start=start, end=end, rank=rank, score=1.0, text_sha256="a" * 64)


def test_language_pair_values():
    assert language_pair(make_case("u", [])) == "en->none"
    assert language_pair(make_case("m", [span(EN_QUOTE, 0), span(AR_QUOTE, 20)])) == "en->mixed"


C1 = make_case("c1", [span(EN_QUOTE, 0)])
C2 = make_case("c2", [span(AR_QUOTE, 20)])
C3 = make_case("c3", [span(AR_QUOTE, 20)])
U = make_case("u", [])
HITS = {
    "c1": [(0, 11)],
    "c2": [(0, 11)],
    "c3": [(0, 11), (30, 35), (20, 25)],
    "u": [],
}
KS = (1, 3)


def hits_for(case_id):
    return tuple(hit(i, s, e) for i, (s, e) in enumerate(HITS[case_id], 1))


def test_aggregate_language_pairs_hand_computed():
    rows = [(c, case_metrics(c, hits_for(c.case_id), ks=KS), False) for c in (C1, C2, C3, U)]
    out = aggregate_language_pairs(rows, KS)
    assert [s.group for s in out] == ["pair:en->ar", "pair:en->en", "pair:en->none"]
    ar, en, none = out
    assert ar.n_cases == 2
    assert ar.hit_rate == {1: 0.0, 3: 0.5}
    assert ar.recall == {1: 0.0, 3: 0.5}
    assert ar.mrr == 0.166667
    assert en.mrr == 1.0
    assert none.n_unanswerable == 1
    assert none.hit_rate is None and none.recall is None and none.ndcg is None and none.mrr is None


def test_aggregate_language_pairs_errors_and_split_order():
    held = C1.model_copy(update={"case_id": "h", "split": "held_out"})
    rows = [
        (held, case_metrics(held, hits_for("c1"), ks=KS), False),
        (C1, case_metrics(C1, (), ks=KS), True),
    ]
    out = aggregate_language_pairs(rows, KS)
    assert [(s.split, s.group) for s in out] == [("dev", "pair:en->en"), ("held_out", "pair:en->en")]
    assert out[0].n_errors == 1
    assert out[1].n_errors == 0


def test_aggregate_language_pairs_rejects_bad_ks():
    with pytest.raises(ValueError):
        aggregate_language_pairs([], (0,))


def build():
    dataset = EvalDataset(
        format="rag-foundry.eval-dataset",
        format_version="1",
        dataset_id="t",
        dataset_version="1",
        description="d",
        documents=(EvalDocument(doc_key="d", text_sha256="0" * 64, length=40),),
        cases=(C1, C2, C3, U),
    )
    results = {
        "config": {"ks": [1, 3], "coverage_threshold": 0.8, "techniques": [{"label": "L"}]},
        "cases": [
            {
                "label": "L",
                "case_id": c.case_id,
                "status": "ok",
                "hits": [h.model_dump(mode="json") for h in hits_for(c.case_id)],
            }
            for c in (C1, C2, C3, U)
        ],
    }
    return results, dataset


def test_render_language_pairs():
    results, dataset = build()
    assert set(language_pair_summaries(results, dataset)) == {"L"}
    text = render_language_pairs(results, dataset)
    assert text.startswith("## Language pairs\n")
    assert "### L\n" in text
    assert "| dev | en->ar | 2 | 2 | 0 | 0 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.1667 |" in text.splitlines()
    assert "### Per case" in text
    assert render_language_pairs(results, dataset) == text
