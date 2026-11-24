import pytest
from rag_foundry_answering.claims import validate_citations

S1 = "Tolls are 5 USD per car. This does not apply to bicycles."
S2 = "The bridge opened on 3 March 2019."
PROVIDED = {"S1": S1, "S2": S2}
KNOWN = {"S1", "S2", "S3"}


def test_report_claims_and_status_order():
    answer = (
        "Tolls are 5 USD per car [S1]. The bridge opened in 2019 [S2]. "
        "Bicycles pay 7 USD [S1]. It has a cafe. The cafe sells tea [S3]. Hours are long [S9]."
    )
    report = validate_citations(answer, provided=PROVIDED, known=KNOWN)
    assert len(report.claims) == 6
    assert [claim.status for claim in report.claims] == [
        "supported_lexical",
        "supported_lexical",
        "unsupported_lexical",
        "uncited",
        "invalid_citation",
        "invalid_citation",
    ]
    assert (report.supported, report.unsupported, report.uncited, report.invalid, report.ok) == (2, 1, 1, 2, False)
    assert (report.claims[0].start, report.claims[0].end) == (0, 29)
    assert report.claims[0].overlap == 1.0
    assert [(flag.code, flag.value) for flag in report.claims[2].flags] == [
        ("constraint_missing", "7"),
        ("constraint_missing", "7 usd"),
    ]
    assert report.claims[2].overlap == pytest.approx(2 / 3)
    assert [(flag.code, flag.value) for flag in report.claims[4].flags] == [("not_provided", "S3")]
    assert [(flag.code, flag.value) for flag in report.claims[5].flags] == [("unknown_marker", "S9")]


def test_marker_only_segment_merges_with_previous_claim():
    report = validate_citations("Tolls are 5 USD per car. [S1]", provided=PROVIDED, known=KNOWN)
    assert len(report.claims) == 1
    claim = report.claims[0]
    assert (claim.start, claim.end) == (0, 29)
    assert claim.markers == ("S1",)
    assert claim.status == "supported_lexical"


def test_low_overlap():
    report = validate_citations("Ferries are free [S2].", provided=PROVIDED, known=KNOWN)
    claim = report.claims[0]
    assert claim.status == "unsupported_lexical"
    assert [(flag.code, flag.value) for flag in claim.flags] == [("low_overlap", "0.00")]


def test_lexical_rule_drops_negation():
    # This intentional losing case demonstrates the limits of lexical matching.
    claim = validate_citations("Tolls apply to bicycles [S1].", provided=PROVIDED, known=KNOWN).claims[0]
    assert claim.status == "supported_lexical"
    assert claim.overlap == 1.0


def test_arabic_constraints_and_digit_normalization():
    arabic_s1 = (
        "\u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631 \u0665 "
        "\u062f\u0648\u0644\u0627\u0631. \u0644\u0627 \u062a\u0646\u0637\u0628\u0642 "
        "\u0627\u0644\u0631\u0633\u0648\u0645 \u0639\u0644\u0649 \u0627\u0644\u062f\u0631\u0627\u062c\u0627\u062a."
    )
    provided = {"S1": arabic_s1}
    ascii_five = "\u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631 5 \u062f\u0648\u0644\u0627\u0631 [S1]."
    seven = "\u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631 \u0667 \u062f\u0648\u0644\u0627\u0631 [S1]."
    good = validate_citations(ascii_five, provided=provided, known={"S1"}).claims[0]
    assert good.status == "supported_lexical"
    bad = validate_citations(seven, provided=provided, known={"S1"}).claims[0]
    assert bad.status == "unsupported_lexical"
    assert [(flag.code, flag.value) for flag in bad.flags] == [
        ("constraint_missing", "7"),
        ("constraint_missing", "7 \u062f\u0648\u0644\u0627\u0631"),
    ]


def test_marker_group():
    claim = validate_citations("Tolls are 5 USD per car [S1, S2].", provided=PROVIDED, known=KNOWN).claims[0]
    assert claim.markers == ("S1", "S2")
    assert claim.status == "supported_lexical"


def test_edge_cases_and_determinism():
    with pytest.raises(ValueError):
        validate_citations("x [S1]", provided=PROVIDED, known=KNOWN, min_overlap=0)
    assert validate_citations("", provided=PROVIDED, known=KNOWN).claims == ()
    assert not validate_citations("", provided=PROVIDED, known=KNOWN).ok
    first = validate_citations("Tolls are 5 USD per car [S1].", provided=PROVIDED, known=KNOWN)
    second = validate_citations("Tolls are 5 USD per car [S1].", provided=PROVIDED, known=KNOWN)
    assert first.model_dump_json() == second.model_dump_json()


def test_leading_marker_only_segment_goes_to_next_claim():
    answer = "[S1]" + chr(10) * 2 + "Tolls are 5 USD per car."
    report = validate_citations(answer, provided=PROVIDED, known=KNOWN)
    assert len(report.claims) == 1
    claim = report.claims[0]
    assert (claim.start, claim.end) == (0, len(answer))
    assert claim.markers == ("S1",)
    assert claim.text == "Tolls are 5 USD per car."
    assert claim.status == "supported_lexical"


def test_marker_only_answer_is_a_known_lexical_limit():
    # Known limit of lexical-v1: a claim with no text has no constraints and no terms,
    # so nothing flags it and it counts as supported. P6.3-04 decides how to treat it.
    report = validate_citations("[S1]", provided=PROVIDED, known=KNOWN)
    assert len(report.claims) == 1
    assert report.claims[0].text == ""
    assert report.claims[0].overlap is None
    assert report.claims[0].status == "supported_lexical"
    assert report.ok is True


def test_constraints_do_not_join_across_cited_sources():
    provided = {"S1": "Tolls rose by 5", "S2": "USD fees apply to tolls."}
    report = validate_citations("Tolls are 5 USD [S1, S2].", provided=provided, known={"S1", "S2"})
    assert ("constraint_missing", "5 usd") in [(flag.code, flag.value) for flag in report.claims[0].flags]
    assert report.claims[0].status == "unsupported_lexical"
