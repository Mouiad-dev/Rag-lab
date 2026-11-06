import pytest
from rag_foundry_answering.citations import check_answer, parse_markers, strip_markers


def test_parse_markers_order_dedup_and_bounds():
    assert parse_markers("[S1][S2] [S1, S10] [S0] [s1]") == ("S1", "S2", "S10")
    assert strip_markers("Fact [S1, S2] and [S10].") == "Fact  and ."


@pytest.mark.parametrize(
    "text,finish,known,outcome,cited",
    [
        ("Fact [S1]", "stop", {"S1"}, "ok", ("S1",)),
        ("Fact [S1]", "length", {"S1"}, "truncated", ("S1",)),
        ("INSUFFICIENT_EVIDENCE [S1]", "stop", {"S1"}, "declined", ("S1",)),
        ("   ", "stop", set(), "uncited", ()),
        ("Fact", "stop", {"S1"}, "uncited", ()),
        ("[S1]", "stop", {"S1"}, "uncited", ("S1",)),
        ("Fact [S2]", "stop", {"S1"}, "invalid", ("S2",)),
        ("The passages do not state X [S1]", "stop", {"S1"}, "ok", ("S1",)),
    ],
)
def test_check_answer(text, finish, known, outcome, cited):
    checked = check_answer(text, finish, known)
    assert checked.outcome == outcome
    assert checked.cited == cited
