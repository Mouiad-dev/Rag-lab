import pytest
from rag_foundry_answering.citations import check_answer, normalize_nonce_markers, parse_markers, strip_markers


def test_parse_markers_order_dedup_and_bounds():
    assert parse_markers("[S1][S2] [S1, S10] [S0] [s1]") == ("S1", "S2", "S10")
    assert strip_markers("Fact [S1, S2] and [S10].") == "Fact  and ."


def test_normalize_nonce_markers():
    nonce = "0123456789abcdef"
    assert normalize_nonce_markers(f"A [S1 {nonce}].", nonce) == "A [S1]."
    assert normalize_nonce_markers(f"[S1 {nonce}] [S12 {nonce}] [S2 {nonce}]", nonce) == "[S1] [S12] [S2]"
    for text in (
        "[S1 fedcba9876543210]",
        "[S1 0123456789ABCDEF]",
        "[S1  0123456789abcdef]",
        "S1 0123456789abcdef]",
        "[S1 0123456789abcdef, S2]",
    ):
        assert normalize_nonce_markers(text, nonce) == text
    with pytest.raises(ValueError):
        normalize_nonce_markers("[S1 0123456789abcdef]", "0123456789ABCDEf")


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
