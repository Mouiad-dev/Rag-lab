import json

import pytest
from rag_foundry_cache.features import QuestionFeatures, eligible, features
from rag_foundry_domain.errors import CacheError


def ar(*codes: str) -> str:
    """Build an Arabic word from hex code points (keeps this file ASCII)."""
    return "".join(chr(int(c, 16)) for c in codes)


LA = ar("0644", "0627")
QABL = ar("0642", "0628", "0644")
BAAD = ar("0628", "0639", "062F")
WHAT_YEAR = ar("0645", "0627", "0630", "0627")
ARABIC_INDIC_2000 = ar("0662", "0660", "0660", "0660")
EXT_INDIC_2000 = ar("06F2", "06F0", "06F0", "06F0")


def diff(a: str, b: str) -> tuple[bool, tuple[str, ...]]:
    return eligible(features(a), features(b))


def test_word_order_paraphrase_is_eligible() -> None:
    assert diff("What is the refund policy for premium plans?", "For premium plans, what is the refund policy?") == (
        True,
        (),
    )


def test_same_number_in_ascii_and_arabic_indic_digits() -> None:
    assert diff("Revenue in 2025", "Revenue in " + ar("0662", "0660", "0662", "0665"))[0]
    assert features(ARABIC_INDIC_2000).numbers == ("2000",)
    assert features(EXT_INDIC_2000).numbers == ("2000",)


def test_thousands_separator_and_decimals_canonical() -> None:
    values = {features(q).numbers for q in ("2,000", "2000", ARABIC_INDIC_2000, "2000.0", "02000")}
    assert values == {("2000",)}
    assert features("3.50 and 3.5").numbers == ("3.5", "3.5")
    assert features("1,5").numbers != features("15").numbers


def test_numbers_are_a_multiset() -> None:
    assert features("2 and 2").numbers == ("2", "2")
    assert not diff("pay 2 and 2", "pay 2")[0]


@pytest.mark.parametrize(
    ("a", "b", "name"),
    [
        ("Revenue in 2025", "Revenue in 2026", "numbers"),
        ("Plans with support", "Plans without support", "negation"),
        ("Is the plan active?", "Isn't the plan active?", "negation"),
        (
            WHAT_YEAR + " " + ar("0646", "0639", "0645"),
            WHAT_YEAR + " " + LA + " " + ar("0646", "0639", "0645"),
            "negation",
        ),
        ("Events before the merger", "Events after the merger", "comparatives"),
        (
            QABL + " " + ar("0627", "0644", "0639", "0642", "062F"),
            BAAD + " " + ar("0627", "0644", "0639", "0642", "062F"),
            "comparatives",
        ),
        ("Invoice INV-2041", "Invoice INV-2014", "identifiers"),
        ("Invoice INV-2041", "Invoice inv-2041", "identifiers"),
        ('Define "blue"', 'Define "red"', "quoted"),
        ("Define \u201cBlue\u201d", "Define \u201cblue\u201d", "quoted"),
        ("refund policy", "refund " + ar("0633", "064A", "0627", "0633", "0629"), "script"),
        ("at least 5 items", "at most 5 items", "comparatives"),
    ],
)
def test_not_eligible_names_the_feature(a: str, b: str, name: str) -> None:
    ok, names = diff(a, b)
    assert not ok
    assert name in names


def test_arabic_negation_with_prefix_and_without_nt_rule() -> None:
    assert features(ar("0648", "0644", "0645") + " " + "x").negation == 1
    assert features(ar("0641", "0644", "064A", "0633")).negation == 1
    assert features("They don't and can't").negation == 2
    assert features("Do not, never, nor none").negation == 4


def test_script_values() -> None:
    assert features("hello").script == "latin"
    assert features(LA).script == "arabic"
    assert features("hello " + LA).script == "mixed"
    assert features("12345").script == "other"


def test_documented_limits_stay_eligible() -> None:
    # Known false-hit risks (module docstring): months and units are not features.
    assert diff("Sales in March", "Sales in May")[0]
    assert diff("Distance of 10 km", "Distance of 10 miles")[0]


def test_deterministic_and_whitespace_only() -> None:
    q = 'Is "A b" INV-1 before 2,000 not?'
    assert features(q) == features(q)
    assert features(q).to_json() == features(q).to_json()
    empty = features("   \n\t ")
    assert empty == QuestionFeatures(numbers=(), negation=0, comparatives=(), identifiers=(), quoted=(), script="other")


def test_json_round_trip_and_eligibility_agree() -> None:
    a = features('Is "Blue Sky" INV-2041 not before 2,000 for R2D2?')
    b = features('Is "Blue Sky" INV-2041 not before 2000 for R2D2?')
    c = features('Is "Blue Sky" INV-2014 not before 2000 for R2D2?')
    back = QuestionFeatures.from_json(a.to_json())
    assert back.to_json() == a.to_json()
    assert eligible(a, b) == eligible(back, b) == eligible(back, QuestionFeatures.from_json(b.to_json()))
    assert eligible(a, c) == eligible(back, c) == (False, ("identifiers",))
    assert back.negation == 1
    assert back.comparatives == ("before",)


def test_to_json_stores_no_question_text() -> None:
    text = features('Is "Blue Sky" INV-2041 before 31337 and 3.14159?').to_json()
    for raw in ("Blue Sky", "INV-2041", "2041", "31337", "3.14159", "14159"):
        assert raw not in text
    data = json.loads(text)
    assert all(len(h) == 64 for k in ("numbers", "identifiers", "quoted") for h in data[k])
    assert data["numbers"] == sorted(data["numbers"])
    assert data["comparatives"] == ["before"]
    assert text.isascii()


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[]",
        "{}",
        json.dumps(
            {"numbers": ["2025"], "negation": 0, "comparatives": [], "identifiers": [], "quoted": [], "script": "latin"}
        ),
        json.dumps(
            {"numbers": [], "negation": True, "comparatives": [], "identifiers": [], "quoted": [], "script": "latin"}
        ),
        json.dumps(
            {"numbers": [], "negation": -1, "comparatives": [], "identifiers": [], "quoted": [], "script": "latin"}
        ),
        json.dumps(
            {
                "numbers": [],
                "negation": 0,
                "comparatives": ["bogus"],
                "identifiers": [],
                "quoted": [],
                "script": "latin",
            }
        ),
        json.dumps(
            {"numbers": [], "negation": 0, "comparatives": [], "identifiers": [], "quoted": [], "script": "greek"}
        ),
        json.dumps(
            {
                "numbers": [],
                "negation": 0,
                "comparatives": [],
                "identifiers": [],
                "quoted": [],
                "script": "latin",
                "extra": 1,
            }
        ),
    ],
)
def test_from_json_rejects_invalid(bad: str) -> None:
    with pytest.raises(CacheError) as info:
        QuestionFeatures.from_json(bad)
    assert str(info.value) == "cache_entry_invalid" or "cache_entry_invalid" in str(info.value.args)


def test_minus_sign_is_a_number_feature() -> None:
    ok, names = diff("Temperature -5 C", "Temperature 5 C")
    assert not ok
    assert "numbers" in names
    assert features("-5").numbers == ("-5",)
    assert features("\N{MINUS SIGN}5").numbers == ("-5",)
    assert features("-0").numbers == ("0",)
    assert features("range 2025-2026").numbers == ("2025", "2026")
    assert features("(-1,500.50)").numbers == ("-1500.5",)


def test_quoted_apostrophe_negation_is_detected() -> None:
    assert not diff("The answer is 'no'", "The answer is")[0]
    assert features("The answer is 'not'").negation == 1
    assert features("They don't, 'don't'").negation == 2


def test_cannot_is_negation() -> None:
    ok, names = diff("I cannot pay", "I can pay")
    assert not ok
    assert "negation" in names


def test_arabic_alef_variants_match_for_comparatives_only() -> None:
    plain_more = ar("0627", "0643", "062B", "0631")
    hamza_more = ar("0623", "0643", "062B", "0631")
    assert features(plain_more).comparatives == ("more",)
    assert features(hamza_more).comparatives == ("more",)
    assert features(ar("0625", "0642", "0644")).comparatives == ("less",)


def test_comparatives_are_sorted() -> None:
    f = QuestionFeatures(
        numbers=(), negation=0, comparatives=("less", "after"), identifiers=(), quoted=(), script="other"
    )
    assert f.comparatives == ("after", "less")
