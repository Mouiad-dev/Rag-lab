import random
import re
import unicodedata
from pathlib import Path

import pytest
from rag_foundry_domain.offsets import ATOMIC, ATTACH, CP, MappedRange, OffsetMap
from rag_foundry_ingestion import normalize as nz
from rag_foundry_ingestion.normalize import CANONICAL_V1, SEARCH_AR_EN_V1, apply

FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "contracts" / "fixtures" / "sources"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("utf-8")


def check_map(src: str, out: str, m: OffsetMap) -> None:
    assert (m.n_len, m.o_len) == (len(out), len(src))
    assert OffsetMap.from_json(m.to_json()) == m


# -- single steps -----------------------------------------------------------------------------------------------


def test_newlines():
    out, m = nz.newlines("a\r\nb\rc\nd")
    assert out == "a\nb\nc\nd"
    assert m.to_json() == [[0, 1, 0, 1, CP], [1, 2, 1, 3, ATOMIC], [2, 7, 3, 8, CP]]
    assert nz.newlines("plain\n")[1].is_identity
    assert nz.newlines.lossy is True


def test_nfc_reorders_marks_as_one_atomic_segment():
    out, m = nz.nfc("\u0651\u064e")
    assert out == "\u064e\u0651"
    assert m.to_json() == [[0, 2, 0, 2, ATOMIC]]
    assert nz.nfc.lossy is False


def test_nfc_hangul_jamo_compose():
    out, m = nz.nfc("\u1100\u1161\u11a8")
    assert out == "\uac01"
    assert m.to_json() == [[0, 1, 0, 3, ATOMIC]]


def test_nfc_alef_madda_and_singleton():
    out, m = nz.nfc("\u0627\u0653")
    assert out == "\u0622"
    assert m.to_json() == [[0, 1, 0, 2, ATOMIC]]
    out, m = nz.nfc("\u212b")
    assert out == "\u00c5"
    assert m.to_json() == [[0, 1, 0, 1, CP]]  # one code point becomes one code point


@pytest.mark.parametrize("text", ["\U0001f468\u200d\U0001f469\u200d\U0001f467", "\U0001f4d8", "abc", ""])
def test_nfc_leaves_these_unchanged(text):
    out, m = nz.nfc(text)
    assert out == text
    assert m.is_identity


def test_nfkc_presentation_forms_and_ligatures():
    out, m = nz.nfkc("x\ufb01y\ufe8d")
    assert out == "xfiy\u0627"
    assert m.to_json() == [[0, 1, 0, 1, CP], [1, 3, 1, 2, ATOMIC], [3, 5, 2, 4, CP]]
    assert nz.nfkc.lossy is True


def test_casefold_expands_atomically():
    out, m = nz.casefold("Stra\u00dfe")
    assert out == "strasse"
    assert m.to_json() == [[0, 4, 0, 4, CP], [4, 6, 4, 5, ATOMIC], [6, 7, 5, 6, CP]]
    out, m = nz.casefold("ABC")
    assert out == "abc" and m.is_identity


def test_strip_steps_use_expected_kinds():
    out, m = nz.strip_format("a\u200db\u200b\u200cc")
    assert out == "abc"
    assert m.to_json() == [
        [0, 1, 0, 1, CP],
        [1, 1, 1, 2, ATOMIC],
        [1, 2, 2, 3, CP],
        [2, 2, 3, 5, ATOMIC],
        [2, 3, 5, 6, CP],
    ]
    assert nz.strip_format("a\u034fb")[0] == "ab"
    out, m = nz.strip_tatweel("a\u0640\u0640b")
    assert out == "ab"
    assert m.to_json() == [[0, 1, 0, 1, CP], [1, 1, 1, 3, ATTACH], [1, 2, 3, 4, CP]]
    out, m = nz.strip_harakat("\u0643\u064e\u0651\u0670\u062a")
    assert out == "\u0643\u062a"
    assert m.to_json() == [[0, 1, 0, 1, CP], [1, 1, 1, 4, ATTACH], [1, 2, 4, 5, CP]]


def test_folds_are_one_to_one():
    for step, src, dst in [
        (nz.fold_alef, "\u0622\u0623\u0625\u0671\u0627", "\u0627" * 5),
        (nz.fold_yeh, "\u0649\u06cc\u064a", "\u064a" * 3),
        (nz.fold_kaf, "\u06a9\u0643", "\u0643" * 2),
        (nz.fold_digits, "\u0660\u0669\u06f0\u06f9", "0909"),
    ]:
        out, m = step(src)
        assert out == dst
        assert m.is_identity
        assert step.lossy is True
    assert nz.fold_alef("\u0624\u0626\u0629")[0] == "\u0624\u0626\u0629"  # not folded (ADR-0010 D7)


def test_whitespace_collapse_and_trim():
    out, m = nz.whitespace("a  b")
    assert out == "a b"
    assert m.to_json() == [[0, 1, 0, 1, CP], [1, 2, 1, 3, ATOMIC], [2, 3, 3, 4, CP]]
    out, m = nz.whitespace("  ab \t")
    assert out == "ab"
    assert m.to_json() == [[0, 0, 0, 2, ATOMIC], [0, 2, 2, 4, CP], [2, 2, 4, 6, ATOMIC]]
    out, m = nz.whitespace("a\u00a0b\u2003c")
    assert out == "a b c" and m.is_identity
    assert nz.whitespace("   ")[0] == ""


def test_whitespace_pattern_equals_str_isspace():
    pattern = re.compile(r"\s")
    for c in range(0x110000):
        assert bool(pattern.fullmatch(chr(c))) == chr(c).isspace()


# -- profiles ---------------------------------------------------------------------------------------------------


def test_profiles_are_step_lists_with_ids():
    assert [s.__name__ for s in CANONICAL_V1] == ["newlines", "nfc"]
    assert [s.__name__ for s in SEARCH_AR_EN_V1] == [
        "nfkc", "casefold", "nfkc", "strip_format", "strip_tatweel", "strip_harakat",
        "fold_alef", "fold_yeh", "fold_kaf", "fold_digits", "whitespace",
    ]  # fmt: skip
    r = apply(CANONICAL_V1, "a\r\nb")
    assert (r.text, r.profile_id, r.lossy_steps) == ("a\nb", "canonical-v1", ("newlines",))
    assert r.unicode_version == unicodedata.unidata_version
    s = apply(SEARCH_AR_EN_V1, "A")
    assert s.profile_id == "search-ar-en-v1"
    assert "nfc" not in s.lossy_steps and "casefold" in s.lossy_steps and "whitespace" in s.lossy_steps


def test_apply_empty_text():
    r = apply(SEARCH_AR_EN_V1, "")
    assert r.text == "" and r.map.n_len == 0 and r.map.o_len == 0


# -- fuzz -------------------------------------------------------------------------------------------------------

_ASSIGNED = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) not in ("Cn", "Cs", "Co")]
_MARKS = [c for c in _ASSIGNED if unicodedata.category(c).startswith("M")]
_JAMO = (
    [chr(c) for c in range(0x1100, 0x1113)]
    + [chr(c) for c in range(0x1161, 0x1176)]
    + [chr(c) for c in range(0x11A8, 0x11C3)]
    + [chr(c) for c in range(0x3131, 0x318F)]  # compatibility jamo (NFKC maps them to conjoining jamo)
    + [chr(c) for c in range(0xFFA0, 0xFFDD)]  # halfwidth compatibility jamo
)
_BIASED = _MARKS + _JAMO + [chr(c) for c in range(0x41, 0x5B)] + [chr(c) for c in range(0x0621, 0x064B)]


def random_text(rng: random.Random) -> str:
    n = rng.randint(1, 12)
    return "".join(rng.choice(_BIASED) if rng.random() < 0.6 else rng.choice(_ASSIGNED) for _ in range(n))


@pytest.mark.parametrize("form", ["NFC", "NFKC"])
def test_boundary_invariants_over_all_code_points(form):
    second = nz._second_of_composite()
    for cp in range(0x110000):
        c = chr(cp)
        if nz._is_boundary(form, c):
            r = unicodedata.normalize(form, c)
            assert r and unicodedata.combining(r[0]) == 0, hex(cp)
            assert r[0] not in second, hex(cp)
            assert unicodedata.combining(c) == 0 and c not in second, hex(cp)


def test_nfkc_compatibility_jamo_compose():
    out, m = nz.nfkc("\u3131\u314f")
    assert out == "\uac00"
    assert m.to_json() == [[0, 1, 0, 2, ATOMIC]]
    assert nz.nfkc("\u1100\u314f")[0] == "\uac00"
    assert nz.nfkc("\u1100\u1fa2")[0] == unicodedata.normalize("NFKC", "\u1100\u1fa2")


@pytest.mark.parametrize(("form", "step"), [("NFC", nz.nfc), ("NFKC", nz.nfkc)])
def test_grouping_matches_unicodedata(form, step):
    rng = random.Random(1005)
    for _ in range(4000):
        s = random_text(rng)
        out, m = step(s)  # the internal guard raises NormalizationError if grouping were wrong
        assert out == unicodedata.normalize(form, s)
        check_map(s, out, m)


# Realistic text for the search profile: Arabic block, ASCII, Latin-1, presentation forms, whitespace, ligatures.
# Idempotence cannot hold for arbitrary code points: deleting a format character or tatweel can bring a base and a
# combining mark together that the second NFKC pass then composes or reorders (see test_search_profile_known_gap).
_REALISTIC = (
    [chr(c) for c in range(0x0600, 0x0700) if unicodedata.category(chr(c)) != "Cn"]
    + [chr(c) for c in range(0x20, 0x7F)]
    + [chr(c) for c in range(0xC0, 0x100)]
    + [chr(c) for c in range(0xFE70, 0xFEFD)]
    + ["\n", "\r", "\t", "\u00a0", "\ufb01", "\u1e9e", "\U0001f4d8"]
)


@pytest.mark.parametrize(
    ("profile", "alphabet"),
    [(CANONICAL_V1, _ASSIGNED), (SEARCH_AR_EN_V1, _REALISTIC)],
    ids=["canonical-v1", "search-ar-en-v1"],
)
def test_profiles_are_idempotent(profile, alphabet):
    rng = random.Random(2005)
    for _ in range(1500):
        pool = _BIASED if alphabet is _ASSIGNED and rng.random() < 0.6 else alphabet
        s = "".join(rng.choice(pool) for _ in range(rng.randint(1, 12)))
        once = apply(profile, s)
        twice = apply(profile, once.text)
        assert twice.text == once.text
        check_map(s, once.text, once.map)


def test_search_profile_known_gap():
    # Deleting U+200D leaves a base next to a mark that the second pass composes; search text is derived on demand
    # and never stored, so ADR-0010 does not require idempotence on arbitrary input.
    once = apply(SEARCH_AR_EN_V1, "e\u200d\u0301")
    assert once.text == "e\u0301"
    assert apply(SEARCH_AR_EN_V1, once.text).text == "\u00e9"


# -- done criterion ---------------------------------------------------------------------------------------------

CRAFTED = [
    "\u0643\u0640\u064e\u062a\u064e\u0628\u064f x",
    "ab\r\ncd\ref\n",
    "a\u00a0b\u00a0\u00a0c",
    "\u0663\u0660\u0660 \u0661\u0662 x",
    "\ufb01x \ufe8d\ufe8e y",
    "a \U0001f4d8 b \U0001f468\u200d\U0001f469 c",
    "Stra\u00dfe  GRO\u1e9eE",
    "  \u0627\u0653\u0644\u0652\u0643\u0650\u062a\u064e\u0627\u0628\u064f\u0640\u0640 \u0649 \u06a9 \u06cc  ",
    "\u0651\u064e\u0628 e\u0301 \u00e9",
]


def total_map(text: str):
    canon = apply(CANONICAL_V1, text)
    search = apply(SEARCH_AR_EN_V1, canon.text)
    return canon, search, search.map.compose(canon.map)


def renormalize(piece: str) -> str:
    return apply(SEARCH_AR_EN_V1, apply(CANONICAL_V1, piece).text).text


@pytest.mark.parametrize(
    "text",
    [fixture("en_short.txt"), fixture("ar_short.txt"), fixture("mixed_short.txt"), *CRAFTED],
    ids=["en", "ar", "mixed", *[f"crafted{i}" for i in range(len(CRAFTED))]],
)
def test_word_ranges_map_back_and_renormalize(text):
    _, search, total = total_map(text)
    assert (total.n_len, total.o_len) == (len(search.text), len(text))
    words = list(re.finditer(r"\S+", search.text))
    assert words
    for w in words:
        r = total.to_original(w.start(), w.end())
        assert 0 <= r.start <= r.end <= len(text)
        assert renormalize(text[r.start : r.end]) == search.text[w.start() : w.end()], (w.start(), w.end())


def test_hand_computed_offsets():
    # voweled word with tatweel: full original span, trailing damma included
    t = "\u0643\u0640\u064e\u062a\u064e\u0628\u064f x"
    canon, search, total = total_map(t)
    assert search.text == "\u0643\u062a\u0628 x"
    assert total.to_original(0, 3) == MappedRange(0, 7, True)
    assert total.to_original(1, 2) == MappedRange(3, 5, True)
    assert total.to_original(1, 3) == MappedRange(3, 7, True)
    assert total.to_original(4, 5) == MappedRange(8, 9, True)

    # CRLF: offsets shift by one after the break, canonical text has one LF
    t = "ab\r\ncd"
    canon, search, total = total_map(t)
    assert canon.text == "ab\ncd"
    assert total.to_original(3, 5) == MappedRange(4, 6, True)
    assert total.to_original(0, 2) == MappedRange(0, 2, True)
    assert canon.map.to_original(3, 5) == MappedRange(4, 6, True)

    # NBSP becomes an ordinary space, one for one
    _, search, total = total_map("a\u00a0b")
    assert search.text == "a b"
    assert total.to_original(2, 3) == MappedRange(2, 3, True)

    # Arabic-Indic digits fold one for one
    _, search, total = total_map("\u0663\u0660\u0660 x")
    assert search.text == "300 x"
    assert total.to_original(0, 3) == MappedRange(0, 3, True)

    # ligature: two search code points come from one original code point
    _, search, total = total_map("\ufb01x")
    assert search.text == "fix"
    assert total.to_original(0, 2) == MappedRange(0, 1, True)
    assert total.to_original(1, 2) == MappedRange(0, 1, False)
    assert total.to_original(0, 3) == MappedRange(0, 2, True)

    # whitespace run: one space stands for the whole original run
    _, search, total = total_map("a  \t b")
    assert search.text == "a b"
    assert total.to_original(2, 3) == MappedRange(5, 6, True)
    assert total.to_original(0, 3) == MappedRange(0, 6, True)

    # trimmed ends are detached deletions
    _, search, total = total_map("  ab  ")
    assert search.text == "ab"
    assert total.to_original(0, 2) == MappedRange(2, 4, True)

    # non-BMP emoji is one code point
    _, search, total = total_map("a \U0001f4d8 b")
    assert search.text == "a \U0001f4d8 b"
    assert total.to_original(2, 3) == MappedRange(2, 3, True)

    # casefold expansion
    _, search, total = total_map("Stra\u00dfe")
    assert search.text == "strasse"
    assert total.to_original(4, 6) == MappedRange(4, 5, True)
    assert total.to_original(5, 6) == MappedRange(4, 5, False)

    # canonical reordering: shadda+fatha is stored as fatha+shadda, anchors point at the canonical text
    canon, _, _ = total_map("\u0628\u0651\u064e")
    assert canon.text == "\u0628\u064e\u0651"
    assert canon.map.to_original(1, 2) == MappedRange(0, 3, False)
