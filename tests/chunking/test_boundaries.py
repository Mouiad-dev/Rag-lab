import unicodedata

from rag_foundry_chunking.boundaries import pack, safe_cut_ranges, sentence_ranges, trim, word_ranges


def test_english_sentence_boundaries_and_abbreviations():
    text = 'Dr. Smith arrived. He said "Stop." Then he left!'
    assert sentence_ranges(text, 0, len(text), ("dr",)) == [(0, 18), (19, 34), (35, len(text))]
    text = "J. Smith met the U.S. Army, e.g. this team. Pi is 3.14 today. Wait... What?! Yes."
    ranges = sentence_ranges(text, 0, len(text))
    assert [text[a:b] for a, b in ranges] == [
        "J. Smith met the U.S. Army, e.g. this team.",
        "Pi is 3.14 today.",
        "Wait...",
        "What?!",
        "Yes.",
    ]


def test_arabic_and_mixed_ranges_are_exact_and_nfc():
    text = (
        "\u0645\u0631\u062d\u0628\u0627\u061f \u0646\u0639\u0645\u06d4 "
        "\u0639\u0644\u064a\u060c \u0648\u0633\u0647\u0644\u061b \u062b\u0645 \u062e\u0631\u062c."
    )
    assert unicodedata.is_normalized("NFC", text)
    ranges = sentence_ranges(text, 0, len(text))
    assert [text[a:b] for a, b in ranges] == [
        "\u0645\u0631\u062d\u0628\u0627\u061f",
        "\u0646\u0639\u0645\u06d4",
        "\u0639\u0644\u064a\u060c \u0648\u0633\u0647\u0644\u061b \u062b\u0645 \u062e\u0631\u062c.",
    ]
    quoted = '\u0642\u0627\u0644 "\u0646\u0639\u0645.\u00bb \u0648\u0645\u0634\u0649.'
    assert sentence_ranges(quoted, 0, len(quoted)) == [(0, 10), (11, len(quoted))]
    abbrev = "\u062f. \u0630\u0647\u0628."
    assert sentence_ranges(abbrev, 0, len(abbrev)) == [(0, len(abbrev))]
    mixed = "\u0628\u064e\u0627\u0628. English ends."
    assert unicodedata.is_normalized("NFC", mixed)
    assert sentence_ranges(mixed, 0, len(mixed)) == [(0, 5), (6, len(mixed))]


def test_boundary_helper_ranges_and_safe_cuts():
    assert trim("  abc \n", 0, 7) == (2, 5)
    assert trim(" \n", 0, 2) is None
    text = "aa bb\tcc"
    assert word_ranges(text, 0, len(text)) == [(0, 2), (3, 5), (6, 8)]
    units = [(index, index + 1, 0) for index in range(8)]
    assert pack("", units, 3, 0) == [(0, 3, 0), (3, 6, 0), (6, 8, 0)]
    assert pack("", units, 3, 1) == [(0, 3, 0), (2, 5, 0), (4, 7, 0), (6, 8, 0)]
    assert pack("", units, 4, 2) == [(0, 4, 0), (2, 6, 0), (4, 8, 0)]
    mixed_levels = [(0, 2, 0), (2, 4, 0), (4, 6, 1), (6, 8, 1), (8, 10, 0)]
    assert pack("", mixed_levels, 4, 2) == [(0, 4, 0), (4, 8, 1), (8, 10, 0)]
    assert pack("", [(0, 2, 0), (3, 4, 0)], 4, 0) == [(0, 4, 0)]
    text = "x" * 31 + "\u064e" + "y" * 40
    assert all(text[a] != "\u064e" for a, _ in safe_cut_ranges(text, 0, len(text), 32))
