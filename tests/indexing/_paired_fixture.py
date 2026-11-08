"""Bilingual paired retrieval fixture."""

from tests.indexing._fixture import DISTRACTORS, FACT_AR, FACT_EN, QUERY_AR, QUERY_EN

PREFIX = "\u0631\u0642\u0645 \u0627\u0644\u0642\u0637\u0639\u0629 "
ID1 = PREFIX + "AB-1234 \u0641\u064a \u0627\u0644\u0645\u062e\u0632\u0646 \u0627\u0644\u0634\u0645\u0627\u0644\u064a."
ID2 = PREFIX + "AB-1243 \u0641\u064a \u0627\u0644\u0645\u062e\u0632\u0646 \u0627\u0644\u062c\u0646\u0648\u0628\u064a."
MIXED = "The Oasis Library (\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629) opens its reading room at 08:30."
PARAGRAPHS = (FACT_EN, FACT_AR, *DISTRACTORS, ID1, ID2, MIXED)
DOC = "\n\n".join(PARAGRAPHS)

SPANS = []
_offset = 0
for _paragraph in PARAGRAPHS:
    SPANS.append((_offset, _offset + len(_paragraph), _paragraph))
    _offset += len(_paragraph) + 2
SPANS = tuple(SPANS)
assert all(DOC[start:end] == text for start, end, text in SPANS)

# (case_id, cohort, split, query, relevant paragraph or None)
CASES = (
    ("en-fact", "en", "dev", QUERY_EN, FACT_EN),
    ("ar-fact", "ar", "dev", QUERY_AR, FACT_AR),
    ("id-exact", "identifier", "dev", "AB-1234", ID1),
    ("en-club", "en", "dev", "Which club meets on Tuesday evenings?", DISTRACTORS[2]),
    ("id-ar", "identifier", "validation", "\u0623\u064a\u0646 \u0627\u0644\u0642\u0637\u0639\u0629 AB-1243\u061f", ID2),
    ("ar-diacritic", "ar", "validation", "\u0627\u0644\u0645\u064e\u062e\u0652\u0632\u064e\u0646 \u0627\u0644\u062c\u0646\u0648\u0628\u064a", ID2),
    ("en-variant", "en", "validation", "Port Sallow tide readings", FACT_EN),
    ("mixed", "mixed", "validation", "When does \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 open its reading room?", MIXED),
    ("paraphrase", "en", "validation", "How frequently is the ocean height measured at the harbour station?", FACT_EN),
    ("unanswerable", "unanswerable", "validation", "What is the population of the oasis town?", None),
)
