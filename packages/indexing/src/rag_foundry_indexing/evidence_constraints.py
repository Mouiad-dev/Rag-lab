"""Evidence constraints shared by extractive compression (R26) and evidence packing.

`evidence_constraints` lists the facts a reader must not lose: hard constraints (numbers, quoted text,
negations), month words, and number+unit pairs. Months and units are matched on NFC+casefold text without
alef folding; stopwords and condition words are compared with `lexical_tokens` output, which folds alef.
"""

from __future__ import annotations

import re
import unicodedata

from rag_foundry_indexing.lexical_text import lexical_tokens
from rag_foundry_indexing.query_transform import _AR_NEG, _WORD_RE, hard_constraints

MONTHS = frozenset(
    "january february march april may june july august september october november december".split()
) | frozenset(
    (
        "\u064a\u0646\u0627\u064a\u0631",
        "\u0641\u0628\u0631\u0627\u064a\u0631",
        "\u0645\u0627\u0631\u0633",
        "\u0623\u0628\u0631\u064a\u0644",
        "\u0645\u0627\u064a\u0648",
        "\u064a\u0648\u0646\u064a\u0648",
        "\u064a\u0648\u0644\u064a\u0648",
        "\u0623\u063a\u0633\u0637\u0633",
        "\u0633\u0628\u062a\u0645\u0628\u0631",
        "\u0623\u0643\u062a\u0648\u0628\u0631",
        "\u0646\u0648\u0641\u0645\u0628\u0631",
        "\u062f\u064a\u0633\u0645\u0628\u0631",
    )
)
UNITS = frozenset(
    "% kg g km m cm mm l ml usd eur percent day days hour hours minute minutes year years "
    "metre metres meter meters".split()
) | frozenset(
    (
        "\u062f\u0648\u0644\u0627\u0631",
        "\u064a\u0648\u0645",
        "\u0623\u064a\u0627\u0645",
        "\u0633\u0627\u0639\u0629",
        "\u062f\u0642\u064a\u0642\u0629",
        "\u062f\u0642\u0627\u0626\u0642",
        "\u0633\u0646\u0629",
        "\u0643\u0645",
        "\u0643\u063a",
        "\u0645\u062a\u0631",
    )
)
STOPWORDS = frozenset(
    "the and for with what which when where who how does did are was were that this from about has have "
    "into its than then there their they you your can will".split()
) | frozenset(
    (
        "\u0639\u0644\u0649",
        "\u0627\u0644\u0649",
        "\u0627\u0644\u062a\u064a",
        "\u0627\u0644\u0630\u064a",
        "\u0647\u0630\u0627",
        "\u0647\u0630\u0647",
        "\u0630\u0644\u0643",
    )
)
_CONDITION_WORDS = frozenset({"unless", "except", "only", "if", "\u0627\u0644\u0627", "\u0627\u0630\u0627"})
_UNIT_PAIR_RE = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?("
    + "|".join(re.escape(unit) for unit in sorted(UNITS, key=lambda item: (-len(item), item)))
    + r")(?![^\W_])"
)


def _normal(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _digits_ascii(value: str) -> str:
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in value)


def evidence_constraints(text: str) -> tuple[str, ...]:
    out = list(hard_constraints(text))
    value = _normal(text)
    out.extend(match.group() for match in _WORD_RE.finditer(value) if match.group() in MONTHS)
    out.extend(f"{m.group(1)} {m.group(2)}" for m in _UNIT_PAIR_RE.finditer(_digits_ascii(value)))
    return tuple(dict.fromkeys(out))


def is_qualifier(text: str) -> bool:
    constraints = hard_constraints(text)
    if "not" in constraints or any(item in _AR_NEG for item in constraints):
        return True
    return any(token in _CONDITION_WORDS for token in lexical_tokens(text))


def content_terms(text: str) -> frozenset[str]:
    return frozenset(token for token in lexical_tokens(text) if len(token) >= 3 and token not in STOPWORDS)
