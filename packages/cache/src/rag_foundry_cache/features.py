"""Guarded question features for semantic-cache eligibility (R51, ADR-0021).

A semantic hit may only reuse an answer when the meaning-bearing features of both questions
are equal. Features, not question text, are what a cache entry stores: `to_json` writes
numbers, identifiers and quoted strings as SHA-256 digests of their NFC values.

Known limits. Each one means a possible false hit that only the similarity threshold, the
margin and revalidation can stop:
- U+0645 U+0627 is not treated as negation (it also means "what").
- Entity names without digits are not features.
- Spelled-out numbers ("two", U+0627 U+062B U+0646 U+0627 U+0646) are not parsed.
- Month names (EN "March" vs "May", AR month names) and units ("km" vs "miles", "USD" vs
  "EUR") are not features.
- No stemming. Single-quoted text is not a quoted feature (the apostrophe is ambiguous).
- Arabic words that merely start with the U+0648 or U+0641 prefix plus a negation word
  can be read as negation.
- "nothing" is not negation. European "1.000" reads as 1 (a decimal point).
- A token mixing letters and digits ("10km", "3rd") counts as an identifier, not a number.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
from rag_foundry_domain.errors import CacheError

from rag_foundry_cache.keys import _nfc

_HEX = re.compile(r"[0-9a-f]{64}")
_TAGS = frozenset({"before", "after", "more", "less", "at_least", "at_most", "except"})
_JSON_KEYS = frozenset({"numbers", "negation", "comparatives", "identifiers", "quoted", "script"})

_DIGITS = {0x0660 + i: str(i) for i in range(10)}
_DIGITS.update({0x06F0 + i: str(i) for i in range(10)})
_DIGITS.update({0x066B: ".", 0x066C: ","})

_QUOTED = re.compile(r"\"([^\"]*)\"|\u201c([^\u201d]*)\u201d|\u00ab([^\u00bb]*)\u00bb")
_NUMBER = re.compile(r"(?:(?<!\w)[-\N{MINUS SIGN}])?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")
_ALEF = str.maketrans({0x0623: 0x0627, 0x0625: 0x0627, 0x0622: 0x0627})
_EN_WORD = re.compile(r"[a-z']+")
_AR_WORD = re.compile(r"[\u0621-\u064a\u0671-\u06d3]+")
_TASHKEEL = re.compile(r"[\u064b-\u065f\u0670\u0640]")

_EN_NEGATION = frozenset({"not", "no", "never", "without", "none", "nobody", "neither", "nor", "cannot"})
_AR_NEGATION_BASE = (
    "\u0644\u0627",
    "\u0644\u064a\u0633",
    "\u0644\u064a\u0633\u062a",
    "\u0644\u0645",
    "\u0644\u0646",
    "\u063a\u064a\u0631",
    "\u0628\u062f\u0648\u0646",
)
_AR_NEGATION = frozenset({w for base in _AR_NEGATION_BASE for w in (base, "\u0648" + base, "\u0641" + base)})
_EN_COMPARATIVE = {
    "before": "before",
    "after": "after",
    "more": "more",
    "over": "more",
    "less": "less",
    "fewer": "less",
    "under": "less",
    "except": "except",
}
_AR_COMPARATIVE_RAW = {
    "\u0642\u0628\u0644": "before",
    "\u0628\u0639\u062f": "after",
    "\u0623\u0643\u062b\u0631": "more",
    "\u0623\u0642\u0644": "less",
    "\u0628\u0627\u0633\u062a\u062b\u0646\u0627\u0621": "except",
}
_AR_COMPARATIVE = {k.translate(_ALEF): v for k, v in _AR_COMPARATIVE_RAW.items()}


def _hash(value: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", value).encode("utf-8")).hexdigest()


class QuestionFeatures(BaseModel):
    """Meaning-bearing features of one question.

    Freshly extracted features hold raw values (`hashed=False`). Features read back from
    storage hold SHA-256 digests (`hashed=True`). `eligible` works on both.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    numbers: tuple[str, ...]
    negation: int
    comparatives: tuple[str, ...]
    identifiers: tuple[str, ...]
    quoted: tuple[str, ...]
    script: Literal["latin", "arabic", "mixed", "other"]
    hashed: bool = False

    @field_validator("comparatives")
    @classmethod
    def _sorted_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _check(self) -> "QuestionFeatures":
        if self.negation < 0 or not set(self.comparatives) <= _TAGS:
            raise ValueError("invalid features")
        if self.hashed:
            for values in (self.numbers, self.identifiers, self.quoted):
                if not all(_HEX.fullmatch(v) for v in values):
                    raise ValueError("invalid digest")
        return self

    def _digests(self) -> dict[str, tuple[str, ...]]:
        def conv(values: tuple[str, ...]) -> tuple[str, ...]:
            return values if self.hashed else tuple(sorted(_hash(v) for v in values))

        return {
            "numbers": conv(self.numbers),
            "identifiers": conv(self.identifiers),
            "quoted": conv(self.quoted),
        }

    def to_json(self) -> str:
        """Canonical JSON. No raw number, identifier or quoted string is written."""
        payload: dict[str, Any] = {
            **{k: list(v) for k, v in self._digests().items()},
            "negation": self.negation,
            "comparatives": list(self.comparatives),
            "script": self.script,
        }
        return json.dumps(_nfc(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "QuestionFeatures":
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or set(data) != _JSON_KEYS:
                raise ValueError("bad keys")
            if isinstance(data["negation"], bool) or not isinstance(data["negation"], int):
                raise ValueError("bad negation")
            for name in ("numbers", "comparatives", "identifiers", "quoted"):
                if not isinstance(data[name], list) or not all(isinstance(v, str) for v in data[name]):
                    raise ValueError("bad list")
            return cls(
                numbers=tuple(data["numbers"]),
                negation=data["negation"],
                comparatives=tuple(data["comparatives"]),
                identifiers=tuple(data["identifiers"]),
                quoted=tuple(data["quoted"]),
                script=data["script"],
                hashed=True,
            )
        except (ValueError, TypeError, ValidationError, RecursionError):
            raise CacheError("cache_entry_invalid") from None


def _canonical_number(raw: str) -> str:
    negative = raw[0] in "-\N{MINUS SIGN}"
    whole, _, frac = raw.lstrip("-\N{MINUS SIGN}").replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0")
    value = f"{whole}.{frac}" if frac else whole
    return "-" + value if negative and value != "0" else value


def _script(text: str) -> Literal["latin", "arabic", "mixed", "other"]:
    latin = arabic = False
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            arabic = True
        elif unicodedata.name(ch, "").startswith("LATIN"):
            latin = True
    if latin and arabic:
        return "mixed"
    return "latin" if latin else "arabic" if arabic else "other"


def _comparatives(en_words: list[str], ar_words: list[str]) -> tuple[str, ...]:
    tags: set[str] = set()
    index = 0
    while index < len(en_words):
        pair = en_words[index : index + 2]
        if pair in (["at", "least"], ["at", "most"]):
            tags.add("at_" + pair[1])
            index += 2
            continue
        tag = _EN_COMPARATIVE.get(en_words[index])
        if tag:
            tags.add(tag)
        index += 1
    tags.update(t for w in ar_words if (t := _AR_COMPARATIVE.get(w)))
    return tuple(sorted(tags))


def features(question: str) -> QuestionFeatures:
    """Extract features deterministically. Identifiers and quoted text keep their case."""
    text = unicodedata.normalize("NFC", question).translate(_DIGITS)
    quoted = sorted(next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(text))
    rest = _QUOTED.sub(" ", text)
    identifiers: list[str] = []
    numbers: list[str] = []
    for raw_token in rest.split():
        token = raw_token.strip("".join(c for c in set(raw_token) if not c.isalnum()))
        if not token:
            continue
        if any(c.isalpha() for c in token) and any(c.isdigit() for c in token):
            identifiers.append(token)
        else:
            numbers.extend(_canonical_number(n) for n in _NUMBER.findall(raw_token))
    en_found = _EN_WORD.findall(rest.lower().replace("\u2019", "'"))
    en_words = [w for w in (w.strip("'") for w in en_found) if w]
    ar_words = _AR_WORD.findall(_TASHKEEL.sub("", rest).translate(_ALEF))
    negation = sum(1 for w in en_words if w in _EN_NEGATION or w.endswith("n't"))
    negation += sum(1 for w in ar_words if w in _AR_NEGATION)
    return QuestionFeatures(
        numbers=tuple(sorted(numbers)),
        negation=negation,
        comparatives=_comparatives(en_words, ar_words),
        identifiers=tuple(sorted(identifiers)),
        quoted=tuple(quoted),
        script=_script(text),
    )


def eligible(a: QuestionFeatures, b: QuestionFeatures) -> tuple[bool, tuple[str, ...]]:
    """Return `(True, ())` when all features are equal, else the differing feature names."""
    da, db = a._digests(), b._digests()
    differing = [name for name in ("numbers", "identifiers", "quoted") if da[name] != db[name]]
    for name in ("negation", "comparatives", "script"):
        if getattr(a, name) != getattr(b, name):
            differing.append(name)
    return (not differing, tuple(sorted(differing)))


__all__ = ["QuestionFeatures", "eligible", "features"]
