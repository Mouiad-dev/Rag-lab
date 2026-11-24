"""Deterministic text normalization steps with offset maps (ADR-0010 D6/D7).

Every step is `str -> (str, OffsetMap)` where the map goes from the step's output (normalized) to its input (original).
Steps are flagged `lossy`: positions map back, but text is never reconstructed from a lossy step. Source files are ASCII
only; every non-ASCII character is written as an escape.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from rag_foundry_domain.offsets import OffsetMap, OffsetMapBuilder

Step = Callable[[str], tuple[str, OffsetMap]]


class NormalizationError(RuntimeError):
    """An internal invariant failed. Carries no text content."""


def _step(*, lossy: bool) -> Callable[[Step], Step]:
    def mark(fn: Step) -> Step:
        fn.lossy = lossy  # type: ignore[attr-defined]
        return fn

    return mark


def _identity(text: str) -> OffsetMap:
    return OffsetMap.identity(len(text))


def _scan(text: str, pattern: re.Pattern[str], emit: Callable[[OffsetMapBuilder, str], None]) -> OffsetMap:
    b = OffsetMapBuilder()
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            b.cp(m.start() - pos)
        emit(b, m.group())
        pos = m.end()
    if pos < len(text):
        b.cp(len(text) - pos)
    return b.build()


# -- newlines ---------------------------------------------------------------------------------------------------

_NEWLINE = re.compile(r"\r\n?")


@_step(lossy=True)
def newlines(text: str) -> tuple[str, OffsetMap]:
    if "\r" not in text:
        return text, _identity(text)
    return _NEWLINE.sub("\n", text), _scan(text, _NEWLINE, lambda b, g: b.atomic(1, 2) if len(g) == 2 else b.cp(1))


# -- NFC / NFKC -------------------------------------------------------------------------------------------------


@cache
def _second_of_composite() -> frozenset[str]:
    """Code points that can be the second element of a primary composite (canonical pair decompositions)."""
    seconds = {chr(c) for c in range(0x1161, 0x1176)} | {chr(c) for c in range(0x11A8, 0x11C3)}
    for cp in range(0x110000):
        d = unicodedata.decomposition(chr(cp))
        if d and not d.startswith("<"):
            parts = d.split()
            if len(parts) == 2:
                seconds.add(chr(int(parts[1], 16)))
    return frozenset(seconds)


_BOUNDARY_CACHE: dict[str, dict[str, bool]] = {"NFC": {}, "NFKC": {}}


def _is_boundary(form: str, c: str) -> bool:
    cache_ = _BOUNDARY_CACHE[form]
    v = cache_.get(c)
    if v is None:
        v = False
        if unicodedata.combining(c) == 0 and c not in _second_of_composite():
            r = unicodedata.normalize(form, c)
            v = bool(r) and unicodedata.combining(r[0]) == 0 and r[0] not in _second_of_composite()
        cache_[c] = v
    return v


def _unicode_normalize(form: str, text: str) -> tuple[str, OffsetMap]:
    if unicodedata.is_normalized(form, text):
        return text, _identity(text)
    n = len(text)
    starts = [i for i, c in enumerate(text) if _is_boundary(form, c)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(n)
    b = OffsetMapBuilder()
    pieces: list[str] = []
    run = 0  # pending unchanged code points
    normalize = unicodedata.normalize
    for k in range(len(starts) - 1):
        group = text[starts[k] : starts[k + 1]]
        r = normalize(form, group)
        pieces.append(r)
        if r == group or (len(group) == 1 and len(r) == 1):
            run += len(group)
        else:
            if run:
                b.cp(run)
                run = 0
            b.atomic(len(r), len(group))
    if run:
        b.cp(run)
    out = "".join(pieces)
    if out != normalize(form, text):
        raise NormalizationError(f"{form} grouping disagrees with unicodedata.normalize")
    return out, b.build()


@_step(lossy=False)
def nfc(text: str) -> tuple[str, OffsetMap]:
    return _unicode_normalize("NFC", text)


@_step(lossy=True)
def nfkc(text: str) -> tuple[str, OffsetMap]:
    return _unicode_normalize("NFKC", text)


# -- casefold ---------------------------------------------------------------------------------------------------


@cache
def _casefold_expanding() -> re.Pattern[str]:
    chars = [chr(c) for c in range(0x110000) if len(chr(c).casefold()) != 1]
    return re.compile("[" + "".join(re.escape(c) for c in chars) + "]")


@_step(lossy=True)
def casefold(text: str) -> tuple[str, OffsetMap]:
    out = text.casefold()
    if len(out) == len(text):  # casefold never deletes, so equal length means every code point maps 1:1
        return out, _identity(text)
    return out, _scan(text, _casefold_expanding(), lambda b, g: b.atomic(len(g.casefold()), 1))


# -- deletions --------------------------------------------------------------------------------------------------


@cache
def _format_chars() -> re.Pattern[str]:
    chars = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Cf"]
    chars.append("\u034f")
    return re.compile("[" + "".join(re.escape(c) for c in sorted(set(chars))) + "]+")


@_step(lossy=True)
def strip_format(text: str) -> tuple[str, OffsetMap]:
    pattern = _format_chars()
    return pattern.sub("", text), _scan(text, pattern, lambda b, g: b.atomic(0, len(g)))


_TATWEEL = re.compile(r"\u0640+")
_HARAKAT = re.compile(r"[\u064b-\u065f\u0670]+")


@_step(lossy=True)
def strip_tatweel(text: str) -> tuple[str, OffsetMap]:
    return _TATWEEL.sub("", text), _scan(text, _TATWEEL, lambda b, g: b.attach(len(g)))


@_step(lossy=True)
def strip_harakat(text: str) -> tuple[str, OffsetMap]:
    return _HARAKAT.sub("", text), _scan(text, _HARAKAT, lambda b, g: b.attach(len(g)))


# -- 1:1 folds --------------------------------------------------------------------------------------------------

_ALEF = {0x0622: 0x0627, 0x0623: 0x0627, 0x0625: 0x0627, 0x0671: 0x0627}
_YEH = {0x0649: 0x064A, 0x06CC: 0x064A}
_KAF = {0x06A9: 0x0643}
_DIGITS = {c: 0x30 + (c - 0x0660) for c in range(0x0660, 0x066A)} | {
    c: 0x30 + (c - 0x06F0) for c in range(0x06F0, 0x06FA)
}


def _fold(table: dict[int, int]) -> Step:
    def fold(text: str) -> tuple[str, OffsetMap]:
        return text.translate(table), _identity(text)

    return fold


fold_alef = _step(lossy=True)(_fold(_ALEF))
fold_yeh = _step(lossy=True)(_fold(_YEH))
fold_kaf = _step(lossy=True)(_fold(_KAF))
fold_digits = _step(lossy=True)(_fold(_DIGITS))
for _name, _fn in (
    ("fold_alef", fold_alef),
    ("fold_yeh", fold_yeh),
    ("fold_kaf", fold_kaf),
    ("fold_digits", fold_digits),
):
    _fn.__name__ = _name


# -- whitespace -------------------------------------------------------------------------------------------------

_SPACE_RUN = re.compile(r"\s+")  # for str patterns \s is exactly str.isspace


@_step(lossy=True)
def whitespace(text: str) -> tuple[str, OffsetMap]:
    n = len(text)
    b = OffsetMapBuilder()
    pieces: list[str] = []
    pos = 0
    for m in _SPACE_RUN.finditer(text):
        s, e = m.span()
        if s > pos:
            b.cp(s - pos)
            pieces.append(text[pos:s])
        if s == 0 or e == n:
            b.atomic(0, e - s)  # trimmed end: detached deletion
        elif e - s == 1:
            b.cp(1)  # one code point becomes U+0020 (1:1)
            pieces.append(" ")
        else:
            b.atomic(1, e - s)
            pieces.append(" ")
        pos = e
    if pos < n:
        b.cp(n - pos)
        pieces.append(text[pos:])
    return "".join(pieces), b.build()


# -- profiles ---------------------------------------------------------------------------------------------------


class Profile(tuple):
    """An ordered list of steps with an identifier. Iterating yields the steps."""

    profile_id: str

    def __new__(cls, profile_id: str, steps: list[Step]) -> "Profile":
        self = super().__new__(cls, steps)
        self.profile_id = profile_id
        return self


CANONICAL_V1 = Profile("canonical-v1", [newlines, nfc])
SEARCH_AR_EN_V1 = Profile(
    "search-ar-en-v1",
    [
        nfkc,
        casefold,
        nfkc,
        strip_format,
        strip_tatweel,
        strip_harakat,
        # The strip steps can bring a base next to a mark it composes with; re-normalize so the profile is idempotent.
        nfkc,
        fold_alef,
        fold_yeh,
        fold_kaf,
        fold_digits,
        whitespace,
    ],
)


@dataclass(frozen=True)
class NormalizedText:
    """`map` goes from `text` (normalized) to the input of the first step (original)."""

    text: str
    map: OffsetMap
    profile_id: str
    unicode_version: str
    lossy_steps: tuple[str, ...]


def apply(profile: Profile, text: str) -> NormalizedText:
    acc = _identity(text)
    current = text
    lossy: list[str] = []
    for step in profile:
        out, m = step(current)
        if m.n_len != len(out) or m.o_len != len(current):
            raise NormalizationError(f"step {step.__name__} produced an inconsistent offset map")
        if not m.is_identity:
            acc = m.compose(acc)
        current = out
        if step.lossy:  # type: ignore[attr-defined]
            lossy.append(step.__name__)
    return NormalizedText(current, acc, profile.profile_id, unicodedata.unidata_version, tuple(dict.fromkeys(lossy)))
