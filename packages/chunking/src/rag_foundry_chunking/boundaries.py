"""Shared text boundary and packing helpers."""

from __future__ import annotations

import unicodedata

TERMINATORS = (".", "!", "?", "\u061f", "\u06d4", "\u2026")
CLOSERS = ('"', "'", ")", "]", "}", "\u201d", "\u2019", "\u00bb", "\u203a")
DEFAULT_ABBREVIATIONS = tuple("mr mrs ms dr prof sr jr st vs e.g i.e fig no approx dept".split())
_TOKEN_PREFIX = frozenset("\"')]}([{\u00ab")
_EXTENDERS = frozenset(("\u200d", "\u200c", "\ufe0e", "\ufe0f"))


def trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return None if start == end else (start, end)


def sentence_ranges(text: str, start: int, end: int, abbreviations=DEFAULT_ABBREVIATIONS) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start

    def emit(boundary: int) -> None:
        nonlocal cursor
        item = trim(text, cursor, boundary)
        if item is not None:
            result.append(item)
        cursor = boundary

    i = start
    while i < end:
        if text[i] == "\n":
            j = i + 1
            while j < end and text[j] != "\n" and text[j].isspace():
                j += 1
            if j < end and text[j] == "\n":
                emit(i + 1)
                i = j + 1
                cursor = i
                continue
        if text[i] not in TERMINATORS:
            i += 1
            continue
        run_end = i + 1
        while run_end < end and text[run_end] in TERMINATORS:
            run_end += 1
        boundary = run_end
        while boundary < end and text[boundary] in CLOSERS:
            boundary += 1
        if boundary < end and not text[boundary].isspace():
            i = run_end
            continue
        exceptional = False
        if run_end == i + 1 and text[i] == ".":
            token_start = i
            while token_start > cursor and not text[token_start - 1].isspace():
                token_start -= 1
            token = text[token_start:i]
            while token and token[0] in _TOKEN_PREFIX:
                token = token[1:]
            token = token.lower()
            last_part = token.rsplit(".", 1)[-1]
            next_char = boundary
            while next_char < end and text[next_char].isspace():
                next_char += 1
            exceptional = (
                token in abbreviations
                or (len(last_part) == 1 and last_part.isalpha())
                or (next_char < end and text[next_char].islower())
            )
        if not exceptional:
            emit(boundary)
        i = max(run_end, boundary)
    item = trim(text, cursor, end)
    if item is not None:
        result.append(item)
    return result


def word_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result = []
    i = start
    while i < end:
        while i < end and text[i].isspace():
            i += 1
        begin = i
        while i < end and not text[i].isspace():
            i += 1
        if begin < i:
            result.append((begin, i))
    return result


def _regional(char: str) -> bool:
    return "\U0001f1e6" <= char <= "\U0001f1ff"


def _unsafe(text: str, i: int, start: int) -> bool:
    if i <= start:
        return False
    char = text[i]
    if (
        unicodedata.combining(char)
        or char in _EXTENDERS
        or text[i - 1] == "\u200d"
        or "\U0001f3fb" <= char <= "\U0001f3ff"
        or (text[i - 1] == "\r" and char == "\n")
    ):
        return True
    if _regional(text[i - 1]) and _regional(char):
        count = 0
        j = i - 1
        while j >= start and _regional(text[j]):
            count += 1
            j -= 1
        return count % 2 == 1
    return False


def safe_cut_ranges(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    ranges = []
    cursor = start
    while cursor < end:
        cut = min(cursor + size, end)
        if cut < end:
            safe = cut
            while safe > cursor and _unsafe(text, safe, start):
                safe -= 1
            cut = safe if safe > cursor else min(cursor + size, end)
        ranges.append((cursor, cut))
        cursor = cut
    return ranges


def pack(
    text: str,
    units: list[tuple[int, int, int]],
    size: int,
    overlap_units: int,
    overlap_level: int = 0,
) -> list[tuple[int, int, int]]:
    if not units:
        return []
    output: list[tuple[int, int, int]] = []
    previous_units: list[tuple[int, int, int]] = []
    next_index = 0
    while next_index < len(units):
        new_start_index = next_index
        carried: list[tuple[int, int, int]] = []
        if output and overlap_units and units[next_index][2] <= overlap_level:
            for unit in reversed(previous_units):
                if len(carried) == overlap_units or unit[2] > overlap_level:
                    break
                carried.insert(0, unit)
            while carried and (carried[0][0] <= output[-1][0] or units[next_index][1] - carried[0][0] > size):
                carried.pop(0)

        chunk_units = carried[:]
        chunk_start = chunk_units[0][0] if chunk_units else units[next_index][0]
        while next_index < len(units) and units[next_index][1] - chunk_start <= size:
            chunk_units.append(units[next_index])
            next_index += 1
        if not chunk_units:
            chunk_units = [units[next_index]]
            next_index += 1
        start, end = chunk_units[0][0], chunk_units[-1][1]
        output.append((start, end, max(unit[2] for unit in chunk_units)))
        previous_units = chunk_units
        if next_index == new_start_index:
            next_index += 1
    return output
