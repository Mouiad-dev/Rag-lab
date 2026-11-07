"""Text units used by fixed-size chunkers."""

from __future__ import annotations

import unicodedata
from typing import Protocol

Unit = tuple[int, int]


def char_units(text: str) -> tuple[Unit, ...]:
    if not text:
        return ()
    starts = [0]
    for index in range(1, len(text)):
        char = text[index]
        if not (
            unicodedata.combining(char) != 0
            or char in ("\u200d", "\ufe0e", "\ufe0f")
            or (text[index - 1] == "\r" and char == "\n")
        ):
            starts.append(index)
    return tuple((start, end) for start, end in zip(starts, (*starts[1:], len(text)), strict=True))


def word_units(text: str) -> tuple[Unit, ...]:
    units: list[Unit] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        start = index
        index += 1
        while index < len(text) and not text[index].isspace():
            index += 1
        units.append((start, index))
    return tuple(units)


class Tokenizer(Protocol):
    name: str

    def token_spans(self, text: str) -> tuple[Unit, ...]: ...


class LabRegexTokenizer:
    """Offline stand-in for a model tokenizer; its tokens are deterministic lab units."""

    name = "lab-regex-v1"

    def token_spans(self, text: str) -> tuple[Unit, ...]:
        tokens: list[Unit] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue
            category = unicodedata.category(char)
            if category.startswith(("L", "M")):
                start = index
                index += 1
                while index < len(text) and unicodedata.category(text[index]).startswith(("L", "M")):
                    index += 1
                tokens.append((start, index))
            elif category == "Nd":
                start = index
                while index < len(text) and unicodedata.category(text[index]) == "Nd":
                    end = min(index + 3, len(text))
                    while end > index and unicodedata.category(text[end - 1]) != "Nd":
                        end -= 1
                    tokens.append((index, end))
                    index = end
            else:
                tokens.append((index, index + 1))
                index += 1
        return tuple(tokens)
