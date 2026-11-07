"""Shared unit-window packing for fixed-size chunkers."""

from __future__ import annotations

from rag_foundry_techniques import ChunkSpan

from rag_foundry_chunking._units import Unit


def pack_windows(
    text: str,
    units: tuple[Unit, ...],
    weights: tuple[int, ...],
    size: int,
    overlap: int,
) -> tuple[ChunkSpan, ...]:
    if not units:
        return ()
    spans: list[ChunkSpan] = []
    start_index = 0
    previous_end = -1
    while start_index < len(units):
        end_index = start_index
        total = weights[end_index]
        while end_index + 1 < len(units) and total + weights[end_index + 1] <= size:
            end_index += 1
            total += weights[end_index]

        start, end = units[start_index][0], units[end_index][1]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end and end > previous_end:
            spans.append(ChunkSpan(start=start, end=end, split_level=0))
            previous_end = end

        if end_index == len(units) - 1:
            break
        next_index = end_index + 1
        suffix_weight = total - weights[start_index]
        candidate = start_index + 1
        while candidate <= end_index and suffix_weight > overlap:
            suffix_weight -= weights[candidate]
            candidate += 1
        if candidate <= end_index:
            next_index = candidate
        start_index = next_index
    spans.sort(key=lambda span: (span.start, span.end))
    return tuple(spans)
