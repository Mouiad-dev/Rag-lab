"""Hierarchical parent and child paragraph chunking (C18)."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, ParentedSpan, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import pack, safe_cut_ranges, sentence_ranges, trim, word_ranges
from rag_foundry_chunking.paragraph import paragraph_ranges
from rag_foundry_chunking.structure import Flags, pack_units


class ParentChildConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_size: int = Field(default=2000, ge=200, le=20000)
    child_size: int = Field(default=400, ge=50, le=4000)
    respect_sections: bool = True
    oversized: Literal["split", "keep"] = "split"

    @model_validator(mode="after")
    def _sizes_ordered(self):
        if self.child_size >= self.parent_size:
            raise ValueError("child_size must be less than parent_size")
        return self


class ParentSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    path: tuple[str, ...] = ()
    flags: Flags = ()

    @field_validator("end")
    @classmethod
    def _after_start(cls, value, info):
        if value <= info.data.get("start", -1):
            raise ValueError("end must be greater than start")
        return value

    @model_validator(mode="after")
    def _flags_ordered(self):
        if tuple(sorted(set(self.flags))) != self.flags:
            raise ValueError("flags must be unique and sorted")
        return self


class ChildSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    split_level: int = Field(ge=0, le=63)
    parent_id: str
    flags: Flags = ()

    @field_validator("end")
    @classmethod
    def _after_start(cls, value, info):
        if value <= info.data.get("start", -1):
            raise ValueError("end must be greater than start")
        return value

    @model_validator(mode="after")
    def _flags_ordered(self):
        if tuple(sorted(set(self.flags))) != self.flags:
            raise ValueError("flags must be unique and sorted")
        return self

    def to_span(self) -> ChunkSpan:
        return ChunkSpan(start=self.start, end=self.end, split_level=self.split_level)


class Hierarchy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parents: tuple[ParentSpan, ...]
    children: tuple[ChildSpan, ...]

    @model_validator(mode="after")
    def _sorted(self):
        if tuple(sorted(self.parents, key=lambda item: item.start)) != self.parents:
            raise ValueError("parents must be sorted by start")
        if tuple(sorted(self.children, key=lambda item: item.start)) != self.children:
            raise ValueError("children must be sorted by start")
        return self

    def parent_of(self, child: ChildSpan) -> ParentSpan:
        for parent in self.parents:
            if parent.parent_id == child.parent_id and parent.start <= child.start and child.end <= parent.end:
                return parent
        raise ValueError("child does not belong to a parent in this hierarchy")


def parent_id_for(text: str, start: int, end: int) -> str:
    content_hash = hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
    payload = json.dumps(["c18-parent-v1", start, end, content_hash], separators=(",", ":"))
    return "par_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class ParentChildChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C18",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ParentChildConfig,
    )

    def __init__(self, config: ParentChildConfig) -> None:
        self.config = config

    def chunk_hierarchy(self, doc: ChunkInput) -> Hierarchy:
        text = doc.text
        units = [item for a, b in paragraph_ranges(doc, use_blocks=True) if (item := trim(text, a, b))]
        groups: list[tuple[tuple[str, ...], list[tuple[int, int]]]] = []
        current_path: tuple[str, ...] = ()
        last_path: tuple[str, ...] = ()
        have_path = False
        block_index = 0
        for unit in units:
            path = last_path
            if self.config.respect_sections and doc.blocks:
                while block_index < len(doc.blocks) and doc.blocks[block_index].end <= unit[0]:
                    block_index += 1
                block = None
                if block_index < len(doc.blocks):
                    candidate = doc.blocks[block_index]
                    if candidate.start < unit[1] and unit[0] < candidate.end:
                        block = candidate
                if block is not None:
                    path = block.section_path
                    last_path = path
                    have_path = True
                elif not have_path:
                    path = ()
            else:
                path = ()
            if not groups or path != current_path:
                groups.append((path, []))
                current_path = path
            groups[-1][1].append(unit)

        parent_records: list[ParentSpan] = []
        for path, group_units in groups:
            for start, end, _level, flags in pack_units(
                text, group_units, self.config.parent_size, self.config.oversized
            ):
                parent = ParentSpan(
                    parent_id=parent_id_for(text, start, end), start=start, end=end, path=path, flags=flags
                )
                parent_records.append(parent)

        parents = tuple(sorted(parent_records, key=lambda item: item.start))
        children: list[ChildSpan] = []
        for parent in parent_records:
            sentence_units = sentence_ranges(text, parent.start, parent.end)
            child_units: list[tuple[int, int, int]] = []
            for start, end in sentence_units:
                if end - start <= self.config.child_size:
                    child_units.append((start, end, 0))
                    continue
                for word_start, word_end in word_ranges(text, start, end):
                    if word_end - word_start <= self.config.child_size:
                        child_units.append((word_start, word_end, 1))
                    else:
                        child_units.extend(
                            (a, b, 2) for a, b in safe_cut_ranges(text, word_start, word_end, self.config.child_size)
                        )
            for start, end, level in pack(text, child_units, self.config.child_size, overlap_units=0):
                children.append(ChildSpan(start=start, end=end, split_level=level, parent_id=parent.parent_id))
        return Hierarchy(parents=parents, children=tuple(sorted(children, key=lambda item: item.start)))

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        return tuple(child.to_span() for child in self.chunk_hierarchy(doc).children)

    def chunk_with_parents(self, doc: ChunkInput) -> tuple[ParentedSpan, ...]:
        hierarchy = self.chunk_hierarchy(doc)
        return tuple(
            ParentedSpan(
                start=child.start,
                end=child.end,
                split_level=child.split_level,
                parent_start=(parent := hierarchy.parent_of(child)).start,
                parent_end=parent.end,
            )
            for child in hierarchy.children
        )
