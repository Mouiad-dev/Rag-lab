"""Declaration and Python AST code chunkers (C13/C14)."""

from __future__ import annotations

import ast
import re
from bisect import bisect_right
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import trim
from rag_foundry_chunking.structure import (
    ContextSpan,
    StructuredChunk,
    StructureParseError,
    StructureResult,
    line_ranges,
    pack_units,
)


class CodeSeparatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    language: Literal["python", "javascript", "typescript", "java", "go", "rust"] = "python"


class AstCodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: int = Field(default=1200, ge=32, le=8000)
    oversized: Literal["split", "keep"] = "split"
    language: str = "python"
    on_syntax_error: Literal["raise", "fallback"] = "raise"


_PATTERNS = {
    "python": (re.compile(r"^(?:class|def|async def) "), re.compile(r"^\s+(?:def|async def) ")),
    "javascript": (re.compile(r"^(?:export\b|(?:async\s+)?function\b|class\b)"),),
    "typescript": (re.compile(r"^(?:export\b|(?:async\s+)?function\b|class\b|interface\b|type\b)"),),
    "java": (re.compile(r"^(?:\s*(?:public|protected|private|static|final|class|interface|enum)\b)"),),
    "go": (re.compile(r"^(?:func\b|type\s+\w+\s+struct\b|type\s+\w+\s+interface\b)"),),
    "rust": (re.compile(r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|mod)\b"),),
}
_DECORATOR = re.compile(r"^\s*@")


def _lines(text: str):
    return [(a, e, text[a:e]) for a, e, _ in line_ranges(text)]


def _line_start(lines, index: int, floor: int, comment: tuple[str, ...]) -> int:
    """Start of a declaration line, moved up over decorator/annotation and comment lines directly above it."""
    start = lines[index][0]
    i = index - 1
    while i >= 0 and lines[i][0] >= floor:
        line = lines[i][2].lstrip()
        if not (_DECORATOR.match(line) or line.startswith(comment)):
            break
        start = lines[i][0]
        i -= 1
    return start


_COMMENTS = {"python": ("#",)}
_C_COMMENTS = ("//", "/*", "*")


class CodeSeparatorChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C13",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=CodeSeparatorConfig,
    )

    def __init__(self, config: CodeSeparatorConfig) -> None:
        self.config = config

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        text, lines, size = doc.text, _lines(doc.text), self.config.chunk_size
        if not text.strip():
            return StructureResult(chunks=())
        patterns = _PATTERNS[self.config.language]
        declarations = []
        for regex in patterns:
            entries = [(a, line, len(line) - len(line.lstrip())) for a, _e, line in lines if regex.match(line)]
            declarations.append(([a for a, _line, _indent in entries], entries))

        comment = _COMMENTS.get(self.config.language, _C_COMMENTS)
        starts = [a for a, _e, _line in lines]

        # Cut before declaration lines of the highest level first; oversized units are refined by the next level.
        def emit_region(start: int, end: int, depth: int) -> list[StructuredChunk]:
            boundaries = [start, end]
            if depth < len(patterns):
                regex = patterns[depth]
                for i in range(bisect_right(starts, start), bisect_right(starts, end - 1)):
                    if regex.match(lines[i][2]):
                        boundaries.append(_line_start(lines, i, start, comment))
            boundaries = sorted(set(boundaries))
            units = [u for u in (trim(text, a, b) for a, b in zip(boundaries, boundaries[1:], strict=False)) if u]
            out: list[StructuredChunk] = []
            run: list[tuple[int, int]] = []

            def flush() -> None:
                for x, y, lvl, flags in pack_units(text, run, size, self.config.oversized):
                    out.append(self._chunk(text, x, y, depth + lvl, flags, declarations, starts))
                run.clear()

            for a, b in units:
                if b - a > size and depth + 1 < len(patterns):
                    flush()
                    out.extend(emit_region(a, b, depth + 1))
                else:
                    run.append((a, b))
            flush()
            return out

        chunks = emit_region(0, len(text), 0)
        chunks.sort(key=lambda c: c.start)
        return StructureResult(chunks=tuple(chunks))

    def _chunk(self, text, a, b, level, flags, declarations, line_starts):
        names = []
        target = a
        line_end = text.find("\n", a, b)
        first_line = text[a : b if line_end < 0 else line_end]
        line_start = line_starts[bisect_right(line_starts, a) - 1]
        indentation = a - line_start if not text[line_start:a].strip() else 0
        if _DECORATOR.match(first_line):
            candidates = [
                starts[index]
                for starts, _entries in declarations
                if (index := bisect_right(starts, a)) < len(starts) and starts[index] < b
            ]
            if candidates:
                target = min(candidates)
        floor = 0  # a deeper level only looks inside the declaration found at the level above
        for starts, entries in declarations:
            index = bisect_right(starts, target) - 1
            while index >= 0 and entries[index][0] >= floor and entries[index][2] > indentation:
                index -= 1
            if index < 0 or entries[index][0] < floor:
                break
            names.append(entries[index][1].strip())
            floor = entries[index][0]
        item = trim(text, a, b)
        return StructuredChunk(
            start=item[0],
            end=item[1],
            split_level=min(level, 63),
            kind="code",
            path=tuple(names),
            attrs=(("language", self.config.language),),
            flags=flags,
        )

    def chunk(self, doc: ChunkInput):
        return tuple(c.to_span() for c in self.chunk_structured(doc).chunks)


def _offset(text: str, lines, line_no: int, col: int) -> int:
    """Code-point offset of an `ast` position (1-based line, UTF-8 byte column)."""
    start, end, _ = lines[line_no - 1]
    return start + len(text[start:end].encode("utf-8")[:col].decode("utf-8"))


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _node_start(text: str, lines, node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        line_start, line_end, _ = lines[min(d.lineno for d in decorators) - 1]
        return trim(text, line_start, line_end)[0]  # the '@' is the first token of its line
    return _offset(text, lines, node.lineno, node.col_offset)


def _node_end(text: str, lines, node: ast.stmt) -> int:
    return _offset(text, lines, node.end_lineno, node.end_col_offset)


def _header_end(text: str, lines, node: ast.stmt, first: ast.stmt) -> int:
    """End of a def/class header (signature through docstring), before the first body statement.

    Blank and comment-only lines between the header and `first` are left to `first`, like all gap text.
    """
    first_start = _node_start(text, lines, first)
    index = first.lineno - 1
    if lines[index][0] < first_start and text[lines[index][0] : first_start].strip():
        return first_start  # the body starts on the header's own line: `def f(): return 1`
    index -= 1
    while index >= node.lineno and (
        not text[lines[index][0] : lines[index][1]].strip() or _comment_only(text, lines[index])
    ):
        index -= 1
    end = lines[index][1]
    if _docstring(node):  # a docstring line that looks like a comment must not end the header early
        end = max(end, _node_end(text, lines, node.body[0]))
    return end


def _comment_only(text: str, line) -> bool:
    return text[line[0] : line[1]].lstrip().startswith("#")


def _docstring(node) -> bool:
    body = node.body
    return bool(
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


class AstCodeChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C14",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=AstCodeConfig,
    )

    def __init__(self, config: AstCodeConfig) -> None:
        self.config = config

    def chunk_structured(self, doc: ChunkInput) -> StructureResult:
        if self.config.language != "python":
            raise StructureParseError("unsupported_language", f"unsupported language: {self.config.language}")
        text, size = doc.text, self.config.chunk_size
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            if self.config.on_syntax_error == "raise":
                line = getattr(exc, "lineno", None)
                raise StructureParseError("syntax_error", f"syntax error at line {line}") from exc
            fallback = CodeSeparatorChunker(CodeSeparatorConfig(chunk_size=size, oversized=self.config.oversized))
            return StructureResult(
                chunks=tuple(
                    c.model_copy(update={"flags": tuple(sorted({*c.flags, "unparsed"}))})
                    for c in fallback.chunk_structured(doc).chunks
                )
            )
        lines = line_ranges(text)
        chunks: list[StructuredChunk] = []

        def emit(a, b, level, flags, node, path, context):
            chunks.append(
                StructuredChunk(
                    start=a,
                    end=b,
                    split_level=min(level, 63),
                    kind="code",
                    path=path,
                    context=context,
                    attrs=(("node_type", type(node).__name__), ("qualname", ".".join(path))),
                    flags=flags,
                )
            )

        def walk(statements, region_start, region_end, context, qual, level):
            # Tile the region: text between statements (comments, blank lines, ';') joins the next statement,
            # trailing text joins the last one, so nothing is lost and cuts stay at statement boundaries.
            units = []
            cursor = region_start
            for index, node in enumerate(statements):
                end = region_end if index == len(statements) - 1 else _node_end(text, lines, node)
                item = trim(text, cursor, end)
                if item is not None:
                    units.append((node, *item))
                cursor = end
            i = 0
            while i < len(units):
                node, a, b = units[i]
                body = node.body[1:] if isinstance(node, _DEFS) and _docstring(node) else getattr(node, "body", [])
                if isinstance(node, _DEFS) and b - a > size and body:
                    path = (*qual, node.name)
                    header_end = _header_end(text, lines, node, body[0])
                    header = trim(text, a, header_end)
                    for x, y, piece_level, flags in pack_units(text, [header], size, self.config.oversized):
                        emit(x, y, level + piece_level, flags, node, path, context)
                    line_start, line_end, _ = lines[node.lineno - 1]
                    declaration = trim(text, line_start, min(line_end, header[1]))
                    walk(
                        body,
                        header_end,
                        b,
                        (*context, ContextSpan(start=declaration[0], end=declaration[1], role="declaration")),
                        path,
                        level + 1,
                    )
                    i += 1
                    continue
                group = []
                while i < len(units):
                    node, a, b = units[i]
                    if isinstance(node, _DEFS) and b - a > size and len(node.body) > int(_docstring(node)):
                        break
                    group.append(units[i])
                    i += 1
                cursor = 0  # pieces come in order: a moving index finds each piece's first unit
                for x, y, piece_level, flags in pack_units(
                    text, [(a, b) for _, a, b in group], size, self.config.oversized
                ):
                    while group[cursor][2] <= x:
                        cursor += 1
                    first = group[cursor][0]
                    path = (*qual, first.name) if isinstance(first, _DEFS) else qual
                    emit(x, y, level + piece_level, flags, first, path, context)

        if tree.body:
            walk(tree.body, 0, len(text), (), (), 0)
        elif (item := trim(text, 0, len(text))) is not None:  # only comments
            for x, y, level, flags in pack_units(text, [item], size, self.config.oversized):
                emit(x, y, level, flags, tree, (), ())
        return StructureResult(chunks=tuple(chunks))

    def chunk(self, doc: ChunkInput):
        return tuple(c.to_span() for c in self.chunk_structured(doc).chunks)
