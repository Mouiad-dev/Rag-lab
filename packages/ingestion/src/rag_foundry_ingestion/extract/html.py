"""HTML extractor (`html`): selectolax (Lexbor, HTML5 parser) plus a source alignment layer (ADR-0010 D2).

The parser gives text nodes in document order. Their characters are aligned one by one with the decoded source, so
every raw character maps to exact source offsets. Anything that cannot be aligned raises `alignment_failed`; offsets
are never guessed. No network, no script execution, no external entities (the parser only builds a tree).

Known limits (all fail with `alignment_failed`, never mis-align): `<template>` content, any text in svg/math when
the source contains `<![CDATA[` anywhere (the check is global), any foster-parented text (parser-moved
table content), and any construct the parser rewrites.
"""

import html as _html
import importlib.metadata
import platform
import re
from os.path import commonprefix

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionAlignmentError, ExtractionError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_domain.offsets import ATOMIC, CP, OffsetMapBuilder

from rag_foundry_ingestion.extract.base import (
    EXTRACTORS,
    ExtractionLimits,
    ExtractionResult,
    RawBlock,
    decode_text,
    finish,
    register,
)

try:
    from selectolax.lexbor import LexborHTMLParser
except ImportError:  # pragma: no cover - exercised only without the 'html' extra
    LexborHTMLParser = None  # type: ignore[assignment,misc]

_SKIP_TEXT_TAGS = frozenset({"head", "script", "style", "template", "noscript"})
_RAW_TEXT_TAGS = frozenset(
    {"script", "style", "title", "textarea", "noscript", "xmp", "iframe", "noembed", "noframes", "plaintext"}
)
# Raw-text (not RCDATA) elements keep `&amp;` literal; title and textarea decode entities.
_LITERAL_ENTITY_TAGS = _RAW_TEXT_TAGS - {"title", "textarea"}
_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "section", "article"}
    | {"header", "footer", "br"}
)
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_PRESERVE_TAGS = frozenset({"pre", "textarea"})
_FOREIGN_TAGS = frozenset({"svg", "math"})
_WS = " \t\n\f"  # node whitespace (the parser has already turned CR into LF)
_SRC_WS = " \t\n\r\f"

_ENTITY = re.compile(r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[A-Za-z][A-Za-z0-9]*;?)")
_RUN = re.compile(r"[^&\r\n\t\f \x00]+")
_RUN_PRE = re.compile(r"[^&\r\x00]+")
_TAG_NAME = re.compile(r"[^\t\n\f\r />]*")
_SKIP_WS_SLASH = re.compile(r"[\t\n\f\r /]*")
_SKIP_WS = re.compile(r"[\t\n\f\r ]*")
_ATTR_REST = re.compile(r"[^\t\n\f\r />=]*")
_UNQUOTED = re.compile(r"[^\t\n\f\r >]*")
_COMMENT_END = re.compile(r"--!?>")
_MARKUP_START = re.compile(r"<(?:[A-Za-z!?]|/.)", re.DOTALL)
_RAW_END: dict[str, re.Pattern[str]] = {}


def _raw_end(name: str) -> re.Pattern[str]:
    pat = _RAW_END.get(name)
    if pat is None:
        pat = _RAW_END[name] = re.compile(r"</" + re.escape(name) + r"(?=[\t\n\f\r />])", re.IGNORECASE)
    return pat


def _ascii_letter(ch: str) -> bool:
    return "a" <= ch <= "z" or "A" <= ch <= "Z"


class _Aligner:
    """Cursor over the decoded source plus the raw-text builder."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0
        self.map = OffsetMapBuilder()
        self.parts: list[str] = []
        self.raw_name: str | None = None  # a raw-text start tag was skipped; its content may follow
        # block state
        self.cur_start: int | None = None
        self.cur_kind = BlockKind.PARAGRAPH
        self.cur_level: int | None = None
        self.cur_text: list[str] = []
        self.space_tail = False
        self.blocks: list[RawBlock] = []
        self.path: list[tuple[int, str]] = []
        self.decode_entities = True
        self.literal_markup = False
        self.tables_skipped = 0  # `<table` start tags consumed by the skipper

    # -- source side ------------------------------------------------------------------------------------------

    def _try_match(self, t: str, i: int) -> tuple[int, int, int] | None:
        """(node chars used, source length, kind) when the node text at `i` matches the source at the cursor."""
        src, pos = self.src, self.pos
        if pos >= len(src):
            return None
        c = src[pos]
        ch = t[i]
        if c == "&" and self.decode_entities:
            m = _ENTITY.match(src, pos)
            if m:
                text = m.group()
                u = _html.unescape(text)
                if u != text and t.startswith(u, i):
                    return len(u), len(text), ATOMIC
        if ch == "\n":
            if src.startswith("\r\n", pos):
                return 1, 2, ATOMIC
            if c == "\r":
                return 1, 1, ATOMIC
        if c == ch:
            if c == "<" and self.raw_name is None and not self.literal_markup and _MARKUP_START.match(src, pos):
                return None
            return 1, 1, CP
        if c == "\x00" and ch == "\ufffd":
            return 1, 1, ATOMIC
        return None

    def _skip(self) -> None:
        """Skip one markup token or one dropped whitespace/NUL character as a detached deletion."""
        src, pos = self.src, self.pos
        n = len(src)
        if pos >= n:
            raise ExtractionAlignmentError()
        c = src[pos]
        if self.raw_name is not None and c not in _SRC_WS and c != "\x00":
            # content of a raw-text element that the parser did not emit: skip up to its end tag
            m = _raw_end(self.raw_name).search(src, pos)
            end = m.start() if m else n
            self.raw_name = None
            if end > pos:
                self._delete(end - pos)
                return
            c = src[pos]
        if c in _SRC_WS or c == "\x00":
            self._delete(1)
            return
        if c != "<" or pos + 1 >= n:
            raise ExtractionAlignmentError()
        nxt = src[pos + 1]
        if _ascii_letter(nxt):
            end, name = self._tag(pos, False)
            self.raw_name = name if name in _RAW_TEXT_TAGS else None
            if name == "table":
                self.tables_skipped += 1
        elif nxt == "/" and pos + 2 < n and _ascii_letter(src[pos + 2]):
            end, _ = self._tag(pos, True)
            self.raw_name = None
        elif nxt == "/" or nxt == "?":
            if nxt == "/" and pos + 2 >= n:
                raise ExtractionAlignmentError()
            end = self._to_gt(pos + 2)
            self.raw_name = None
        elif nxt == "!":
            if src.startswith("<!--", pos):
                if src.startswith(">", pos + 4):
                    end = pos + 5
                elif src.startswith("->", pos + 4):
                    end = pos + 6
                else:
                    m = _COMMENT_END.search(src, pos + 4)
                    end = m.end() if m else n
            else:
                end = self._to_gt(pos + 2)
            self.raw_name = None
        else:
            raise ExtractionAlignmentError()
        self._delete(end - pos)

    def _to_gt(self, start: int) -> int:
        e = self.src.find(">", start)
        return len(self.src) if e < 0 else e + 1

    def _tag(self, pos: int, end_tag: bool) -> tuple[int, str]:
        """End of the tag token at `pos` (quotes only count in attribute values) and its lower-case name."""
        s = self.src
        n = len(s)
        j = pos + (2 if end_tag else 1)
        m = _TAG_NAME.match(s, j)
        assert m is not None
        name = m.group().lower()
        j = m.end()
        while True:
            j = _SKIP_WS_SLASH.match(s, j).end()  # type: ignore[union-attr]
            if j >= n:
                return n, name
            if s[j] == ">":
                return j + 1, name
            j = _ATTR_REST.match(s, j + 1).end()  # type: ignore[union-attr]
            k = _SKIP_WS.match(s, j).end()  # type: ignore[union-attr]
            if k < n and s[k] == "=":
                k = _SKIP_WS.match(s, k + 1).end()  # type: ignore[union-attr]
                if k < n and s[k] in "\"'":
                    e = s.find(s[k], k + 1)
                    if e < 0:
                        return n, name
                    j = e + 1
                else:
                    j = _UNQUOTED.match(s, k).end()  # type: ignore[union-attr]

    def _delete(self, length: int) -> None:
        self.map.atomic(0, length)
        self.pos += length

    def _advance(self, t: str, i: int) -> tuple[int, int, int]:
        while True:
            r = self._try_match(t, i)
            if r is not None:
                self.raw_name = None
                return r
            self._skip()

    def finish_source(self) -> None:
        while self.pos < len(self.src):
            self._skip()

    # -- raw text side ----------------------------------------------------------------------------------------

    def _begin_block(self, kind: BlockKind, level: int | None) -> None:
        if self.cur_start is None:
            if self.map.n > 0:
                self.map.atomic(2, 0)  # block separator, no source
                self.parts.append("\n\n")
            self.cur_start = self.map.n
            self.cur_kind, self.cur_level = kind, level
            self.cur_text = []
            self.space_tail = False

    def flush(self) -> None:
        start = self.cur_start
        if start is None:
            return
        end = self.map.n - (1 if self.space_tail else 0)
        self.cur_start = None
        self.space_tail = False
        if end <= start:
            return
        level = self.cur_level
        if self.cur_kind is BlockKind.HEADING and level is not None:
            title = "".join(self.cur_text)[: end - start].strip()
            while self.path and self.path[-1][0] >= level:
                self.path.pop()
            self.path.append((level, title))
        path = tuple(t for _, t in self.path)
        self.blocks.append(RawBlock(self.cur_kind, start, end, section_path=path, heading_level=level))

    def text(
        self, t: str, *, emit: bool, preserve: bool, kind: BlockKind, level: int | None, decode_entities: bool = True
    ) -> None:
        self.decode_entities = decode_entities
        self.literal_markup = not decode_entities  # raw-text elements: `<b>` in the source is text
        i, n = 0, len(t)
        run_re = _RUN_PRE if preserve else _RUN
        while i < n:
            ch = t[i]
            if not preserve and ch in _WS:
                used, olen, _ = self._advance(t, i)
                self.pos += olen
                if not emit or self.cur_start is None or self.space_tail:
                    self.map.atomic(0, olen)
                else:
                    if olen == 1 and used == 1 and self.src[self.pos - 1] == " ":
                        self.map.cp(1)
                    else:
                        self.map.atomic(1, olen)
                    self.parts.append(" ")
                    self.cur_text.append(" ")
                    self.space_tail = True
                i += used
                continue
            m = run_re.match(t, i)
            if m is not None:
                piece = m.group()
                k = len(commonprefix([piece, self.src[self.pos : self.pos + len(piece)]]))
                if k and not self.literal_markup and self.raw_name is None:
                    mk = _MARKUP_START.search(self.src, self.pos, self.pos + k + 2)
                    if mk is not None and mk.start() < self.pos + k:
                        k = mk.start() - self.pos  # a source '<' that starts markup is never text here
                if k:
                    self.pos += k
                    self.raw_name = None
                    self._emit(piece[:k], k, CP, emit, kind, level)
                    i += k
                    continue
            used, olen, k = self._advance(t, i)
            self.pos += olen
            self._emit(t[i : i + used], olen, k, emit, kind, level)
            i += used

    def _emit(self, piece: str, olen: int, kind_map: int, emit: bool, kind: BlockKind, level: int | None) -> None:
        if not emit:
            self.map.atomic(0, olen)
            return
        self._begin_block(kind, level)
        if kind_map == CP:
            self.map.cp(len(piece))
        else:
            self.map.atomic(len(piece), olen)
        self.parts.append(piece)
        self.cur_text.append(piece)
        self.space_tail = False


class HtmlExtractor:
    name = "html"
    media_types = (MediaType.TEXT_HTML,)

    def __init__(self) -> None:
        self.version = f"1+selectolax=={importlib.metadata.version('selectolax')}+py{platform.python_version()}"

    def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
        src = decode_text(data, text_encoding, limits)
        parser = LexborHTMLParser(src, scripting=True)
        al = _Aligner(src)
        has_cdata = "<![CDATA[" in src
        open_tags: list[str] = []  # entered elements, for context
        skip_depth = preserve_depth = foreign_depth = tables_entered = 0

        def enter(tag: str) -> None:
            nonlocal skip_depth, preserve_depth, foreign_depth, tables_entered
            if tag == "table":
                tables_entered += 1
            if tag in _BLOCK_TAGS:
                al.flush()
            if tag in _SKIP_TEXT_TAGS:
                skip_depth += 1
            if tag in _PRESERVE_TAGS:
                preserve_depth += 1
            if tag in _FOREIGN_TAGS:
                foreign_depth += 1
            open_tags.append(tag)

        def leave() -> None:
            nonlocal skip_depth, preserve_depth, foreign_depth
            tag = open_tags.pop()
            if tag in _SKIP_TEXT_TAGS:
                skip_depth -= 1
            if tag in _PRESERVE_TAGS:
                preserve_depth -= 1
            if tag in _FOREIGN_TAGS:
                foreign_depth -= 1
            if tag in _BLOCK_TAGS:
                al.flush()

        def context() -> tuple[BlockKind, int | None]:
            for tag in reversed(open_tags):
                if tag in _HEADINGS:
                    return BlockKind.HEADING, _HEADINGS[tag]
                if tag == "pre":
                    return BlockKind.CODE, None
            return BlockKind.PARAGRAPH, None

        def handle_text(t: str) -> None:
            if not t:
                return
            if foreign_depth and has_cdata and t.strip(_WS):
                raise ExtractionAlignmentError()  # CDATA in svg/math cannot be told apart from markup
            emit = skip_depth == 0 and (foreign_depth == 0 or bool(t.strip(_WS)))
            kind, level = context()
            al.text(
                t,
                emit=emit,
                preserve=preserve_depth > 0,
                kind=kind,
                level=level,
                decode_entities=not (open_tags and open_tags[-1] in _LITERAL_ENTITY_TAGS),
            )
            if al.tables_skipped > tables_entered:
                # The skipper passed a `<table` start tag that the walk has not entered: this text was moved by
                # foster parenting. It cannot be aligned without guessing between equal texts, so fail explicitly.
                raise ExtractionAlignmentError()

        stack: list = []
        node = parser.root.child if parser.root is not None else None
        while node is not None:
            if node.is_text_node:
                handle_text(node.text_content or "")
            elif node.is_element_node:
                enter((node.tag or "").lower())
                if node.child is not None:
                    stack.append(node)
                    node = node.child
                    continue
                leave()
            while True:
                nxt = node.next
                if nxt is not None:
                    node = nxt
                    break
                if not stack:
                    node = None
                    break
                node = stack.pop()
                leave()
        al.finish_source()
        al.flush()
        raw = "".join(al.parts)
        return finish(
            raw,
            al.map.build(),
            al.blocks,
            extractor=self.name,
            extractor_version=self.version,
            source_space="decoded_text",
        )


if LexborHTMLParser is not None:
    EXTRACTOR = HtmlExtractor()
    register(EXTRACTOR)
else:  # pragma: no cover - exercised only without the 'html' extra
    # Fail explicitly and name the missing extra (never a silent fallback).
    class _MissingHtmlExtra:
        name = "html"
        version = "unavailable"
        media_types = (MediaType.TEXT_HTML,)

        def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
            raise ExtractionError("unsupported_extractor", "HTML extraction needs the 'html' extra (selectolax)")

    EXTRACTORS.setdefault(MediaType.TEXT_HTML, _MissingHtmlExtra())
