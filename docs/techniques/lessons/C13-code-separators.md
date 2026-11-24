# C13: Code separators

## Intuition
Source code has natural cut points: the line where a class or function starts. Cut **before** those lines (never after a separator), keep decorators and the comments directly above with their declaration, and only fall back to smaller cuts when a declaration is too large.

## Mechanism
`CodeSeparatorChunker` is in `packages/chunking/src/rag_foundry_chunking/code.py`. `language` (`python`, `javascript`, `typescript`, `java`, `go`, `rust`) selects an ordered list of line-start regular expressions (Python: level 0 `class`/`def`/`async def` at column 0, level 1 indented `def`/`async def`).

- The text is cut before every matching line of the current level; decorator/annotation lines (`@...`) and comment lines (`#` for Python, `//`, `/*`, `*` otherwise) directly above move with the declaration.
- Units that fit are packed in order within `chunk_size`. A unit that is too large is cut again with the next level; after the last level it follows `oversized`: `keep` (one chunk flagged `oversized`) or `split` (lines, then words, then safe cuts; pieces flagged `oversized` + `split_part`). Nothing is dropped.
- Each chunk has `kind="code"`, `attrs` `language`, and `path`: the declaration lines (one per level) that enclose the chunk start, found with the same regular expressions.

## Worked example
Input (`chunk_size=80`):
```python
class Cart:
    """A shopping cart."""

    def add(self, item: dict[str, int]) -> None:
        # keep insertion order
        self.items.append(item)
        self.total += item['price']

    @property
    def size(self) -> int:
        return len(self.items)
```

Real output (`path`, `flags`, chunk text):
```text
('class Cart:',) () 'class Cart:\n    """A shopping cart."""'
('class Cart:', 'def add(self, item: dict[str, int]) -> None:') ('oversized', 'split_part') 'def add(self, item: dict[str, int]) -> None:\n        # keep insertion order'
('class Cart:', 'def add(self, item: dict[str, int]) -> None:') ('oversized', 'split_part') "self.items.append(item)\n        self.total += item['price']"
('class Cart:', 'def size(self) -> int:') () '@property\n    def size(self) -> int:\n        return len(self.items)'
```
`@property` stays with `size`. `add` (about 140 characters) is larger than 80 and Python has no separator level below methods, so it is split by lines and both pieces are flagged. The comment ends up with the signature, not with the line it describes; C14 does better here.

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import CodeSeparatorChunker as K, CodeSeparatorConfig as C; from rag_foundry_techniques import ChunkInput; t='class Cart:\n    \"\"\"A shopping cart.\"\"\"\n\n    def add(self, item: dict[str, int]) -> None:\n        # keep insertion order\n        self.items.append(item)\n        self.total += item[\'price\']\n\n    @property\n    def size(self) -> int:\n        return len(self.items)\n'; [print(c.path, c.flags, repr(t[c.start:c.end])) for c in K(C(chunk_size=80)).chunk_structured(ChunkInput(text=t)).chunks]"
```

## Failure case
Regular expressions do not know brackets. A function with a multi-line list and `chunk_size=32` is cut inside the list (real output: flags, text):
```text
('oversized', 'split_part') 'def f():\n    x = [\n        1,'
('oversized', 'split_part') '2,\n    ]\n    return x'
```

## Limitations
Line patterns, not a parser: strings or comments that look like declarations can create cuts, nested functions and multi-line signatures are not understood, and Python is the only language with two levels. Paths are declaration lines, not qualified names. Chunk metadata is not yet stored in the index or embedded; `chunk()` gives offsets only.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
