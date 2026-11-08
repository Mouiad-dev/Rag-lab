# C14: AST-aware code

## Intuition
A real parser knows where every statement, function and class starts and ends, including multi-line signatures, brackets and strings that line patterns get wrong. Cut only at syntax-node boundaries, and when a class or function is too large, keep its header as context for the pieces of its body.

## Mechanism
`AstCodeChunker` is in `packages/chunking/src/rag_foundry_chunking/code.py`. It calls only `ast.parse` from the Python standard library; the code is never compiled to bytecode or executed.

- `ast` positions are 1-based lines and **UTF-8 byte** columns; they are converted to code-point offsets in the canonical text (tested with Arabic identifiers and strings, CRLF and CR line ends).
- Each statement list is tiled: text between statements (comments, blank lines, `;`) joins the next statement, trailing text joins the last one, so nothing is lost. Decorators belong to their function or class.
- Statements that fit are packed in order within `chunk_size`. A `class`/`def` that is too large becomes a header chunk (decorators, signature and docstring) and its body is chunked the same way, one level deeper; body chunks carry `declaration` context spans for each enclosing header line (`class Cart:`, `def add(...)`), so `render()` shows where they are.
- `path` is the qualified name (`("Cart", "add")`); `attrs` hold `node_type` (for example `FunctionDef`) and `qualname`.
- A single statement larger than `chunk_size` follows `oversized` (`keep`: flagged `oversized`; `split`: pieces flagged `oversized` + `split_part`).
- Errors are explicit: `language` other than `python` raises `StructureParseError("unsupported_language")`; a syntax error (or a NUL byte) raises `StructureParseError("syntax_error")` with the line, or, with `on_syntax_error="fallback"`, returns C13 chunks flagged `unparsed` (never labelled as AST chunks).

## Worked example
Same input as the C13 lesson (`chunk_size=80`). Real output (`path`, `render()`):
```text
('Cart',) 'class Cart:\n    """A shopping cart."""'
('Cart', 'add') 'class Cart:\ndef add(self, item: dict[str, int]) -> None:'
('Cart', 'add') 'class Cart:\ndef add(self, item: dict[str, int]) -> None:\n# keep insertion order\n        self.items.append(item)'
('Cart', 'add') "class Cart:\ndef add(self, item: dict[str, int]) -> None:\nself.total += item['price']"
('Cart', 'size') 'class Cart:\n@property\n    def size(self) -> int:\n        return len(self.items)'
```
Compared with C13, the comment stays with the statement it describes, and every body piece shows its class and method header (from source offsets, not invented text).

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import AstCodeChunker as A, AstCodeConfig as C, render; from rag_foundry_techniques import ChunkInput; t='class Cart:\n    \"\"\"A shopping cart.\"\"\"\n\n    def add(self, item: dict[str, int]) -> None:\n        # keep insertion order\n        self.items.append(item)\n        self.total += item[\'price\']\n\n    @property\n    def size(self) -> int:\n        return len(self.items)\n'; [print(c.path, repr(render(t, c))) for c in A(C(chunk_size=80)).chunk_structured(ChunkInput(text=t)).chunks]"
```

## Failure case
A syntax tree does not make one long statement small. `x = [1, 2]` written over five lines with `chunk_size=32` (real output: flags, text):
```text
() 'def f():'
('oversized', 'split_part') 'x = [\n        1,\n\n        2,'
('oversized', 'split_part') ']'
() 'return x'
```
The list is split as plain text and a lone `]` chunk is useless. Invalid code is refused: `StructureParseError syntax_error syntax error at line 1`.

## Limitations
Python only. Other languages need a real parser such as tree-sitter, a new dependency (and grammar downloads) that needs the owner's OK. A comment at the end of a class body joins the next top-level statement, because it is outside the class node. Chunk metadata is not yet stored in the index or embedded; `chunk()` gives offsets only.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
