# C10: JSON structure

## Intuition
JSON already says what belongs together: object members and array elements. Split along those boundaries and keep each piece's JSONPath, so a retrieved value like `2` can be read back as `$['store']['books'][2]['price']`.

## Mechanism
`JsonStructureChunker` is in `packages/chunking/src/rag_foundry_chunking/json_structure.py`. An offset-tracking walker reads the canonical text with the stdlib decoder (`json.JSONDecoder.raw_decode` for keys and scalars, `json.loads` as the final validity check) and records, for every member and element, its exact code-point range and its normalized JSONPath (`$`, `['key']` with `'` and `\` escaped, `[0]`).

- A document that fits `chunk_size` is one chunk with path `$`.
- A larger container is split: its members/elements are packed in order within `chunk_size`; a member that is itself a large container is split the same way, deeper.
- Each chunk has `path` (first and last unit), `attrs` (`container_path`, `first_key`/`last_key` or `first_index`/`last_index`) and `container` context spans: the source key tokens (`"store":`, `"books":`) that lead to it, so `render()` shows the field names without inventing text.
- The key tokens of a split container are in the body of the next chunk (or in a short `part=keys` chunk when they do not fit with it), so no key text is only in context.
- A scalar larger than `chunk_size` (a long string) is flagged, never dropped: `keep` gives one `oversized` chunk; `split` gives `oversized` + `split_part` pieces with the same path.
- `json_path_at(text, result, offset)` returns the deepest JSONPath whose range contains `offset`.
- Invalid JSON, trailing text and `NaN`/`Infinity` raise `StructureParseError("invalid_json")`; nesting deeper than `max_depth` (default 64) raises `StructureParseError("too_deep")`.

## Worked example
Input: `{"store": {"name": "Shelf", "books": [4 books with title and price]}}`, `chunk_size=64`.

Real output (`path`, offsets, `render()`), then the path of the price value of book 2:
```text
("$['store']['name']",) (1, 26) '"store": {"name": "Shelf"'
("$['store']['books']",) (28, 36) '"store":\n"books":'
("$['store']['books'][0]", "$['store']['books'][1]") (38, 102) '"store":\n"books":\n{"title": "Book 0", "price": 0}, {"title": "Book 1", "price": 1}'
("$['store']['books'][2]", "$['store']['books'][3]") (104, 168) '"store":\n"books":\n{"title": "Book 2", "price": 2}, {"title": "Book 3", "price": 3}'
$['store']['books'][2]['price'] '2'
```
The second chunk is the `"books":` key on its own (`part=keys`), because it did not fit together with the first two books.

## Runnable demo
```bash
uv run python -c "import json; from rag_foundry_chunking import JsonStructureChunker as J, JsonStructureConfig as C, render, json_path_at; from rag_foundry_techniques import ChunkInput; t=json.dumps({'store': {'name': 'Shelf', 'books': [{'title': 'Book %d' % i, 'price': i} for i in range(4)]}}); r=J(C(chunk_size=64)).chunk_structured(ChunkInput(text=t)); [print(c.path, (c.start, c.end), repr(render(t, c))) for c in r.chunks]; o=t.index('\"price\": 2')+9; print(json_path_at(t, r, o), repr(t[o]))"
```

## Failure case
A long free-text field is one JSON value, so structure cannot help. `{"id": 7, "note": "<90 characters>"}` with `chunk_size=40` (real output: path, flags, offsets):
```text
("$['id']",) () (1, 8)
("$['note']",) ('oversized', 'split_part') (10, 48)
("$['note']",) ('oversized', 'split_part') (49, 88)
("$['note']",) ('oversized', 'split_part') (89, 110)
```
The pieces keep their path but are plain text cuts. Also, `{"a": NaN}` is refused (`StructureParseError invalid_json`) instead of being chunked as text.

## Limitations
One JSON document per view; JSON Lines are not supported. Structural punctuation (`{ } [ ] , :`) between units is outside chunk bodies. Chunk metadata (paths, context) is not yet stored in the index or embedded; `chunk()` gives offsets only. JSON files are ingested through the plain-text extractor today; the canonical text must still be valid JSON.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
