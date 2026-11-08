# C08: Markdown sections

## Intuition
A Markdown heading starts a topic. Keep each heading section together when it fits, and remember the headings above it, so a retrieved chunk still says where it belongs ("Guide > Setup > Linux").

## Mechanism
`MarkdownSectionChunker` is in `packages/chunking/src/rag_foundry_chunking/markdown.py`. It scans the canonical text (the Markdown view keeps markup) for ATX headings (`#` to `######`) and fenced code (```` ``` ```` or `~~~`). A `#` line inside a fence is code, not a heading.

- A section runs from its heading line to the next heading line. `path` is the list of heading titles; `context` holds the source offsets of the ancestor heading lines, so `render()` can show them above the chunk without inventing text.
- A heading with no body of its own (for example `## Setup` directly followed by `### Linux`) is merged into its first child when the result fits `chunk_size`.
- A section larger than `chunk_size` is split into paragraphs and whole fences, packed again; a fence that fits is never cut. Later chunks of the section also get the section's own heading as context.
- A single paragraph or fence larger than `chunk_size` is flagged, never dropped: `oversized="keep"` keeps it whole with flag `oversized`; `oversized="split"` (default) cuts it at lines, then words, then safe cut points, and every piece gets `oversized` and `split_part`.

`chunk()` returns plain `ChunkSpan`s for the index; `chunk_structured()` returns `StructuredChunk`s with `path`, `context`, `attrs` and `flags`.

## Worked example
Input (`chunk_size=40`):
````text
# Guide

Intro.

## Setup
### Linux
Run make.

```sh
# not a heading
make
```
````

Real output (`path`, offsets, kind, `render()`):
```text
('Guide',) (0, 15) section '# Guide\n\nIntro.'
('Guide', 'Setup') (17, 25) section '# Guide\n## Setup'
('Guide', 'Setup', 'Linux') (26, 45) section '# Guide\n## Setup\n### Linux\nRun make.'
('Guide', 'Setup', 'Linux') (47, 77) code '# Guide\n## Setup\n### Linux\n```sh\n# not a heading\nmake\n```'
```
`## Setup` could not merge into `### Linux` because the Linux section (with its fence) is larger than 40, so it stays a short chunk of its own. The fence stays whole and `# not a heading` is not a heading.

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import MarkdownSectionChunker as M, MarkdownSectionConfig as C, render; from rag_foundry_techniques import ChunkInput; t='# Guide\n\nIntro.\n\n## Setup\n### Linux\nRun make.\n\n\`\`\`sh\n# not a heading\nmake\n\`\`\`'; [print(c.path, (c.start, c.end), c.kind, repr(render(t, c))) for c in M(C(chunk_size=40)).chunk_structured(ChunkInput(text=t)).chunks]"
```

## Failure case
Setext headings (a title underlined with `=====`) are not recognised, so the whole document is one section with an empty path:
```text
[((), 'Guide\n=====\n\nSetext heading text.')]
```
Very short sections also give very short chunks (`'# Guide\n## Setup'` above), which carry little meaning on their own.

## Limitations
A CommonMark subset: ATX headings and fences only (same rules as the Markdown extractor). Lists, block quotes, tables and setext headings are body text. Section paths and context are not yet stored in the index or used for embedding; `chunk()` gives offsets only.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
