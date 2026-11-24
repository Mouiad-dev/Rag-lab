# C09: HTML structure

## Intuition
A web page has sections (headings) and noise (menus repeated in the header and footer). Chunk by section, keep the locators that say where each chunk came from (section path and block ids), and leave out repeated menu copies without losing their text or identity.

## Mechanism
`HtmlStructureChunker` is in `packages/chunking/src/rag_foundry_chunking/html_structure.py`. The HTML view's canonical text has no tags, so C09 works on the extraction blocks: since P4.3-03, `ChunkInput.blocks` (`TextBlock`) carries the stored block `kind`, `section_path`, `heading_level` and `page_index`, and the index service passes them through. Without kinds, C09 refuses: `StructureParseError("missing_structure")`.

- A heading block starts a new section; consecutive blocks with the same `section_path` form one section. Blocks are whole units packed within `chunk_size`, never across sections.
- Each chunk has `path` (the section path), `block_ids` (its locators), and `heading` context spans for the ancestor headings (plus its own heading for later chunks of a long section), taken from source offsets.
- Repeated boilerplate: a short non-heading block (`boilerplate_max_chars`, default 120) whose whitespace-collapsed text occurs in at least `boilerplate_min_repeats` blocks (default 2) is kept at its **first** place; later copies are left out of chunk bodies and reported in `StructureResult.skipped` with block id, offsets and reason `repeated_boilerplate`. `drop_repeated_boilerplate=False` turns this off.
- A block larger than `chunk_size` is flagged, never dropped (`keep`: `oversized`; `split`: `oversized` + `split_part`).

## Worked example
Input (through the real HTML extractor, `chunk_size=80`):
```html
<header><ul><li>Home</li><li>Docs</li></ul></header>
<h1>Guide</h1><h2>Install</h2><p>Run the installer.</p>
<h2>Use</h2><p>Open the app.</p>
<footer><ul><li>Home</li><li>Docs</li></ul></footer>
```

Real output (`path`, `block_ids`, body, context), then the skipped report:
```text
() ('b00000', 'b00001') 'Home\n\nDocs' []
('Guide',) ('b00002',) 'Guide' []
('Guide', 'Install') ('b00003', 'b00004') 'Install\n\nRun the installer.' ['Guide']
('Guide', 'Use') ('b00005', 'b00006') 'Use\n\nOpen the app.' ['Guide']
skipped [('b00007', 'repeated_boilerplate', 'Home'), ('b00008', 'repeated_boilerplate', 'Docs')]
```
The footer copy of the menu is not indexed again, but its blocks and offsets are still reported, so nothing disappears silently.

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import HtmlStructureChunker as H, HtmlStructureConfig as C; from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor; from rag_foundry_techniques import ChunkInput, TextBlock; m='<ul><li>Home</li><li>Docs</li></ul>'; x=get_extractor('text/html').extract(('<header>'+m+'</header><h1>Guide</h1><h2>Install</h2><p>Run the installer.</p><h2>Use</h2><p>Open the app.</p><footer>'+m+'</footer>').encode(), text_encoding=None, limits=ExtractionLimits()); t=x.canonical_text; b=tuple(TextBlock(block_id=k.block_id, start=k.start, end=k.end, kind=k.kind.value, section_path=k.section_path, heading_level=k.heading_level) for k in x.blocks); r=H(C(chunk_size=80)).chunk_structured(ChunkInput(text=t, blocks=b)); [print(c.path, c.block_ids, repr(t[c.start:c.end]), [t[s.start:s.end] for s in c.context]) for c in r.chunks]; print('skipped', [(s.block_id, s.reason, t[s.start:s.end]) for s in r.skipped])"
```

## Failure case
A menu that appears only once is not noise to this heuristic. `<nav><ul><li>Home</li><li>Docs</li></ul></nav><h1>Guide</h1><p>Only one menu.</p>` (real output):
```text
() ('b00000', 'b00001') 'Home\n\nDocs' []
('Guide',) ('b00002', 'b00003') 'Guide\n\nOnly one menu.' []
skipped []
```
The `<nav>` is indexed as content. Also, a heading with no text of its own (`Guide` above) becomes a tiny chunk.

## Limitations
Boilerplate detection is a repeated-block heuristic, not landmark (`<nav>`, `role=navigation`) detection; that needs the HTML extractor to record landmarks in the view (an extractor version change, later step). HTML tables are flattened to cell blocks by the current extractor, so C12 does not see HTML table rows. Skipped ranges and chunk metadata are not yet stored in the index; `chunk()` gives offsets only, and the index service does not yet read `skipped`.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
