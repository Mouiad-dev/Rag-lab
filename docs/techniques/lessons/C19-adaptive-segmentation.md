# C19: Adaptive segmentation

## Intuition
One document can hold prose, lists, tables, code and mixed Arabic/English text. One fixed chunker treats them all the same. C19 looks at each region first and picks a segmentation policy for it.

## Mechanism
`AdaptiveChunker` is in `packages/chunking/src/rag_foundry_chunking/adaptive.py`. It is synchronous and makes no model calls (`router="rules"` is the only router in this version).

1. Regions: extraction blocks (`use_blocks=True`) or blank-line paragraphs; neighbouring regions of the same kind are merged.
2. Each region gets features (`list_line_share`, `table_line_share`, `code_line_share`, `arabic_share`, `latin_share`, lines, chars) and one kind by explicit rules in this order: table, code, list, mixed script, prose.
3. The kind maps to a policy (default: prose C06, mixed script C21, list C07, table C07, code C05). The policy chunks only that region; its spans are shifted back to document offsets, so no chunk crosses a region.
4. If a policy raises or returns a span outside its region, the region is marked `failed=True` with the error and `fallback_policy` (C05) is used.
5. `segment(doc)` returns spans plus one `RegionDecision` per region (kind, policy, reason, features, failure). `compare_with_fixed(doc, config, "C05")` counts how many fixed-policy chunks cross a region boundary.

## Worked example
Real output of the demo below (`chunk_size=64`):

```text
kind | policy | start | end | chunks | reason
prose | C06 | 0 | 42 | 1 | no structural or mixed-script rule matched
list | C07 | 44 | 70 | 1 | at least half of lines are list items
table | C07 | 72 | 118 | 1 | at least half of lines have table separators
code | C05 | 124 | 147 | 1 | at least half of lines look like code
mixed_script | C21 | 149 | 168 | 1 | Arabic and Latin letters both meet the minimum share
{'adaptive_chunks': 5, 'fixed_chunks': 4, 'fixed_policy': 'C05', 'fixed_cross_region': 1, 'failures': 0, 'oversized_adaptive': 0, 'oversized_fixed': 0}
fixed C05 chunks: ['English prose paragraph. Another sentence.', '- first item\n- second item', 'Name | Value | Extra\n--- | --- | ---\na | b | c', 'def run():\n    return 1\n\nEnglish كلمات عربي.']
```

The fixed C05 policy puts the code and the Arabic/English paragraph into one chunk (`fixed_cross_region=1`); C19 keeps them apart.

## Runnable demo
```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/c19_adaptive_demo.py
```

## Failure case
The code region starts at 124, not at 120 where the line `    def run():` starts: regions are trimmed of edge whitespace, so the first code line loses its indentation inside the chunk (`'def run():\n    return 1'`). For Python this changes how the chunk reads. The rules are also hand-set: one pipe per line is not a table, and a list where only 1 of 3 lines has a marker is prose.

## Limitations
- Thresholds (0.5 line shares, `mixed_min_share=0.2`) are hand-set, not learned or evaluated.
- No model router yet; the catalogue entry sits in the model-based family, but this version routes by rules only.
- Tables and code go to generic policies (C07, C05) until the P4.3 structure-aware chunkers (C08-C14) can be routed to.
- After merging same-kind regions, the features are recomputed on the merged text but the kind is kept.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
