# C18: Hierarchical parent-child

## Intuition
Store broad parent passages and smaller child spans. A retrieved child can point to the larger passage that explains it.

## Mechanism
`ParentChildChunker` is in `packages/chunking/src/rag_foundry_chunking/hierarchy.py`. Parents are paragraph packs up to `parent_size` code points; children are sentence packs up to `child_size` inside one parent (a long sentence falls back to words, then safe cuts: `split_level` 0, 1, 2). `chunk()` returns the children as plain spans; `chunk_hierarchy()` returns both levels. Each parent gets a stable `parent_id` from its offsets and a hash of its text. With `respect_sections=True` and extraction blocks that carry a `section_path`, a parent never spans two sections. Without blocks (as in the demo below) the path is empty.

## Worked example
A short English + Arabic text. The child that holds the English fact points to its parent, and the parent shows the text around it.

## Runnable demo
```bash
PYTHONIOENCODING=utf-8 uv run python -c "from rag_foundry_chunking.hierarchy import ParentChildChunker, ParentChildConfig; from rag_foundry_techniques import ChunkInput; t='## Guide\n\nThe sensor records temperature.\n\n## دليل\n\nيقيس المستشعر درجة الحرارة.'; h=ParentChildChunker(ParentChildConfig(parent_size=200, child_size=50)).chunk_hierarchy(ChunkInput(text=t)); [print('parent',p.parent_id,p.start,p.end,p.path) for p in h.parents]; [print('child',c.start,c.end,c.parent_id) for c in h.children]; c=next(c for c in h.children if 'sensor' in t[c.start:c.end]); print('fact:',repr(t[c.start:c.end])); print('expanded parent:',repr(t[h.parent_of(c).start:h.parent_of(c).end]))"
```

Actual output:
```text
parent par_72e4f7d41939f0cd4d1aeb1c 0 79 ()
child 0 50 par_72e4f7d41939f0cd4d1aeb1c
child 52 79 par_72e4f7d41939f0cd4d1aeb1c
fact: '## Guide\n\nThe sensor records temperature.\n\n## دليل'
expanded parent: '## Guide\n\nThe sensor records temperature.\n\n## دليل\n\nيقيس المستشعر درجة الحرارة.'
```

## Failure case
A fact that crosses a parent boundary is split. Expanding to parents then needs both parents, so the context grows.

```bash
PYTHONIOENCODING=utf-8 uv run python -c "from rag_foundry_chunking.hierarchy import ParentChildChunker, ParentChildConfig; from rag_foundry_techniques import ChunkInput; t='The relevant identifier is blue ' + 'x '*64 + '\n\nlantern in the storage room. ' + 'y '*64; h=ParentChildChunker(ParentChildConfig(parent_size=200, child_size=60)).chunk_hierarchy(ChunkInput(text=t)); s=t.index('blue'); e=t.index('lantern')+7; ps=[p for p in h.parents if p.start<e and s<p.end]; print('split fact: blue ... lantern'); print('parents returned:', len(ps)); print('parent context chars:', sum(p.end-p.start for p in ps)); [print('parent', p.start, p.end, repr(t[p.start:p.start+40])) for p in ps]"
```

Actual output:
```text
split fact: blue ... lantern
parents returned: 2
parent context chars: 315
parent 0 159 'The relevant identifier is blue x x x x '
parent 162 318 'lantern in the storage room. y y y y y y'
```

The span report shows the same trade-off at a larger scale. On the 2586 code point fixture, `parent_size` 1600 gives only two parents, so parent expansion at k=3 returns the whole document (`context_chars@3` = 2583.0 in every cohort). Its `span_recall@3` of 1.0 for ar comes from that, not from better ranking: the plain C18 children give `span_recall@3` 0.0 for ar with 1130.0 context characters. With parent/child sizes 600/200, children have `span_recall@3` / `context_chars@3` of ar 0.0 / 395.0, en 0.5 / 435.2, and mixed 0.625 / 363.25; expansion changes these to ar 1.0 / 1530.0, en 1.0 / 1694.8, and mixed 0.75 / 1545.5. C07 paragraphs at 1600 have 1.0 / 2583.0 in all three cohorts, so the smaller C18 expansion uses less context at k=3 but does not match its recall in every cohort.

## Limitations
Parents are not yet stored in the index. Query-time parent return is R22/R25 (P6.2). `parent_id` is a chunk-level reference, not a source anchor. The span report uses a lexical ranker and one small fixture. At `parent_size` 600, expansion at k=3 still returns 1530.0-1694.8 context characters across cohorts, more than half of the 2586 code point fixture, so this comparison does not establish a ranking winner.

## Baseline comparison
[Span comparison](../../evidence/P4.5/span_comparison.md), rows "C18 parent 1600 child 400" and "+ parent expansion": span_recall@3 / context_chars@3 — children: ar 0.0 / 1130.0, en 1.0 / 1071.6, mixed 0.625 / 1136.0; parent expansion: ar 1.0 / 2583.0, en 1.0 / 2583.0, mixed 1.0 / 2583.0. Offline span metric only.

Additional baseline rows in the same format: "C07 paragraphs at 1600": ar 1.0 / 2583.0, en 1.0 / 2583.0, mixed 1.0 / 2583.0. "C18 parent 600 child 200": ar 0.0 / 395.0, en 0.5 / 435.2, mixed 0.625 / 363.25; its "+ parent expansion" row: ar 1.0 / 1530.0, en 1.0 / 1694.8, mixed 0.75 / 1545.5. Offline span metric only.

## Status
Draft.
