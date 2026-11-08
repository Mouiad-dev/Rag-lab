# C04: Overlap and sliding windows

## Intuition
Repeat a small part of each chunk in its neighbor so facts near a boundary can remain together.

## Mechanism
`SlidingWindowChunker` is in `packages/chunking/src/rag_foundry_chunking/sliding.py`. `unit` defaults to `words`; `window_size` defaults to 200 (range 2 to 8000); `overlap` defaults to 50 and cannot exceed half the window. Character windows require at least 8 characters. Token windows use `lab-regex-v1` by default.

## Worked example
Input: `A short sentence has useful context here.`

Output from the demo:
```text
[(0, 20, 'A short sentence has'), (8, 35, 'sentence has useful context'), (21, 41, 'useful context here.')]
WindowStats(chunks=3, stored_code_points=67, unique_code_points=41, duplicated_code_points=26, storage_ratio=1.6341463414634145)
```

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import SlidingWindowChunker, SlidingWindowConfig, window_stats; from rag_foundry_techniques import ChunkInput; t='A short sentence has useful context here.'; s=SlidingWindowChunker(SlidingWindowConfig(unit='words',window_size=4,overlap=2)).chunk(ChunkInput(text=t)); print([(x.start,x.end,t[x.start:x.end]) for x in s]); print(window_stats(s))"
```

## Failure case
Overlap duplicates stored text and creates more chunks to embed. Here 41 unique code points become 67 stored code points across three chunks, a storage ratio of 1.6341463414634145.

## Limitations
Overlap does not guarantee retrieval quality and increases embedding and storage work. Token counts use the lab tokenizer.

## Baseline comparison
See [span comparison](../../evidence/P4.5/span_comparison.md): span_recall@3 / context_chars@3 — ar 0.0 / 1200.0, en 0.7 / 1072.2, mixed 0.625 / 1146.5.

## Status
Draft.
