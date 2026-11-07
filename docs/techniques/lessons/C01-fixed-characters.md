# C01: Fixed characters

## Intuition
Give every chunk a fixed character budget. This is simple and predictable.

## Mechanism
`FixedCharChunker` is in `packages/chunking/src/rag_foundry_chunking/fixed.py`. `chunk_size` defaults to 800 (range 8 to 8000). It uses grapheme-like character units and trims boundary whitespace.

## Worked example
Input: `A short sentence has context.`

Output from the demo:
```text
[(0, 7, 'A short'), (8, 16, 'sentence'), (17, 24, 'has con'), (24, 29, 'text.')]
```

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import FixedCharChunker, FixedCharConfig; from rag_foundry_techniques import ChunkInput; t='A short sentence has context.'; print([(s.start,s.end,t[s.start:s.end]) for s in FixedCharChunker(FixedCharConfig(chunk_size=8)).chunk(ChunkInput(text=t))])"
```

## Failure case
The example cuts `sentence` from adjacent words, and cuts the word `context` into `con` and `text.`. A fixed character budget does not preserve word or sentence boundaries.

## Limitations
The configured budget counts code points grouped into combining clusters, not model tokens.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
