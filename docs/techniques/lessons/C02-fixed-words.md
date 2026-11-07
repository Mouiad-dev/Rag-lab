# C02: Fixed words

## Intuition
Pack a configured number of whitespace-separated words into each chunk.

## Mechanism
`FixedWordChunker` is in `packages/chunking/src/rag_foundry_chunking/fixed.py`. `words_per_chunk` defaults to 150 (range 1 to 2000). Punctuation stays attached because whitespace defines word boundaries.

## Worked example
Input: `A short sentence has context.`

Output from the demo:
```text
[(0, 7, 'A short'), (8, 20, 'sentence has'), (21, 29, 'context.')]
```

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import FixedWordChunker, FixedWordConfig; from rag_foundry_techniques import ChunkInput; t='A short sentence has context.'; print([(s.start,s.end,t[s.start:s.end]) for s in FixedWordChunker(FixedWordConfig(words_per_chunk=2)).chunk(ChunkInput(text=t))])"
```

Arabic and mixed-script example (`words_per_chunk=2`), real output:
```text
[(0, 10, 'مرحبا بكم،'), (11, 18, 'هل أنتم'), (19, 30, 'بخير؟ RAGهو'), (31, 35, 'test')]
```
The Arabic comma `،` and question mark `؟` stay attached to their word, and `RAGهو` (no space between scripts) counts as one word.
Demo: `PYTHONIOENCODING=utf-8 uv run python -c "from rag_foundry_chunking import FixedWordChunker, FixedWordConfig; from rag_foundry_techniques import ChunkInput; t='مرحبا بكم، هل أنتم بخير؟ RAGهو test'; print([(s.start,s.end,t[s.start:s.end]) for s in FixedWordChunker(FixedWordConfig(words_per_chunk=2)).chunk(ChunkInput(text=t))])"`

## Failure case
Text without spaces, such as `averylongunbrokenstring`, is one huge word. Punctuation also stays attached to the word, including punctuation that may matter for downstream processing.

## Limitations
Whitespace splitting is language-agnostic and is not a linguistic tokenizer.

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
