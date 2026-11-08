# C03: Fixed tokens

## Intuition
Keep chunks within a selected tokenizer's token budget.

## Mechanism
`FixedTokenChunker` is in `packages/chunking/src/rag_foundry_chunking/fixed.py`. `tokens_per_chunk` defaults to 256 (range 8 to 8192), and `tokenizer` defaults to `lab-regex-v1`.

## Worked example
Input: `A short sentence has context.`

Output from the demo:
```text
[(0, 29, 'A short sentence has context.')]
```

Characters vs tokens with `lab-regex-v1` (real output):

| Text | Code points | Tokens |
|---|---|---|
| `A short sentence has context.` | 29 | 6 |
| `كَتَبَ الطَّالِبُ الدَّرْسَ.` (with harakat) | 28 | 4 |
| `Year 2026, page 12.` | 19 | 7 (`2026` becomes `202`, `6`) |

Harakat stay inside their word token, so the Arabic line has many code points but few tokens. A real model tokenizer would give different counts.
Demo: `PYTHONIOENCODING=utf-8 uv run python -c "from rag_foundry_chunking import LabRegexTokenizer; tk=LabRegexTokenizer(); [print(len(t), len(tk.token_spans(t)), [t[a:b] for a,b in tk.token_spans(t)]) for t in ('A short sentence has context.', 'كَتَبَ الطَّالِبُ الدَّرْسَ.', 'Year 2026, page 12.')]"`

## Runnable demo
```bash
uv run python -c "from rag_foundry_chunking import FixedTokenChunker, FixedTokenConfig; from rag_foundry_techniques import ChunkInput; t='A short sentence has context.'; print([(s.start,s.end,t[s.start:s.end]) for s in FixedTokenChunker(FixedTokenConfig(tokens_per_chunk=8)).chunk(ChunkInput(text=t))])"
```

## Failure case
`lab-regex-v1` is not a model tokenizer, so its counts can differ from the embedding model's counts.

## Limitations
Use the tokenizer expected by the downstream model before relying on token limits.

## Baseline comparison
See [span comparison](../../evidence/P4.5/span_comparison.md): span_recall@3 / context_chars@3 — ar 1.0 / 1481.0, en 0.8 / 1903.4, mixed 0.625 / 1689.5.

## Status
Draft.
