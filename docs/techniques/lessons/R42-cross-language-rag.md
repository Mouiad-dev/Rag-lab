# R42: Cross-language RAG

## Intuition

A user may ask in English about an Arabic document, or the reverse. R42 is an evaluation cohort, not a retriever: it reports retrieval metrics for each question-language to document-language pair separately, so a weak pair is not hidden inside an average.

## Language pairs

A pair is the question language -> the document language, for example `en->ar`. Cases whose relevant span is absent (unanswerable) map to `none`.

## Script detection

The document language is the script of the relevant span quotes. Only Arabic and Latin letters are counted: if Arabic letters are at least 0.8 of Arabic+Latin letters it is `ar`, at most 0.2 is `en`, anything between is `mixed`, and text with neither script is `unknown`. Any Latin script counts as `en`. This is script detection, not language identification: French or German text would also be labelled `en`.

## Fixture

`cross-language-fixture` version 1: 9 cases, dev split only. No tuning is done on it, and it is not a benchmark.

## Offline rows (quoted exactly)

From the "Language pairs" section of `docs/evidence/P6.4/cross_language_pairs.md` (the file also has a summary table with the same `### R02` headings). R02 uses the test-only hashed bag-of-words `LexicalTestEmbedder`; R03 is BM25.

```text
### R02

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

### R03

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
```

## Live rows (quoted exactly)

One run, not an evaluation, from `docs/evidence/P6.4/cross_language_live.md`. Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_cross_language_live.py -m live -v -s`, 2026-11-12 (about 11:53 UTC), Ollama local, cost 0. R02 dense used `bge-m3`. Result: 1 passed in 33.02s. Ollama embeddings are not guaranteed bit-identical across runs, so this is not a frozen reproducible result.

```text
#### R02

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->ar | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### R03

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
```

The evidence file also has R06 rows (its dense branch used `bge-m3` too) and a per-case table.

## Honest reading

- The offline lexical runs can only match shared tokens (`AB-1234`, a shared Arabic word). The losing rows above (`ar->en` 0.0000 and half of `en->ar` missed) are the expected result of that.
- A real cross-script match needs a multilingual embedder. In the one live run, R02 with `bge-m3` found the relevant span at rank 1 for every answerable case, while R03 (BM25) still missed `xl-en-ar` and `xl-ar-en`, likely because question and span share no token.
- n is tiny (1 or 2 cases per pair) and the fixture is dev-only. These are observations, not rates; no significance is claimed.

## Cohorts

Names, numbers and negation appear as tags in the per-case table of the evidence file. The two negation cases are unanswerable, so they test abstention signals (returned hits and top-1 score, with no threshold evaluated), not negation handling.

## Fidelity

Evaluation cohort and metrics only. R42 adds no retrieval or translation step.

## Limitations

Per-span declared document language (dataset v2) is not implemented, so the document language comes from script detection. The fixture has 9 cases, dev only, and the live run was not repeated.

## Status

Implemented; not validated or benchmarked.
