# P6.4 R42 cross-language live run (bge-m3)

One run, not an evaluation. Ollama embeddings are not guaranteed bit-identical across runs; this live run is not a frozen reproducible result. The dataset is the tiny dev-only `cross-language-fixture` v1 (9 cases); it is not a benchmark. Compare with the offline lexical run in [cross_language_pairs.md](cross_language_pairs.md).

- Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_cross_language_live.py -m live -v -s`
- Date: 2026-11-12 (about 11:53 UTC), working tree on commit 9fc23d5 plus the uncommitted P6.4-04 files (the code version in the run id is `-dirty`)
- Models: R02 dense with `bge-m3` (Ollama, local, cost 0); R03 BM25 (no embedder; seed config k=5); R06 hybrid whose dense branch also uses `bge-m3` (seed config k=5, branch_k=10)
- Result: 1 passed in 33.02s

Observed in this one run (read from the tables below): with bge-m3, R02 found the relevant span at rank 1 for every answerable case, including both en->ar cases and ar->en; R03 (BM25) still missed the cross-script cases xl-en-ar and xl-ar-en (likely because question and span share no token); R06 missed xl-en-ar and found xl-ar-en at rank 3. With n=1 or 2 per pair, these are observations, not rates.

## Printed output

Markdown headings demoted one level to fit this file; text otherwise as printed.

### Language pairs

A pair is the question language -> the document language. The document language is the script of the relevant span quotes (>= 80% Arabic letters is ar, <= 20% is en, else mixed; any Latin script counts as en); unanswerable cases map to none.

#### R02

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->mixed | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | en->ar | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | mixed->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### R03

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | ar->mixed | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| dev | ar->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| dev | en->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | mixed->ar | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.3333 |

#### R06

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.3333 |
| dev | ar->mixed | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| dev | ar->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| dev | en->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | mixed->ar | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |

#### Per case

| case | label | pair | tags | found_ranks | RR |
|---|---|---|---|---|---|
| xl-en-en | R02 | en->en | name | 1 | 1.0000 |
| xl-ar-ar | R02 | ar->ar | name | 1 | 1.0000 |
| xl-en-ar | R02 | en->ar | name,number | 1 | 1.0000 |
| xl-ar-en | R02 | ar->en | number | 1 | 1.0000 |
| xl-en-ar-id | R02 | en->ar | identifier | 1 | 1.0000 |
| xl-mixed-ar | R02 | mixed->ar | number | 1 | 1.0000 |
| xl-ar-mixed | R02 | ar->mixed | - | 1 | 1.0000 |
| xl-en-neg | R02 | en->none | negation | - | - |
| xl-ar-neg | R02 | ar->none | negation | - | - |
| xl-en-en | R03 | en->en | name | 1 | 1.0000 |
| xl-ar-ar | R03 | ar->ar | name | 1 | 1.0000 |
| xl-en-ar | R03 | en->ar | name,number | - | 0.0000 |
| xl-ar-en | R03 | ar->en | number | - | 0.0000 |
| xl-en-ar-id | R03 | en->ar | identifier | 1 | 1.0000 |
| xl-mixed-ar | R03 | mixed->ar | number | 3 | 0.3333 |
| xl-ar-mixed | R03 | ar->mixed | - | 2 | 0.5000 |
| xl-en-neg | R03 | en->none | negation | - | - |
| xl-ar-neg | R03 | ar->none | negation | - | - |
| xl-en-en | R06 | en->en | name | 1 | 1.0000 |
| xl-ar-ar | R06 | ar->ar | name | 1 | 1.0000 |
| xl-en-ar | R06 | en->ar | name,number | - | 0.0000 |
| xl-ar-en | R06 | ar->en | number | 3 | 0.3333 |
| xl-en-ar-id | R06 | en->ar | identifier | 1 | 1.0000 |
| xl-mixed-ar | R06 | mixed->ar | number | 2 | 0.5000 |
| xl-ar-mixed | R06 | ar->mixed | - | 2 | 0.5000 |
| xl-en-neg | R06 | en->none | negation | - | - |
| xl-ar-neg | R06 | ar->none | negation | - | - |

run_id=run-fb5a4e786067e4d7
index_build_usage=calls=19 input_tokens=303 output_tokens=0 cache_read_input_tokens=0 cache_write_input_tokens=0 cost_usd=Decimal('0') unknown_cost_calls=0 estimated_token_calls=0
