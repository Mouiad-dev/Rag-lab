# P6.4 cross-language fixture run: R02, R03, R06

**Tiny offline fixture. Not a benchmark; no significance claim.**

- Run id: run-20e5f206b628026c
- Dataset: cross-language-fixture version 1, sha256 60ca2db35d9dbbbca6278c897059baf6ed6909e24b0eb15fea84a578c3d53708
- Code version: 94d6af410bf98bef279efe60035deac909c7945a-dirty
- Index fingerprint: ac9527f5acf4d5469a50f59ec90ace446b7ab545aa66dbfbcf35959fc5c92d0b
- ks: 1, 3, 5
- Coverage threshold: 0.8
- Splits run: dev, validation
- Techniques: R02 (R02 {'k': 5}), R03 (R03 {'b': 0.75, 'k': 5, 'k1': 1.2, 'max_chunks': 20000}), R06 (R06 {'branch_k': 10, 'fts_operator': 'or', 'fusion': 'R07', 'k': 5, 'lexical': 'R03', 'lexical_weight': 0.5, 'rrf_k': 60})

## Metric definitions

A relevant span counts as found at rank k when the union of the top-k hits covers at least the coverage threshold of its code points; hits from other documents are ignored. HitRate@k is 1 when at least one span is found within k, else 0. Recall@k is the share of labelled spans found within k. RR is 1 divided by the smallest rank at which any span is found (0 when none is found), and MRR is the mean RR. nDCG@k uses grade gains (2^grade - 1) and a log2(rank + 1) discount, against the ideal ordering of the labelled grades. Spans found at the same rank are credited once, at the highest grade, so when one chunk covers several labelled spans, nDCG cannot reach 1 even when Recall is 1. Unanswerable cases are excluded from every denominator and counted separately; averages are means over answerable cases only. A dash means the value is undefined (n=0 answerable).

## Summary per label

### R02

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 9 | 7 | 2 | 0 | 0.4286 | 0.5714 | 0.5714 | 0.4286 | 0.5714 | 0.5714 | 0.4286 | 0.5187 | 0.5187 | 0.5000 |
| dev | language:en | 4 | 3 | 1 | 0 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| dev | language:ar | 4 | 3 | 1 | 0 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.5436 | 0.5436 | 0.5000 |
| dev | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 2 | 0 | 2 | 0 | - | - | - | - | - | - | - | - | - | - |

### R03

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 9 | 7 | 2 | 0 | 0.4286 | 0.7143 | 0.7143 | 0.4286 | 0.7143 | 0.7143 | 0.4286 | 0.5901 | 0.5901 | 0.5476 |
| dev | language:en | 4 | 3 | 1 | 0 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| dev | language:ar | 4 | 3 | 1 | 0 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.5436 | 0.5436 | 0.5000 |
| dev | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.5000 | 0.3333 |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 2 | 0 | 2 | 0 | - | - | - | - | - | - | - | - | - | - |

### R06

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 9 | 7 | 2 | 0 | 0.4286 | 0.5714 | 0.5714 | 0.4286 | 0.5714 | 0.5714 | 0.4286 | 0.5187 | 0.5187 | 0.5000 |
| dev | language:en | 4 | 3 | 1 | 0 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| dev | language:ar | 4 | 3 | 1 | 0 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.3333 | 0.5436 | 0.5436 | 0.5000 |
| dev | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 2 | 0 | 2 | 0 | - | - | - | - | - | - | - | - | - | - |

## Per case

| case | split | language | tags | label | found_ranks | RR | Recall@5 | nDCG@5 | top-1 | status |
|---|---|---|---|---|---|---|---|---|---|---|
| xl-en-en | dev | en | name | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| xl-ar-ar | dev | ar | name | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| xl-en-ar | dev | en | name,number | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:888-953 | ok |
| xl-ar-en | dev | ar | number | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:557-595 | ok |
| xl-en-ar-id | dev | en | identifier | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| xl-mixed-ar | dev | mixed | number | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:888-953 | ok |
| xl-ar-mixed | dev | ar | - | R02 | 2 | 0.5000 | 1.0000 | 0.6309 | paired-fixture:75-134 | ok |
| xl-en-neg | dev | en | negation | R02 | - | - | - | - | paired-fixture:888-953 | ok; returned_hits=5; top1_score=0.4811 |
| xl-ar-neg | dev | ar | negation | R02 | - | - | - | - | paired-fixture:75-134 | ok; returned_hits=5; top1_score=0.4243 |
| xl-en-en | dev | en | name | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| xl-ar-ar | dev | ar | name | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| xl-en-ar | dev | en | name,number | R03 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:888-953 | ok |
| xl-ar-en | dev | ar | number | R03 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:557-595 | ok |
| xl-en-ar-id | dev | en | identifier | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| xl-mixed-ar | dev | mixed | number | R03 | 3 | 0.3333 | 1.0000 | 0.5000 | paired-fixture:888-953 | ok |
| xl-ar-mixed | dev | ar | - | R03 | 2 | 0.5000 | 1.0000 | 0.6309 | paired-fixture:75-134 | ok |
| xl-en-neg | dev | en | negation | R03 | - | - | - | - | paired-fixture:888-953 | ok; returned_hits=2; top1_score=12.5297 |
| xl-ar-neg | dev | ar | negation | R03 | - | - | - | - | paired-fixture:75-134 | ok; returned_hits=2; top1_score=4.4089 |
| xl-en-en | dev | en | name | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| xl-ar-ar | dev | ar | name | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| xl-en-ar | dev | en | name,number | R06 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:888-953 | ok |
| xl-ar-en | dev | ar | number | R06 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:557-595 | ok |
| xl-en-ar-id | dev | en | identifier | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| xl-mixed-ar | dev | mixed | number | R06 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:888-953 | ok |
| xl-ar-mixed | dev | ar | - | R06 | 2 | 0.5000 | 1.0000 | 0.6309 | paired-fixture:75-134 | ok |
| xl-en-neg | dev | en | negation | R06 | - | - | - | - | paired-fixture:888-953 | ok; returned_hits=5; top1_score=0.0328 |
| xl-ar-neg | dev | ar | negation | R06 | - | - | - | - | paired-fixture:75-134 | ok; returned_hits=5; top1_score=0.0328 |

For unanswerable cases the table shows returned_hits and top1_score. Scores are on different scales per technique and no abstention threshold is evaluated.

## Leakage check

pairs_checked: 0; jaccard threshold: 0.8

Findings: none

## Usage

| label | stage | calls | input_tokens | output_tokens | cost_usd |
|---|---|---|---|---|---|
| R02 | query_embedding | 9 | 0 | 0 | 0 |
| R03 | - | 0 | 0 | 0 | - |
| R06 | query_embedding | 9 | 0 | 0 | 0 |
| index build | document_embedding | 19 | 0 | 0 | 0 |

## Notes

- R02 uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), so it can only match shared tokens; no semantic cross-language match is possible offline.
- Expect cross-script cases to fail, except where a token is shared (AB-1234, or the Arabic word in the mixed question).
- A shared token only makes a hit possible, not certain: in the mixed question the shared English words can pull the English-heavy MIXED paragraph above the Arabic one. Read the per-case table for what each technique found.
- The document language is script-detected, and Latin script is labelled en.
- n is tiny and all 9 cases are dev-only; no tuning is done on this fixture. It is not a benchmark.
- The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when this file was generated, so the run id is not reproducible from that commit alone.
- The real cross-language check needs a multilingual embedder: the live bge-m3 run in P6.4-04.

## Language pairs

A pair is the question language -> the document language. The document language is the script of the relevant span quotes (>= 80% Arabic letters is ar, <= 20% is en, else mixed; any Latin script counts as en); unanswerable cases map to none.

### R02

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | ar->mixed | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| dev | ar->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| dev | en->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | mixed->ar | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### R03

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

### R06

| split | pair | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | ar->ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | ar->en | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | ar->mixed | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 |
| dev | ar->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | en->ar | 2 | 2 | 0 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| dev | en->en | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | en->none | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - |
| dev | mixed->ar | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Per case

| case | label | pair | tags | found_ranks | RR |
|---|---|---|---|---|---|
| xl-en-en | R02 | en->en | name | 1 | 1.0000 |
| xl-ar-ar | R02 | ar->ar | name | 1 | 1.0000 |
| xl-en-ar | R02 | en->ar | name,number | - | 0.0000 |
| xl-ar-en | R02 | ar->en | number | - | 0.0000 |
| xl-en-ar-id | R02 | en->ar | identifier | 1 | 1.0000 |
| xl-mixed-ar | R02 | mixed->ar | number | - | 0.0000 |
| xl-ar-mixed | R02 | ar->mixed | - | 2 | 0.5000 |
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
| xl-ar-en | R06 | ar->en | number | - | 0.0000 |
| xl-en-ar-id | R06 | en->ar | identifier | 1 | 1.0000 |
| xl-mixed-ar | R06 | mixed->ar | number | - | 0.0000 |
| xl-ar-mixed | R06 | ar->mixed | - | 2 | 0.5000 |
| xl-en-neg | R06 | en->none | negation | - | - |
| xl-ar-neg | R06 | ar->none | negation | - | - |
