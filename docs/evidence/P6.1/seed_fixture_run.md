# P6.1 seed fixture run: R02, R03, R06

**Tiny offline fixture. Not a benchmark; no significance claim.**

- Run id: run-fb3a2edbae695746
- Dataset: seed-fixture version 1, sha256 fc919a23b4c860a34ff8c6cf054b41a0cd01cd6e7e26fafa8b3dc454ea854db1
- Code version: 47bcb6e8a58514997288cebed5419a5c394aa538-dirty
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
| dev | all | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 0.8774 | 0.8400 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 0.8774 | 0.8400 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3869 | 0.2000 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### R03

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 5 | 5 | 0 | 0 | 0.8000 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 0.8000 | 0.9262 | 0.9262 | 0.9000 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.8000 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 0.8000 | 0.9262 | 0.9262 | 0.9000 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6309 | 0.6309 | 0.5000 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### R06

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 0.8861 | 0.8500 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 | 0.8861 | 0.8500 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4307 | 0.2500 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

## Per case

| case | split | language | tags | label | found_ranks | RR | Recall@5 | nDCG@5 | top-1 | status |
|---|---|---|---|---|---|---|---|---|---|---|
| en-fact | dev | en | - | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| en-variant | dev | en | - | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| paraphrase | dev | en | paraphrase | R02 | 5 | 0.2000 | 1.0000 | 0.3869 | paired-fixture:500-555 | ok |
| id-exact | dev | en | identifier | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| en-club | dev | en | - | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:249-300 | ok |
| ar-fact | validation | ar | - | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| mixed | validation | mixed | - | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:888-953 | ok |
| unanswerable | validation | en | - | R02 | - | - | - | - | paired-fixture:454-498 | ok; returned_hits=5; top1_score=0.5000 |
| en-fact | dev | en | - | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| en-variant | dev | en | - | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| paraphrase | dev | en | paraphrase | R03 | 2 | 0.5000 | 1.0000 | 0.6309 | paired-fixture:500-555 | ok |
| id-exact | dev | en | identifier | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| en-club | dev | en | - | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:249-300 | ok |
| ar-fact | validation | ar | - | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| mixed | validation | mixed | - | R03 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:888-953 | ok |
| unanswerable | validation | en | - | R03 | - | - | - | - | paired-fixture:500-555 | ok; returned_hits=5; top1_score=3.2962 |
| en-fact | dev | en | - | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| en-variant | dev | en | - | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| paraphrase | dev | en | paraphrase | R06 | 4 | 0.2500 | 1.0000 | 0.4307 | paired-fixture:500-555 | ok |
| id-exact | dev | en | identifier | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| en-club | dev | en | - | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:249-300 | ok |
| ar-fact | validation | ar | - | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| mixed | validation | mixed | - | R06 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:888-953 | ok |
| unanswerable | validation | en | - | R06 | - | - | - | - | paired-fixture:500-555 | ok; returned_hits=5; top1_score=0.0325 |

For unanswerable cases the table shows returned_hits and top1_score. Scores are on different scales per technique and no abstention threshold is evaluated.

## Leakage check

pairs_checked: 31; jaccard threshold: 0.8

Findings: none

## Usage

| label | stage | calls | input_tokens | output_tokens | cost_usd |
|---|---|---|---|---|---|
| R02 | query_embedding | 8 | 0 | 0 | 0 |
| R03 | - | 0 | 0 | 0 | - |
| R06 | query_embedding | 8 | 0 | 0 | 0 |
| index build | document_embedding | 19 | 0 | 0 | 0 |

## Notes

- R02 here uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), not a semantic model.
- n is tiny: dev 5 answerable cases, validation 2 answerable and 1 unanswerable case.
- The table and OCR cohorts are absent (n=0).
- The 2 held_out cases were not run; held_out stays locked.
- R04 is absent (it needs PostgreSQL).
- The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when this file was generated, so the run id is not reproducible from that commit alone.
