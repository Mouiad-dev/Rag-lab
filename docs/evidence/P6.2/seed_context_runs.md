<!-- generated 2026-11-10 by `RAG_WRITE_EVIDENCE=1 uv run pytest tests/evaluation/test_p62_report.py`; offline, test-only lexical embedder, tiny fixture, not a benchmark -->
# P6.2 run A (sentence index): R02, R21, R21-cross

**Tiny offline fixture. Not a benchmark; no significance claim.**

- Run id: run-d34a9671ade60511
- Dataset: seed-fixture version 1, sha256 fc919a23b4c860a34ff8c6cf054b41a0cd01cd6e7e26fafa8b3dc454ea854db1
- Code version: 32e6edd64c8f339597b1744359f38cc53804a005-dirty
- Index fingerprint: ac9527f5acf4d5469a50f59ec90ace446b7ab545aa66dbfbcf35959fc5c92d0b
- ks: 1, 3, 5
- Coverage threshold: 0.8
- Splits run: dev, validation
- Techniques: R02 (R02 {'k': 5}), R21 (R21 {'max_seeds': 20, 'max_total_chars': 8000, 'max_window_chars': 2000, 'stay_in_paragraph': True, 'window': 1}), R21-cross (R21 {'max_seeds': 20, 'max_total_chars': 8000, 'max_window_chars': 2000, 'stay_in_paragraph': False, 'window': 1})

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

### R21

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

### R21-cross

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
| en-fact | dev | en | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| en-variant | dev | en | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| paraphrase | dev | en | paraphrase | R21 | 5 | 0.2000 | 1.0000 | 0.3869 | paired-fixture:500-555 | ok |
| id-exact | dev | en | identifier | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| en-club | dev | en | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:249-300 | ok |
| ar-fact | validation | ar | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-134 | ok |
| mixed | validation | mixed | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:888-953 | ok |
| unanswerable | validation | en | - | R21 | - | - | - | - | paired-fixture:454-498 | ok; returned_hits=5; top1_score=0.5000 |
| en-fact | dev | en | - | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-134 | ok |
| en-variant | dev | en | - | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-134 | ok |
| paraphrase | dev | en | paraphrase | R21-cross | 4 | 0.2500 | 1.0000 | 0.4307 | paired-fixture:398-595 | ok |
| id-exact | dev | en | identifier | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:769-953 | ok |
| en-club | dev | en | - | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:75-352 | ok |
| ar-fact | validation | ar | - | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| mixed | validation | mixed | - | R21-cross | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:849-953 | ok |
| unanswerable | validation | en | - | R21-cross | - | - | - | - | paired-fixture:398-595 | ok; returned_hits=3; top1_score=0.5000 |

For unanswerable cases the table shows returned_hits and top1_score. Scores are on different scales per technique and no abstention threshold is evaluated.

## Leakage check

pairs_checked: 31; jaccard threshold: 0.8

Findings: none

## Usage

| label | stage | calls | input_tokens | output_tokens | cost_usd |
|---|---|---|---|---|---|
| R02 | query_embedding | 8 | 0 | 0 | 0 |
| R21 | query_embedding | 8 | 0 | 0 | 0 |
| R21-cross | query_embedding | 8 | 0 | 0 | 0 |
| index build | document_embedding | 19 | 0 | 0 | 0 |

## Notes

- The R02 base retriever uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), not a semantic model.
- n is tiny: dev 5 answerable cases, validation 2 answerable and 1 unanswerable case. This is a tiny fixture, not a benchmark; no difference here is a measured win.
- Seed paragraphs are single sentences, so R21 with `stay_in_paragraph` can equal R02 (window 1 stays inside one paragraph).
- `R21-cross` only adds neighbour paragraphs (it lets the window cross paragraph boundaries).
- The base retriever config (R02, k=5) is fixed in the test factory and is NOT part of RunConfig; the run id does not cover it.
- Each run uses one index; the compared labels change one factor at a time. The held_out split was not run.
- The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when this file was generated.

### Context size (returned code points, `end - start` summed over hits)

| label | split | cases | hits | sum | mean per hit |
|---|---|---|---|---|---|
| R02 | dev | 5 | 25 | 1301 | 52.04 |
| R02 | validation | 3 | 15 | 860 | 57.33 |
| R21 | dev | 5 | 25 | 1301 | 52.04 |
| R21 | validation | 3 | 15 | 860 | 57.33 |
| R21-cross | dev | 5 | 17 | 2810 | 165.29 |
| R21-cross | validation | 3 | 11 | 1715 | 155.91 |

# P6.2 run B (C18 child index): R02, R22, R25, R25-min1

**Tiny offline fixture. Not a benchmark; no significance claim.**

- Run id: run-d33f239253af6536
- Dataset: seed-fixture version 1, sha256 fc919a23b4c860a34ff8c6cf054b41a0cd01cd6e7e26fafa8b3dc454ea854db1
- Code version: 32e6edd64c8f339597b1744359f38cc53804a005-dirty
- Index fingerprint: 5d0b33671a083fd0522923256e167a1b32c2b40986dc3b0e0710b22a1f4a6377
- ks: 1, 3, 5
- Coverage threshold: 0.8
- Splits run: dev, validation
- Techniques: R02 (R02 {'k': 5}), R22 (R22 {'max_parent_chars': 4000, 'max_seeds': 20, 'max_total_chars': 8000, 'on_missing_parent': 'error'}), R25 (R25 {'max_index_chunks': 5000, 'max_parent_chars': 4000, 'max_seeds': 20, 'max_total_chars': 8000, 'merge_ratio': 0.5, 'min_children': 2}), R25-min1 (R25 {'max_index_chunks': 5000, 'max_parent_chars': 4000, 'max_seeds': 20, 'max_total_chars': 8000, 'merge_ratio': 0.5, 'min_children': 1})

## Metric definitions

A relevant span counts as found at rank k when the union of the top-k hits covers at least the coverage threshold of its code points; hits from other documents are ignored. HitRate@k is 1 when at least one span is found within k, else 0. Recall@k is the share of labelled spans found within k. RR is 1 divided by the smallest rank at which any span is found (0 when none is found), and MRR is the mean RR. nDCG@k uses grade gains (2^grade - 1) and a log2(rank + 1) discount, against the ideal ordering of the labelled grades. Spans found at the same rank are credited once, at the highest grade, so when one chunk covers several labelled spans, nDCG cannot reach 1 even when Recall is 1. Unanswerable cases are excluded from every denominator and counted separately; averages are means over answerable cases only. A dash means the value is undefined (n=0 answerable).

## Summary per label

### R02

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 5 | 5 | 0 | 0 | 0.2000 | 0.6000 | 0.6000 | 0.2000 | 0.6000 | 0.6000 | 0.2000 | 0.4262 | 0.4262 | 0.3667 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.2000 | 0.6000 | 0.6000 | 0.2000 | 0.6000 | 0.6000 | 0.2000 | 0.4262 | 0.4262 | 0.3667 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.1934 | 0.1000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3869 | 0.2000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### R22

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

### R25

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

### R25-min1

| split | group | n_cases | n_answerable | n_unanswerable | n_errors | HitRate@1 | HitRate@3 | HitRate@5 | Recall@1 | Recall@3 | Recall@5 | nDCG@1 | nDCG@3 | nDCG@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | all | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 |
| dev | language:en | 5 | 5 | 0 | 0 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 |
| dev | language:ar | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | language:mixed | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:identifier | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dev | tag:paraphrase | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| dev | unanswerable | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | all | 3 | 2 | 1 | 0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| validation | language:en | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | language:ar | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | language:mixed | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| validation | tag:identifier | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:paraphrase | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:diacritics | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:table | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | tag:ocr | 0 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - | - |
| validation | unanswerable | 1 | 0 | 1 | 0 | - | - | - | - | - | - | - | - | - | - |

## Per case

| case | split | language | tags | label | found_ranks | RR | Recall@5 | nDCG@5 | top-1 | status |
|---|---|---|---|---|---|---|---|---|---|---|
| en-fact | dev | en | - | R02 | 3 | 0.3333 | 1.0000 | 0.5000 | paired-fixture:0-48 | ok |
| en-variant | dev | en | - | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:0-48 | ok |
| paraphrase | dev | en | paraphrase | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:500-547 | ok |
| id-exact | dev | en | identifier | R02 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:810-847 | ok |
| en-club | dev | en | - | R02 | 2 | 0.5000 | 1.0000 | 0.6309 | paired-fixture:283-300 | ok |
| ar-fact | validation | ar | - | R02 | 5 | 0.2000 | 1.0000 | 0.3869 | paired-fixture:49-98 | ok |
| mixed | validation | mixed | - | R02 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:898-946 | ok |
| unanswerable | validation | en | - | R02 | - | - | - | - | paired-fixture:454-498 | ok; returned_hits=5; top1_score=0.5000 |
| en-fact | dev | en | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| en-variant | dev | en | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| paraphrase | dev | en | paraphrase | R22 | 4 | 0.2500 | 1.0000 | 0.4307 | paired-fixture:500-681 | ok |
| id-exact | dev | en | identifier | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:683-847 | ok |
| en-club | dev | en | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:188-352 | ok |
| ar-fact | validation | ar | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| mixed | validation | mixed | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:849-953 | ok |
| unanswerable | validation | en | - | R22 | - | - | - | - | paired-fixture:354-498 | ok; returned_hits=4; top1_score=0.5000 |
| en-fact | dev | en | - | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| en-variant | dev | en | - | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| paraphrase | dev | en | paraphrase | R25 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:500-547 | ok |
| id-exact | dev | en | identifier | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:683-847 | ok |
| en-club | dev | en | - | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:188-352 | ok |
| ar-fact | validation | ar | - | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| mixed | validation | mixed | - | R25 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:898-946 | ok |
| unanswerable | validation | en | - | R25 | - | - | - | - | paired-fixture:454-498 | ok; returned_hits=4; top1_score=0.5000 |
| en-fact | dev | en | - | R25-min1 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| en-variant | dev | en | - | R25-min1 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| paraphrase | dev | en | paraphrase | R25-min1 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:500-547 | ok |
| id-exact | dev | en | identifier | R25-min1 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:683-847 | ok |
| en-club | dev | en | - | R25-min1 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:188-352 | ok |
| ar-fact | validation | ar | - | R25-min1 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| mixed | validation | mixed | - | R25-min1 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:898-946 | ok |
| unanswerable | validation | en | - | R25-min1 | - | - | - | - | paired-fixture:454-498 | ok; returned_hits=4; top1_score=0.5000 |

For unanswerable cases the table shows returned_hits and top1_score. Scores are on different scales per technique and no abstention threshold is evaluated.

## Leakage check

pairs_checked: 31; jaccard threshold: 0.8

Findings: none

## Usage

| label | stage | calls | input_tokens | output_tokens | cost_usd |
|---|---|---|---|---|---|
| R02 | query_embedding | 8 | 0 | 0 | 0 |
| R22 | query_embedding | 8 | 0 | 0 | 0 |
| R25 | query_embedding | 8 | 0 | 0 | 0 |
| R25-min1 | query_embedding | 8 | 0 | 0 | 0 |
| index build | document_embedding | 23 | 0 | 0 | 0 |

## Notes

- The R02 base retriever uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), not a semantic model.
- n is tiny: dev 5 answerable cases, validation 2 answerable and 1 unanswerable case. This is a tiny fixture, not a benchmark; no difference here is a measured win.
- The index holds C18 children (parent_size 200, child_size 50); R22 and R25 read the stored parent spans.
- R25-min1 equals R25 on this fixture: the two labels return the same hits and the same context sizes.
- The base retriever config (R02, k=5) is fixed in the test factory and is NOT part of RunConfig; the run id does not cover it.
- Each run uses one index; the compared labels change one factor at a time. The held_out split was not run.
- The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when this file was generated.

### Context size (returned code points, `end - start` summed over hits)

| label | split | cases | hits | sum | mean per hit |
|---|---|---|---|---|---|
| R02 | dev | 5 | 25 | 967 | 38.68 |
| R02 | validation | 3 | 15 | 633 | 42.20 |
| R22 | dev | 5 | 19 | 3060 | 161.05 |
| R22 | validation | 3 | 13 | 2007 | 154.38 |
| R25 | dev | 5 | 19 | 1460 | 76.84 |
| R25 | validation | 3 | 13 | 806 | 62.00 |
| R25-min1 | dev | 5 | 19 | 1460 | 76.84 |
| R25-min1 | validation | 3 | 13 | 806 | 62.00 |
