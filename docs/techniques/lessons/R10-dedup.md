# R10: Evidence deduplication

## Intuition

Retrieval often returns the same passage more than once: the same text in two views, overlapping spans, or a lightly edited copy. Deduplication removes repeats so the context budget is spent on distinct evidence. It must not remove a passage that disagrees on a figure.

## Mechanism

`EvidenceDedup` takes one R03 or R02 style result and compares each candidate with every hit already kept, in input order. Three rules run in this order: `exact_text` (SHA-256 of NFC text), `span_overlap` (same `extraction_view_id`, overlap of at least 0.5 of the shorter span) and `near_text` (Jaccard of word 3-gram shingles, at least 0.8). The first rule that matches drops the candidate. Each drop records the rule, the similarity, the view and span, and the kept hit that absorbed it. Cost is at most n(n-1)/2 pairs, so the input should stay small (about 50 hits).

## Config and defaults

`DedupConfig`: `rules` (all three), `near_threshold` 0.8, `shingle_size` 3, `span_overlap_min` 0.5, `protect_numbers` true, `k` none.

## Fidelity

Adaptation. The catalogue says "source/overlap or explicit semantic rule". This is the deterministic part only. There is no MinHash or locality-sensitive hashing: shingle sets are compared exactly, pair by pair. There is no semantic (embedding) rule. Number protection compares the ordered number-bearing tokens of the two texts; if they differ, `near_text` does not merge them, so "10 dollars" and "12 dollars" stay apart.

## Runnable demo

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p53_selection_demo.py
```

Full output: [selection_and_filters.md](../../evidence/P5.3/selection_and_filters.md). It is an offline fixture; no model; not a benchmark.

## Success case

Rows quoted exactly from the evidence file (hand list of 6, default config):

```text
| counts | 6 | 3 | 3 | 0.500 | 0.000 |
| c2 | exact_text | 1.000 | view-b [0:287] | c1 |
| c3 | span_overlap | 1.000 | view-a [10:50] | c1 |
| c4 | near_text | 0.979 | view-c [0:293] | c1 |
| c5 | view-d | [0:287] | The annual service plan costs 12 dollars per customer and th |
```

Three repeats are removed (redundancy 0.500 before, 0.000 after re-running on the output) and each drop keeps its provenance. The 12-dollar hit `c5` survives because of number protection. With `protect_numbers=False` the same list gives:

```text
| counts | 6 | 2 | 4 | 0.667 | 0.000 |
| c5 | near_text | 0.880 | view-d [0:287] | c1 |
```

## Losing case

A conflict stated in words, not numbers, is still merged:

```text
| counts | 2 | 1 | 1 | 0.500 | 0.000 |
| c2 | near_text | 0.812 | view-b [0:203] | c1 |
```

The two hits differ only in "approved" against "rejected". There is no number token to protect, and the shingle overlap (0.812) is above the 0.8 threshold, so the contradicting hit `c2` is dropped. Number protection is not conflict detection. Long near-identical passages are the risky case; raising `near_threshold` or turning `near_text` off trades this for more repeats.

## Limitations

Quadratic pair count; word shingles only (no stemming, no cross-language matching); number protection looks only at tokens containing a digit. `truncated` lists survivors cut by `k`.

## Status

Implemented; not validated or benchmarked.
