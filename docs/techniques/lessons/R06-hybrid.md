# R06: Hybrid retrieval

## Intuition

Hybrid retrieval runs a lexical branch and a dense branch, then fuses their ranked candidate lists. R03 is the offline lexical option; R04 uses PostgreSQL full-text search.

## Mechanism

Both branches use the same ready index version and scope, and they run one after the other. The implementation fails explicitly if either branch errors; it does not silently return a partial fusion. RRF (R07) is the default fusion and R08 weighted min-max is optional. Every fused hit keeps each branch's own components with their source and rank.

## Runnable demo

The script builds a four-chunk in-memory index with the test-only `LexicalTestEmbedder` (not a semantic model) and asks for the part number `AB-1234`.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/r06_hybrid_demo.py
```

Actual output:

```text
scoring: hybrid[R03+R02]/R07:rrf-k60
1 0.032787 'رقم القطعة AB-1234' [('R03', 1, 0.016393), ('R02', 1, 0.016393)]
2 0.032258 'رقم القطعة AB-1243' [('R03', 2, 0.016129), ('R02', 2, 0.016129)]
3 0.015873 'chess club meets on tuesday' [('R02', 3, 0.015873)]
4 0.015625 'the tidal observatory records sea levels' [('R02', 4, 0.015625)]
losing case: query '?!' -> no_query_terms (no lexical terms; the dense branch is not used alone)
```

Success: the right part number is first, and each hit shows both branch contributions.

Losing cases: the wrong part number `AB-1243` is a close second (0.032258 against 0.032787), because RRF only sees ranks. BM25 scored `AB-1234` much higher than `AB-1243` (3.163 against 0.707 on this fixture), and that gap is lost after fusion. A query with no lexical terms fails the whole call instead of quietly falling back to dense only.

## Paired observations

From [the paired comparison](../../evidence/P5.2/paired_vs_dense.md), quoted exactly:

```text
| en-fact | en | dev | R02 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-fact | en | dev | R06+R07 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| paraphrase | en | validation | R02 | 5 | 0.2000 | 0.0000 | tie | Children fly kites in the park when the  |
| paraphrase | en | validation | R03 | 2 | 0.5000 | 0.3000 | win | Children fly kites in the park when the  |
| paraphrase | en | validation | R06+R07 | 4 | 0.2500 | 0.0500 | win | Children fly kites in the park when the  |
```

On `paraphrase`, R06+R07 beats R02 (rank 4 against 5) but trails R03 alone (rank 2). On `en-fact` all tie at rank 1. The dense branch there is the same test-only lexical embedder, and the dev weight sweep for R08 was flat (every weight gives dev MRR 1.0000), so these rows compare lexical scoring variants on a tiny fixture and support no general claim.

## Limitations

The branches run sequentially, so latency adds. Candidate complementarity depends on the embedding model and corpus; these fixture observations do not establish production performance.

## Status

Implemented; not validated or benchmarked.
