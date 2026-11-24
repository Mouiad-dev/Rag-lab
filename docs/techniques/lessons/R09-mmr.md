# R09: MMR diversity selection

## Intuition

Maximal Marginal Relevance picks hits one at a time. Each pick balances how relevant a candidate is against how similar it is to what was already picked, so repeated passages give way to different ones.

## Mechanism

`MmrSelector` takes one retrieval result and greedily selects up to `k` hits. For each remaining candidate the score is `lambda * relevance - (1 - lambda) * max_similarity_to_selected`. Relevance is the input hit score min-max normalised over the input list (1.0 for all if the scores are equal). It is not query similarity: the query is not seen again. Similarity between hits is either the cosine of stored index embeddings (`embedding`, default; needs a `ChunkVectorReader` and fails if a vector is missing) or the cosine of lexical term-count vectors (`lexical`). Ties are broken by the lower input rank. Each output hit keeps its components and adds `mmr_relevance`, `mmr_max_similarity` and `mmr_score`.

## Config and defaults

`MmrConfig`: `k` 5 (1-50), `lambda_mult` 0.5 (0.0-1.0), `similarity` `embedding`. The input must be a single corpus result from the same ready index version.

## Fidelity

Adaptation of Carbonell and Goldstein's MMR. The relevance term reuses the earlier retriever's score instead of a fresh query-document similarity, and the similarity term is a stored-vector or lexical cosine. Primary source: https://doi.org/10.1145/290941.291025 (SIGIR 1998), checked in `docs/techniques/SOURCE_LINK_AUDIT.md`.

## Runnable demo

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p53_selection_demo.py
```

Full output: [selection_and_filters.md](../../evidence/P5.3/selection_and_filters.md). Lexical mode, same 6-hit list as R10. It is an offline fixture; no model; not a benchmark.

## Success case

```text
| 2 | 1.0 | mmr[R03]:lexical:lambda=1.00 | c1, c2 | 2 | 1 | - |
| 2 | 0.5 | mmr[R03]:lexical:lambda=0.50 | c1, c3 | 1 | 2 | c2 |
```

At k=2 and lambda 1.0 the top two are `c1` and its exact copy `c2` (1 distinct text). At lambda 0.5 the copy is replaced by `c3`, a different passage (2 distinct texts). That is the novelty gain.

## Losing case

The same rows show the cost. `c2` was relevant (score 0.9, an exact copy in another view) and is lost at lambda 0.5. The count of distinct `extraction_view_id` also falls from 2 to 1, because `c3` comes from the same view as `c1`. At lambda 0.0 with k=3:

```text
| 3 | 0.0 | mmr[R03]:lexical:lambda=0.00 | c1, c3, c6 | 2 | 3 | c2 |
```

the least relevant hit `c6` enters the top 3 because only diversity counts after the first pick. At k=3 and lambda 0.5 the exact copy `c2` is still selected:

```text
| 3 | 0.5 | mmr[R03]:lexical:lambda=0.50 | c1, c3, c2 | 2 | 2 | - |
```

MMR does not guarantee de-duplication; use R10 for that.

The tie rule is shown by equal input scores:

```text
| equal scores | 0.5, 0.5 | c1, c2 |
```

## Limitations

Lexical similarity sees shared words, not meaning; embedding similarity depends on the embedder. Selection is greedy. Relevance depends on the quality and scale of the input scores.

## Status

Implemented; not validated or benchmarked.
