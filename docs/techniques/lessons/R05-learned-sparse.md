# R05: Learned sparse retrieval

## Intuition

A learned sparse encoder gives each text a weighted bag of vocabulary terms, like a lexical index, but the weights and the extra terms come from a trained model. A document can therefore match a query that shares no words with it. R05 is not BM25 (R03) and not PostgreSQL full-text search (R04): there is no term-frequency or IDF formula, only a dot product of learned weights.

## Mechanism

The SPLADE family (arXiv 2107.05720) is the research source. This model, `opensearch-neural-sparse-encoding-multilingual-v1`, is a document-side learned sparse encoder with inference-free queries: a query is only tokenized and given static per-token weights, and only documents run the transformer. R05 is therefore a fidelity adaptation of the SPLADE idea, not a SPLADE reimplementation.

The model card says the model is meant for OpenSearch. We use it through the sentence-transformers `SparseEncoder` with `local_files_only`, so nothing is downloaded at run time. Document vectors are stored in `chunk_sparse_vectors`, and the dot product is computed in SQL (`learned-sparse-dot`). If the `[sparse]` extra or the local model is missing, R05 raises `TechniqueUnavailableError`; it never falls back to another retriever. The 684 MB model is not part of CI, so the tests that use it are opt-in.

## Runnable demo

The demo is opt-in and needs the approved local model directory. Without `RAG_SPARSE_MODEL_DIR` it prints a SKIP line and exits 0.

```bash
PYTHONIOENCODING=utf-8 RAG_SPARSE_MODEL_DIR=var/models/opensearch-neural-sparse-encoding-multilingual-v1 uv run python docs/techniques/lessons/demos/r05_sparse_demo.py
```

The full output is in [the evidence file](../../evidence/P5.1/learned_sparse_observations.md), generated 2026-11-09 on CPU with torch 2.15.0 and sentence-transformers 6.2.0. The demo header prints the revision as "(not recorded)" because the model directory has no revision file; the evidence file header gives the real revision `1e0f096c2b51c234f1d20725c793e1b5b6d556db` (local files only). It is one tiny local fixture of 8 documents; it is not a benchmark.

How to read the tables: `term:q_weight*d_weight` is one matched term; the score is the sum of those products over all matched terms (only the top 5 are shown). The last column, "terms the doc vector adds beyond its text", is counted per row: it is the number of matched terms of that row (all matched terms, not only the top 5 shown) that are not in the document's own tokenizer output or casefolded words. A count of 0 means every matched term appears in the document text; a higher count means some matched terms come from the model's document-side expansion. Query-side expansion is not counted because queries are inference-free.

## Success cases

English query, quoted exactly (columns: rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text):

```text
1 | 0 | 15.519 | basal:3.394*1.825, sample:3.105*1.160, collected:2.612*0.908, where:1.695*0.947, ##t:0.922*1.500 | 1
2 | 7 | 1.402 | ##t:0.922*0.884, collected:2.612*0.181, ?:1.611*0.058, was:1.147*0.019 | 4
```

Doc 0 ranks first (15.519) and the next document scores 1.402.

Arabic query:

```text
## AR: متى وصلت القافلة إلى مدينة صور؟
1 | 1 | 18.158 | ##فل:1.935*1.653, ##ور:1.532*1.885, ##قا:1.575*1.689, ص:1.397*1.898, وصل:2.514*0.904 | 1
```

Doc 1 ranks first (18.158) and every other document scores 0.000.

Expansion success, an English paraphrase with no shared content words:

```text
## EN paraphrase: Which volcanic rock was gathered close to the icy north?
1 | 0 | 6.514 | north:1.582*1.190, volcanic:3.338*0.412, rock:2.351*0.561, gathered:3.054*0.266, close:2.012*0.199 | 6
```

The question shares no content words with doc 0, yet doc 0 is first (6.514) and the column shows 6 matched terms that are not in the document's own tokens, so part of this match is document-side expansion. The column does not say which six terms they are, and the top 5 shown (`north`, `volcanic`, `rock`, `gathered`, `close`) may include terms that are in the document text (the document contains `northern`), so only the count, not any single term, is evidence of expansion here.

Robustness, Arabic with diacritics and the Arabic plural:

```text
## AR with diacritics: متى وصلت القَافِلَة إلى مَدِينَة صور؟
1 | 1 | 18.158 | ##فل:1.935*1.653, ##ور:1.532*1.885, ##قا:1.575*1.689, ص:1.397*1.898, وصل:2.514*0.904 | 1
## AR plural: متى وصلت القوافل إلى مدينة صور؟
1 | 1 | 14.305 | ##فل:1.935*1.653, ##ور:1.532*1.885, ص:1.397*1.898, وصل:2.514*0.904, مدينة:2.110*0.957 | 1
```

The diacritics query gives the same 18.158 as the plain query, and the plural still ranks doc 1 first (14.305), lower than the singular.

## Losing cases

Negation, quoted exactly:

```text
## Negation: Which sample was not collected near the glacier?
1 | 0 | 12.899 | glacier:3.631*1.358, sample:3.105*1.160, collected:2.612*0.908, near:1.477*0.858, which:1.295*0.280 | 1
```

The question asks for the sample that was not collected near the glacier, but doc 0 ranks first (12.899; it matches `glacier`). The sparse dot product ignores "not".

Shared words, wrong fact:

```text
## Shared words, wrong fact: Where was the copper sample collected?
1 | 0 | 7.943 | sample:3.105*1.160, collected:2.612*0.908, where:1.695*0.947, the:0.723*0.451, was:1.147*0.034 | 1
2 | 2 | 5.854 | copper:3.166*1.800, the:0.723*0.133, ?:1.611*0.035 | 1
```

The fixture has no document that answers this question correctly. The basalt doc 0 (7.943) still ranks above the copper doc 2 (5.854) because it shares the words `sample`, `collected` and `where` with the question.

## Memory estimate

The numbers are an estimate for term-string plus float32 storage (`terms * (average UTF-8 term bytes across the fixture + 4)`), not a measured database size. A 1024-dim float32 dense vector is 4096 bytes.

```text
doc | nonzero terms | estimated bytes
---:|---:|---:
0 | 241 | 2308
1 | 112 | 1073
2 | 340 | 3257
3 | 316 | 3027
4 | 177 | 1695
5 | 427 | 4090
6 | 264 | 2529
7 | 341 | 3266
```

Most sparse vectors here are smaller than the dense vector, but doc 5 (4090 bytes) is close to it. Vector size depends on the document, so on longer chunks the estimate can pass the dense size; this fixture does not show that either way.

## Limitations

- The demo reads the adapter's private tokenizer (`encoder._encoder.tokenizer`) for the expansion column; an adapter refactor can break the demo, not R05.

Eight documents and a handful of queries: these observations support no general claim about quality or speed. Negation and shared-word queries are not handled by the dot product. Stored vectors grow with the number of expanded terms. The model needs the `[sparse]` extra and a local copy of the model.

## Status

Implemented; not validated or benchmarked.
