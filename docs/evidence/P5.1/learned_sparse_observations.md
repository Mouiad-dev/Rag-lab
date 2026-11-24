> Generated 2026-11-09 by the orchestrator: `PYTHONIOENCODING=utf-8 RAG_SPARSE_MODEL_DIR=var/models/opensearch-neural-sparse-encoding-multilingual-v1 uv run python docs/techniques/lessons/demos/r05_sparse_demo.py`; model revision `1e0f096c2b51c234f1d20725c793e1b5b6d556db` (local files only), CPU torch 2.15.0, sentence-transformers 6.2.0. One tiny local fixture; not a benchmark. Body below is the unedited script output.

# R05 learned sparse demo: opensearch-neural-sparse-encoding-multilingual-v1 (not recorded)
One tiny local fixture (8 docs); not a benchmark.

## Memory estimate
Estimate for term-string plus float32 storage; not measured DB size.
Compare with a 1024-dim float32 dense vector: 4096 bytes.

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

## EN: Where was the basalt sample collected?
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 0 | 15.519 | basal:3.394*1.825, sample:3.105*1.160, collected:2.612*0.908, where:1.695*0.947, ##t:0.922*1.500 | 1
2 | 7 | 1.402 | ##t:0.922*0.884, collected:2.612*0.181, ?:1.611*0.058, was:1.147*0.019 | 4
3 | 6 | 0.732 | where:1.695*0.269, the:0.723*0.382 | 1
4 | 3 | 0.339 | the:0.723*0.468 | 0
5 | 2 | 0.153 | the:0.723*0.133, ?:1.611*0.035 | 1
6 | 5 | 0.119 | ?:1.611*0.057, the:0.723*0.038 | 1
7 | 1 | 0.000 | (none) | 0
8 | 4 | 0.000 | (none) | 0
expected doc: 0; rank of expected doc: 1

## AR: متى وصلت القافلة إلى مدينة صور؟
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 1 | 18.158 | ##فل:1.935*1.653, ##ور:1.532*1.885, ##قا:1.575*1.689, ص:1.397*1.898, وصل:2.514*0.904 | 1
2 | 0 | 0.000 | (none) | 0
3 | 2 | 0.000 | (none) | 0
4 | 3 | 0.000 | (none) | 0
5 | 4 | 0.000 | (none) | 0
6 | 5 | 0.000 | (none) | 0
7 | 6 | 0.000 | (none) | 0
8 | 7 | 0.000 | (none) | 0
expected doc: 1; rank of expected doc: 1

## EN paraphrase: Which volcanic rock was gathered close to the icy north?
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 0 | 6.514 | north:1.582*1.190, volcanic:3.338*0.412, rock:2.351*0.561, gathered:3.054*0.266, close:2.012*0.199 | 6
2 | 3 | 0.783 | close:2.012*0.221, the:0.723*0.468 | 1
3 | 6 | 0.473 | the:0.723*0.382, close:2.012*0.097 | 1
4 | 7 | 0.351 | to:0.839*0.281, ?:1.611*0.058, was:1.147*0.019 | 3
5 | 2 | 0.285 | to:0.839*0.158, the:0.723*0.133, ?:1.611*0.035 | 2
6 | 5 | 0.242 | to:0.839*0.138, ?:1.611*0.057, the:0.723*0.038, ##y:1.317*0.006 | 3
7 | 4 | 0.084 | close:2.012*0.042 | 1
8 | 1 | 0.000 | (none) | 0
expected doc: 0; rank of expected doc: 1

## Shared words, wrong fact: Where was the copper sample collected?
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 0 | 7.943 | sample:3.105*1.160, collected:2.612*0.908, where:1.695*0.947, the:0.723*0.451, was:1.147*0.034 | 1
2 | 2 | 5.854 | copper:3.166*1.800, the:0.723*0.133, ?:1.611*0.035 | 1
3 | 6 | 0.732 | where:1.695*0.269, the:0.723*0.382 | 1
4 | 7 | 0.587 | collected:2.612*0.181, ?:1.611*0.058, was:1.147*0.019 | 3
5 | 3 | 0.339 | the:0.723*0.468 | 0
6 | 5 | 0.317 | copper:3.166*0.063, ?:1.611*0.057, the:0.723*0.038 | 2
7 | 1 | 0.000 | (none) | 0
8 | 4 | 0.000 | (none) | 0
expected doc: none (top doc shown above); rank of expected doc: n/a

## Negation: Which sample was not collected near the glacier?
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 0 | 12.899 | glacier:3.631*1.358, sample:3.105*1.160, collected:2.612*0.908, near:1.477*0.858, which:1.295*0.280 | 1
2 | 4 | 0.840 | near:1.477*0.568 | 1
3 | 7 | 0.587 | collected:2.612*0.181, ?:1.611*0.058, was:1.147*0.019 | 3
4 | 3 | 0.575 | the:0.723*0.468, near:1.477*0.160 | 1
5 | 6 | 0.276 | the:0.723*0.382 | 0
6 | 2 | 0.153 | the:0.723*0.133, ?:1.611*0.035 | 1
7 | 5 | 0.119 | ?:1.611*0.057, the:0.723*0.038 | 1
8 | 1 | 0.000 | (none) | 0
expected doc: none (top doc shown above); rank of expected doc: n/a

## AR plural: متى وصلت القوافل إلى مدينة صور؟
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 1 | 14.305 | ##فل:1.935*1.653, ##ور:1.532*1.885, ص:1.397*1.898, وصل:2.514*0.904, مدينة:2.110*0.957 | 1
2 | 0 | 0.000 | (none) | 0
3 | 2 | 0.000 | (none) | 0
4 | 3 | 0.000 | (none) | 0
5 | 4 | 0.000 | (none) | 0
6 | 5 | 0.000 | (none) | 0
7 | 6 | 0.000 | (none) | 0
8 | 7 | 0.000 | (none) | 0
expected doc: 1; rank of expected doc: 1

## AR with diacritics: متى وصلت القَافِلَة إلى مَدِينَة صور؟
rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | terms the doc vector adds beyond its text
---:|---:|---:|---|---:
1 | 1 | 18.158 | ##فل:1.935*1.653, ##ور:1.532*1.885, ##قا:1.575*1.689, ص:1.397*1.898, وصل:2.514*0.904 | 1
2 | 0 | 0.000 | (none) | 0
3 | 2 | 0.000 | (none) | 0
4 | 3 | 0.000 | (none) | 0
5 | 4 | 0.000 | (none) | 0
6 | 5 | 0.000 | (none) | 0
7 | 6 | 0.000 | (none) | 0
8 | 7 | 0.000 | (none) | 0
expected doc: 1; rank of expected doc: 1
