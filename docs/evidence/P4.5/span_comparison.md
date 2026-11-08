# Span relevance comparison

Method: lexical token overlap (not BM25); source-span coverage.
Fixture version: 1.

Metrics: best coverage and precision use the best overlapping chunk (ties choose the shorter chunk); contained means one chunk covers the span; pieces counts overlapping chunks.
Ranked span recall uses union coverage >= 0.8 at k=1,3,5. Hit@k means any span covered; MRR is reciprocal of first successful k, zero when none. Context counts code points.
Unanswerable cases report context and top-1 score only; excluded from recall, hit, and MRR denominators.

## Per-cohort means

| Cohort | Chunker | Status | n | Means |
|---|---|---|---:|---|
| ar | C01 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.75, span_recall@5=1.0, hit@1=0.0, hit@3=0.75, hit@5=1.0, mrr=0.3, context_chars@1=400.0, context_chars@3=1199.0, context_chars@5=1999.0 |
| en | C01 400 chars | ok | 5 | span_recall@1=0.4, span_recall@3=0.8, span_recall@5=1.0, hit@1=0.4, hit@3=0.8, hit@5=1.0, mrr=0.5733, context_chars@1=399.8, context_chars@3=1070.8, context_chars@5=1870.4 |
| mixed | C01 400 chars | ok | 4 | span_recall@1=0.75, span_recall@3=0.75, span_recall@5=1.0, hit@1=0.75, hit@3=0.75, hit@5=1.0, mrr=0.8, context_chars@1=400.0, context_chars@3=1092.0, context_chars@5=1892.0 |
| unanswerable | C01 400 chars | ok | 1 | context_chars@1=185.0, context_chars@3=985.0, context_chars@5=1785.0 |
| ar | C02 67 words (~400 chars) | ok | 4 | span_recall@1=0.0, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.0, hit@3=1.0, hit@5=1.0, mrr=0.3333, context_chars@1=451.0, context_chars@3=1266.0, context_chars@5=2099.0 |
| en | C02 67 words (~400 chars) | ok | 5 | span_recall@1=0.2, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.2, hit@3=1.0, hit@5=1.0, mrr=0.4667, context_chars@1=273.8, context_chars@3=1040.0, context_chars@5=1885.0 |
| mixed | C02 67 words (~400 chars) | ok | 4 | span_recall@1=0.5, span_recall@3=0.5, span_recall@5=1.0, hit@1=0.5, hit@3=0.5, hit@5=1.0, mrr=0.6, context_chars@1=423.0, context_chars@3=1092.0, context_chars@5=1903.0 |
| unanswerable | C02 67 words (~400 chars) | ok | 1 | context_chars@1=425.0, context_chars@3=897.0, context_chars@5=1707.0 |
| ar | C03 128 tokens (~400 chars) | ok | 4 | span_recall@1=0.0, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.0, hit@3=1.0, hit@5=1.0, mrr=0.3333, context_chars@1=673.0, context_chars@3=1481.0, context_chars@5=2583.0 |
| en | C03 128 tokens (~400 chars) | ok | 5 | span_recall@1=0.4, span_recall@3=0.8, span_recall@5=1.0, hit@1=0.4, hit@3=0.8, hit@5=1.0, mrr=0.5733, context_chars@1=662.2, context_chars@3=1903.4, context_chars@5=2583.0 |
| mixed | C03 128 tokens (~400 chars) | ok | 4 | span_recall@1=0.625, span_recall@3=0.625, span_recall@5=1.0, hit@1=0.75, hit@3=0.75, hit@5=1.0, mrr=0.8, context_chars@1=494.5, context_chars@3=1689.5, context_chars@5=2583.0 |
| unanswerable | C03 128 tokens (~400 chars) | ok | 1 | context_chars@1=667.0, context_chars@3=1757.0, context_chars@5=2583.0 |
| ar | C04 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=400.0, context_chars@3=1200.0, context_chars@5=1999.0 |
| en | C04 400 chars | ok | 5 | span_recall@1=0.5, span_recall@3=0.7, span_recall@5=0.9, hit@1=0.6, hit@3=0.8, hit@5=1.0, mrr=0.7067, context_chars@1=400.0, context_chars@3=1072.2, context_chars@5=1872.2 |
| mixed | C04 400 chars | ok | 4 | span_recall@1=0.375, span_recall@3=0.625, span_recall@5=0.625, hit@1=0.5, hit@3=0.75, hit@5=0.75, mrr=0.5833, context_chars@1=400.0, context_chars@3=1146.5, context_chars@5=1893.0 |
| unanswerable | C04 400 chars | ok | 1 | context_chars@1=187.0, context_chars@3=987.0, context_chars@5=1787.0 |
| ar | C05 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=265.0, context_chars@3=909.0, context_chars@5=1579.0 |
| en | C05 400 chars | ok | 5 | span_recall@1=0.4, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.4, hit@3=1.0, hit@5=1.0, mrr=0.6, context_chars@1=341.8, context_chars@3=1032.2, context_chars@5=1685.2 |
| mixed | C05 400 chars | ok | 4 | span_recall@1=0.375, span_recall@3=0.625, span_recall@5=0.625, hit@1=0.5, hit@3=0.75, hit@5=0.75, mrr=0.5833, context_chars@1=317.5, context_chars@3=996.5, context_chars@5=1654.5 |
| unanswerable | C05 400 chars | ok | 1 | context_chars@1=370.0, context_chars@3=1033.0, context_chars@5=1614.0 |
| ar | C06 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=398.0, context_chars@3=1130.0, context_chars@5=1916.0 |
| en | C06 400 chars | ok | 5 | span_recall@1=0.6, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.6, hit@3=1.0, hit@5=1.0, mrr=0.7333, context_chars@1=348.2, context_chars@3=1071.6, context_chars@5=1820.6 |
| mixed | C06 400 chars | ok | 4 | span_recall@1=0.625, span_recall@3=0.625, span_recall@5=0.875, hit@1=0.75, hit@3=0.75, hit@5=1.0, mrr=0.8, context_chars@1=389.0, context_chars@3=1136.0, context_chars@5=1887.25 |
| unanswerable | C06 400 chars | ok | 1 | context_chars@1=370.0, context_chars@3=1115.0, context_chars@5=1804.0 |
| ar | C07 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=265.0, context_chars@3=875.0, context_chars@5=1526.0 |
| en | C07 400 chars | ok | 5 | span_recall@1=0.4, span_recall@3=0.8, span_recall@5=0.8, hit@1=0.4, hit@3=0.8, hit@5=0.8, mrr=0.5333, context_chars@1=324.6, context_chars@3=986.4, context_chars@5=1600.2 |
| mixed | C07 400 chars | ok | 4 | span_recall@1=0.375, span_recall@3=0.625, span_recall@5=0.625, hit@1=0.5, hit@3=0.75, hit@5=0.75, mrr=0.5833, context_chars@1=298.0, context_chars@3=956.25, context_chars@5=1586.5 |
| unanswerable | C07 400 chars | ok | 1 | context_chars@1=370.0, context_chars@3=993.0, context_chars@5=1554.0 |
| ar | C08 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=245.0, context_chars@3=856.0, context_chars@5=1230.0 |
| en | C08 400 chars | ok | 5 | span_recall@1=0.6, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.6, hit@3=1.0, hit@5=1.0, mrr=0.7333, context_chars@1=238.6, context_chars@3=821.0, context_chars@5=1356.2 |
| mixed | C08 400 chars | ok | 4 | span_recall@1=0.5, span_recall@3=0.625, span_recall@5=0.625, hit@1=0.5, hit@3=0.75, hit@5=0.75, mrr=0.5833, context_chars@1=241.0, context_chars@3=741.75, context_chars@5=1341.5 |
| unanswerable | C08 400 chars | ok | 1 | context_chars@1=187.0, context_chars@3=801.0, context_chars@5=1362.0 |
| ar | C15 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| en | C15 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| mixed | C15 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| unanswerable | C15 model | requires_models | 0 | context_chars@1=None, context_chars@3=None, context_chars@5=None |
| ar | C16 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| en | C16 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| mixed | C16 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| unanswerable | C16 model | requires_models | 0 | context_chars@1=None, context_chars@3=None, context_chars@5=None |
| ar | C17 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| en | C17 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| mixed | C17 model | requires_models | 0 | span_recall@1=None, span_recall@3=None, span_recall@5=None, hit@1=None, hit@3=None, hit@5=None, mrr=None, context_chars@1=None, context_chars@3=None, context_chars@5=None |
| unanswerable | C17 model | requires_models | 0 | context_chars@1=None, context_chars@3=None, context_chars@5=None |
| ar | C18 parent 1600 child 400 | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=398.0, context_chars@3=1130.0, context_chars@5=1916.0 |
| en | C18 parent 1600 child 400 | ok | 5 | span_recall@1=0.6, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.6, hit@3=1.0, hit@5=1.0, mrr=0.7333, context_chars@1=348.2, context_chars@3=1071.6, context_chars@5=1820.6 |
| mixed | C18 parent 1600 child 400 | ok | 4 | span_recall@1=0.625, span_recall@3=0.625, span_recall@5=0.875, hit@1=0.75, hit@3=0.75, hit@5=1.0, mrr=0.8, context_chars@1=389.0, context_chars@3=1136.0, context_chars@5=1887.25 |
| unanswerable | C18 parent 1600 child 400 | ok | 1 | context_chars@1=370.0, context_chars@3=1115.0, context_chars@5=1804.0 |
| ar | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 4 | span_recall@1=1.0, span_recall@3=1.0, span_recall@5=1.0, hit@1=1.0, hit@3=1.0, hit@5=1.0, mrr=1.0, context_chars@1=1534.0, context_chars@3=2583.0, context_chars@5=2583.0 |
| en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 5 | span_recall@1=1.0, span_recall@3=1.0, span_recall@5=1.0, hit@1=1.0, hit@3=1.0, hit@5=1.0, mrr=1.0, context_chars@1=1437.0, context_chars@3=2583.0, context_chars@5=2583.0 |
| mixed | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 4 | span_recall@1=0.75, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.75, hit@3=1.0, hit@5=1.0, mrr=0.8333, context_chars@1=1170.25, context_chars@3=2583.0, context_chars@5=2583.0 |
| unanswerable | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | context_chars@1=1049.0, context_chars@3=2583.0, context_chars@5=2583.0 |
| ar | C19 adaptive | ok | 4 | span_recall@1=0.0, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.0, hit@3=1.0, hit@5=1.0, mrr=0.3333, context_chars@1=772.0, context_chars@3=2211.0, context_chars@5=2581.0 |
| en | C19 adaptive | ok | 5 | span_recall@1=0.4, span_recall@3=0.8, span_recall@5=1.0, hit@1=0.4, hit@3=0.8, hit@5=1.0, mrr=0.5733, context_chars@1=666.2, context_chars@3=1832.2, context_chars@5=2581.0 |
| mixed | C19 adaptive | ok | 4 | span_recall@1=0.5, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.5, hit@3=1.0, hit@5=1.0, mrr=0.6667, context_chars@1=624.5, context_chars@3=2010.0, context_chars@5=2581.0 |
| unanswerable | C19 adaptive | ok | 1 | context_chars@1=370.0, context_chars@3=1809.0, context_chars@5=2581.0 |
| ar | C21 400 chars | ok | 4 | span_recall@1=0.0, span_recall@3=0.0, span_recall@5=1.0, hit@1=0.0, hit@3=0.0, hit@5=1.0, mrr=0.2, context_chars@1=398.0, context_chars@3=1130.0, context_chars@5=1916.0 |
| en | C21 400 chars | ok | 5 | span_recall@1=0.6, span_recall@3=1.0, span_recall@5=1.0, hit@1=0.6, hit@3=1.0, hit@5=1.0, mrr=0.7333, context_chars@1=348.2, context_chars@3=1071.6, context_chars@5=1820.6 |
| mixed | C21 400 chars | ok | 4 | span_recall@1=0.625, span_recall@3=0.625, span_recall@5=0.875, hit@1=0.75, hit@3=0.75, hit@5=1.0, mrr=0.8, context_chars@1=389.0, context_chars@3=1136.0, context_chars@5=1887.25 |
| unanswerable | C21 400 chars | ok | 1 | context_chars@1=370.0, context_chars@3=1115.0, context_chars@5=1804.0 |

## Per-case

| Case | Cohort | Chunker | Status | Spans | Recall@5 | Hit@5 | MRR | Top-1 score |
|---|---|---|---|---:|---:|---:|---:|---:|
| case-01 | en | C01 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.2949 |
| case-01 | en | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.4082 |
| case-01 | en | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.2 | 0.3180 |
| case-01 | en | C04 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.4009 |
| case-01 | en | C05 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.4472 |
| case-01 | en | C06 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.4286 |
| case-01 | en | C07 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.4472 |
| case-01 | en | C08 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.4575 |
| case-01 | en | C21 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.4286 |
| case-01 | en | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.3333 | 0.4286 |
| case-01 | en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.4286 |
| case-01 | en | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-01 | en | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-01 | en | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-01 | en | C19 adaptive | ok | 1 | 1.0 | 1 | 0.2 | 0.3375 |
| case-02 | en | C01 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.2697 |
| case-02 | en | C02 67 words (~400 chars) | ok | 2 | 1.0 | 1 | 0.3333 | 0.3015 |
| case-02 | en | C03 128 tokens (~400 chars) | ok | 2 | 1.0 | 1 | 0.3333 | 0.2309 |
| case-02 | en | C04 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.2774 |
| case-02 | en | C05 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.2981 |
| case-02 | en | C06 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.2774 |
| case-02 | en | C07 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.2981 |
| case-02 | en | C08 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.3050 |
| case-02 | en | C21 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.2774 |
| case-02 | en | C18 parent 1600 child 400 | ok | 2 | 1.0 | 1 | 1.0 | 0.2774 |
| case-02 | en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 2 | 1.0 | 1 | 1.0 | 0.2774 |
| case-02 | en | C15 model | requires_models | 2 |  |  |  | 0.0000 |
| case-02 | en | C16 model | requires_models | 2 |  |  |  | 0.0000 |
| case-02 | en | C17 model | requires_models | 2 |  |  |  | 0.0000 |
| case-02 | en | C19 adaptive | ok | 2 | 1.0 | 1 | 0.3333 | 0.2774 |
| case-03 | en | C01 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.5394 |
| case-03 | en | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.5443 |
| case-03 | en | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.4240 |
| case-03 | en | C04 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5345 |
| case-03 | en | C05 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5963 |
| case-03 | en | C06 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5714 |
| case-03 | en | C07 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5963 |
| case-03 | en | C08 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.6100 |
| case-03 | en | C21 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5714 |
| case-03 | en | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 1.0 | 0.5714 |
| case-03 | en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.5714 |
| case-03 | en | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-03 | en | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-03 | en | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-03 | en | C19 adaptive | ok | 1 | 1.0 | 1 | 1.0 | 0.4500 |
| case-04 | en | C01 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.2722 |
| case-04 | en | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.3015 |
| case-04 | en | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.2309 |
| case-04 | en | C04 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.2673 |
| case-04 | en | C05 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.2981 |
| case-04 | en | C06 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.2857 |
| case-04 | en | C07 400 chars | ok | 1 | 0.0 | 0 | 0.0 | 0.2981 |
| case-04 | en | C08 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.3430 |
| case-04 | en | C21 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.2857 |
| case-04 | en | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 1.0 | 0.2857 |
| case-04 | en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.2857 |
| case-04 | en | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-04 | en | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-04 | en | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-04 | en | C19 adaptive | ok | 1 | 1.0 | 1 | 1.0 | 0.2250 |
| case-05 | ar | C01 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-05 | ar | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-05 | ar | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-05 | ar | C04 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C05 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C06 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C07 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C08 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C21 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-05 | ar | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.0000 |
| case-05 | ar | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-05 | ar | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-05 | ar | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-05 | ar | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-06 | ar | C01 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-06 | ar | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-06 | ar | C04 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C05 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C06 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C07 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C08 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C21 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-06 | ar | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.0000 |
| case-06 | ar | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-06 | ar | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-06 | ar | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-06 | ar | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-07 | ar | C01 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-07 | ar | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-07 | ar | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-07 | ar | C04 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C05 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C06 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C07 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C08 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C21 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-07 | ar | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.0000 |
| case-07 | ar | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-07 | ar | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-07 | ar | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-07 | ar | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-08 | ar | C01 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-08 | ar | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-08 | ar | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-08 | ar | C04 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C05 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C06 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C07 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C08 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C21 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.2 | 0.0000 |
| case-08 | ar | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.0000 |
| case-08 | ar | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-08 | ar | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-08 | ar | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-08 | ar | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.0000 |
| case-09 | mixed | C01 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5898 |
| case-09 | mixed | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.6100 |
| case-09 | mixed | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.6325 |
| case-09 | mixed | C04 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6325 |
| case-09 | mixed | C05 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6405 |
| case-09 | mixed | C06 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.6576 |
| case-09 | mixed | C07 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6667 |
| case-09 | mixed | C08 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.7276 |
| case-09 | mixed | C21 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.6576 |
| case-09 | mixed | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 1.0 | 0.6576 |
| case-09 | mixed | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.6576 |
| case-09 | mixed | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-09 | mixed | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-09 | mixed | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-09 | mixed | C19 adaptive | ok | 1 | 1.0 | 1 | 1.0 | 0.4887 |
| case-10 | mixed | C01 400 chars | ok | 2 | 1.0 | 1 | 1.0 | 0.1474 |
| case-10 | mixed | C02 67 words (~400 chars) | ok | 2 | 1.0 | 1 | 0.2 | 0.1925 |
| case-10 | mixed | C03 128 tokens (~400 chars) | ok | 2 | 1.0 | 1 | 1.0 | 0.1581 |
| case-10 | mixed | C04 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.1581 |
| case-10 | mixed | C05 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.1601 |
| case-10 | mixed | C06 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.1644 |
| case-10 | mixed | C07 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.1667 |
| case-10 | mixed | C08 400 chars | ok | 2 | 0.5 | 1 | 0.3333 | 0.1715 |
| case-10 | mixed | C21 400 chars | ok | 2 | 0.5 | 1 | 1.0 | 0.1644 |
| case-10 | mixed | C18 parent 1600 child 400 | ok | 2 | 0.5 | 1 | 1.0 | 0.1644 |
| case-10 | mixed | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 2 | 1.0 | 1 | 1.0 | 0.1644 |
| case-10 | mixed | C15 model | requires_models | 2 |  |  |  | 0.0000 |
| case-10 | mixed | C16 model | requires_models | 2 |  |  |  | 0.0000 |
| case-10 | mixed | C17 model | requires_models | 2 |  |  |  | 0.0000 |
| case-10 | mixed | C19 adaptive | ok | 2 | 1.0 | 1 | 1.0 | 0.1222 |
| case-11 | mixed | C01 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.3651 |
| case-11 | mixed | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.3849 |
| case-11 | mixed | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.4743 |
| case-11 | mixed | C04 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.4743 |
| case-11 | mixed | C05 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.4804 |
| case-11 | mixed | C06 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.3288 |
| case-11 | mixed | C07 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.3333 |
| case-11 | mixed | C08 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.5000 |
| case-11 | mixed | C21 400 chars | ok | 1 | 1.0 | 1 | 1.0 | 0.3288 |
| case-11 | mixed | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 1.0 | 0.3288 |
| case-11 | mixed | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.3288 |
| case-11 | mixed | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-11 | mixed | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-11 | mixed | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-11 | mixed | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.2774 |
| case-12 | mixed | C01 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.1443 |
| case-12 | mixed | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.2 | 0.1374 |
| case-12 | mixed | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 0.2 | 0.1140 |
| case-12 | mixed | C04 400 chars | ok | 1 | 0.0 | 0 | 0.0 | 0.1443 |
| case-12 | mixed | C05 400 chars | ok | 1 | 0.0 | 0 | 0.0 | 0.1768 |
| case-12 | mixed | C06 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.1459 |
| case-12 | mixed | C07 400 chars | ok | 1 | 0.0 | 0 | 0.0 | 0.1768 |
| case-12 | mixed | C08 400 chars | ok | 1 | 0.0 | 0 | 0.0 | 0.1826 |
| case-12 | mixed | C21 400 chars | ok | 1 | 1.0 | 1 | 0.2 | 0.1459 |
| case-12 | mixed | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.2 | 0.1459 |
| case-12 | mixed | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 0.3333 | 0.1459 |
| case-12 | mixed | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-12 | mixed | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-12 | mixed | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-12 | mixed | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.1066 |
| case-13 | unanswerable | C01 400 chars | ok | 0 |  |  |  | 0.3849 |
| case-13 | unanswerable | C02 67 words (~400 chars) | ok | 0 |  |  |  | 0.3050 |
| case-13 | unanswerable | C03 128 tokens (~400 chars) | ok | 0 |  |  |  | 0.3464 |
| case-13 | unanswerable | C04 400 chars | ok | 0 |  |  |  | 0.3849 |
| case-13 | unanswerable | C05 400 chars | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C06 400 chars | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C07 400 chars | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C08 400 chars | ok | 0 |  |  |  | 0.3922 |
| case-13 | unanswerable | C21 400 chars | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C18 parent 1600 child 400 | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 0 |  |  |  | 0.4160 |
| case-13 | unanswerable | C15 model | requires_models | 0 |  |  |  | 0.0000 |
| case-13 | unanswerable | C16 model | requires_models | 0 |  |  |  | 0.0000 |
| case-13 | unanswerable | C17 model | requires_models | 0 |  |  |  | 0.0000 |
| case-13 | unanswerable | C19 adaptive | ok | 0 |  |  |  | 0.4160 |
| case-14 | en | C01 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.5898 |
| case-14 | en | C02 67 words (~400 chars) | ok | 1 | 1.0 | 1 | 0.3333 | 0.7625 |
| case-14 | en | C03 128 tokens (~400 chars) | ok | 1 | 1.0 | 1 | 1.0 | 0.6928 |
| case-14 | en | C04 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.7454 |
| case-14 | en | C05 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6100 |
| case-14 | en | C06 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6761 |
| case-14 | en | C07 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6325 |
| case-14 | en | C08 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 1.4142 |
| case-14 | en | C21 400 chars | ok | 1 | 1.0 | 1 | 0.3333 | 0.6761 |
| case-14 | en | C18 parent 1600 child 400 | ok | 1 | 1.0 | 1 | 0.3333 | 0.6761 |
| case-14 | en | C18 parent 1600 child 400 + parent expansion (offline span metric, not R22/R25) | ok | 1 | 1.0 | 1 | 1.0 | 0.6761 |
| case-14 | en | C15 model | requires_models | 1 |  |  |  | 0.0000 |
| case-14 | en | C16 model | requires_models | 1 |  |  |  | 0.0000 |
| case-14 | en | C17 model | requires_models | 1 |  |  |  | 0.0000 |
| case-14 | en | C19 adaptive | ok | 1 | 1.0 | 1 | 0.3333 | 0.6108 |

## Limits

This measures one small authored fixture with a lexical ranker and does not measure answer quality.
