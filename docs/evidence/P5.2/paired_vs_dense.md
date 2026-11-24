# Paired comparison: dense and lexical retrieval

Method: source-span relevance; a retrieved chunk hits when it covers at least 80% of a relevant span.
Dense branch uses `LexicalTestEmbedder` (hashed bag of words, 256 buckets, test-only). It is not a semantic model, so these rows compare lexical scoring variants, not semantic recall.
Hit@k uses k=1, 3, 5. Per-case RR@5 is 1/first_rank, where first_rank is the smallest r in 1..5 whose hits[:r] union covers at least 80% of a relevant span; mean MRR@5 is the mean of these reciprocal ranks.
R04 absent (needs PostgreSQL).
min-max has no fitted parameters; dev and validation report denominators are computed from the cases. dev sweep is flat (every weight gives dev MRR 1.0000); wl=0.5 chosen by tie-break, not learned

## Dev sweep

| w | dev MRR |
|---:|---:|
| 0.0 | 1.0000 |
| 0.1 | 1.0000 |
| 0.2 | 1.0000 |
| 0.3 | 1.0000 |
| 0.4 | 1.0000 |
| 0.5 | 1.0000 |
| 0.6 | 1.0000 |
| 0.7 | 1.0000 |
| 0.8 | 1.0000 |
| 0.9 | 1.0000 |
| 1.0 | 1.0000 |

## Per-split summary

| split | label | n | mean hit@1 | mean hit@5 | mean MRR@5 | W | T | L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev | R02 | 4 | 1.0000 | 1.0000 | 1.0000 | 0 | 4 | 0 |
| dev | R03 | 4 | 1.0000 | 1.0000 | 1.0000 | 0 | 4 | 0 |
| dev | R06+R07 | 4 | 1.0000 | 1.0000 | 1.0000 | 0 | 4 | 0 |
| dev | R06+R08(wl=0.5) | 4 | 1.0000 | 1.0000 | 1.0000 | 0 | 4 | 0 |
| validation | R02 | 5 | 0.8000 | 1.0000 | 0.8400 | 0 | 5 | 0 |
| validation | R03 | 5 | 0.8000 | 1.0000 | 0.9000 | 1 | 4 | 0 |
| validation | R06+R07 | 5 | 0.8000 | 1.0000 | 0.8500 | 1 | 4 | 0 |
| validation | R06+R08(wl=0.5) | 5 | 0.8000 | 1.0000 | 0.8667 | 1 | 4 | 0 |

## Per-case results

| case | cohort | split | label | first relevant rank | RR@5 | delta vs R02 | outcome | top-1 chunk (first 40 chars) |
|---|---|---|---|---:|---:|---:|---|---|
| en-fact | en | dev | R02 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-fact | en | dev | R03 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-fact | en | dev | R06+R07 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-fact | en | dev | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| ar-fact | ar | dev | R02 | 1 | 1.0000 | 0.0000 | tie | تحفظ مكتبة الواحة ثلاثة آلاف ومئتي مخطوط |
| ar-fact | ar | dev | R03 | 1 | 1.0000 | 0.0000 | tie | تحفظ مكتبة الواحة ثلاثة آلاف ومئتي مخطوط |
| ar-fact | ar | dev | R06+R07 | 1 | 1.0000 | 0.0000 | tie | تحفظ مكتبة الواحة ثلاثة آلاف ومئتي مخطوط |
| ar-fact | ar | dev | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | تحفظ مكتبة الواحة ثلاثة آلاف ومئتي مخطوط |
| id-exact | identifier | dev | R02 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1234 في المخزن الشمالي. |
| id-exact | identifier | dev | R03 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1234 في المخزن الشمالي. |
| id-exact | identifier | dev | R06+R07 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1234 في المخزن الشمالي. |
| id-exact | identifier | dev | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1234 في المخزن الشمالي. |
| en-club | en | dev | R02 | 1 | 1.0000 | 0.0000 | tie | Members of the chess club meet on Tuesda |
| en-club | en | dev | R03 | 1 | 1.0000 | 0.0000 | tie | Members of the chess club meet on Tuesda |
| en-club | en | dev | R06+R07 | 1 | 1.0000 | 0.0000 | tie | Members of the chess club meet on Tuesda |
| en-club | en | dev | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | Members of the chess club meet on Tuesda |
| id-ar | identifier | validation | R02 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| id-ar | identifier | validation | R03 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| id-ar | identifier | validation | R06+R07 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| id-ar | identifier | validation | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| ar-diacritic | ar | validation | R02 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| ar-diacritic | ar | validation | R03 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| ar-diacritic | ar | validation | R06+R07 | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| ar-diacritic | ar | validation | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | رقم القطعة AB-1243 في المخزن الجنوبي. |
| en-variant | en | validation | R02 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-variant | en | validation | R03 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-variant | en | validation | R06+R07 | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| en-variant | en | validation | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | The tidal observatory at Port Sallow rec |
| mixed | mixed | validation | R02 | 1 | 1.0000 | 0.0000 | tie | The Oasis Library (مكتبة الواحة) opens i |
| mixed | mixed | validation | R03 | 1 | 1.0000 | 0.0000 | tie | The Oasis Library (مكتبة الواحة) opens i |
| mixed | mixed | validation | R06+R07 | 1 | 1.0000 | 0.0000 | tie | The Oasis Library (مكتبة الواحة) opens i |
| mixed | mixed | validation | R06+R08(wl=0.5) | 1 | 1.0000 | 0.0000 | tie | The Oasis Library (مكتبة الواحة) opens i |
| paraphrase | en | validation | R02 | 5 | 0.2000 | 0.0000 | tie | Children fly kites in the park when the  |
| paraphrase | en | validation | R03 | 2 | 0.5000 | 0.3000 | win | Children fly kites in the park when the  |
| paraphrase | en | validation | R06+R07 | 4 | 0.2500 | 0.0500 | win | Children fly kites in the park when the  |
| paraphrase | en | validation | R06+R08(wl=0.5) | 3 | 0.3333 | 0.1333 | win | Children fly kites in the park when the  |
| unanswerable | unanswerable | validation | R02 | - | - | - | - | top-1 score 0.5000; context_chars@5 265 |
| unanswerable | unanswerable | validation | R03 | - | - | - | - | top-1 score 3.2962; context_chars@5 265 |
| unanswerable | unanswerable | validation | R06+R07 | - | - | - | - | top-1 score 0.0325; context_chars@5 265 |
| unanswerable | unanswerable | validation | R06+R08(wl=0.5) | - | - | - | - | top-1 score 0.9192; context_chars@5 265 |

Unanswerable rows report top-1 score and context_chars@5; top-1 scores use different scales and are not comparable across labels; no abstention threshold is evaluated. They are excluded from hit/MRR denominators: 1 case(s).

## Limits

Dev and validation share relevant documents (FACT_EN is used by en-fact, en-variant, and paraphrase; ID1/ID2 are near-duplicates), and a dev query is close to a validation query. OCR and table cohorts are absent. n=5 supports no significance claim. The dense branch uses a test-only lexical embedder; results describe these queries and source spans only.
