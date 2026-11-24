# P6.3 demo | uv run python docs/techniques/lessons/demos/p63_packing_citations_demo.py | 2026-11-10 | offline; no model; tiny hand-made fixtures; not a benchmark

## R26 success (EN)
| sentence | overlap | reason | constraints |
|---|---:|---|---|
| The bridge opened on 3 March 2019. | 1 | overlap | 3, 2019, march |
| Tolls are 5 USD per car. | 1 | overlap | 5, 5 usd |
| This does not apply to bicycles. | 0 | qualifier | not |
| The cafe sells tea. | 0 | no_overlap | () |
lost_constraints: ()

## R26 success (AR)
| sentence | overlap | reason | constraints |
|---|---:|---|---|
| افتتح الجسر عام ٢٠١٩. | 1 | overlap | 2019 |
| رسوم الجسر ٥ دولار. | 2 | overlap | 5, 5 دولار |
| لا تنطبق الرسوم على الدراجات. | 0 | qualifier | لا |
| المقهى يبيع الشاي. | 0 | no_overlap | () |
lost_constraints: ()

## R26 losing
| case | kept | lost_constraints |
|---|---|---|
| keep_qualifiers=False | The bridge opened on 3 March 2019. Tolls are 5 USD per car. | ('not',) |
| synonym miss: How much is the fee? | no hit | ('3', '2019', '5', 'not', 'march', '5 usd') |

## R27
| order | ranks in output order |
|---|---|
| rank | (1, 2, 3, 4, 5) |
| lost_in_middle | (1, 3, 5, 4, 2) |
| source_order | (2, 3, 1, 4, 5) |

Budget-20 drops:
| rank | reason | cost |
|---:|---|---:|
| 4 | budget | 4 |
| 5 | budget | 7 |

R27 losing: oversized rank-1 hit; packed items=()
| rank | reason | cost |
|---:|---|---:|
| 1 | budget | 7 |

## Citation validator
| claim | status | flags | overlap |
|---|---|---|---:|
| Tolls are 5 USD per car . | supported_lexical | () | 1.0 |
| The bridge opened in 2019 . | supported_lexical | () | 1.0 |
| Bicycles pay 7 USD . | unsupported_lexical | (('constraint_missing', '7'), ('constraint_missing', '7 usd')) | 0.6666666666666666 |
| It has a cafe. | uncited | () | None |
| The cafe sells tea . | invalid_citation | (('not_provided', 'S3'),) | None |
| Hours are long . | invalid_citation | (('unknown_marker', 'S9'),) | None |
False positive: `Tolls apply to bicycles [S1].` => `supported_lexical`, overlap `1.0`. Lexical rule, not entailment.

## Usage
0 model calls. Tokenizer: `lab-regex-v1` (an estimate, not a model tokenizer).
