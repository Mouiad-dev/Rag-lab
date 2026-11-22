# P7.3 R48 KAG-style retrieval: demo output

Command: `uv run python docs/techniques/lessons/demos/p73_kag_demo.py`. Date: 2026-11-22. Offline; scripted extraction and plans; shows schema checks, logical forms and grounding, not extraction or planning quality.

Scripted chat (fixed JSON replies, not a model), test-only lexical embedder, in-memory store. R48 here is an adaptation of the KAG idea (schema-constrained extraction, logical-form planning, deterministic execution), not OpenSPG. The derived answer lives in the trace only and is never evidence; hits are original chunks. None of this is a benchmark.

## 1. Schema

Paragraph labels: K0 to K8 carry facts, F0 to F8 are filler; 18 paragraphs, one chunk each. Offsets are Unicode code points in the view text.

Schema `demo` version `1`, sha256 `37ffa8cf1acecf3f45db80002f376fa8234b2f3540e1f8cd53ed51a05dfc434b`.

| entity type | property | value kind | cardinality |
|---|---|---|---|
| Organization | founded_year | year | one |
| Person | - | - | - |

| relation | domain | range | cardinality |
|---|---|---|---|
| founded | Person | Organization | many |
| supplies | Organization | Organization | many |
| leads | Person | Organization | one |

Prompt SHAs (stable): extraction `1a9fab19a9c4e872d4207f61f75ff07789ed61118801fbc96be54662d3fbc016`, plan `c3fdb6a2eb6dc09dd4c387b82cf30a55b0448ee9b76bf9d30a72ea2911af52d7`, re-plan `917ec9c8ddb6b58c554e654beb714db95873828212340077e985b727af39aac3`.

## 2. Graph

9 entities and 9 facts kept from 18 chunks; 11 items rejected; 1 conflict(s). Every kept entity and fact carries source spans; nothing is kept without a quote that occurs exactly once in its chunk.

**Entities** (one row per mention)

| entity | type | paragraph and span | quote |
|---|---|---|---|
| Lina Haddad | Person | K0 [0, 11) | Lina Haddad |
| Cedar Labs | Organization | K0 [20, 30) | Cedar Labs |
| Cedar Labs | Organization | K1 [117, 127) | Cedar Labs |
| Harbor Authority | Organization | K1 [157, 173) | Harbor Authority |
| Harbor Authority | Organization | K2 [274, 290) | Harbor Authority |
| Harbor Authority | Organization | K3 [418, 434) | Harbor Authority |
| Harbor Authority | Organization | K7 [874, 890) | Harbor Authority |
| Omar Nasser | Person | K2 [252, 263) | Omar Nasser |
| Delta Works | Organization | K3 [370, 381) | Delta Works |
| Delta Works | Organization | K6 [750, 761) | Delta Works |
| Blue Reef | Organization | K4 [513, 522) | Blue Reef |
| Blue Reef | Organization | K5 [623, 632) | Blue Reef |
| Nour Saleh | Person | K6 [733, 743) | Nour Saleh |
| Gulf Marine | Organization | K7 [844, 855) | Gulf Marine |
| شركة النور | Organization | K8 [974, 984) | شركة النور |

**Facts** (one row per evidence span)

| fact | paragraph and span | quote |
|---|---|---|
| Lina Haddad founded Cedar Labs | K0 [0, 39) | Lina Haddad founded Cedar Labs in 2019. |
| Cedar Labs founded_year 2019 | K0 [20, 38) | Cedar Labs in 2019 |
| Cedar Labs supplies Harbor Authority | K1 [117, 174) | Cedar Labs supplies tide sensors to the Harbor Authority. |
| Omar Nasser leads Harbor Authority | K2 [252, 291) | Omar Nasser leads the Harbor Authority. |
| Delta Works founded_year 2010 | K3 [370, 398) | Delta Works, founded in 2010 |
| Delta Works supplies Harbor Authority | K3 [370, 435) | Delta Works, founded in 2010, also supplies the Harbor Authority. |
| Blue Reef founded_year 2012 | K4 [513, 543) | Blue Reef was founded in 2012. |
| Blue Reef founded_year 2014 | K5 [623, 653) | Blue Reef was founded in 2014. |
| شركة النور founded_year 2015 | K8 [974, 999) | شركة النور تأسست عام 2015 |

**Rejected items** (kept out of the graph, by reason)

| paragraph | kind | item index | reason |
|---|---|---|---|
| K0 | entity | 2 | unknown_type |
| K0 | fact | 2 | unknown_predicate |
| K1 | entity | 2 | ambiguous_quote |
| K1 | fact | 1 | unknown_object |
| K2 | fact | 1 | range_violation |
| K3 | fact | 2 | value_not_in_quote |
| K4 | fact | 1 | bad_literal |
| K5 | fact | 1 | unknown_subject |
| K6 | fact | 0 | quote_not_found |
| K7 | fact | 0 | name_not_in_quote |
| K8 | entity | 10 | too_many_items |

Two rejections matter later. K6 has a scripted fact whose quote is not in the chunk (`quote_not_found`), so the graph does not know who leads Delta Works. K7 has a scripted fact whose subject is not in its quote (`name_not_in_quote`), so the graph does not know Gulf Marine supplies the Harbor Authority.

**Conflict** (reported, never resolved)

| entity | property (cardinality one) | value | paragraph and span |
|---|---|---|---|
| Blue Reef | founded_year | 2012 | K4 [513, 543) |
| Blue Reef | founded_year | 2014 | K5 [623, 653) |

**Mutual index** (facts to chunks and chunks to facts) for two paragraphs

| chunk | fact (from `facts_for_chunk`) | chunks (from `chunks_for_fact`) |
|---|---|---|
| K0 | Cedar Labs founded_year 2019 | K0 |
| K0 | Lina Haddad founded Cedar Labs | K0 |
| K1 | Cedar Labs supplies Harbor Authority | K1 |

Store round trip (build, store, read again): graph equal = True. JSON round trip: graph equal = True. Every kept span quote equals the view slice: True.

## 3. Questions answered by the executor

Each plan is the scripted chat reply, parsed and checked against the schema before it runs. Bindings are shown by name. The derived answer is computed from graph facts; the supporting facts and quotes below are the evidence behind it, and the hits are the original chunks of those facts. Hit order follows the hash-derived fact order, not relevance.

**Q0**: Who founded Cedar Labs?

```json
{"steps": [{"id": "s1", "object": {"entity": {"name": "Cedar Labs", "type": "Organization"}}, "op": "retrieve", "predicate": "founded", "subject": null}, {"id": "s2", "inputs": ["s1"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | Lina Haddad | - |
| s2 | output | ok | Lina Haddad | - |

Final status `answered`. Derived answer (from the trace, not evidence): Lina Haddad. Hits (original chunks): K0.

| supporting fact | paragraph and span | quote |
|---|---|---|
| Lina Haddad founded Cedar Labs | K0 [0, 39) | Lina Haddad founded Cedar Labs in 2019. |

**Q1**: In which year was Cedar Labs founded?

```json
{"steps": [{"id": "s1", "object": null, "op": "retrieve", "predicate": "founded_year", "subject": {"entity": {"name": "Cedar Labs", "type": "Organization"}}}, {"id": "s2", "inputs": ["s1"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | 2019 | - |
| s2 | output | ok | 2019 | - |

Final status `answered`. Derived answer (from the trace, not evidence): 2019. Hits (original chunks): K0.

| supporting fact | paragraph and span | quote |
|---|---|---|
| Cedar Labs founded_year 2019 | K0 [20, 38) | Cedar Labs in 2019 |

**Q2**: How many organizations supply the Harbor Authority?

```json
{"steps": [{"id": "s1", "object": {"entity": {"name": "Harbor Authority", "type": "Organization"}}, "op": "retrieve", "predicate": "supplies", "subject": null}, {"id": "s2", "input": "s1", "op": "count"}, {"id": "s3", "inputs": ["s2"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | Cedar Labs, Delta Works | - |
| s2 | count | ok | 2 | - |
| s3 | output | ok | 2 | - |

Final status `answered`. Derived answer (from the trace, not evidence): 2. Hits (original chunks): K3, K1.

| supporting fact | paragraph and span | quote |
|---|---|---|
| Cedar Labs supplies Harbor Authority | K1 [117, 174) | Cedar Labs supplies tide sensors to the Harbor Authority. |
| Delta Works supplies Harbor Authority | K3 [370, 435) | Delta Works, founded in 2010, also supplies the Harbor Authority. |

**Q3 (multi-hop)**: Who leads the organization that Cedar Labs supplies?

```json
{"steps": [{"id": "s1", "object": null, "op": "retrieve", "predicate": "supplies", "subject": {"entity": {"name": "Cedar Labs", "type": "Organization"}}}, {"id": "s2", "object": {"step": "s1"}, "op": "retrieve", "predicate": "leads", "subject": null}, {"id": "s3", "inputs": ["s2"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | Harbor Authority | - |
| s2 | retrieve | ok | Omar Nasser | - |
| s3 | output | ok | Omar Nasser | - |

Final status `answered`. Derived answer (from the trace, not evidence): Omar Nasser. Hits (original chunks): K2, K1.

| supporting fact | paragraph and span | quote |
|---|---|---|
| Cedar Labs supplies Harbor Authority | K1 [117, 174) | Cedar Labs supplies tide sensors to the Harbor Authority. |
| Omar Nasser leads Harbor Authority | K2 [252, 291) | Omar Nasser leads the Harbor Authority. |

**count with filter**: How many organizations founded after 2015 supply the Harbor Authority?

```json
{"steps": [{"id": "s1", "object": {"entity": {"name": "Harbor Authority", "type": "Organization"}}, "op": "retrieve", "predicate": "supplies", "subject": null}, {"cmp": "gt", "id": "s2", "input": "s1", "op": "filter", "predicate": "founded_year", "value": 2015}, {"id": "s3", "input": "s2", "op": "count"}, {"id": "s4", "inputs": ["s3"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | Cedar Labs, Delta Works | - |
| s2 | filter | ok | Cedar Labs | - |
| s3 | count | ok | 1 | - |
| s4 | output | ok | 1 | - |

Final status `answered`. Derived answer (from the trace, not evidence): 1. Hits (original chunks): K0, K1.

| supporting fact | paragraph and span | quote |
|---|---|---|
| Cedar Labs founded_year 2019 | K0 [20, 38) | Cedar Labs in 2019 |
| Cedar Labs supplies Harbor Authority | K1 [117, 174) | Cedar Labs supplies tide sensors to the Harbor Authority. |

**Arabic**: متى تأسست شركة النور؟

```json
{"steps": [{"id": "s1", "object": null, "op": "retrieve", "predicate": "founded_year", "subject": {"entity": {"name": "شركة النور", "type": "Organization"}}}, {"id": "s2", "inputs": ["s1"], "op": "output"}]}
```

| step | op | status | bindings | missing |
|---|---|---|---|---|
| s1 | retrieve | ok | 2015 | - |
| s2 | output | ok | 2015 | - |

Final status `answered`. Derived answer (from the trace, not evidence): 2015. Hits (original chunks): K8.

| supporting fact | paragraph and span | quote |
|---|---|---|
| شركة النور founded_year 2015 | K8 [974, 999) | شركة النور تأسست عام 2015 |

## 4. Re-planning

At most one extra plan call, only when the plan itself failed (`invalid_logical_form`, `unsupported_operator`, `schema_violation`, `unknown_entity`). Data problems such as missing facts or conflicts are not re-planned. Re-plan prompt sha256 `917ec9c8ddb6b58c554e654beb714db95873828212340077e985b727af39aac3`.

**Success: a misspelled entity is fixed by the second plan** (Who leads the organization that Cedar Labs supplies?)

| attempt | prompt | reason or execution status | detail |
|---|---|---|---|
| 1 | plan prompt | incomplete (an entity not found in the graph counts as a plan failure) | sent to re-plan: reason `unknown_entity`, detail: steps with an entity not found in the graph: s1 |
| 2 | re-plan prompt | answered | - |

Plan calls: 2 (usage records 2). Final status `answered`. Derived answer: Omar Nasser. Hits: K2, K1.

**Failure: the second plan is also unusable** (Is Cedar Labs bigger than Delta Works?)

| attempt | prompt | reason or execution status | detail |
|---|---|---|---|
| 1 | plan prompt | unsupported_operator | unsupported operator: deduce |
| 2 | re-plan prompt | invalid_logical_form | model output was not a complete JSON object |

Plan calls: 2 (usage records 2). Final status `rejected`. Derived answer: none. Hits: none.

## 5. R48 against R38 and R02

Metric: span recall = gold paragraph spans fully inside one hit / number of gold spans, printed as `recall (hits=n)`. Gold counts (denominators): Q0 1, Q1 1, Q2 3, Q3 2, count with filter 2, Q4 1. R38 is built in memory from scripted entity-relation replies on the same text; R02 is dense with k=3 over the lexical test embedder. Six hand-written questions on a fixture built for the R48 tests, not a benchmark.

| question | gold spans | R48 recall (hits=n) | R38 recall (hits=n) | R02 k=3 recall (hits=n) |
|---|---|---|---|---|
| Q0 | 1 | 1.00 (hits=1), status answered | 1.00 (hits=2) | 1.00 (hits=3) |
| Q1 | 1 | 1.00 (hits=1), status answered | 1.00 (hits=2) | 1.00 (hits=3) |
| Q2 | 3 | 0.67 (hits=2), status answered | 0.67 (hits=3) | 0.67 (hits=3) |
| Q3 | 2 | 1.00 (hits=2), status answered | 0.50 (hits=2) | 1.00 (hits=3) |
| count with filter | 2 | 1.00 (hits=2), status answered | 0.50 (hits=3) | 0.50 (hits=3) |
| Q4 | 1 | 0.00 (hits=0), status incomplete | 0.00 (hits=1) | 1.00 (hits=3) |

R38 and R02 return chunks only: they cannot count, filter or compare, so on the count questions their recall says nothing about the answer value. Per row on span recall: R48 ties R02 on Q0, Q1, Q2 and Q3; it beats both R38 and R02 on the count with filter (1.00 against 0.50 for each); it beats R38 on Q3 (1.00 against 0.50); and it loses only on Q4, the extraction miss (0.00 against 1.00 for R02). These are six scripted cases on one fixture, not a benchmark and not evidence about real corpora. R48 also returns a derived answer in its trace (section 3) and abstains explicitly.

## 6. Losing cases

| case | question | status | derived answer | hits | rejection reasons by attempt | plan calls |
|---|---|---|---|---|---|---|
| Extraction miss (Q4) | Who leads Delta Works? | incomplete | none | 0 | - | 1 |
| Partial graph (Q2 with the Gulf Marine fact missing) | How many organizations supply the Harbor Authority? | answered | 2 | 2 | - | 1 |
| Conflict | When was Blue Reef founded? | conflict | none | 0 | - | 1 |
| `unsupported_operator` | Is Cedar Labs bigger than Delta Works? | rejected | none | 0 | unsupported_operator then unsupported_operator | 2 |
| `schema_violation` | Who owns Cedar Labs? | rejected | none | 0 | schema_violation then schema_violation | 2 |
| Invalid plan | Who runs the port? | rejected | none | 0 | invalid_logical_form then invalid_logical_form | 2 |

The `unsupported_operator` row uses the shared scripted re-plan reply (the same `deduce` plan again), so both attempts fail alike. Section 4 asks the same question with a different scripted re-plan reply (text that is not JSON), so its second attempt fails differently (`invalid_logical_form`).

- Extraction miss: the scripted extraction dropped the Delta Works leadership fact, so R48 says `incomplete` and returns no hits. Dense R02 with k=3 still returns the K6 paragraph (recall 1.00, section 5). Abstaining is honest here but loses the evidence.
- Partial graph: the true count of suppliers is 3 (K1, K3, K7) but the graph holds 2, so the derived answer is wrong yet the status is `answered`, with 2 of 3 supplier spans as hits. Nothing in the trace shows it is incomplete; only the source text would.
- Conflict: Blue Reef has two founded_year values (2012 in K4, 2014 in K5) and the property has cardinality one, so the executor reports `conflict`, returns no answer and no hits, and does not pick one.
- `unsupported_operator`: `deduce` is not in the grammar. `schema_violation`: the relation `owns` is not declared. Invalid plan: the reply is not JSON. Per the P7.3-08 notes, filter or sort on a `many` property is also a `schema_violation` (not run in this demo). In each case the single re-plan did not help here (2 plan calls), the status is `rejected`, and there is no fallback to chunk retrieval that could be mistaken for R48.

The cited-answer path (answer service, citations of the original chunks) is covered by `tests/answering/test_kag_answer.py`; this demo does not rebuild the answering stack.

## 7. Cost

Scripted chat; token counts are the fake model's estimates, not provider figures.

| stage | chat calls | input tokens | output tokens |
|---|---|---|---|
| R48 extraction (build, one call per chunk) | 18 | 3697 | 781 |

| section | question (plan calls incl. re-plans) | calls | input tokens | output tokens |
|---|---|---|---|---|
| 3 | Who founded Cedar Labs? | 1 | 230 | 48 |
| 3 | In which year was Cedar Labs founded? | 1 | 233 | 51 |
| 3 | How many organizations supply the Harbor Authority? | 1 | 233 | 51 |
| 3 | Who leads the organization that Cedar Labs supplies? | 1 | 234 | 52 |
| 3 | How many organizations founded after 2015 supply the Harbor Authority? | 1 | 236 | 54 |
| 3 | متى تأسست شركة النور؟ | 1 | 230 | 48 |
| 4 | Who leads the organization that Cedar Labs supplies? | 2 | 580 | 152 |
| 4 | Is Cedar Labs bigger than Delta Works? | 2 | 543 | 115 |
| 5 | Who founded Cedar Labs? | 1 | 230 | 48 |
| 5 | In which year was Cedar Labs founded? | 1 | 233 | 51 |
| 5 | How many organizations supply the Harbor Authority? | 1 | 233 | 51 |
| 5 | Who leads the organization that Cedar Labs supplies? | 1 | 234 | 52 |
| 5 | How many organizations founded after 2015 supply the Harbor Authority? | 1 | 236 | 54 |
| 5 | Who leads Delta Works? | 1 | 230 | 48 |
| 6 | Who leads Delta Works? | 1 | 230 | 48 |
| 6 | How many organizations supply the Harbor Authority? | 1 | 233 | 51 |
| 6 | When was Blue Reef founded? | 1 | 231 | 49 |
| 6 | Is Cedar Labs bigger than Delta Works? | 2 | 543 | 115 |
| 6 | Who owns Cedar Labs? | 2 | 556 | 128 |
| 6 | Who runs the port? | 2 | 542 | 114 |

Executor runs make no model call. A question run in more than one section is listed once per run. Update cost = full rebuild; not measured separately.

## Live run (orchestrator)

<orchestrator>
Live run result goes here (P7.3-10). Live output is not deterministic and is recorded separately.
</orchestrator>

