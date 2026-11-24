# R12: Self-query filters

## Intuition

A user question may carry a filter ("in the first report"). One chat call turns the question into a metadata filter. The model output is untrusted: it is checked against an allowlist and can only narrow the caller's scope.

## Mechanism

`SelfQuery.parse` sends the question and the offered documents (media type, creation date, name) to the chat port. The model sees short refs `d1..dn`, never document UUIDs. The reply must be one JSON object with exactly the keys `document_refs`, `media_types`, `created_from`, `created_to`. Unknown keys give `unknown_filter_field`; wrong types or values give `invalid_filter_value`; unparseable or truncated output gives `invalid_model_output`. Any UUID-looking string, or a ref that was not offered, always raises `filter_scope_widening` whatever `on_invalid` says. The result carries `status` (`applied`, `no_filter`, `rejected`), the filter, usage, the prompt SHA-256 and at most 500 characters of raw text. `retrieve_with_self_query` hands the filter to R11 as `extra_filter`, which intersects it.

## Config and defaults

`SelfQueryConfig`: `max_output_tokens` 256, `temperature` 0.0, `on_invalid` `fail` (or `no_filter`), `max_choices` 50. Cost is one chat call, recorded as usage on the result so it can be metered separately from retrieval.

## Fidelity

Adaptation. A fixed small filter schema over the stored metadata instead of an open-ended query language.

## Runnable demo

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p53_selection_demo.py
```

Full output: [selection_and_filters.md](../../evidence/P5.3/selection_and_filters.md). In the demo the chat is a scripted responder, not a model; the token counts come from the fake chat profile. Offline fixture; no model; not a benchmark.

## Success case

```text
| d1 | report-2024 |
| document_refs [d1] | fail | applied | None | document_ids -> report-2024 | 101/37 |
| retrieve_with_self_query | applied | 2 | report-2024 |
```

The scripted reply names ref `d1`; the filter narrows to one document and the end-to-end call returns only its 2 hits.

## Losing case

```text
| extra key language | no_filter | rejected | unknown_filter_field | filter=None | 96/32 |
| UUID instead of ref | no_filter | - | - | raised filter_scope_widening | - |
| UUID instead of ref | fail | - | - | raised filter_scope_widening | - |
```

An invented field is rejected, and in `no_filter` mode the retrieval runs without a self-query filter (nothing widens). A UUID raises in both modes. The cost of the safeguard is that a model reply with an extra key is thrown away even if the rest was usable.

## Live smoke run

One live run was done on 2026-11-09 with owner OK: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_self_query_live.py -m live -v -s`, 1 passed. Model `gemma4:e4b`, status `applied`, ref `d1`, 178 input / 82 output tokens, 17.1 s, cost 0. For a general question with one offered document, the model filled every field (seen in the test's `-s` output: refs, media type and a date range; recorded in `docs/tasks/P5.3-04-r12-self-query.md`). This is one smoke run, not an evaluation; no accuracy claim follows from it.

## Limitations

The filter quality depends on the model; the allowlist guards safety, not correctness. Only document, media type and date fields exist; `language` and `section` are not available (see R11).

## Status

Implemented; not validated or benchmarked.
