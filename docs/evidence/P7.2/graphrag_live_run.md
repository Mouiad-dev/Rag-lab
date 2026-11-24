# P7.2-09 live smoke run (one run, not an evaluation)

Command: RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_graphrag_live.py -m live -v -s
Date: 2026-11-22. Model: gemma4:e4b on local Ollama (cost 0, no pull). Embeddings: offline scripted vectors.
Result: 1 passed in 95.49s. Output is not bit-identical across runs; no quality claim.
Summary: the test asserts structure only, so "passed" says nothing about R38 quality. R38 on QUESTION: `no_path_within_hops`, recall 0.00 on the model-built graph (42 entities, 22 relations, 2 rejected items). R39: answered, 1 traced sentence. 44 chat calls (17 extraction, 13 summaries, 13 map, 1 reduce).

```
tests/live/test_graphrag_live.py::test_graphrag_live_smoke graph entities=42 relations=22 chunks=17
extraction rejected=2 reasons=['unknown_endpoint']
stage=extraction calls=17 input_tokens=3234 output_tokens=3001 latency_ms=48856
r38 status=no_path_within_hops mode=path recall=0.00
communities=20 summaries=13 skipped_small=7 rejected_claims=0 rejected_summaries=0
stage=summaries calls=13 input_tokens=4983 output_tokens=2711 latency_ms=37308
r39 status=answered reason=None sentences=1
r39 rejected_points=0 invalid_map_replies=0 dropped_sentences=0
r39 answer=["Cedar Labs supplies tide sensors to the Harbor Authority."]
stage=map calls=13 input_tokens=4149 output_tokens=208 latency_ms=7818
stage=reduce calls=1 input_tokens=191 output_tokens=45 latency_ms=838
ledger calls=44 input_tokens=12557 output_tokens=5965
PASSED
```
