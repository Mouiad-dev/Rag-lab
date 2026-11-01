# ADR-0006: Observability with OpenTelemetry and LangSmith

Date: 2026-10-26
Status: accepted

## Context
The app needs traces of every AI step (retrieval, rerank, generation, judge, tools) to debug, compare techniques and measure cost and latency. Open-source users must be able to run without any SaaS account; some users already rely on LangSmith for LangChain/LangGraph tracing and evaluation.

## Decision
Use both, as independent and optional exporters over one instrumentation layer:
- **OpenTelemetry (default, always available):** the app emits OTel spans with GenAI semantic attributes (model, tokens, cost, latency). Users can export to any OTLP backend (for example Jaeger, Grafana Tempo, Phoenix, Langfuse) or to the console.
- **LangSmith (optional):** enabled when `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` are set; used for LangChain/LangGraph traces, datasets and evaluation runs.

Both can be on at the same time. Prompt/response content is redacted by default and only exported when the user opts in.

## Consequences
- Instrumentation lives in one shared package; code never calls LangSmith or an OTel exporter directly.
- Evaluation results are stored in the app database regardless of exporter, so no feature depends on LangSmith.
- Tests check that traces are emitted with no exporter configured and that redaction works.
