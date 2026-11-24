# R46: CRAG research reproduction (blocked)

Status: blocked. Source: arXiv 2401.15884, "Corrective Retrieval Augmented Generation". ADR-0022, D6.

## What the method needs
CRAG grades retrieved documents with a trained retrieval evaluator, then takes one of three actions (use, correct, or search the web), and refines the kept knowledge by decomposing it into strips, filtering the strips and recomposing them.

## Why we stop
A reproduction needs the authors' components. We have none of them, and a general reranker or LLM grader is a different technique: it is R29 (P8.1), not R46. Calling such a grader CRAG would misstate fidelity.

## Exact requirements (ADR-0022, D6)
- The authors' fine-tuned T5-large retrieval evaluator (link and licence from the official repo).
- Knowledge refinement: decompose, filter, recompose.
- Web search through an approved, budgeted provider in enriched mode.

## Owner decision
R46 stays `blocked` (decided 2026-11-23, recorded in ADR-0022). No code, demo or model result exists for it.

## Losing and success cases
None run. The catalogue entry stays with status `blocked` and is not removed.

## Open items
The faithful path above, once the evaluator, refinement and a web search budget are approved.
