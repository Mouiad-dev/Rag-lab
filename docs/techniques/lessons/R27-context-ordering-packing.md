# R27: Context ordering and packing

## Intuition
Select whole hits in rank order within an estimated token budget, then arrange placed evidence for the prompt.

## Mechanism
The packer computes budget as window minus reserved output, prompt overhead, and fixed-text tokens. It accounts for each hit plus per-item overhead, drops contained hits when enabled, and after the first budget overflow drops remaining candidates. `lost_in_middle` places alternating rank-order items at the front and reverses the rest; `source_order` sorts by view and offsets.

## Config and defaults
`order="rank"`; `context_window_tokens=None` (window required from config or call, configured value must not exceed model window; 256..2,000,000); `prompt_overhead_tokens=256` (0..100000); `per_item_overhead_tokens=16` (0..1000); `drop_contained=True`. At most 50 unique hits/ranks are accepted. Output reserve and window constraints are checked at packing time.

## Fidelity
ADAPTATION: `lost_in_middle` order follows the finding of Liu et al. 2023, "Lost in the Middle: How Language Models Use Long Contexts"; the paper measures, it does not prescribe this order. Token counts are an estimate (owner accepted `lab-regex-v1`).

## Metering
Zero model calls. The tokenizer is `lab-regex-v1`, not a model tokenizer.

## Runnable demo
`uv run --no-sync python docs/techniques/lessons/demos/p63_packing_citations_demo.py`

## Observations (quoted exactly)
Success: `| lost_in_middle | (1, 3, 5, 4, 2) |`

Losing case: `| 1 | budget | 7 |`

## Limitations
Estimated costs can differ from the eventual model tokenizer. The ordering is a heuristic, and these tiny fixtures do not measure answer quality. Provenance retains the exact original hits and their ranks; nothing synthetic is created.

## Status
Implemented; [primary source](https://arxiv.org/abs/2307.03172). Evidence: `docs/evidence/P6.3/packing_and_citations.md`.
