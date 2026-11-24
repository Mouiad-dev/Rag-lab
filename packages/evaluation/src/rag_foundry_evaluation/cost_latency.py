"""Deterministic cost and latency report: joins per-stage usage with measured call timings."""

import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .report import _table
from .timing import CallTiming

_HEADER = (
    "Latency is wall-clock time on this machine for this run. It is not reproducible and is not part of "
    "results_json. Cold is the first call per stage and model in this process; warm is every later call. No answer "
    "or embedding cache exists, so cold versus warm reflects only process, connection and model-load effects. "
    "TTFT (time to first token) exists only for streamed calls; complete calls show not_applicable."
)


class StageLatency(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str
    calls: int
    failed: int
    cold_ms: float | None
    warm_median_ms: float | None
    total_median_ms: float | None
    total_max_ms: float | None
    ttft_median_ms: float | None
    ttft_status: str


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 3) if values else None


def summarize_timings(records: Sequence[CallTiming]) -> list[StageLatency]:
    by_stage: dict[str, list[CallTiming]] = {}
    for record in records:
        by_stage.setdefault(record.stage, []).append(record)
    rows = []
    for stage in sorted(by_stage):
        calls = by_stage[stage]
        ok = [c for c in calls if c.ok]
        cold = [c.total_ms for c in ok if c.cold]
        ttfts = [c.ttft_ms for c in ok if c.ttft_ms is not None]
        if not any(c.mode == "stream" for c in calls):
            status = "not_applicable"
        elif ttfts:
            status = "measured"
        else:
            status = "no_tokens"
        rows.append(
            StageLatency(
                stage=stage,
                calls=len(calls),
                failed=len(calls) - len(ok),
                cold_ms=round(cold[0], 3) if cold else None,
                warm_median_ms=_median([c.total_ms for c in ok if not c.cold]),
                total_median_ms=_median([c.total_ms for c in ok]),
                total_max_ms=round(max(c.total_ms for c in ok), 3) if ok else None,
                ttft_median_ms=_median(ttfts),
                ttft_status=status,
            )
        )
    return rows


def summarize_runner_timings(timings: dict) -> dict[str, dict]:
    out = {}
    for label, cases in timings.items():
        values = [float(row["total_ms"]) for row in cases.values()]
        out[label] = {
            "n": len(values),
            "cold_ms": round(values[0], 3) if values else None,
            "warm_median_ms": _median(values[1:]),
            "max_ms": round(max(values), 3) if values else None,
            "ttft": "not_applicable",
        }
    return out


def _ms(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_cost_latency(
    *,
    title: str,
    usage: dict[str, dict],
    timings: Sequence[CallTiming],
    runner_timings: dict | None = None,
    notes: Sequence[str] = (),
) -> str:
    latency = {row.stage: row for row in summarize_timings(timings)}
    rows = []
    for stage in sorted(set(usage) | set(latency)):
        u = usage.get(stage)
        lat = latency.get(stage)
        if u is None:
            tokens = ["-", "-", "-", "-"]
        else:
            cost = u.get("cost_usd")
            tokens = [
                u["input_tokens"],
                u["output_tokens"],
                "unknown" if cost is None else cost,
                u.get("estimated_token_calls", "-"),
            ]
        if lat is None:
            calls, failed = (u["calls"] if u else "-"), "-"
            ms = ["-", "-", "-", "-", "-"]
            ttft = "not_measured"
        else:
            calls, failed = lat.calls, lat.failed
            ms = [
                _ms(lat.cold_ms),
                _ms(lat.warm_median_ms),
                _ms(lat.total_median_ms),
                _ms(lat.total_max_ms),
                _ms(lat.ttft_median_ms),
            ]
            ttft = lat.ttft_status
        rows.append([stage, calls, failed, *tokens, *ms, ttft])
    header = [
        "stage",
        "calls",
        "failed",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "estimated_token_calls",
        "cold_ms",
        "warm_median_ms",
        "total_median_ms",
        "total_max_ms",
        "ttft_median_ms",
        "ttft",
    ]
    lines = [f"# {title}", "", _HEADER, ""]
    lines += _table(header, rows)
    if runner_timings is not None:
        lines += ["", "## Retrieval timings", ""]
        summary = summarize_runner_timings(runner_timings)
        lines += _table(
            ["label", "n", "cold_ms", "warm_median_ms", "max_ms", "ttft"],
            [
                [label, s["n"], _ms(s["cold_ms"]), _ms(s["warm_median_ms"]), _ms(s["max_ms"]), s["ttft"]]
                for label, s in summary.items()
            ],
        )
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"
