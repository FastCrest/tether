"""A/B comparison helper — 6-gate ship-worthy threshold rule.

Compare two BenchReports (baseline OFF vs candidate ON) against the
ship-rule from cuda-graphs research sidecar Lens 5. Reusable by every
perf feature (cuda-graphs, compile-cache, fp8-thor, action-similarity-fast-path,
cross-request-pipelining, dkv-cache-for-experts, language-layer-pruning-adaptive).

Gates (in evaluation order):

1. CI no-overlap — 95% CIs on means must not overlap (statistical signal)
2. Mean improvement floor — improvement >= mean_floor_pct (default 5%)
3. p99 regression cap — p99 must not regress > p99_cap_pct (default 5%)
4. p99.9 regression cap — p99.9 must not regress > p99_9_cap_pct (default 10%)
5. Jitter cap — jitter must not increase > jitter_cap_pct (default 20%)
6. Parity cos — captured output must match eager (>= 0.9999) when parity present

First failing gate determines the verdict. If all pass: SHIP.

The eager-fallback gate from the cuda-graphs research sidecar is intentionally
out-of-scope for this helper; that signal lives in Prometheus counters during
the bench run, not in the BenchReport. Caller can check it separately.

Reference: features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_research.md Lens 5
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from tether.bench.methodology import LatencyStats
from tether.bench.report import BenchReport


@dataclass(frozen=True)
class CompareGate:
    """Result of evaluating a single ship-rule gate."""
    name: str
    passed: bool
    measured: float
    threshold: float
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BenchCompareResult:
    """Result of A/B comparison between two BenchReports."""
    mean_delta_ms: float
    mean_delta_pct: float  # negative = improvement, positive = regression
    p99_delta_ms: float
    p99_delta_pct: float
    p99_9_delta_ms: float
    p99_9_delta_pct: float
    jitter_delta_pct: float
    ci_overlap: bool
    gates: tuple[CompareGate, ...]
    verdict: str  # one of: SHIP, NO_SIGNAL, INSUFFICIENT_GAIN, TAIL_REGRESSION, JITTER_REGRESSION, PARITY_FAILURE
    schema_version: int = 1

    @property
    def passes_ship_gates(self) -> bool:
        return self.verdict == "SHIP"

    @property
    def failed_gates(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "mean_delta_ms": self.mean_delta_ms,
            "mean_delta_pct": self.mean_delta_pct,
            "p99_delta_ms": self.p99_delta_ms,
            "p99_delta_pct": self.p99_delta_pct,
            "p99_9_delta_ms": self.p99_9_delta_ms,
            "p99_9_delta_pct": self.p99_9_delta_pct,
            "jitter_delta_pct": self.jitter_delta_pct,
            "ci_overlap": self.ci_overlap,
            "gates": [g.to_dict() for g in self.gates],
            "verdict": self.verdict,
        }


def _ci_overlap(stats_a: LatencyStats, stats_b: LatencyStats) -> bool:
    """True if the two 95% CIs on the means overlap (no statistical signal)."""
    a_low, a_high = stats_a.ci95_low_ms, stats_a.ci95_high_ms
    b_low, b_high = stats_b.ci95_low_ms, stats_b.ci95_high_ms
    return not (a_high < b_low or b_high < a_low)


def _delta_pct(before: float, after: float) -> float:
    """Percentage change: positive = `after` is larger (worse for latency)."""
    if before <= 0 or math.isnan(before):
        return float("nan")
    return (after - before) / before * 100.0


def compare_reports(
    report_off: BenchReport,
    report_on: BenchReport,
    *,
    mean_floor_pct: float = 5.0,
    p99_cap_pct: float = 5.0,
    p99_9_cap_pct: float = 10.0,
    jitter_cap_pct: float = 20.0,
    parity_floor: float = 0.9999,
) -> BenchCompareResult:
    """Compare two BenchReports against the 6-gate ship-worthy rule.

    Convention: `report_off` is the baseline (feature OFF); `report_on` has the
    feature enabled. Improvement = `report_on` faster than `report_off` (negative
    delta on latency metrics).

    Returns a BenchCompareResult with the verdict set by the FIRST failing gate.
    All gates are evaluated regardless of which fails first (so callers can see
    the full picture in `.gates`).
    """
    s_off = report_off.stats
    s_on = report_on.stats

    mean_delta_ms = s_on.mean_ms - s_off.mean_ms
    mean_delta_pct = _delta_pct(s_off.mean_ms, s_on.mean_ms)
    p99_delta_ms = s_on.p99_ms - s_off.p99_ms
    p99_delta_pct = _delta_pct(s_off.p99_ms, s_on.p99_ms)
    p99_9_delta_ms = s_on.p99_9_ms - s_off.p99_9_ms
    p99_9_delta_pct = _delta_pct(s_off.p99_9_ms, s_on.p99_9_ms)
    jitter_delta_pct = _delta_pct(s_off.jitter, s_on.jitter)
    ci_overlap = _ci_overlap(s_off, s_on)

    gates: list[CompareGate] = []

    gates.append(CompareGate(
        name="ci_no_overlap",
        passed=not ci_overlap,
        measured=0.0,
        threshold=0.0,
        detail=(
            f"95% CIs overlap: off=[{s_off.ci95_low_ms:.2f}, {s_off.ci95_high_ms:.2f}], "
            f"on=[{s_on.ci95_low_ms:.2f}, {s_on.ci95_high_ms:.2f}]"
            if ci_overlap else "95% CIs do not overlap"
        ),
    ))

    improvement_pct = -mean_delta_pct  # positive = improvement
    gates.append(CompareGate(
        name="mean_improvement_floor",
        passed=improvement_pct >= mean_floor_pct,
        measured=improvement_pct,
        threshold=mean_floor_pct,
        detail=f"mean improvement {improvement_pct:+.2f}% (floor {mean_floor_pct}%)",
    ))

    gates.append(CompareGate(
        name="p99_regression_cap",
        passed=p99_delta_pct <= p99_cap_pct,
        measured=p99_delta_pct,
        threshold=p99_cap_pct,
        detail=f"p99 delta {p99_delta_pct:+.2f}% (cap {p99_cap_pct}%)",
    ))

    gates.append(CompareGate(
        name="p99_9_regression_cap",
        passed=p99_9_delta_pct <= p99_9_cap_pct,
        measured=p99_9_delta_pct,
        threshold=p99_9_cap_pct,
        detail=f"p99.9 delta {p99_9_delta_pct:+.2f}% (cap {p99_9_cap_pct}%)",
    ))

    gates.append(CompareGate(
        name="jitter_increase_cap",
        passed=jitter_delta_pct <= jitter_cap_pct,
        measured=jitter_delta_pct,
        threshold=jitter_cap_pct,
        detail=f"jitter delta {jitter_delta_pct:+.2f}% (cap {jitter_cap_pct}%)",
    ))

    if report_on.parity is not None:
        cos = float(report_on.parity.get("cos", float("nan")))
        gates.append(CompareGate(
            name="parity_cos_floor",
            passed=(not math.isnan(cos)) and cos >= parity_floor,
            measured=cos,
            threshold=parity_floor,
            detail=f"parity cos {cos:.6f} (floor {parity_floor:.6f})",
        ))

    # First failing gate determines verdict (precedence: ci > mean > tail > jitter > parity)
    if not gates[0].passed:
        verdict = "NO_SIGNAL"
    elif not gates[1].passed:
        verdict = "INSUFFICIENT_GAIN"
    elif not gates[2].passed or not gates[3].passed:
        verdict = "TAIL_REGRESSION"
    elif not gates[4].passed:
        verdict = "JITTER_REGRESSION"
    elif len(gates) > 5 and not gates[5].passed:
        verdict = "PARITY_FAILURE"
    else:
        verdict = "SHIP"

    return BenchCompareResult(
        mean_delta_ms=mean_delta_ms,
        mean_delta_pct=mean_delta_pct,
        p99_delta_ms=p99_delta_ms,
        p99_delta_pct=p99_delta_pct,
        p99_9_delta_ms=p99_9_delta_ms,
        p99_9_delta_pct=p99_9_delta_pct,
        jitter_delta_pct=jitter_delta_pct,
        ci_overlap=ci_overlap,
        gates=tuple(gates),
        verdict=verdict,
    )


def render_compare_markdown(
    result: BenchCompareResult,
    label_off: str = "OFF",
    label_on: str = "ON",
) -> str:
    """Render the comparison as Markdown (PR-friendly)."""
    lines = [
        "# Tether Bench A/B Comparison",
        "",
        f"## Verdict: **{result.verdict}**",
        "",
        f"Comparing **{label_off}** (baseline) vs **{label_on}** (feature enabled).",
        "",
        "## Latency deltas",
        "",
        "| Metric | Delta (ms) | Delta (%) |",
        "|---|---:|---:|",
        f"| Mean | {result.mean_delta_ms:+.2f} | {result.mean_delta_pct:+.2f}% |",
        f"| p99  | {result.p99_delta_ms:+.2f} | {result.p99_delta_pct:+.2f}% |",
        f"| p99.9| {result.p99_9_delta_ms:+.2f} | {result.p99_9_delta_pct:+.2f}% |",
        f"| Jitter (relative) | n/a | {result.jitter_delta_pct:+.2f}% |",
        "",
        "_Negative delta = improvement; positive = regression._",
        "",
        "## Ship-worthy gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for g in result.gates:
        status = "PASS" if g.passed else "FAIL"
        lines.append(f"| `{g.name}` | {status} | {g.detail} |")
    lines.append("")
    return "\n".join(lines)
