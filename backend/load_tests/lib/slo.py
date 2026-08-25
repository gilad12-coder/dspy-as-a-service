"""Explicit pass/fail service-level objectives for load-test scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import ScenarioResult


@dataclass(frozen=True)
class ScenarioSLO:
    """Thresholds that turn a latency report into a release gate."""

    max_error_rate_percent: float
    max_p95_ms: float
    max_p99_ms: float
    min_rps: float

    def as_dict(self) -> dict[str, float]:
        """Return thresholds with report-friendly names.

        Returns:
            A JSON-serializable threshold mapping.
        """
        return {
            "max_error_rate_percent": self.max_error_rate_percent,
            "max_p95_ms": self.max_p95_ms,
            "max_p99_ms": self.max_p99_ms,
            "min_rps": self.min_rps,
        }


def evaluate_slo(result: ScenarioResult, slo: ScenarioSLO) -> list[str]:
    """Return every SLO violation found in a scenario result.

    Args:
        result: Completed scenario metrics.
        slo: Maximum error/latency and minimum throughput thresholds.

    Returns:
        Human-readable violations; an empty list means the SLO passed.
    """
    if result.total == 0:
        return ["no requests completed"]

    violations: list[str] = []
    error_rate = result.errors / result.total * 100.0
    if error_rate > slo.max_error_rate_percent:
        violations.append(
            f"error rate {error_rate:.2f}% exceeded {slo.max_error_rate_percent:.2f}%",
        )
    if result.latency_p95_ms > slo.max_p95_ms:
        violations.append(
            f"p95 {result.latency_p95_ms:.1f}ms exceeded {slo.max_p95_ms:.1f}ms",
        )
    if result.latency_p99_ms > slo.max_p99_ms:
        violations.append(
            f"p99 {result.latency_p99_ms:.1f}ms exceeded {slo.max_p99_ms:.1f}ms",
        )
    if result.rps < slo.min_rps:
        violations.append(f"throughput {result.rps:.1f} rps fell below {slo.min_rps:.1f} rps")
    return violations


def apply_slo(result: ScenarioResult, slo: ScenarioSLO, *, extra_violations: list[str] | None = None) -> None:
    """Attach one SLO evaluation to a mutable scenario result.

    Args:
        result: Result to annotate for console, JSON, Markdown, and exit status.
        slo: Core HTTP thresholds to evaluate.
        extra_violations: Scenario-specific failures such as an SSE readiness miss.
    """
    violations = evaluate_slo(result, slo)
    violations.extend(extra_violations or [])
    result.slo_passed = not violations
    result.slo_violations = violations
    result.slo_thresholds = slo.as_dict()
