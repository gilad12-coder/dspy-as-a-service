"""Latency + throughput metrics for load-test scenarios.

Keeps the public surface small: callers record per-request samples, then
ask :class:`ScenarioMetrics` for a :class:`ScenarioResult` snapshot at the
end. Percentiles are computed via :class:`statistics.quantiles` for cheap,
dependency-free accuracy at the sample sizes we run (thousands, not millions).
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioResult:
    """Immutable snapshot returned by :meth:`ScenarioMetrics.finish`."""

    name: str
    total: int
    errors: int
    duration_seconds: float
    rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_max_ms: float
    status_codes: dict[int, int]
    extras: dict[str, Any] = field(default_factory=dict)
    slo_passed: bool | None = None
    slo_violations: list[str] = field(default_factory=list)
    slo_thresholds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view used by the reporter.

        Returns:
            A dict with snake_case keys mirroring this dataclass.
        """
        return {
            "name": self.name,
            "total": self.total,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 3),
            "rps": round(self.rps, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_p99_ms": round(self.latency_p99_ms, 1),
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "latency_max_ms": round(self.latency_max_ms, 1),
            "status_codes": dict(sorted(self.status_codes.items())),
            "extras": self.extras,
            "slo_passed": self.slo_passed,
            "slo_violations": list(self.slo_violations),
            "slo_thresholds": dict(self.slo_thresholds),
        }


class ScenarioMetrics:
    """Per-scenario metric collector.

    Not thread-safe; scenarios drive concurrency via ``asyncio.gather`` so
    samples land on the event-loop thread without needing a lock.
    """

    def __init__(self, name: str) -> None:
        """Initialise an empty collector.

        Args:
            name: Human-readable scenario name surfaced in the report.
        """
        self.name = name
        self._latencies_ms: list[float] = []
        self._status_codes: Counter[int] = Counter()
        self._errors = 0
        self._t_start = time.monotonic()
        self.extras: dict[str, Any] = {}

    def record(self, *, status_code: int, latency_seconds: float) -> None:
        """Record one request sample.

        Args:
            status_code: HTTP status (``0`` for a transport-level failure).
            latency_seconds: Wall-clock duration of the request.
        """
        self._latencies_ms.append(latency_seconds * 1000.0)
        self._status_codes[status_code] += 1
        if status_code == 0 or status_code >= 400:
            self._errors += 1

    def add_extra(self, key: str, value: Any) -> None:
        """Attach a scenario-specific KPI to the final report.

        Used for invariants like ``"unique_jobs_created": 1`` (idempotency)
        or ``"orphans_recovered": 3`` (failure injection) — values the
        latency view alone would not expose.

        Args:
            key: Name of the extra metric.
            value: JSON-serializable value to record.
        """
        self.extras[key] = value

    def finish(self) -> ScenarioResult:
        """Compute and return the immutable result snapshot.

        Returns:
            A :class:`ScenarioResult` capturing percentiles + counts at the
            moment of the call. Safe to call multiple times.
        """
        duration = max(time.monotonic() - self._t_start, 1e-9)
        total = sum(self._status_codes.values())
        latencies = sorted(self._latencies_ms)
        return ScenarioResult(
            name=self.name,
            total=total,
            errors=self._errors,
            duration_seconds=duration,
            rps=total / duration,
            latency_p50_ms=_pct(latencies, 0.50),
            latency_p95_ms=_pct(latencies, 0.95),
            latency_p99_ms=_pct(latencies, 0.99),
            latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
            latency_max_ms=latencies[-1] if latencies else 0.0,
            status_codes=dict(self._status_codes),
            extras=dict(self.extras),
        )


def _pct(sorted_values: list[float], p: float) -> float:
    """Return the ``p``-quantile of a presorted list using nearest-rank.

    Nearest-rank is good enough at thousands of samples — :mod:`statistics`
    quantile interpolation adds dependencies and noise for negligible
    accuracy gain at our sample sizes.

    Args:
        sorted_values: Pre-sorted samples in ascending order.
        p: Quantile in ``[0, 1]``.

    Returns:
        The p-th percentile value, or ``0.0`` for an empty input.
    """
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]
