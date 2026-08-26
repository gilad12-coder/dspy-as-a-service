"""Unit tests for the load-test release-gate contract."""

from __future__ import annotations

import pytest

from load_tests.lib.metrics import ScenarioResult
from load_tests.lib.slo import ScenarioSLO, apply_slo, evaluate_slo
from load_tests.scenarios.mixed_realistic import MixedRealisticConfig


def _result(**overrides: float | int) -> ScenarioResult:
    """Build a passing result with selected numeric fields overridden.

    Args:
        **overrides: Numeric result fields to replace.

    Returns:
        A complete :class:`ScenarioResult` fixture.
    """
    values: dict[str, object] = {
        "name": "mixed_realistic",
        "total": 1000,
        "errors": 1,
        "duration_seconds": 10.0,
        "rps": 100.0,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 500.0,
        "latency_p99_ms": 900.0,
        "latency_mean_ms": 150.0,
        "latency_max_ms": 1200.0,
        "status_codes": {200: 999, 500: 1},
    }
    values.update(overrides)
    return ScenarioResult(**values)


def _slo() -> ScenarioSLO:
    """Return the unit-test SLO thresholds.

    Returns:
        A representative release gate.
    """
    return ScenarioSLO(
        max_error_rate_percent=0.5,
        max_p95_ms=1500.0,
        max_p99_ms=3000.0,
        min_rps=40.0,
    )


def test_evaluate_slo_accepts_a_result_within_every_threshold() -> None:
    """Return no violations when all four SLO dimensions pass."""
    assert evaluate_slo(_result(), _slo()) == []


def test_evaluate_slo_reports_every_breached_dimension() -> None:
    """Report errors, p95, p99, and throughput together after a bad run."""
    violations = evaluate_slo(
        _result(errors=10, rps=20.0, latency_p95_ms=2000.0, latency_p99_ms=4000.0),
        _slo(),
    )

    assert len(violations) == 4
    assert any("error rate" in violation for violation in violations)
    assert any("p95" in violation for violation in violations)
    assert any("p99" in violation for violation in violations)
    assert any("throughput" in violation for violation in violations)


def test_apply_slo_persists_thresholds_and_scenario_violations() -> None:
    """Attach a failed verdict when a scenario-specific invariant breaks."""
    result = _result()

    apply_slo(result, _slo(), extra_violations=["only 23/24 SSE streams opened"])

    assert result.slo_passed is False
    assert result.slo_violations == ["only 23/24 SSE streams opened"]
    assert result.slo_thresholds["max_error_rate_percent"] == 0.5


def test_evaluate_slo_rejects_an_empty_result() -> None:
    """Treat a scenario that issued no requests as a failed release gate."""
    assert evaluate_slo(_result(total=0, errors=0), _slo()) == ["no requests completed"]


def test_mixed_config_rejects_more_streams_than_submitters() -> None:
    """Reject an impossible workload shape before starting the stack."""
    with pytest.raises(ValueError, match="sse_connections"):
        MixedRealisticConfig(
            api_base_url="http://api",
            mock_lm_url="http://mock-lm/v1",
            frontend_base_url="http://frontend",
            virtual_users=10,
            submitting_users=2,
            sse_connections=3,
            ramp_seconds=1.0,
            soak_seconds=1.0,
            completion_timeout_seconds=1.0,
            think_time_min_seconds=0.1,
            think_time_max_seconds=0.2,
        )
