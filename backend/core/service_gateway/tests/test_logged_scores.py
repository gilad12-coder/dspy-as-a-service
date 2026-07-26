"""Tests for the named-score logging channel (``log_metrics``).

Covers the strict logging contract, per-example capture through
:class:`LoggedScoreRecorder`, macro-averaging, the ``evaluate_on_test``
integration, the ``load_metric_from_code`` namespace injection, and the
validation probe's capture of logged scores (real subprocesses).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.service_gateway.optimization.data import load_metric_from_code
from core.service_gateway.optimization.logged_scores import (
    MAX_LOGGED_METRICS,
    LoggedScoreRecorder,
    aggregate_logged_metrics,
    drain_logged_metrics,
    log_metrics,
    reset_logged_metrics,
)
from core.service_gateway.optimization.optimizers import evaluate_on_test
from core.service_gateway.safe_exec import probe_metric_on_sample


@pytest.fixture(autouse=True)
def _clean_slot() -> None:
    """Reset the thread-local logging slot around every test."""
    reset_logged_metrics()
    yield
    reset_logged_metrics()


class _FloatConvertible:
    """Stand-in for numpy-style scalars that only expose ``__float__``."""

    def __float__(self) -> float:
        """Return the wrapped value."""
        return 0.5


def test_log_metrics_records_and_drains() -> None:
    """Logged values come back from drain and the slot is cleared."""
    log_metrics(precision=0.8, recall=0.5)
    assert drain_logged_metrics() == {"precision": 0.8, "recall": 0.5}
    assert drain_logged_metrics() == {}


def test_log_metrics_relogging_a_name_overwrites() -> None:
    """The last value logged for a name wins."""
    log_metrics(precision=0.1)
    log_metrics(precision=0.9)
    assert drain_logged_metrics() == {"precision": 0.9}


def test_log_metrics_accepts_bool_and_float_convertible() -> None:
    """Bools and ``__float__``-bearing scalars coerce to float."""
    log_metrics(flag=True, np_like=_FloatConvertible())
    assert drain_logged_metrics() == {"flag": 1.0, "np_like": 0.5}


def test_log_metrics_accepts_unicode_names() -> None:
    """Hebrew (and other unicode word) names are valid metric names."""
    log_metrics(**{"דיוק": 1.0, "exact match %": 0.5, "recall@5": 0.25})
    drained = drain_logged_metrics()
    assert drained["דיוק"] == 1.0
    assert drained["exact match %"] == 0.5


@pytest.mark.parametrize(
    "bad_name",
    ["", " leading", "bad!name", "line\nbreak", "x" * 51],
)
def test_log_metrics_rejects_bad_names(bad_name: str) -> None:
    """Names outside the contract raise with a log_metrics-prefixed message."""
    with pytest.raises(ValueError, match="log_metrics"):
        log_metrics(**{bad_name: 1.0})


@pytest.mark.parametrize("bad_value", ["0.5", b"1", None, float("nan"), float("inf"), object()])
def test_log_metrics_rejects_bad_values(bad_value: object) -> None:
    """Strings, non-numbers, and non-finite values raise."""
    with pytest.raises((TypeError, ValueError), match="log_metrics"):
        log_metrics(precision=bad_value)  # type: ignore[arg-type]


def test_log_metrics_caps_distinct_names() -> None:
    """The 21st distinct name raises; overwriting at the cap does not."""
    for i in range(MAX_LOGGED_METRICS):
        log_metrics(**{f"m{i}": 1.0})
    log_metrics(m0=0.0)
    with pytest.raises(ValueError, match="distinct metric names"):
        log_metrics(one_too_many=1.0)


def test_recorder_captures_per_example_without_bleed() -> None:
    """Each example keeps only the scores logged during its own metric call."""

    class _Ex:
        pass

    ex_a, ex_b, ex_c = _Ex(), _Ex(), _Ex()

    def metric(example: object, prediction: object) -> float:
        """Log a per-example precision except for the last example."""
        if example is ex_a:
            log_metrics(precision=1.0)
        elif example is ex_b:
            log_metrics(precision=0.0)
        return 1.0

    recorder = LoggedScoreRecorder(metric)
    assert recorder(ex_a, None) == 1.0
    assert recorder(ex_b, None) == 1.0
    assert recorder(ex_c, None) == 1.0
    assert recorder.scores_for(ex_a) == {"precision": 1.0}
    assert recorder.scores_for(ex_b) == {"precision": 0.0}
    assert recorder.scores_for(ex_c) == {}


def test_recorder_keeps_partial_logs_when_metric_raises() -> None:
    """Scores logged before a metric crash are retained and the error propagates."""

    class _Ex:
        pass

    ex = _Ex()

    def metric(example: object, prediction: object) -> float:
        """Log then fail."""
        log_metrics(recall=0.5)
        raise RuntimeError("boom")

    recorder = LoggedScoreRecorder(metric)
    with pytest.raises(RuntimeError, match="boom"):
        recorder(ex, None)
    assert recorder.scores_for(ex) == {"recall": 0.5}


def test_aggregate_logged_metrics_macro_averages_per_name() -> None:
    """Each name averages over the rows that logged it, in first-seen order."""
    rows = [
        {"logged_metrics": {"precision": 1.0, "recall": 0.0}},
        {"logged_metrics": {"precision": 0.0}},
        {},
        {"logged_metrics": "junk"},
        {"logged_metrics": {"recall": 1.0, "precision": "bad"}},
    ]
    aggregated = aggregate_logged_metrics(rows)
    assert aggregated == {"precision": 0.5, "recall": 0.5}
    assert list(aggregated) == ["precision", "recall"]


def test_aggregate_logged_metrics_empty_rows() -> None:
    """No logged metrics anywhere aggregates to an empty map."""
    assert aggregate_logged_metrics([{"score": 1.0}]) == {}


class _FakeEvalResult:
    """Stand-in for dspy.Evaluate's structured result with score + results."""

    def __init__(self, score: float, results: list) -> None:
        """Store the aggregate score and per-example results list."""
        self.score = score
        self.results = results


class _FakeExample:
    """Stand-in dataset example exposing a single 'answer' label."""

    def labels(self) -> list[str]:
        """Return the canonical label list for this example."""
        return ["answer"]


class _MetricDrivingEvaluator:
    """Fake dspy.Evaluate that actually invokes the metric per example."""

    def __init__(self, **kwargs: object) -> None:
        """Capture the devset and metric evaluate_on_test passes in."""
        self._devset = kwargs["devset"]
        self._metric = kwargs["metric"]

    def __call__(self, program: object) -> _FakeEvalResult:
        """Score every example through the (wrapped) metric."""
        results = []
        for example in self._devset:
            score = self._metric(example, None)
            results.append((example, None, score))
        return _FakeEvalResult(score=100.0, results=results)


def test_evaluate_on_test_rows_carry_logged_metrics() -> None:
    """Per-example rows include logged scores and aggregate cleanly."""
    examples = [_FakeExample(), _FakeExample()]
    values = iter([1.0, 0.0])

    def metric(example: object, prediction: object) -> float:
        """Log a varying precision per example."""
        log_metrics(precision=next(values))
        return 1.0

    with patch(
        "core.service_gateway.optimization.optimizers.dspy.Evaluate",
        _MetricDrivingEvaluator,
    ):
        score, rows = evaluate_on_test(object(), examples, metric, collect_per_example=True)

    assert score == pytest.approx(100.0)
    assert [row["logged_metrics"] for row in rows] == [{"precision": 1.0}, {"precision": 0.0}]
    assert aggregate_logged_metrics(rows) == {"precision": 0.5}


def test_evaluate_on_test_rows_omit_key_when_metric_never_logs() -> None:
    """A metric that never calls log_metrics leaves rows without the key."""
    examples = [_FakeExample()]

    def metric(example: object, prediction: object) -> float:
        """Return a score without logging."""
        return 1.0

    with patch(
        "core.service_gateway.optimization.optimizers.dspy.Evaluate",
        _MetricDrivingEvaluator,
    ):
        _, rows = evaluate_on_test(object(), examples, metric, collect_per_example=True)

    assert "logged_metrics" not in rows[0]


def test_load_metric_from_code_injects_log_metrics() -> None:
    """Metric code can call log_metrics without importing anything."""
    code = "def metric(gold, pred, trace=None):\n    log_metrics(precision=0.5)\n    return 1.0\n"
    metric = load_metric_from_code(code)
    reset_logged_metrics()
    assert metric(None, None) == 1.0
    assert drain_logged_metrics() == {"precision": 0.5}


def test_load_metric_from_code_fallback_ignores_injected_helper() -> None:
    """The single-callable fallback still finds a metric not named 'metric'."""
    code = "def my_scorer(gold, pred, trace=None):\n    return 1.0\n"
    metric = load_metric_from_code(code)
    assert metric.__name__ == "my_scorer"


_LOGGING_METRIC = (
    "def metric(example, prediction, trace=None):\n"
    "    log_metrics(precision=0.75, recall=0.5)\n"
    "    return 1.0\n"
)

_BAD_LOGGING_METRIC = (
    "def metric(example, prediction, trace=None):\n"
    "    log_metrics(precision='high')\n"
    "    return 1.0\n"
)


def test_probe_reports_logged_metrics() -> None:
    """The validation probe surfaces what the metric logged on the sample row."""
    probe = probe_metric_on_sample(
        metric_code=_LOGGING_METRIC,
        example_payload={"question": "q", "answer": "a"},
        prediction_payload={"answer": "a"},
        input_field_names=["question"],
    )
    assert probe.result_kind == "numeric"
    assert probe.logged_metrics == {"precision": 0.75, "recall": 0.5}


def test_probe_reports_logging_contract_violation_as_error() -> None:
    """A log_metrics contract violation fails the probe with the actionable message."""
    probe = probe_metric_on_sample(
        metric_code=_BAD_LOGGING_METRIC,
        example_payload={"question": "q", "answer": "a"},
        prediction_payload={"answer": "a"},
        input_field_names=["question"],
    )
    assert probe.result_kind == "error"
    assert probe.error is not None
    assert "log_metrics" in probe.error
    assert probe.logged_metrics == {}
