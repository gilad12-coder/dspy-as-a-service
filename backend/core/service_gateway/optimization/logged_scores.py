"""Named-score logging channel for user-authored metric code.

A metric hands the optimizer exactly one scalar, so composite metrics
(precision + recall averaged into F1, per-field verdicts averaged into an
overall match rate) lose their components — the run reports "0.72" with no
way to see that recall is what's dragging it down. This module gives metric
code a side channel: ``log_metrics(precision=0.8, recall=0.5)`` records
named scalars for the example currently being scored, without touching the
scalar the optimizer consumes.

The channel is a thread-local dict because the metric call and every
``log_metrics`` call it makes happen synchronously on the same thread, no
matter which executor (dspy.Evaluate workers, GEPA feedback closures) drives
the metric. :class:`LoggedScoreRecorder` resets the slot before each metric
call and drains it after, so values can never bleed between examples;
outside a recorder (training-time calls) the slot is overwritten in place
and stays bounded by the name cap.

The contract is deliberately strict — names and values are validated at log
time and violations raise ``ValueError`` with an actionable message, so the
validation probe (``safe_exec.probe_metric_on_sample``) rejects malformed
logging before a run ever starts.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from typing import Any

MAX_LOGGED_METRICS = 20
MAX_METRIC_NAME_LENGTH = 50

# Word characters (unicode-aware — Hebrew column names become Hebrew metric
# names) plus the punctuation a metric name plausibly needs ("f1_score",
# "recall@5", "exact-match %"). Anything else is rejected so every logged
# name renders cleanly in JSON, logs, and the UI.
_NAME_PATTERN = re.compile(r"\w[\w .%@/\-]*")

_local = threading.local()


def log_metrics(**scores: float) -> None:
    """Record named scalar scores for the example currently being scored.

    Available inside user metric code as a global (injected by
    ``load_metric_from_code``). Call it any number of times per example;
    re-logging a name overwrites the previous value. Use a fixed set of
    names across examples — each name is macro-averaged over the examples
    that logged it.

    Args:
        **scores: Named scalar values, e.g. ``precision=0.8, recall=0.5``.
            Names must be 1–50 characters of word characters (any
            language), spaces, or ``. % @ / -``, starting with a word
            character; values must be finite numbers. At most 20 distinct
            names per example.

    Raises:
        ValueError: When a name is malformed, a value is non-finite, or the
            distinct-name cap is exceeded.
        TypeError: When a value is not a number.
    """
    slot = getattr(_local, "scores", None)
    if slot is None:
        slot = {}
        _local.scores = slot
    for name, value in scores.items():
        if len(name) > MAX_METRIC_NAME_LENGTH or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"log_metrics: invalid metric name {name!r} — use 1-{MAX_METRIC_NAME_LENGTH} "
                "word characters, spaces, or . % @ / -, starting with a word character."
            )
        # float() would happily parse "0.5", so strings are rejected up front;
        # everything else float-convertible (numpy scalars, Decimal) is welcome.
        if isinstance(value, (str, bytes)):
            raise TypeError(
                f"log_metrics: value for {name!r} must be a number, got {type(value).__name__}."
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"log_metrics: value for {name!r} must be a number, got {type(value).__name__}."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"log_metrics: value for {name!r} must be finite, got {value!r}.")
        if name not in slot and len(slot) >= MAX_LOGGED_METRICS:
            raise ValueError(
                f"log_metrics: at most {MAX_LOGGED_METRICS} distinct metric names per example — "
                "log a fixed, small set of names."
            )
        slot[name] = numeric


def reset_logged_metrics() -> None:
    """Clear the current thread's logging slot before a metric call."""
    _local.scores = {}


def drain_logged_metrics() -> dict[str, float]:
    """Return and clear whatever the current thread's metric call logged."""
    slot = getattr(_local, "scores", None) or {}
    _local.scores = {}
    return dict(slot)


class LoggedScoreRecorder:
    """Metric wrapper that captures ``log_metrics`` output per example.

    Composition like ``MinibatchRecorder``: pass this in place of the raw
    metric and the return value is forwarded unmodified. Captured scores are
    keyed by example identity, matching how ``dspy.Evaluate`` hands the same
    example object to the metric and back in its results list.
    """

    def __init__(self, metric: Callable[..., Any]):
        """Wrap ``metric`` with per-example logged-score capture.

        Args:
            metric: The user metric callable to forward calls to.
        """
        self._metric = metric
        self._lock = threading.Lock()
        self._by_example: dict[int, dict[str, float]] = {}

    def __call__(self, example: Any, prediction: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped metric, capturing scores it logs for ``example``.

        Args:
            example: The DSPy ``Example`` being scored.
            prediction: The candidate program's output for ``example``.
            *args: Forwarded positional metric args (``trace`` etc.).
            **kwargs: Forwarded keyword metric args.

        Returns:
            The wrapped metric's return value, unmodified.
        """
        reset_logged_metrics()
        try:
            return self._metric(example, prediction, *args, **kwargs)
        finally:
            logged = drain_logged_metrics()
            if logged:
                with self._lock:
                    self._by_example[id(example)] = logged

    def scores_for(self, example: Any) -> dict[str, float]:
        """Return the scores logged while scoring ``example`` (empty if none).

        Args:
            example: The same example object the metric was invoked with.
        """
        with self._lock:
            return dict(self._by_example.get(id(example), {}))


def aggregate_logged_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Macro-average per-example logged scores into one value per name.

    Each name is averaged over the examples that logged it (not the full
    row count), so a metric that only logs a score when it applies still
    aggregates meaningfully. Name order follows first appearance across
    rows — the order the metric code logged them in.

    Args:
        rows: Per-example result dicts, each optionally carrying a
            ``"logged_metrics"`` map.

    Returns:
        Mapping of metric name to its mean over the rows that logged it.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        logged = row.get("logged_metrics")
        if not isinstance(logged, dict):
            continue
        for name, value in logged.items():
            if not isinstance(value, (int, float)):
                continue
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}
