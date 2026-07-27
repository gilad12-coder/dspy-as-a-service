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

import logging
import math
import re
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

MAX_LOGGED_METRICS = 20
MAX_METRIC_NAME_LENGTH = 50
# Aggregates union names across examples, so a metric that derives names from
# example content (against the documented contract) could mint 20 fresh names
# per row. Cap the union so the result — and the UI columns built from it —
# stays bounded no matter what the metric does.
MAX_AGGREGATE_METRIC_NAMES = 50
_ERROR_TEXT_CAP = 500

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
        self._errors: dict[int, str] = {}

    def __call__(self, example: Any, prediction: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped metric, capturing scores it logs for ``example``.

        A metric crash is recorded (see :meth:`error_for`) and re-raised
        unchanged, so the evaluator's own failure handling — dspy.Evaluate
        scores the row 0 — still applies; the recorded text is what lets the
        per-example results say "the metric crashed" instead of rendering a
        silent zero.

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
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_CAP]
            with self._lock:
                self._errors[id(example)] = text
            raise
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

    def error_for(self, example: Any) -> str | None:
        """Return the crash text recorded while scoring ``example``, if any.

        Args:
            example: The same example object the metric was invoked with.
        """
        with self._lock:
            return self._errors.get(id(example))


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
        Mapping of metric name to its mean over the rows that logged it,
        capped at :data:`MAX_AGGREGATE_METRIC_NAMES` names (first-seen wins).
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
    names = list(sums)
    if len(names) > MAX_AGGREGATE_METRIC_NAMES:
        logger.warning(
            "aggregate_logged_metrics: %d distinct names, keeping the first %d — "
            "log a fixed, small set of names",
            len(names),
            MAX_AGGREGATE_METRIC_NAMES,
        )
        names = names[:MAX_AGGREGATE_METRIC_NAMES]
    return {name: sums[name] / counts[name] for name in names}


def combined_test_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Merge computed classification metrics under the metric's own logged values.

    The metric's ``log_metrics`` values win on a name collision — a metric
    that logs its own ``precision`` knows its task better than the generic
    confusion-matrix computation.

    Args:
        rows: Per-example result dicts from a test-set evaluation.

    Returns:
        One flat name→value map for the result payload's ``*_logged_metrics``.
    """
    return {**classification_metrics(rows), **aggregate_logged_metrics(rows)}


CLASSIFICATION_MAX_CLASSES = 20
"""Above this many distinct gold values a field is prose, not a class column."""

# Two-class vocabularies with an unambiguous positive class. For these,
# positive-class precision/recall is reported (what an analyst expects of a
# binary classifier); any other label set gets a macro average instead.
_BINARY_POSITIVE_BY_CLASSES: dict[frozenset[str], str] = {
    frozenset({"1", "0"}): "1",
    frozenset({"yes", "no"}): "yes",
    frozenset({"true", "false"}): "true",
}


def _canon_label(value: Any) -> str:
    """Normalize a label for comparison the way generated metrics do (case-insensitive exact match)."""
    return str(value).strip().lower()


def _pr_for_class(pairs: list[tuple[str, str]], cls: str) -> tuple[float, float]:
    """Compute one class's precision and recall over (gold, predicted) pairs."""
    tp = sum(1 for g, p in pairs if g == cls and p == cls)
    fp = sum(1 for g, p in pairs if g != cls and p == cls)
    fn = sum(1 for g, p in pairs if g == cls and p != cls)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Corpus-level precision/recall for categorical output fields.

    Unlike :func:`aggregate_logged_metrics` (a macro average of whatever the
    metric chose to log per example), these are true corpus metrics built from
    the stored ``(gold, outputs)`` pairs: positive-class precision/recall for a
    binary field (1/0, yes/no, true/false), macro-averaged one-vs-rest for a
    multiclass field. Fields that don't look categorical — missing gold, more
    than :data:`CLASSIFICATION_MAX_CLASSES` distinct values, a single class,
    or non-scalar values — are skipped, so regression/freeform runs report
    nothing here.

    Args:
        rows: Per-example result dicts carrying ``"gold"`` and ``"outputs"``.

    Returns:
        ``{"precision": …, "recall": …}`` when exactly one output field
        qualifies; ``"precision (<field>)"``-style names when several do;
        empty when none does.
    """
    pairs_by_field: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        gold = row.get("gold")
        outputs = row.get("outputs")
        if not isinstance(gold, dict) or not isinstance(outputs, dict):
            continue
        for field, gold_value in gold.items():
            predicted = outputs.get(field)
            if gold_value is None or predicted is None:
                continue
            if isinstance(gold_value, (dict, list)) or isinstance(predicted, (dict, list)):
                continue
            pairs_by_field.setdefault(field, []).append(
                (_canon_label(gold_value), _canon_label(predicted))
            )

    per_field: dict[str, tuple[float, float]] = {}
    for field, pairs in pairs_by_field.items():
        classes = {g for g, _ in pairs}
        if not 2 <= len(classes) <= CLASSIFICATION_MAX_CLASSES:
            continue
        positive = _BINARY_POSITIVE_BY_CLASSES.get(frozenset(classes))
        if positive is not None:
            per_field[field] = _pr_for_class(pairs, positive)
            continue
        scores = [_pr_for_class(pairs, cls) for cls in sorted(classes)]
        per_field[field] = (
            sum(p for p, _ in scores) / len(scores),
            sum(r for _, r in scores) / len(scores),
        )

    if not per_field:
        return {}
    if len(per_field) == 1:
        precision, recall = next(iter(per_field.values()))
        return {"precision": precision, "recall": recall}
    result: dict[str, float] = {}
    for field, (precision, recall) in per_field.items():
        result[f"precision ({field})"] = precision
        result[f"recall ({field})"] = recall
    return result
