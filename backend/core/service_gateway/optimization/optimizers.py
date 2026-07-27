"""Optimizer compile/evaluate/instantiate helpers for :class:`DspyService`.

Per-strategy plumbing around DSPy's optimizers: detecting whether the
factory accepts ``valset`` / ``metric`` kwargs, validating user-supplied
``optimizer_kwargs`` against the factory signature, evaluating compiled
programs on the test split, and injecting reflection LMs for GEPA.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, Literal, overload

import dspy

from ...constants import (
    COMPILE_TRAINSET_KEY,
    COMPILE_VALSET_KEY,
    OPTIMIZER_LOG_DIR_KEY,
    OPTIMIZER_METRIC_KEY,
    OPTIMIZER_NAME_GEPA,
    OPTIMIZER_REFLECTION_LM_KEY,
)
from ...exceptions import ServiceError
from ...models import ModelConfig
from ..language_models import build_language_model
from .data import DatasetSplits
from .logged_scores import LoggedScoreRecorder, reset_logged_metrics

logger = logging.getLogger(__name__)


PREFLIGHT_SAMPLE_SIZE = 5


def _perfect_prediction_score(metric: Any, example: Any, output_fields: list[str]) -> float:
    """Score a metric against a perfect prediction built from an example's gold outputs.

    Constructs a ``dspy.Prediction`` whose output-field values equal the
    example's gold values, then invokes ``metric(example, pred, trace=None)``.
    A metric return of ``dspy.Prediction`` (or any object exposing ``.score``)
    is unwrapped to its ``.score``; otherwise the result is coerced to ``float``.

    Args:
        metric: The user-supplied DSPy metric callable.
        example: A ``dspy.Example`` carrying gold output values.
        output_fields: Signature output field names to copy from the gold
            example into the perfect prediction.

    Returns:
        The numeric metric score, or ``0.0`` when the metric raises or returns
        a non-numeric, non-``.score`` value. Per-example failures are swallowed
        so the caller can still render the aggregate all-zero verdict.
    """

    perfect_outputs = {field: example.get(field) for field in output_fields}
    perfect_pred = dspy.Prediction(**perfect_outputs)
    # Fresh log_metrics slot: residue from earlier same-thread metric calls
    # could trip the per-example name cap inside the metric, and the except
    # below would misread that crash as "scored 0 on a perfect prediction".
    reset_logged_metrics()
    try:
        result = metric(example, perfect_pred, trace=None)
    except Exception:
        return 0.0
    score = getattr(result, "score", result)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def preflight_metric_check(
    metric: Any,
    examples: list[Any],
    output_fields: list[str],
    *,
    sample_size: int = PREFLIGHT_SAMPLE_SIZE,
) -> None:
    """Abort fast when the metric scores a perfect prediction as 0 for every sample.

    A correct metric ALWAYS scores a perfect prediction (one whose output
    fields equal the gold values) above 0, so this check never blocks a
    legitimately hard task — only a structurally broken metric (wrong field
    names, ``isinstance(gold, dict)`` gating, etc.) trips it. This runs before
    the expensive optimizer so a broken metric fails with an actionable message
    instead of grinding the whole budget at 0%.

    Args:
        metric: The user-supplied DSPy metric callable.
        examples: The trainset (``dspy.Example`` instances carrying gold outputs).
        output_fields: Signature output field names used to build perfect predictions.
        sample_size: Maximum number of examples to score (kept small and cheap).

    Raises:
        ServiceError: When every sampled perfect prediction scores ``<= 0``
            (sample non-empty), indicating the metric mis-reads the data.
    """

    if metric is None or not examples or not output_fields:
        return

    sample = examples[:sample_size]
    scores = [_perfect_prediction_score(metric, example, output_fields) for example in sample]
    if all(score <= 0 for score in scores):
        raise ServiceError(
            f"Pre-flight check failed: the metric scored 0 on a correct (perfect) prediction "
            f"for all {len(sample)} sampled examples. The metric mis-reads the data — e.g. wrong "
            f"field names, or gating on isinstance(gold, dict) (gold is a dspy.Example, not a dict). "
            f"Fix the metric or column mapping and resubmit."
        )


def compile_program(
    *,
    optimizer: Any,
    program: Any,
    splits: DatasetSplits,
    metric: Any | None,
    compile_kwargs: dict[str, Any],
) -> Any:
    """Run optimizer.compile() with the derived trainset/valset.

    Passes ``valset`` only when the optimizer's compile signature accepts it,
    preventing TypeError on optimizers like BootstrapFewShot that do not.

    Args:
        optimizer: An instantiated DSPy optimizer.
        program: The DSPy program to compile.
        splits: Train/val/test partitions.
        metric: Optional metric callable (already wired into the optimizer).
        compile_kwargs: User-supplied kwargs forwarded to ``optimizer.compile``.

    Returns:
        The compiled DSPy program returned by ``optimizer.compile``.

    Raises:
        ServiceError: If ``splits.train`` is empty or the optimizer rejects
            the kwargs.
    """

    if not splits.train:
        raise ServiceError("Training split is empty; increase the train fraction or provide more data.")

    kwargs = dict(compile_kwargs or {})
    if COMPILE_TRAINSET_KEY not in kwargs:
        kwargs[COMPILE_TRAINSET_KEY] = splits.train

    if splits.val and _compile_accepts_valset(optimizer):
        kwargs.setdefault(COMPILE_VALSET_KEY, splits.val)

    try:
        return optimizer.compile(program, **kwargs)
    except TypeError as exc:
        raise ServiceError(f"Optimizer.compile rejected the provided arguments; update compile_kwargs: {exc}") from exc


def _compile_accepts_valset(optimizer: Any) -> bool:
    """Return True if the optimizer's compile() method accepts a valset parameter.

    Args:
        optimizer: An instantiated DSPy optimizer.

    Returns:
        True when ``optimizer.compile`` exposes a ``valset`` parameter.
    """
    compile_method = getattr(optimizer, "compile", None)
    if compile_method is None:
        return False
    try:
        sig = inspect.signature(compile_method)
        return COMPILE_VALSET_KEY in sig.parameters
    except (ValueError, TypeError):
        return False


@overload
def evaluate_on_test(
    program: Any,
    test_examples: list[Any],
    metric: Any,
    *,
    collect_per_example: Literal[True],
) -> tuple[float | None, list[dict]]:
    """Evaluate ``program`` and return aggregate score plus per-example breakdown."""


@overload
def evaluate_on_test(
    program: Any,
    test_examples: list[Any],
    metric: Any,
    *,
    collect_per_example: Literal[False] = ...,
) -> float | None:
    """Evaluate ``program`` and return only the aggregate test score."""


def evaluate_on_test(
    program: Any,
    test_examples: list[Any],
    metric: Any,
    *,
    collect_per_example: bool = False,
) -> tuple[float | None, list[dict]] | float | None:
    """Evaluate a compiled program on the test split using dspy.Evaluate.

    Args:
        program: The compiled DSPy program to evaluate.
        test_examples: Held-out examples to score.
        metric: The DSPy-compatible scoring callable.
        collect_per_example: When True, also return the per-row breakdown.

    Returns:
        The aggregate score as a float (or ``None`` when ``test_examples``
        is empty). When ``collect_per_example=True``, returns
        ``(score, list[dict])`` where each dict contains ``index``,
        ``outputs``, ``score``, ``pass``, and — when the metric called
        ``log_metrics`` — a ``logged_metrics`` name→value map.

    Raises:
        ServiceError: If the evaluator returns a non-numeric score.
    """

    if not test_examples:
        return (None, []) if collect_per_example else None

    recorder = LoggedScoreRecorder(metric) if collect_per_example else None
    evaluator = dspy.Evaluate(
        devset=test_examples,
        metric=recorder if recorder is not None else metric,
        display_progress=True,
    )
    eval_result = evaluator(program)

    raw_results: list[Any]
    if isinstance(eval_result, (int, float)):
        aggregate = float(eval_result)
        raw_results = []
    else:
        score = getattr(eval_result, "score", None)
        if isinstance(score, (int, float)):
            aggregate = float(score)
        else:
            raise ServiceError("Evaluator returned a non-numeric result; ensure the metric's score is a float.")
        raw_results = getattr(eval_result, "results", []) or []

    if not collect_per_example:
        return aggregate

    # Each EvaluationResult.results entry is (example, prediction, score).
    per_example: list[dict] = []
    for i, entry in enumerate(raw_results):
        try:
            example, prediction, ex_score = entry
            # Metric may return a dspy.Prediction with a .score attribute
            if hasattr(ex_score, "score"):
                ex_score = ex_score.score
            ex_score = float(ex_score) if isinstance(ex_score, (int, float, bool)) else 0.0
            # Per-row heartbeat at DEBUG so it surfaces only in the Logs tab's
            # verbose view: these baseline/optimized eval passes sit outside
            # capture_tqdm, so dspy.Evaluate's bar never forwards. Normal mode
            # keeps the single aggregate metric; verbose adds the live per-row
            # progress. Demoted in lockstep with the react/predict heartbeats.
            logger.debug(
                "%s test eval %d/%d score=%.3f pass=%s",
                program.__class__.__name__,
                i + 1,
                len(raw_results),
                ex_score,
                ex_score > 0,
            )
            outputs = {}
            gold = {}
            for k in example.labels():
                outputs[k] = getattr(prediction, k, None) if prediction else None
                # Gold rides along so corpus-level classification metrics
                # (precision/recall) can be computed from the stored rows.
                gold[k] = getattr(example, k, None)
            row: dict[str, Any] = {
                "index": i,
                "outputs": outputs,
                "gold": gold,
                "score": ex_score,
                "pass": ex_score > 0,
            }
            logged = recorder.scores_for(example) if recorder is not None else {}
            if logged:
                row["logged_metrics"] = logged
            # A metric crash was scored 0 by dspy.Evaluate; carry the crash
            # text so the row can say why instead of rendering a silent zero.
            metric_error = recorder.error_for(example) if recorder is not None else None
            if metric_error:
                row["error"] = metric_error
            per_example.append(row)
        except Exception:
            per_example.append({"index": i, "outputs": {}, "score": 0.0, "pass": False})

    return aggregate, per_example


def optimizer_requires_metric(factory: Callable[..., Any]) -> bool:
    """Return True if the optimizer factory (or any wrapped target) accepts a ``metric`` parameter.

    Wrapped callables (``__wrapped__``) and closure cells are also inspected
    so decorated factories report accurately.

    Args:
        factory: The optimizer factory callable.

    Returns:
        True when the factory or one of its wrapped targets accepts ``metric``.
    """

    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return False
    if "metric" in sig.parameters:
        return True

    if _callable_accepts_metric(factory):
        return True
    return any(_callable_accepts_metric(target) for target in _extract_factory_targets(factory))


def validate_optimizer_signature(factory: Callable[..., Any], name: str) -> None:
    """Warn if the optimizer factory is not introspectable.

    Args:
        factory: The optimizer factory callable.
        name: The optimizer's registered name (used in log output).
    """

    try:
        inspect.signature(factory)
    except (ValueError, TypeError):
        logger.warning("Unable to introspect optimizer '%s' signature.", name)


def validate_optimizer_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any], name: str) -> None:
    """Validate user-supplied kwargs against the optimizer factory signature.

    Args:
        factory: The optimizer factory callable.
        kwargs: User-supplied keyword arguments.
        name: The optimizer's registered name (used in error messages).

    Raises:
        ServiceError: When ``kwargs`` cannot be bound to the factory signature.
    """

    if not kwargs:
        return
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return
    try:
        sig.bind_partial(**kwargs)
    except TypeError as exc:
        raise ServiceError(f"optimizer_kwargs contain unsupported entries for '{name}': {exc}") from exc
    # bind_partial is too permissive when the factory accepts **kwargs — every
    # key matches the wildcard. Flag kwargs that aren't in the named params so
    # typos surface instead of silently passing.
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        named = {k for k, p in sig.parameters.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}
        unknown = sorted(k for k in kwargs if k not in named)
        if unknown:
            logger.warning(
                "Optimizer '%s' received kwargs %s not in named parameters — forwarded via **kwargs; verify spelling.",
                name,
                unknown,
            )


def instantiate_optimizer(
    factory: Callable[..., Any],
    optimizer_name: str,
    optimizer_kwargs: dict[str, Any],
    metric: Callable[..., Any],
    reflection_model: ModelConfig | None,
    *,
    reflection_lm: Any | None = None,
    log_dir: str | None = None,
) -> Any:
    """Instantiate an optimizer, injecting language models and metrics as needed.

    Per-optimizer injection rules:
    - All optimizers that expose a ``metric`` parameter receive it automatically.
    - GEPA additionally requires ``reflection_lm`` (built from ``reflection_model``)
      and defaults ``auto`` to ``"light"`` when no budget kwarg is supplied.
    - GEPA receives ``log_dir`` when supplied so it persists per-iteration
      state to ``<log_dir>/gepa_state.bin`` — required by the trajectory
      watcher that surfaces candidate genealogy to the UI.

    Args:
        factory: The optimizer factory callable to invoke.
        optimizer_name: The optimizer's registered name.
        optimizer_kwargs: User-supplied factory kwargs.
        metric: The DSPy-compatible metric callable to inject when needed.
        reflection_model: Configuration for the reflection model (required
            when ``optimizer_name`` is GEPA and no ``reflection_lm`` is
            already provided).
        reflection_lm: Optional pre-built reflection LM instance. When
            supplied (e.g. so the caller can attach a timing callback
            bound to its identity), it bypasses construction from
            ``reflection_model``.
        log_dir: Optional directory GEPA writes ``gepa_state.bin`` into. The
            trajectory watcher polls this file to emit per-candidate
            progress events. Ignored for non-GEPA optimizers.

    Returns:
        An instantiated optimizer ready for ``compile``.

    Raises:
        ServiceError: When GEPA is requested without a reflection model.
    """

    optimizer_key = optimizer_name.lower()
    reflection_required_optimizers = {OPTIMIZER_NAME_GEPA}
    requires_metric = optimizer_requires_metric(factory)
    if not requires_metric and optimizer_key == OPTIMIZER_NAME_GEPA:
        requires_metric = True

    kwargs = dict(optimizer_kwargs or {})
    if requires_metric and OPTIMIZER_METRIC_KEY not in kwargs:
        kwargs[OPTIMIZER_METRIC_KEY] = metric
    # GEPA requires one of auto/max_full_evals/max_metric_calls — default to "light"
    if optimizer_key == OPTIMIZER_NAME_GEPA and not any(
        k in kwargs for k in ("auto", "max_full_evals", "max_metric_calls")
    ):
        kwargs["auto"] = "light"
    if (
        optimizer_key == OPTIMIZER_NAME_GEPA
        and log_dir is not None
        and OPTIMIZER_LOG_DIR_KEY not in kwargs
    ):
        kwargs[OPTIMIZER_LOG_DIR_KEY] = log_dir
    needs_reflection = optimizer_key in reflection_required_optimizers
    if OPTIMIZER_REFLECTION_LM_KEY not in kwargs:
        if reflection_lm is not None and needs_reflection:
            kwargs[OPTIMIZER_REFLECTION_LM_KEY] = reflection_lm
        elif reflection_model and needs_reflection:
            # Caching stays ON: this reflection LM runs inside GEPA's
            # training/eval region, and forcing cache off there suppresses
            # GEPA's recognized tqdm bar (regression from #23/#24).
            kwargs[OPTIMIZER_REFLECTION_LM_KEY] = build_language_model(reflection_model)
        elif needs_reflection:
            raise ServiceError(
                f"Optimizer '{optimizer_name}' requires reflection_model_config "
                "or a preconfigured 'reflection_lm' in optimizer_kwargs."
            )
    # INFO (not DEBUG): the subprocess log forwarder floors at INFO, so this —
    # the single most useful instantiation breadcrumb — was previously invisible
    # in job_logs. Reports which injections were applied, not just key names.
    logger.info(
        "Creating optimizer %s (metric=%s reflection_lm=%s auto=%s log_dir=%s)",
        optimizer_name,
        OPTIMIZER_METRIC_KEY in kwargs,
        OPTIMIZER_REFLECTION_LM_KEY in kwargs,
        kwargs.get("auto"),
        OPTIMIZER_LOG_DIR_KEY in kwargs,
    )
    return factory(**kwargs)


def _callable_accepts_metric(target: Any) -> bool:
    """Return True when the callable exposes a ``metric`` parameter.

    Args:
        target: A callable to introspect.

    Returns:
        True when ``target`` has a ``metric`` parameter.
    """

    if target is None:
        return False
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return False
    return "metric" in sig.parameters


def _extract_factory_targets(factory: Callable[..., Any]) -> list[Any]:
    """Collect potential callable targets from wrappers/closures for metric-detection.

    Args:
        factory: The factory callable to deconstruct.

    Returns:
        A list of inner callables (``__wrapped__`` target, closure cell
        contents, and the factory itself) suitable for metric-detection.
    """

    targets: list[Any] = []
    wrapped = getattr(factory, "__wrapped__", None)
    if wrapped is not None:
        targets.append(wrapped)
    closure_cells = getattr(factory, "__closure__", None)
    if closure_cells:
        targets.extend(cell.cell_contents for cell in closure_cells)
    if callable(factory):
        targets.append(factory)
    return targets
