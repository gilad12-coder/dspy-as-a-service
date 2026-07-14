"""Subprocess isolation for validation-time exec of user-authored code.

User-authored DSPy signature and metric code has to be exec'd at some point
to validate it. Running exec() directly in the API process leaks arbitrary
user code into the web server — a simple ``while True: pass`` would hang a
request worker, and ``import os; os.kill(1, 9)`` would be much worse.

This module wraps validation in a subprocess boundary: each call spawns a
fresh child via ``multiprocessing.spawn``, exec-s the code there, extracts
only the metadata the parent actually needs (field names, callable param
names, a result-shape probe), and returns a pickleable result. If the
child hangs, we kill it after a timeout; if it raises, the parent
re-raises a ``ServiceError``.

Invariant: the child NEVER returns the compiled class or function back.
Dynamically-exec'd classes can't be pickled across processes anyway, and
letting the compiled object cross the boundary would defeat the point.
Callers that need the actual object — the optimization worker — already
run inside their own subprocess and exec directly (see
``worker/subprocess_runner.py``).
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dspy

from ..exceptions import ServiceError
from .optimization.data import (
    extract_signature_fields,
    image_input_field_names,
    load_metric_from_code,
    load_signature_from_code,
    load_transform_from_code,
)

_DEFAULT_PARSE_TIMEOUT_SECONDS = 30.0
_DEFAULT_PROBE_TIMEOUT_SECONDS = 45.0
_TERMINATE_GRACE_SECONDS = 2.0
_QUEUE_READ_SECONDS = 5.0

# Dogpile-safe per-process caches: identical user code is validated once per
# replica, and concurrent submissions of the same code share that one
# subprocess-spawn cost instead of racing N fresh interpreters that all pay
# the multi-second ``import dspy`` price. Keyed on the raw source string —
# the cost of holding source in memory is trivial next to the spawn cost it
# avoids. Per-key locks live in ``_dogpile_locks``; ``_locks_mutex`` guards
# only the locks dict, not the heavy validation itself.
_signature_cache: dict[str, SignatureIntrospection] = {}
_metric_cache: dict[str, MetricIntrospection] = {}
_transform_cache: dict[str, TransformIntrospection] = {}
_dogpile_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()

_VALIDATION_CACHE_MAX_ENTRIES = 256


def _bounded_cache_put(cache: dict[str, Any], key: str, value: Any) -> None:
    """Insert ``key`` → ``value``, evicting the oldest entry once at the cap.

    The memo dicts above are keyed by full user source strings, so without a
    bound they grow with every distinct submission for the life of the
    process — and the API process is long-lived and fork-parent to every job.
    FIFO suffices: the caches absorb same-code bursts, not long-tail reuse.
    Evicting a dogpile lock someone still holds merely lets two concurrent
    submissions of the same code validate twice.
    """
    if key not in cache and len(cache) >= _VALIDATION_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _dogpile_lock(key: str) -> threading.Lock:
    """Return the lock for ``key``, creating it under ``_locks_mutex`` if absent.

    Args:
        key: Stable key (cache namespace plus code string) the caller will use
            to gate its cache miss.

    Returns:
        A ``threading.Lock`` unique to ``key`` across the process.
    """
    with _locks_mutex:
        lock = _dogpile_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _bounded_cache_put(_dogpile_locks, key, lock)
        return lock


@dataclass(frozen=True)
class SignatureIntrospection:
    """Metadata extracted from a user-authored signature class."""

    class_name: str
    input_fields: list[str]
    output_fields: list[str]
    image_input_fields: list[str]


@dataclass(frozen=True)
class MetricIntrospection:
    """Metadata extracted from a user-authored metric callable."""

    callable_name: str
    param_names: list[str]


@dataclass(frozen=True)
class TransformIntrospection:
    """Metadata extracted from a user-authored workflow transform callable."""

    callable_name: str
    param_names: list[str]


@dataclass(frozen=True)
class MetricProbeResult:
    """Shape of a metric invocation on a sample row, captured in a subprocess.

    ``result_kind`` is one of:

    - ``"none"``       — the metric returned ``None``.
    - ``"prediction"`` — the metric returned a ``dspy.Prediction`` with a
      ``score`` attribute (GEPA-shaped).
    - ``"numeric"``    — the metric returned an ``int``, ``float``, or ``bool``.
    - ``"other"``      — the metric returned something else (e.g. a string).
    - ``"error"``      — the metric itself raised when invoked; ``error`` is
      the stringified exception.

    ``score`` is the metric's numeric output as a float (the ``.score`` of a
    prediction, or the numeric value itself), or ``None`` when unavailable.
    """

    result_kind: str
    result_type_name: str
    has_score_attr: bool
    error: str | None
    score: float | None


def _run_in_subprocess(
    target: Callable[..., None],
    args: tuple[Any, ...],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Spawn ``target(*args, queue)`` in a child process and return its dict result.

    Uses the ``spawn`` start method unconditionally — a fresh Python
    interpreter with no inherited file descriptors, sockets, or memory
    from the parent. This is the whole point of the isolation.

    Args:
        target: Worker function executed in the child process.
        args: Positional arguments for ``target`` (queue is appended).
        timeout_seconds: Wall-clock budget before the child is terminated.

    Returns:
        Dict result the child placed on the queue.

    Raises:
        ServiceError: When the child times out, exits without a result,
            or returns a non-dict payload.
    """

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(_TERMINATE_GRACE_SECONDS)
        if proc.is_alive():
            proc.kill()
            proc.join(_TERMINATE_GRACE_SECONDS)
        raise ServiceError(f"user code exceeded the {timeout_seconds:.0f}s validation timeout and was terminated.")

    try:
        result = queue.get(timeout=_QUEUE_READ_SECONDS)
    except Exception as exc:  # queue.Empty or manager teardown: child died before emitting
        raise ServiceError("validation subprocess exited without returning a result.") from exc
    if not isinstance(result, dict):
        raise ServiceError("validation subprocess returned an unexpected value.")
    return result


def _raise_child_error(result: dict[str, Any]) -> None:
    """Translate a ``{"ok": False, ...}`` child payload into a ``ServiceError``.

    Args:
        result: The error payload emitted by ``_error_payload`` in the child.

    Raises:
        ServiceError: Always; the message is built from the payload.
    """

    error_type = result.get("error_type", "")
    error_msg = result.get("error", "user code failed")
    if error_type == "ServiceError":
        raise ServiceError(error_msg)
    if error_type:
        raise ServiceError(f"{error_type}: {error_msg}")
    raise ServiceError(error_msg)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """Build the ``{"ok": False, ...}`` dict that workers put on the queue.

    Args:
        exc: The exception caught inside the child process.

    Returns:
        A pickleable error payload with class name and traceback.
    """

    return {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }


def _signature_worker(code: str, queue: Any) -> None:
    """Child-side entry point for ``validate_signature_code``.

    Args:
        code: User-authored signature class source.
        queue: Multiprocessing queue used to return a result dict.
    """
    try:
        cls = load_signature_from_code(code)
        inputs, outputs = extract_signature_fields(cls)
        queue.put(
            {
                "ok": True,
                "class_name": cls.__name__,
                "input_fields": inputs,
                "output_fields": outputs,
                "image_input_fields": sorted(image_input_field_names(cls)),
            }
        )
    except BaseException as exc:  # user code is arbitrary — any failure is reported, not raised
        queue.put(_error_payload(exc))


def validate_signature_code(
    code: str,
    *,
    timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS,
) -> SignatureIntrospection:
    """Parse user-authored signature code in a subprocess and return its shape.

    Memoized per-process: concurrent submissions of the same signature share
    one subprocess spawn instead of dogpiling N spawn+import-dspy cycles.

    Args:
        code: User-authored signature class source.
        timeout_seconds: Maximum time to wait for the child to finish.

    Returns:
        Field metadata extracted from the compiled signature class.

    Raises:
        ServiceError: When the user code fails to load or the child errors.
    """
    cached = _signature_cache.get(code)
    if cached is not None:
        return cached
    with _dogpile_lock(f"sig:{code}"):
        cached = _signature_cache.get(code)
        if cached is not None:
            return cached
        result = _run_in_subprocess(
            _signature_worker,
            (code,),
            timeout_seconds=timeout_seconds,
        )
        if not result.get("ok"):
            _raise_child_error(result)
        introspection = SignatureIntrospection(
            class_name=result["class_name"],
            input_fields=list(result["input_fields"]),
            output_fields=list(result["output_fields"]),
            image_input_fields=list(result.get("image_input_fields") or []),
        )
        _bounded_cache_put(_signature_cache, code, introspection)
        return introspection


def _metric_worker(code: str, queue: Any) -> None:
    """Child-side entry point for ``validate_metric_code``.

    Args:
        code: User-authored metric callable source.
        queue: Multiprocessing queue used to return a result dict.
    """
    try:
        metric = load_metric_from_code(code)
        sig = inspect.signature(metric)
        param_names = [p.name for p in sig.parameters.values()]
        queue.put(
            {
                "ok": True,
                "callable_name": getattr(metric, "__name__", "metric"),
                "param_names": param_names,
            }
        )
    except BaseException as exc:  # user code is arbitrary — any failure is reported, not raised
        queue.put(_error_payload(exc))


def validate_metric_code(
    code: str,
    *,
    timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS,
) -> MetricIntrospection:
    """Parse user-authored metric code in a subprocess and return its shape.

    Memoized per-process: concurrent submissions of the same metric share
    one subprocess spawn instead of dogpiling N spawn+import-dspy cycles.

    Args:
        code: User-authored metric callable source.
        timeout_seconds: Maximum time to wait for the child to finish.

    Returns:
        Callable name plus parameter names extracted via ``inspect``.

    Raises:
        ServiceError: When the user code fails to load or the child errors.
    """
    cached = _metric_cache.get(code)
    if cached is not None:
        return cached
    with _dogpile_lock(f"met:{code}"):
        cached = _metric_cache.get(code)
        if cached is not None:
            return cached
        result = _run_in_subprocess(
            _metric_worker,
            (code,),
            timeout_seconds=timeout_seconds,
        )
        if not result.get("ok"):
            _raise_child_error(result)
        introspection = MetricIntrospection(
            callable_name=result["callable_name"],
            param_names=list(result["param_names"]),
        )
        _bounded_cache_put(_metric_cache, code, introspection)
        return introspection


def _transform_worker(code: str, queue: Any) -> None:
    """Child-side entry point for ``validate_transform_code``.

    Args:
        code: User-authored transform callable source.
        queue: Multiprocessing queue used to return a result dict.
    """
    try:
        transform = load_transform_from_code(code)
        sig = inspect.signature(transform)
        param_names = [p.name for p in sig.parameters.values()]
        queue.put(
            {
                "ok": True,
                "callable_name": getattr(transform, "__name__", "transform"),
                "param_names": param_names,
            }
        )
    except BaseException as exc:  # user code is arbitrary — any failure is reported, not raised
        queue.put(_error_payload(exc))


def validate_transform_code(
    code: str,
    *,
    timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS,
) -> TransformIntrospection:
    """Parse user-authored transform code in a subprocess and return its shape.

    Memoized per-process like the signature/metric validators: concurrent
    submissions of the same transform share one subprocess spawn.

    Args:
        code: User-authored transform callable source.
        timeout_seconds: Maximum time to wait for the child to finish.

    Returns:
        Callable name plus parameter names extracted via ``inspect``.

    Raises:
        ServiceError: When the user code fails to load or the child errors.
    """
    cached = _transform_cache.get(code)
    if cached is not None:
        return cached
    with _dogpile_lock(f"tra:{code}"):
        cached = _transform_cache.get(code)
        if cached is not None:
            return cached
        result = _run_in_subprocess(
            _transform_worker,
            (code,),
            timeout_seconds=timeout_seconds,
        )
        if not result.get("ok"):
            _raise_child_error(result)
        introspection = TransformIntrospection(
            callable_name=result["callable_name"],
            param_names=list(result["param_names"]),
        )
        _bounded_cache_put(_transform_cache, code, introspection)
        return introspection


def _probe_worker(
    metric_code: str,
    example_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
    input_field_names: list[str],
    image_input_fields: list[str],
    queue: Any,
) -> None:
    """Child-side entry point for ``probe_metric_on_sample``.

    Args:
        metric_code: User-authored metric callable source.
        example_payload: Field values for a sample row.
        prediction_payload: Field values for a fake prediction.
        input_field_names: Inputs that should be marked on the example.
        image_input_fields: Subset of inputs that need ``dspy.Image`` wrapping.
        queue: Multiprocessing queue used to return a result dict.
    """
    try:
        metric = load_metric_from_code(metric_code)
        prepared_payload = dict(example_payload)
        image_type = getattr(dspy, "Image", None)
        if image_type is not None:
            for field_name in image_input_fields:
                value = prepared_payload.get(field_name)
                if value is None or isinstance(value, image_type):
                    continue
                prepared_payload[field_name] = image_type(url=value)
        example = dspy.Example(**prepared_payload).with_inputs(*input_field_names)
        prediction = dspy.Prediction(**prediction_payload)
        try:
            result = metric(example, prediction, trace=None)
        except BaseException as call_exc:
            queue.put(
                {
                    "ok": True,
                    "result_kind": "error",
                    "result_type_name": type(call_exc).__name__,
                    "has_score_attr": False,
                    "error": str(call_exc),
                    "score": None,
                }
            )
            return

        if result is None:
            kind = "none"
        elif isinstance(result, dspy.Prediction) and hasattr(result, "score"):
            kind = "prediction"
        elif isinstance(result, (int, float, bool)):
            kind = "numeric"
        else:
            kind = "other"

        score: float | None
        try:
            if kind == "prediction":
                score = float(result.score)
            elif kind == "numeric":
                score = float(result)
            else:
                score = None
        except BaseException:  # non-numeric .score / unconvertible value — treat as no score
            score = None

        queue.put(
            {
                "ok": True,
                "result_kind": kind,
                "result_type_name": type(result).__name__,
                "has_score_attr": hasattr(result, "score"),
                "error": None,
                "score": score,
            }
        )
    except BaseException as exc:  # code failed to parse or dspy setup failed — report, don't raise
        queue.put(_error_payload(exc))


def probe_metric_on_sample(
    *,
    metric_code: str,
    example_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
    input_field_names: list[str],
    image_input_fields: list[str] | None = None,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> MetricProbeResult:
    """Invoke user-authored metric code on a sample row inside a subprocess.

    A metric that simply raised on the sample row is reported via
    ``MetricProbeResult.error`` — not as an exception from this function.
    Only ``ServiceError`` (parse failure, subprocess crash, timeout) escapes.

    Args:
        metric_code: User-authored metric callable source.
        example_payload: Field values for a sample row.
        prediction_payload: Field values for a fake prediction.
        input_field_names: Inputs to mark on the example.
        image_input_fields: Subset of inputs needing ``dspy.Image`` wrapping.
        timeout_seconds: Maximum time to wait for the child to finish.

    Returns:
        Shape and outcome of invoking the metric on the sample.

    Raises:
        ServiceError: When the metric fails to load or the child errors out.
    """
    result = _run_in_subprocess(
        _probe_worker,
        (
            metric_code,
            example_payload,
            prediction_payload,
            input_field_names,
            list(image_input_fields or []),
        ),
        timeout_seconds=timeout_seconds,
    )
    if not result.get("ok"):
        _raise_child_error(result)
    raw_score = result.get("score")
    return MetricProbeResult(
        result_kind=str(result["result_kind"]),
        result_type_name=str(result["result_type_name"]),
        has_score_attr=bool(result["has_score_attr"]),
        error=result.get("error"),
        score=float(raw_score) if raw_score is not None else None,
    )
