"""Dataset loading, signature introspection, and split utilities.

Pure helpers used by :class:`DspyService` to: load a user-authored
:class:`dspy.Signature` and metric callable from source, convert raw
rows into :class:`dspy.Example` instances, and partition the example
list into train/val/test slices.

Splits are size-based only: see ``datasets/planner.py`` for the
research-grounded tier policy. Per-class stratification was removed —
GEPA's reflection LM consumes free-form trajectories and the Pareto
frontier scores against an aggregate metric, so balanced sampling
buys nothing here.
"""

from __future__ import annotations

import random
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dspy

from ...exceptions import ServiceError
from ...models import ColumnMapping, SplitFractions


@dataclass
class DatasetSplits:
    """Container for train/validation/test partitions."""

    train: list[Any]
    val: list[Any]
    test: list[Any]


def extract_signature_fields(signature_cls: type[dspy.Signature]) -> tuple[list[str], list[str]]:
    """Inspect a DSPy signature class and return its input and output field names.

    Args:
        signature_cls: The signature class to introspect.

    Returns:
        A tuple ``(input_field_names, output_field_names)``.

    Raises:
        ServiceError: When the signature has no input or no output fields.
    """

    inputs: list[str] = []
    outputs: list[str] = []

    fields_mapping = getattr(signature_cls, "model_fields", None) or getattr(signature_cls, "__pydantic_fields__", None)
    if fields_mapping:
        for field_name, field_info in fields_mapping.items():
            if _is_signature_field(field_info, field_type="input"):
                inputs.append(field_name)
            elif _is_signature_field(field_info, field_type="output"):
                outputs.append(field_name)
    else:
        for attr_name, attr_value in signature_cls.__dict__.items():
            if _is_signature_field(attr_value, field_type="input"):
                inputs.append(attr_name)
            elif _is_signature_field(attr_value, field_type="output"):
                outputs.append(attr_name)
    if not inputs or not outputs:
        raise ServiceError("Signature must declare at least one input and one output field.")
    return inputs, outputs


def rows_to_examples(
    dataset: list[dict[str, Any]],
    mapping: ColumnMapping,
    *,
    image_input_fields: set[str] | None = None,
    extra_columns: set[str] | None = None,
) -> list[Any]:
    """Convert dataset rows into DSPy Example instances using the column mapping.

    ``image_input_fields`` lists signature input field names whose cell
    values must be wrapped in ``dspy.Image(url=...)`` before the example is
    built — pass the result of :func:`image_input_field_names` on the loaded
    signature class.

    ``extra_columns`` lists raw source columns to carry onto the example as
    plain attributes *after* construction — without ``with_inputs``/output
    marking — so a reward function can read ``gold.<col>`` while the value
    never feeds the program. Each named column present in a row is set on the
    example; columns absent from a row are silently skipped (extras are
    advisory, not required like mapped inputs/outputs).

    Args:
        dataset: Raw dataset rows.
        mapping: Column mapping describing inputs and outputs.
        image_input_fields: Optional set of signature field names whose
            values are coerced into ``dspy.Image`` instances.
        extra_columns: Optional set of raw source-column names to attach to
            each example as unmarked attributes for reward-side reads.

    Returns:
        A list of populated :class:`dspy.Example` instances.

    Raises:
        ServiceError: If a row is not a dict or a mapped column is missing.
    """

    image_fields = image_input_fields or set()
    extras = extra_columns or set()
    examples: list[Any] = []
    for row_idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ServiceError(f"Row {row_idx} is not a mapping: {row!r}")
        payload: dict[str, Any] = {}
        for signature_field, column_name in mapping.inputs.items():
            try:
                value = row[column_name]
            except KeyError as exc:
                raise ServiceError(f"Missing input column '{column_name}' for row {row_idx}") from exc
            payload[signature_field] = _coerce_image(value) if signature_field in image_fields else value
        for signature_field, column_name in mapping.outputs.items():
            try:
                payload[signature_field] = row[column_name]
            except KeyError as exc:
                raise ServiceError(f"Missing output column '{column_name}' for row {row_idx}") from exc

        example = dspy.Example(**payload)
        if hasattr(example, "with_inputs"):
            example = example.with_inputs(*mapping.inputs.keys())
        if mapping.outputs:
            if hasattr(example, "with_outputs"):
                example = example.with_outputs(*mapping.outputs.keys())
            elif hasattr(example, "with_targets"):
                example = example.with_targets(*mapping.outputs.keys())
        # Attach extras after input/output marking so they stay out of
        # ``example.inputs()`` — reward code reads ``gold.<col>`` but the
        # program never receives the value.
        for column_name in extras:
            if column_name in row:
                setattr(example, column_name, row[column_name])
        examples.append(example)

    return examples


def image_input_field_names(signature_cls: type[dspy.Signature]) -> set[str]:
    """Return the names of input fields annotated as ``dspy.Image``.

    Inspects the signature's Pydantic field annotations directly. Only
    matches the bare ``dspy.Image`` annotation — not ``Optional[dspy.Image]``
    or ``list[dspy.Image]`` — because the LLM is instructed to emit the
    bare form for image columns.

    Args:
        signature_cls: The signature class to introspect.

    Returns:
        Names of input fields whose annotation is ``dspy.Image``.
    """
    image_type = getattr(dspy, "Image", None)
    if image_type is None:
        return set()
    fields_mapping = getattr(signature_cls, "model_fields", None) or getattr(
        signature_cls, "__pydantic_fields__", None
    )
    if not fields_mapping:
        return set()
    out: set[str] = set()
    for field_name, field_info in fields_mapping.items():
        if not _is_signature_field(field_info, field_type="input"):
            continue
        if _annotation_is_image(getattr(field_info, "annotation", None), image_type):
            out.add(field_name)
    return out


def _annotation_is_image(annotation: Any, image_type: type) -> bool:
    """Return True when ``annotation`` denotes the bare ``dspy.Image`` type.

    Handles both eagerly-evaluated annotations (``dspy.Image`` resolved to
    the actual class) and lazy ``ForwardRef``s — the latter occurs when a
    signature is loaded via ``exec`` from inside a module that uses
    ``from __future__ import annotations``, which leaves Pydantic with
    ``ForwardRef('dspy.Image')`` instead of the resolved class.

    Args:
        annotation: The Pydantic field annotation to check.
        image_type: The resolved ``dspy.Image`` class.

    Returns:
        True when ``annotation`` denotes ``dspy.Image``.
    """
    if annotation is image_type:
        return True
    forward_arg = getattr(annotation, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        return forward_arg in {"dspy.Image", "Image"}
    if isinstance(annotation, str):
        return annotation in {"dspy.Image", "Image"}
    return False


def _coerce_image(value: Any) -> Any:
    """Wrap a raw cell value into a ``dspy.Image`` instance.

    Already-wrapped ``dspy.Image`` instances pass through unchanged so
    callers that pre-coerce don't double-wrap.

    Args:
        value: The raw cell value (URL string, data URI, or pre-wrapped image).

    Returns:
        A ``dspy.Image`` instance (or ``value`` unchanged when ``dspy.Image``
        is unavailable).
    """
    image_type = getattr(dspy, "Image", None)
    if image_type is None:
        return value
    if isinstance(value, image_type):
        return value
    return image_type(url=value)


def load_signature_from_code(code: str) -> type[dspy.Signature]:
    """Execute user-provided code and return the single DSPy signature class it defines.

    Trust boundary: the body of ``code`` runs in-process via ``exec``. Optimization
    runs (``DspyService.run`` / ``run_grid_search``) reach this through
    ``backend/core/worker/subprocess_runner.py``, which sandboxes per-job execution
    in a separate process. Validation paths (the ``code_validation`` router,
    ``validate_payload``, the code agent) intentionally exec in the API process
    so the user gets fast feedback — anyone who can call those endpoints already
    has authenticated arbitrary-code-execution rights against the gateway. Do not
    expose those endpoints beyond the optimization team without sandboxing first.

    Args:
        code: User-authored signature source code.

    Returns:
        The single :class:`dspy.Signature` subclass defined by ``code``.

    Raises:
        ServiceError: When the code has a syntax error, defines no Signature
            subclass, or defines more than one.
    """

    namespace: dict[str, Any] = {"dspy": dspy}
    try:
        # exec: user-supplied signature code. Isolation is context-dependent:
        # the DspyService.run/run_grid_search paths execute inside
        # worker/subprocess_runner.py (separate process boundary), but the
        # validation paths (code_validation router, validate_payload,
        # code_agent) run in the API process. Treat this as class-body
        # evaluation only — the signature class itself is invoked later under
        # the subprocess boundary. See backend/core/worker/subprocess_runner.py.
        #
        # dont_inherit=True: this module's ``from __future__ import annotations``
        # (PEP 563) would otherwise leak into the exec'd code and stringize the
        # signature's field annotations. DSPy resolves bare builtins (``str``,
        # ``int``) but leaves dotted generics like ``typing.List[str]`` as an
        # unresolved ForwardRef, which later crashes GEPA's signature rebuild
        # ("Field types must be types, but received: ForwardRef(...)").
        exec(compile(code, "<signature_code>", "exec", dont_inherit=True), namespace)
    except SyntaxError as exc:
        raise ServiceError(f"signature_code has a syntax error: {exc}") from exc
    signature_classes = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, dspy.Signature) and obj is not dspy.Signature
    ]
    if not signature_classes:
        raise ServiceError("signature_code must define a dspy.Signature subclass.")
    if len(signature_classes) > 1:
        raise ServiceError("signature_code must define exactly one dspy.Signature subclass.")
    signature_cls = signature_classes[0]
    _resolve_signature_annotations(signature_cls, namespace)
    return signature_cls


def _resolve_signature_annotations(
    signature_cls: type[dspy.Signature],
    namespace: dict[str, Any],
) -> None:
    """Force-resolve a signature's field annotations into concrete types in place.

    When the authored code carries ``from __future__ import annotations`` (PEP 563)
    Pydantic stores each field annotation as a ``ForwardRef`` string instead of the
    resolved type. DSPy's ``with_instructions`` / ``make_signature`` rebuild the
    signature from ``cls.fields`` and reject those refs ("Field types must be types,
    but received: ForwardRef(...)"). Resolving names against the exec namespace as
    ``globalns`` (so ``typing``/``List`` etc. bind) and writing the concrete types
    back onto the Pydantic ``FieldInfo`` objects makes DSPy's rebuild — which reads
    ``cls.fields`` (those FieldInfos), not the raw class ``__annotations__`` — accept
    them, regardless of future-import state. We deliberately do not call
    ``model_rebuild``: Pydantic re-evaluates the stringized ``__annotations__``
    against the class's own module globals (where ``List`` etc. are absent) and
    would crash. Resolution failures are left untouched so a signature that merely
    references an unresolvable name still loads as before.

    Args:
        signature_cls: The loaded ``dspy.Signature`` subclass to mutate in place.
        namespace: The exec namespace the signature was defined in, used as the
            global scope for name resolution.
    """
    try:
        hints = typing.get_type_hints(signature_cls, globalns=namespace)
    except Exception:
        return
    fields_mapping = getattr(signature_cls, "model_fields", None) or getattr(
        signature_cls, "__pydantic_fields__", None
    )
    if not fields_mapping:
        return
    for field_name, field_info in fields_mapping.items():
        resolved = hints.get(field_name)
        if resolved is not None and getattr(field_info, "annotation", None) is not resolved:
            field_info.annotation = resolved


def load_metric_from_code(code: str) -> Callable[..., Any]:
    """Execute user-provided code and return the metric callable it defines.

    Same trust boundary as :func:`load_signature_from_code`: the metric body
    runs in-process via ``exec``. Optimization runs route through the per-job
    subprocess sandbox (``worker/subprocess_runner.py``); validation calls
    intentionally exec in the API process. Treat callers as having full code
    execution against the gateway.

    Args:
        code: User-authored metric source code.

    Returns:
        The metric callable extracted from the namespace.

    Raises:
        ServiceError: When the code has a syntax error or defines no callable.
    """

    namespace: dict[str, Any] = {"dspy": dspy}
    try:
        # exec: user-supplied metric code. Same security boundary as
        # load_signature_from_code — isolated when called from
        # DspyService.run/run_grid_search (via worker/subprocess_runner.py),
        # but runs in-process during validation (code_validation router,
        # validate_payload, code_agent). Module-level def evaluation only;
        # the metric callable itself is invoked later inside the subprocess.
        #
        # dont_inherit=True: keep this module's PEP 563 ``from __future__ import
        # annotations`` from stringizing the metric's annotations, matching
        # load_signature_from_code so user code always sees eager annotations.
        exec(compile(code, "<metric_code>", "exec", dont_inherit=True), namespace)
    except SyntaxError as exc:
        raise ServiceError(f"metric_code has a syntax error: {exc}") from exc
    metric = namespace.get("metric")
    if not callable(metric):
        callables = [obj for obj in namespace.values() if callable(obj)]
        if len(callables) == 1:
            metric = callables[0]
    if not callable(metric):
        raise ServiceError("metric_code must define a callable named 'metric'.")
    return metric


def load_transform_from_code(code: str) -> Callable[..., Any]:
    """Execute user-provided code and return the transform callable it defines.

    Transforms are the non-LLM steps of a workflow graph: plain functions
    taking the node's input ports as keyword arguments and returning a dict
    keyed by the node's declared output ports. Same trust boundary as
    :func:`load_metric_from_code` — isolated behind the per-job subprocess
    for optimization runs, exec'd in-process on validation paths.

    Args:
        code: User-authored transform source code.

    Returns:
        The transform callable extracted from the namespace.

    Raises:
        ServiceError: When the code has a syntax error or defines no callable.
    """

    namespace: dict[str, Any] = {"dspy": dspy}
    try:
        # dont_inherit=True for the same reason as load_signature_from_code:
        # keep this module's PEP 563 future-import from stringizing the
        # user function's annotations.
        exec(compile(code, "<transform_code>", "exec", dont_inherit=True), namespace)
    except SyntaxError as exc:
        raise ServiceError(f"transform_code has a syntax error: {exc}") from exc
    transform = namespace.get("transform")
    if not callable(transform):
        callables = [
            obj for obj in namespace.values() if callable(obj) and not isinstance(obj, type) and obj is not dspy
        ]
        if len(callables) == 1:
            transform = callables[0]
    if not callable(transform):
        raise ServiceError("transform_code must define a callable named 'transform'.")
    return transform


def split_examples(
    examples: list[Any],
    fractions: SplitFractions,
    *,
    shuffle: bool,
    seed: int | None,
) -> DatasetSplits:
    """Split examples into train/val/test partitions.

    Args:
        examples: Examples to partition.
        fractions: Train/val/test fractions summing to 1.0.
        shuffle: Whether to shuffle before slicing.
        seed: Optional RNG seed for deterministic splits.

    Returns:
        A populated :class:`DatasetSplits` with train/val/test lists.

    Raises:
        ServiceError: If the dataset is too small to produce the requested
            val split.
    """

    total = len(examples)
    if total == 0:
        return DatasetSplits(train=[], val=[], test=[])

    ordered = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    train_count = int(total * fractions.train)
    val_count = int(total * fractions.val)
    val_end = train_count + val_count
    splits = DatasetSplits(
        train=ordered[:train_count],
        val=ordered[train_count:val_end],
        test=ordered[val_end:],
    )

    if fractions.val > 0 and len(splits.val) == 0:
        raise ServiceError(
            f"Dataset has {total} examples — too small for a val split of "
            f"{fractions.val:.0%}. Add more data or set val fraction to 0."
        )

    return splits


def _is_signature_field(value: Any, *, field_type: str) -> bool:
    """Return True when a signature attribute represents the requested field type.

    Args:
        value: The signature attribute to inspect.
        field_type: Either ``"input"`` or ``"output"``.

    Returns:
        True when ``value`` is a DSPy field of the requested type.
    """

    type_attr = getattr(dspy, "InputField" if field_type == "input" else "OutputField", None)
    if type_attr is not None:
        try:
            if isinstance(value, type_attr):
                return True
        except TypeError:
            pass

    marker = getattr(value, "__dspy_field_type", None)
    if marker == field_type:
        return True

    extra = getattr(value, "json_schema_extra", None)
    return bool(isinstance(extra, dict) and extra.get("__dspy_field_type") == field_type)
