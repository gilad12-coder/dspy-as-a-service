import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import dspy
from ..exceptions import ServiceError
from ..models import ColumnMapping, SplitFractions


@dataclass
class DatasetSplits:
    """Container for train/validation/test partitions."""

    train: List[Any]
    val: List[Any]
    test: List[Any]


def extract_signature_fields(signature_cls: type[dspy.Signature]) -> tuple[List[str], List[str]]:
    """Inspect a DSPy signature class to derive input and output field names.

    Args:
        signature_cls: Compiled ``dspy.Signature`` subclass to introspect.

    Returns:
        tuple[list[str], list[str]]: Names of input fields followed by output fields.
    """

    inputs: List[str] = []
    outputs: List[str] = []

    fields_mapping = getattr(signature_cls, "model_fields", None) or getattr(
        signature_cls, "__pydantic_fields__", None
    )
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


def rows_to_examples(dataset: List[Dict[str, Any]], mapping: ColumnMapping) -> List[Any]:
    """Convert dataframe-like rows into DSPy Example instances.

    Args:
        dataset: Iterable of row dictionaries representing user data.
        mapping: Column mapping aligning dataframe columns with signature fields.

    Returns:
        list[Any]: DSPy ``Example`` objects annotated with inputs/outputs.

    Raises:
        ServiceError: If a required column is missing from any row.
    """

    examples: List[Any] = []
    for row_idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ServiceError(f"Row {row_idx} is not a mapping: {row!r}")
        payload: Dict[str, Any] = {}
        for signature_field, column_name in mapping.inputs.items():
            try:
                payload[signature_field] = row[column_name]
            except KeyError as exc:
                raise ServiceError(
                    f"Missing input column '{column_name}' for row {row_idx}"
                ) from exc
        for signature_field, column_name in mapping.outputs.items():
            try:
                payload[signature_field] = row[column_name]
            except KeyError as exc:
                raise ServiceError(
                    f"Missing output column '{column_name}' for row {row_idx}"
                ) from exc

        example = dspy.Example(**payload)
        if hasattr(example, "with_inputs"):
            example = example.with_inputs(*mapping.inputs.keys())
        if mapping.outputs:
            if hasattr(example, "with_outputs"):
                example = example.with_outputs(*mapping.outputs.keys())
            elif hasattr(example, "with_targets"):
                example = example.with_targets(*mapping.outputs.keys())
        examples.append(example)

    return examples


def load_signature_from_code(code: str) -> type[dspy.Signature]:
    """Execute user-provided code and return the defined DSPy signature class.

    Args:
        code: Source code string that defines exactly one ``dspy.Signature``.

    Returns:
        type[dspy.Signature]: Parsed signature class extracted from the code.

    Raises:
        ServiceError: If zero or multiple signature classes are defined.
    """

    namespace: Dict[str, Any] = {"dspy": dspy}
    exec(code, namespace)
    signature_classes = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, dspy.Signature) and obj is not dspy.Signature
    ]
    if not signature_classes:
        raise ServiceError("signature_code must define a dspy.Signature subclass.")
    if len(signature_classes) > 1:
        raise ServiceError("signature_code must define exactly one dspy.Signature subclass.")
    return signature_classes[0]


def load_metric_from_code(code: str) -> Callable[..., Any]:
    """Execute user-provided code and return the metric callable.

    Args:
        code: Source code string expected to define a callable metric.

    Returns:
        Callable[..., Any]: Metric callable extracted from the namespace.

    Raises:
        ServiceError: If no callable metric is found.
    """

    namespace: Dict[str, Any] = {"dspy": dspy}
    exec(code, namespace)
    metric = namespace.get("metric")
    if not callable(metric):
        callables = [obj for obj in namespace.values() if callable(obj)]
        if len(callables) == 1:
            metric = callables[0]
    if not callable(metric):
        raise ServiceError("metric_code must define a callable named 'metric'.")
    return metric


def split_examples(
    examples: List[Any],
    fractions: SplitFractions,
    *,
    shuffle: bool,
    seed: Optional[int],
) -> DatasetSplits:
    """Split examples into train, val, and test partitions.

    Args:
        examples: Sequence of DSPy example objects.
        fractions: Desired split fractions that must sum to one.
        shuffle: Whether to shuffle the dataset prior to splitting.
        seed: Optional deterministic seed for the shuffle step.

    Returns:
        DatasetSplits: Structured container holding train/val/test lists.
    """

    total = len(examples)
    if total == 0:
        return DatasetSplits(train=[], val=[], test=[])

    ordered = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    train_end = int(total * fractions.train)
    val_end = train_end + int(total * fractions.val)
    train_split = ordered[:train_end]
    val_split = ordered[train_end:val_end]
    test_split = ordered[val_end:]

    return DatasetSplits(train=train_split, val=val_split, test=test_split)


def _is_signature_field(value: Any, *, field_type: str) -> bool:
    """Return True when a signature attribute represents the requested field type.

    Args:
        value: Attribute pulled from the signature class dict.
        field_type: Either ``"input"`` or ``"output"``.

    Returns:
        bool: True when the attribute is marked as the requested type.
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
    if isinstance(extra, dict) and extra.get("__dspy_field_type") == field_type:
        return True

    return False
