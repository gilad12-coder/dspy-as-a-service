"""Pure functions that describe an uploaded dataset.

The profiler walks the raw row list once and produces a ``DatasetProfile``
summarizing its shape, the nature of every output column, duplicate
counts by input columns, and any warnings the user should see before
submitting an optimization. No side effects; safe to call from request
handlers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ...exceptions import ValidationError
from ...i18n import t
from ...i18n_en import t_en
from ...i18n_keys import I18nKey
from ...models.common import ColumnMapping
from ...models.dataset import (
    DatasetProfile,
    InputColumnProfile,
    ProfileWarning,
    ProfileWarningCode,
    TargetColumnProfile,
)

# Adaptive categorical-vs-freeform thresholds. Up to ``MIN_UNIQUE`` distinct
# values a column is always treated as categorical (the cheap, common case).
# Between ``MIN_UNIQUE`` and ``MAX_UNIQUE`` we additionally require the
# unique-to-row ratio to be small enough that each class has roughly five+
# examples on average — otherwise the "classes" are just per-row labels.
# Above ``MAX_UNIQUE`` no column is categorical — hundreds of micro-classes
# carry no useful structure for display or warnings.
CATEGORICAL_MIN_UNIQUE = 20
CATEGORICAL_MAX_UNIQUE = 100
CATEGORICAL_UNIQUE_RATIO = 0.2

RARE_CLASS_THRESHOLD = 5
IMBALANCE_RATIO_THRESHOLD = 10.0
MIN_RECOMMENDED_ROWS = 30
FREEFORM_AVG_LENGTH = 40

# Cell-level image detection. A column is classified ``image`` only when EVERY
# non-empty cell matches one of these patterns — mixed text/image columns stay
# ``text`` so we don't silently drop the textual rows on the runtime side.
_IMAGE_URL_RE = re.compile(r"^https?://\S+\.(?:png|jpe?g|gif|webp)(?:\?\S*)?$", re.IGNORECASE)
_IMAGE_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def profile_dataset(dataset: list[dict[str, Any]], mapping: ColumnMapping) -> DatasetProfile:
    """Return a structural summary and warning list for a raw dataset.

    Args:
        dataset: Raw dataset rows.
        mapping: Column mapping declaring inputs and outputs.

    Returns:
        A populated :class:`DatasetProfile` describing shape, targets,
        inputs, duplicates, and warnings.

    Raises:
        ValidationError: When ``dataset`` is empty.
    """
    if not dataset:
        raise ValidationError(
            t_en(I18nKey.DATASET_PROFILE_EMPTY),
            code=I18nKey.DATASET_PROFILE_EMPTY.value,
        )

    row_count = len(dataset)
    columns: set[str] = set()
    for row in dataset:
        columns.update(row.keys())

    warnings: list[ProfileWarning] = []
    targets = _profile_all_targets(dataset, mapping, warnings)
    inputs = _profile_all_inputs(dataset, mapping)
    primary_target = _select_primary_target(targets)

    if row_count < MIN_RECOMMENDED_ROWS:
        warnings.append(
            ProfileWarning(
                code=ProfileWarningCode.too_small,
                message=t("dataset.profile.too_small", row_count=row_count),
                details={"row_count": row_count, "minimum_recommended": MIN_RECOMMENDED_ROWS},
            )
        )

    duplicate_count = _count_duplicates(dataset, mapping)
    if duplicate_count > 0:
        warnings.append(
            ProfileWarning(
                code=ProfileWarningCode.duplicates,
                message=t("dataset.profile.duplicates", duplicate_count=duplicate_count),
                details={"duplicate_count": duplicate_count},
            )
        )

    return DatasetProfile(
        row_count=row_count,
        column_count=len(columns),
        target=primary_target,
        targets=targets,
        inputs=inputs,
        duplicate_count=duplicate_count,
        warnings=warnings,
    )


def _profile_all_targets(
    dataset: list[dict[str, Any]],
    mapping: ColumnMapping,
    warnings: list[ProfileWarning],
) -> list[TargetColumnProfile]:
    """Profile every output column declared in ``mapping``.

    Each output gets its own categorical/numeric/freeform classification,
    histogram, and per-column warnings. Warning details always include
    the originating ``target_column`` so downstream consumers (the
    planner) can attribute findings back to a specific column. Profile-level
    warnings are appended to ``warnings`` in-place.

    Args:
        dataset: Raw dataset rows.
        mapping: Column mapping whose ``outputs`` are profiled.
        warnings: Mutable warning list extended with target-level findings.

    Returns:
        A list of :class:`TargetColumnProfile` instances, one per output column.
    """
    profiles: list[TargetColumnProfile] = []
    for column_name in mapping.outputs.values():
        profile = _profile_single_target(dataset, column_name, warnings)
        if profile is not None:
            profiles.append(profile)
    return profiles


def _profile_all_inputs(
    dataset: list[dict[str, Any]],
    mapping: ColumnMapping,
) -> list[InputColumnProfile]:
    """Profile every input column declared in ``mapping``.

    Each input gets a single ``kind`` — ``image`` when every non-empty
    cell parses as an image URL or ``data:image/...`` URI, otherwise
    ``text``. The wizard renders an image badge on ``image`` columns;
    the signature generator emits ``dspy.Image`` typed ``InputField``
    for them.

    Args:
        dataset: Raw dataset rows.
        mapping: Column mapping whose ``inputs`` are profiled.

    Returns:
        A list of :class:`InputColumnProfile` instances, one per input column.
    """
    profiles: list[InputColumnProfile] = []
    for column_name in mapping.inputs.values():
        kind = _infer_input_kind(_collect_values(dataset, column_name))
        profiles.append(InputColumnProfile(name=column_name, kind=kind))
    return profiles


def _collect_values(dataset: list[dict[str, Any]], column_name: str) -> list[Any]:
    """Return the non-empty values of ``column_name`` across ``dataset``.

    Skips ``None`` and whitespace-only strings.

    Args:
        dataset: Raw dataset rows.
        column_name: Column whose values are collected.

    Returns:
        The list of non-empty values for ``column_name``.
    """
    out: list[Any] = []
    for row in dataset:
        value = row.get(column_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        out.append(value)
    return out


def _infer_input_kind(values: list[Any]) -> str:
    """Classify an input column as ``image`` or ``text``.

    Returns ``image`` only when every non-empty cell is a string that
    matches an HTTPS image URL (``.png``/``.jpg``/``.jpeg``/``.gif``/
    ``.webp``) or a base64 ``data:image/...`` URI. Empty columns and
    mixed columns fall back to ``text`` so we never silently coerce
    non-image rows into ``dspy.Image`` at runtime.

    Args:
        values: Non-empty cell values for the column.

    Returns:
        ``"image"`` when every value parses as an image reference, else ``"text"``.
    """
    if not values:
        return "text"
    for value in values:
        if not isinstance(value, str):
            return "text"
        candidate = value.strip()
        if not (_IMAGE_URL_RE.match(candidate) or _IMAGE_DATA_URI_RE.match(candidate)):
            return "text"
    return "image"


def _profile_single_target(
    dataset: list[dict[str, Any]],
    column_name: str,
    warnings: list[ProfileWarning],
) -> TargetColumnProfile | None:
    """Summarize a single output column and append its warnings.

    Missing-value, rare-class, and class-imbalance warnings are appended
    to ``warnings`` in-place. Returns ``None`` when the column had no
    usable data.

    Args:
        dataset: Raw dataset rows.
        column_name: The output column to profile.
        warnings: Mutable warning list extended with column-level findings.

    Returns:
        A :class:`TargetColumnProfile` describing the column.
    """
    values: list[Any] = []
    missing = 0
    for row in dataset:
        value = row.get(column_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing += 1
        else:
            values.append(value)

    if missing > 0:
        warnings.append(
            ProfileWarning(
                code=ProfileWarningCode.missing_target,
                message=t(
                    "dataset.profile.missing_target",
                    missing=missing,
                    column_name=column_name,
                ),
                details={"missing_count": missing, "target_column": column_name},
            )
        )

    kind = _infer_target_kind(values)
    histogram: dict[str, int] = {}
    unique_values = len({_stringify(v) for v in values})

    if kind == "categorical" and values:
        counts = Counter(_stringify(v) for v in values)
        histogram = dict(counts.most_common())
        rare_classes = {k: v for k, v in histogram.items() if v < RARE_CLASS_THRESHOLD}
        if rare_classes:
            warnings.append(
                ProfileWarning(
                    code=ProfileWarningCode.rare_class,
                    message=t(
                        "dataset.profile.rare_class",
                        column_name=column_name,
                        rare_classes=", ".join(sorted(rare_classes)),
                    ),
                    details={"rare_classes": rare_classes, "target_column": column_name},
                )
            )
        if len(histogram) >= 2:
            majority = max(histogram.values())
            minority = min(histogram.values())
            if minority > 0 and majority / minority > IMBALANCE_RATIO_THRESHOLD:
                warnings.append(
                    ProfileWarning(
                        code=ProfileWarningCode.class_imbalance,
                        message=t(
                            "dataset.profile.class_imbalance",
                            column_name=column_name,
                            ratio=majority // minority,
                        ),
                        details={
                            "majority": majority,
                            "minority": minority,
                            "target_column": column_name,
                        },
                    )
                )

    return TargetColumnProfile(
        name=column_name,
        kind=kind,
        unique_values=unique_values,
        class_histogram=histogram,
    )


def _select_primary_target(
    targets: list[TargetColumnProfile],
) -> TargetColumnProfile | None:
    """Pick the target column to surface as the profile's representative output.

    Prefers categorical columns; among categoricals, picks the one with
    the fewest unique values, since a smaller class set typically
    indicates a cleaner label space worth highlighting in the UI summary.
    Falls back to the first declared output when no categoricals exist.

    Args:
        targets: Per-output target profiles.

    Returns:
        The chosen primary target profile, or ``None`` when ``targets`` is empty.
    """
    if not targets:
        return None
    categoricals = [t for t in targets if t.kind == "categorical"]
    if categoricals:
        return min(categoricals, key=lambda t: t.unique_values or 0)
    return targets[0]


def _infer_target_kind(values: list[Any]) -> str:
    """Classify the target column as ``categorical`` / ``numeric`` / ``freeform``.

    Empty lists fall back to ``freeform``. Lists of pure numerics are
    ``numeric``, unless only two distinct values appear (a binary class
    column). Strings are ``categorical`` when:

    - the average value length is short (otherwise they're prose), AND
    - either there are very few distinct values (cheap path), or there
      are at most ``CATEGORICAL_MAX_UNIQUE`` distinct values AND the
      unique-to-total ratio is at most ``CATEGORICAL_UNIQUE_RATIO`` (each
      "class" repeats often enough to mean something).

    Args:
        values: Non-empty cell values for the target column.

    Returns:
        One of ``"categorical"``, ``"numeric"``, or ``"freeform"``.
    """
    if not values:
        return "freeform"

    n = len(values)
    numeric_count = sum(1 for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if numeric_count == n:
        # Two distinct numeric values are a binary class column (1/0 labels,
        # e.g. from the tagger's exports), not a regression target.
        if len({_stringify(v) for v in values}) <= 2:
            return "categorical"
        return "numeric"

    avg_len = sum(len(_stringify(v)) for v in values) / n
    if avg_len > FREEFORM_AVG_LENGTH:
        return "freeform"

    n_unique = len({_stringify(v) for v in values})
    if n_unique <= CATEGORICAL_MIN_UNIQUE:
        return "categorical"
    if n_unique <= CATEGORICAL_MAX_UNIQUE and n_unique / n <= CATEGORICAL_UNIQUE_RATIO:
        return "categorical"
    return "freeform"


def _stringify(value: Any) -> str:
    """Coerce a value to a stable string key for hashing and display.

    Args:
        value: Any cell value.

    Returns:
        The trimmed string form of ``value`` (empty string when ``value`` is None).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _count_duplicates(dataset: list[dict[str, Any]], mapping: ColumnMapping) -> int:
    """Count rows whose input-column tuple has appeared earlier in the list.

    Returns 0 when the mapping declares no input columns, which should
    never happen in practice (``ColumnMapping`` rejects that at construction).

    Args:
        dataset: Raw dataset rows.
        mapping: Column mapping whose ``inputs`` define duplicate identity.

    Returns:
        The number of rows whose input tuple already appeared earlier.
    """
    columns = list(mapping.inputs.values())
    if not columns:
        return 0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in dataset:
        key = tuple(_stringify(row.get(col)) for col in columns)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates
