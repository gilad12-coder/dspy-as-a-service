"""Tests for ``core.service_gateway.datasets.profiler.profile_dataset``."""

from __future__ import annotations

import pytest

from core.exceptions import ValidationError
from core.i18n_en import t_en
from core.i18n_keys import I18nKey
from core.models import ColumnMapping
from core.service_gateway.datasets.profiler import profile_dataset


def _mapping(inputs: dict[str, str] | None = None, outputs: dict[str, str] | None = None) -> ColumnMapping:
    """Build a ``ColumnMapping`` with sensible defaults for tests."""
    return ColumnMapping(
        inputs=inputs or {"question": "q"},
        outputs=outputs or {"answer": "a"},
    )


def _rows(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Build a row list from ``(question, answer)`` pairs."""
    return [{"q": q, "a": a} for q, a in pairs]


def test_profile_dataset_empty_raises() -> None:
    """An empty dataset raises ``ValidationError`` with English detail and i18n code."""
    with pytest.raises(ValidationError) as exc_info:
        profile_dataset([], _mapping())
    exc = exc_info.value
    assert str(exc) == t_en(I18nKey.DATASET_PROFILE_EMPTY)
    assert exc.code == I18nKey.DATASET_PROFILE_EMPTY.value


def test_profile_dataset_reports_basic_shape() -> None:
    """Row and column counts are reported on the profile."""
    rows = _rows([("q1", "yes"), ("q2", "no"), ("q3", "yes")])
    profile = profile_dataset(rows, _mapping())

    assert profile.row_count == 3
    assert profile.column_count == 2


def test_profile_dataset_detects_categorical_target() -> None:
    """Short low-cardinality target values produce a categorical profile and histogram."""
    rows = _rows([("q1", "yes"), ("q2", "no"), ("q3", "yes"), ("q4", "no"), ("q5", "yes")])
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "categorical"
    assert profile.target.class_histogram == {"yes": 3, "no": 2}


def test_profile_dataset_detects_numeric_target() -> None:
    """Pure numeric target values produce a numeric profile with empty histogram."""
    rows = [{"q": f"q{i}", "a": float(i)} for i in range(10)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "numeric"
    assert profile.target.class_histogram == {}


def test_profile_dataset_two_value_numeric_target_is_categorical() -> None:
    """A 1/0 label column parsed as numbers is a binary class, not a regression target."""
    rows = [{"q": f"q{i}", "a": i % 2} for i in range(10)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "categorical"


def test_profile_dataset_detects_freeform_target() -> None:
    """Long-text target values produce a freeform profile."""
    long_text = "a" * 80
    rows = [{"q": f"q{i}", "a": f"{long_text} {i}"} for i in range(10)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "freeform"


def test_profile_dataset_counts_duplicates() -> None:
    """Repeated input rows increment the duplicate count."""
    rows = _rows([("q1", "yes"), ("q1", "no"), ("q2", "yes")])
    profile = profile_dataset(rows, _mapping())

    assert profile.duplicate_count == 1


def test_profile_dataset_without_output_mapping_has_no_target() -> None:
    """A mapping with no outputs returns profile.target == None."""
    rows = _rows([("q1", "yes"), ("q2", "no")])
    profile = profile_dataset(rows, ColumnMapping(inputs={"question": "q"}))

    assert profile.target is None
    assert profile.targets == []


def test_profile_dataset_adaptive_categorical_above_min_unique() -> None:
    """A column with 25 short labels in 1000 rows is categorical via the ratio rule."""
    # 25 unique short genres × 40 rows each = 1000 rows. Ratio 0.025 ≪ 0.2.
    rows = [{"q": f"q{i}", "a": f"genre_{i % 25}"} for i in range(1000)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "categorical"
    assert profile.target.unique_values == 25


def test_profile_dataset_adaptive_freeform_when_unique_ratio_too_high() -> None:
    """25 unique short labels in only 30 rows is freeform — each class is just a label."""
    rows = [{"q": f"q{i}", "a": f"label_{i % 25}"} for i in range(30)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    # 25 / 30 = 0.83 > CATEGORICAL_UNIQUE_RATIO=0.2 → freeform.
    assert profile.target.kind == "freeform"


def test_profile_dataset_freeform_when_unique_count_above_max() -> None:
    """A column with more than CATEGORICAL_MAX_UNIQUE distinct values is never categorical."""
    rows = [{"q": f"q{i}", "a": f"id_{i}"} for i in range(1000)]
    profile = profile_dataset(rows, _mapping())

    assert profile.target is not None
    assert profile.target.kind == "freeform"


def test_profile_dataset_profiles_every_output_column() -> None:
    """A mapping with two outputs produces a TargetColumnProfile for each."""
    rows = [{"q": f"q{i}", "label": "yes" if i % 2 == 0 else "no", "tag": f"t{i % 3}"} for i in range(30)]
    profile = profile_dataset(
        rows,
        ColumnMapping(
            inputs={"question": "q"},
            outputs={"label": "label", "tag": "tag"},
        ),
    )

    assert len(profile.targets) == 2
    names = {t.name for t in profile.targets}
    assert names == {"label", "tag"}


def test_profile_dataset_primary_target_prefers_categorical() -> None:
    """When outputs mix kinds, the primary target is the smallest-cardinality categorical."""
    rows = [{"q": f"q{i}", "label": "yes" if i % 2 == 0 else "no", "summary": "a" * 80} for i in range(20)]
    profile = profile_dataset(
        rows,
        ColumnMapping(
            inputs={"question": "q"},
            outputs={"summary": "summary", "label": "label"},
        ),
    )

    assert profile.target is not None
    assert profile.target.name == "label"
    assert profile.target.kind == "categorical"


def test_profile_dataset_inputs_default_to_text_for_plain_strings() -> None:
    """A column of plain text values is profiled as kind='text'."""
    rows = _rows([("Why?", "yes"), ("How?", "no")])
    profile = profile_dataset(rows, _mapping())

    assert len(profile.inputs) == 1
    assert profile.inputs[0].name == "q"
    assert profile.inputs[0].kind == "text"


def test_profile_dataset_detects_image_url_column() -> None:
    """A column where every cell is an HTTPS image URL is profiled as kind='image'."""
    rows = [
        {"q": "https://cdn.example.com/cat.png", "a": "cat"},
        {"q": "http://images.example.org/dog.JPG?v=2", "a": "dog"},
        {"q": "https://example.com/path/to/pic.webp", "a": "bird"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "image"


def test_profile_dataset_detects_data_uri_image_column() -> None:
    """A column where every cell is a data:image/...;base64 URI is image."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    rows = [
        {"q": data_uri, "a": "x"},
        {"q": data_uri.replace("png", "jpeg"), "a": "y"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "image"


def test_profile_dataset_mixed_image_and_text_falls_back_to_text() -> None:
    """A column mixing image URLs and plain text stays kind='text' (deliberate fallback)."""
    rows = [
        {"q": "https://cdn.example.com/cat.png", "a": "yes"},
        {"q": "no image just words", "a": "no"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "text"


def test_profile_dataset_empty_input_column_is_text() -> None:
    """A column with only None / blank values is profiled as kind='text'."""
    rows: list[dict[str, object]] = [
        {"q": None, "a": "yes"},
        {"q": "   ", "a": "no"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "text"


def test_profile_dataset_image_url_with_query_string_detected() -> None:
    """An image URL with query parameters still detects as image."""
    rows = [
        {"q": "https://cdn.example.com/photo.jpg?w=512&h=512", "a": "x"},
        {"q": "https://cdn.example.com/photo2.gif?token=abc", "a": "y"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "image"


def test_profile_dataset_non_image_url_classified_as_text() -> None:
    """An HTTPS URL that doesn't end in an image extension stays text."""
    rows = [
        {"q": "https://example.com/page.html", "a": "x"},
        {"q": "https://api.example.com/items/42", "a": "y"},
    ]
    profile = profile_dataset(rows, _mapping())

    assert profile.inputs[0].kind == "text"


def test_profile_dataset_profiles_every_input_column() -> None:
    """A mapping with two inputs produces an InputColumnProfile for each, with per-column kind."""
    rows = [
        {
            "q": f"q{i}",
            "img": f"https://cdn.example.com/{i}.png",
            "a": "yes" if i % 2 == 0 else "no",
        }
        for i in range(4)
    ]
    profile = profile_dataset(
        rows,
        ColumnMapping(
            inputs={"question": "q", "picture": "img"},
            outputs={"answer": "a"},
        ),
    )

    by_name = {p.name: p.kind for p in profile.inputs}
    assert by_name == {"q": "text", "img": "image"}
