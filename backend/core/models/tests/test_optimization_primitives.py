"""Tests for OptimizationStatus, ColumnMapping, SplitFractions, and ModelConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models.common import ColumnMapping, ModelConfig, OptimizationStatus, SplitFractions


def test_optimization_status_complete_set() -> None:
    """Verify OptimizationStatus exposes the full set of expected status values."""
    assert {s.value for s in OptimizationStatus} == {
        "pending",
        "validating",
        "running",
        "success",
        "failed",
        "cancelled",
        "paused",
    }


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("pending", "pending"),
        ("validating", "validating"),
        ("running", "running"),
        ("success", "success"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("paused", "paused"),
    ],
    ids=["pending", "validating", "running", "success", "failed", "cancelled", "paused"],
)
def test_optimization_status_each_value(member: str, value: str) -> None:
    """Verify each OptimizationStatus member maps to its expected string value."""
    assert OptimizationStatus[member].value == value


def test_optimization_status_is_str_subclass() -> None:
    """Verify OptimizationStatus members are subclasses of str (StrEnum)."""
    assert isinstance(OptimizationStatus.pending, str)


def test_column_mapping_accepts_disjoint_columns() -> None:
    """Verify ColumnMapping accepts disjoint input and output column sets."""
    m = ColumnMapping(inputs={"q": "question"}, outputs={"a": "answer"})

    assert m.inputs == {"q": "question"}
    assert m.outputs == {"a": "answer"}


def test_column_mapping_requires_at_least_one_input() -> None:
    """Verify ColumnMapping rejects an empty inputs dict."""
    with pytest.raises(ValidationError, match="At least one input"):
        ColumnMapping(inputs={}, outputs={"answer": "answer"})


def test_column_mapping_rejects_shared_column_values() -> None:
    """Verify ColumnMapping rejects when input and output values overlap."""
    with pytest.raises(ValidationError, match="must not reuse the same columns"):
        ColumnMapping(inputs={"q": "col"}, outputs={"a": "col"})


def test_column_mapping_empty_outputs_accepted() -> None:
    """Verify ColumnMapping accepts an empty outputs dict."""
    m = ColumnMapping(inputs={"q": "question"}, outputs={})

    assert m.outputs == {}


def test_column_mapping_multiple_shared_columns_reported() -> None:
    """Verify ColumnMapping rejects multiple overlapping column values."""
    with pytest.raises(ValidationError, match="must not reuse the same columns"):
        ColumnMapping(inputs={"q": "x", "r": "y"}, outputs={"a": "x", "b": "y"})


def test_column_mapping_rejects_shared_signature_field_name() -> None:
    """Verify ColumnMapping rejects when a signature field appears in inputs and outputs."""
    with pytest.raises(ValidationError, match="cannot be both input and output"):
        ColumnMapping(inputs={"shared": "col_a"}, outputs={"shared": "col_b"})


def test_split_fractions_defaults_sum_to_one() -> None:
    """Verify default SplitFractions train/val/test sum to 1.0."""
    s = SplitFractions()

    assert abs(s.train + s.val + s.test - 1.0) < 1e-6


def test_split_fractions_defaults_values() -> None:
    """Verify default SplitFractions are 0.7/0.15/0.15."""
    s = SplitFractions()

    assert s.train == pytest.approx(0.7)
    assert s.val == pytest.approx(0.15)
    assert s.test == pytest.approx(0.15)


def test_split_fractions_rejects_negative_train() -> None:
    """Verify SplitFractions rejects a negative train fraction."""
    with pytest.raises(ValidationError, match="non-negative"):
        SplitFractions(train=-0.1, val=0.55, test=0.55)


def test_split_fractions_rejects_negative_val() -> None:
    """Verify SplitFractions rejects a negative val fraction."""
    with pytest.raises(ValidationError, match="non-negative"):
        SplitFractions(train=0.7, val=-0.05, test=0.35)


def test_split_fractions_rejects_negative_test() -> None:
    """Verify SplitFractions rejects a negative test fraction."""
    with pytest.raises(ValidationError, match="non-negative"):
        SplitFractions(train=0.8, val=0.2, test=-0.0001)


def test_split_fractions_rejects_wrong_total() -> None:
    """Verify SplitFractions rejects fractions that do not sum to 1.0."""
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        SplitFractions(train=0.5, val=0.25, test=0.26)


def test_split_fractions_zero_val_accepted() -> None:
    """Verify SplitFractions accepts a zero val fraction."""
    s = SplitFractions(train=0.85, val=0.0, test=0.15)

    assert s.val == pytest.approx(0.0)


def test_split_fractions_custom_valid_split() -> None:
    """Verify SplitFractions accepts a custom 0.6/0.2/0.2 split."""
    s = SplitFractions(train=0.6, val=0.2, test=0.2)

    assert abs(s.train + s.val + s.test - 1.0) < 1e-6


def test_model_config_minimal_defaults() -> None:
    """Verify ModelConfig defaults all optional fields to None or empty."""
    m = ModelConfig(name="gpt-4o-mini")

    assert m.temperature is None
    assert m.base_url is None
    assert m.max_tokens is None
    assert m.extra == {}


def test_model_config_normalized_identifier_strips_leading_slash() -> None:
    """Verify normalized_identifier strips a leading slash from the name."""
    assert ModelConfig(name="/gpt-4o-mini").normalized_identifier() == "gpt-4o-mini"


def test_model_config_normalized_identifier_strips_trailing_slash() -> None:
    """Verify normalized_identifier strips a trailing slash from the name."""
    assert ModelConfig(name="gpt-4o-mini/").normalized_identifier() == "gpt-4o-mini"


def test_model_config_normalized_identifier_strips_both_slashes() -> None:
    """Verify normalized_identifier strips both leading and trailing slashes."""
    assert ModelConfig(name="/gpt-4o-mini/").normalized_identifier() == "gpt-4o-mini"


def test_model_config_normalized_identifier_no_slashes_unchanged() -> None:
    """Verify normalized_identifier returns the name unchanged when no slashes."""
    assert ModelConfig(name="gpt-4o-mini").normalized_identifier() == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("temp", "valid"),
    [
        (-0.1, False),
        (0.0, True),
        (1.0, True),
        (2.0, True),
        (2.1, False),
    ],
    ids=["below_min", "min", "mid", "max", "above_max"],
)
def test_model_config_temperature_boundary(temp: float, valid: bool) -> None:
    """Verify ModelConfig.temperature accepts [0.0, 2.0] and rejects values outside."""
    if valid:
        m = ModelConfig(name="x", temperature=temp)
        assert m.temperature == pytest.approx(temp)
    else:
        with pytest.raises(ValidationError):
            ModelConfig(name="x", temperature=temp)


@pytest.mark.parametrize(
    ("max_tokens", "valid"),
    [
        (0, False),
        (1, True),
        (4096, True),
    ],
    ids=["zero", "min", "large"],
)
def test_model_config_max_tokens_boundary(max_tokens: int, valid: bool) -> None:
    """Verify ModelConfig.max_tokens requires >= 1 and rejects 0 or negative."""
    if valid:
        m = ModelConfig(name="x", max_tokens=max_tokens)
        assert m.max_tokens == max_tokens
    else:
        with pytest.raises(ValidationError):
            ModelConfig(name="x", max_tokens=max_tokens)
