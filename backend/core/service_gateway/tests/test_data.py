"""Tests for ``core.service_gateway.optimization.data``.

Covers split helpers, signature loading, metric loading, row-to-example
conversion, and image-field detection.
"""

from __future__ import annotations

import typing

import dspy
import pytest

from core.exceptions import ServiceError
from core.models import ColumnMapping, SplitFractions
from core.service_gateway.optimization.data import (
    _coerce_image,
    _is_signature_field,
    extract_signature_fields,
    image_input_field_names,
    load_metric_from_code,
    load_signature_from_code,
    rows_to_examples,
    split_examples,
)


def _fractions(train: float, val: float, test: float) -> SplitFractions:
    """Build a ``SplitFractions`` with the provided ratios."""
    return SplitFractions(train=train, val=val, test=test)


def _items(n: int) -> list[int]:
    """Return ``[0, 1, ..., n-1]`` as a stand-in dataset of trivial items."""
    return list(range(n))


@pytest.mark.parametrize(
    ("n", "train_f", "val_f", "test_f", "exp_train", "exp_val", "exp_test"),
    [
        (10, 0.7, 0.15, 0.15, 7, 1, 2),
        (100, 0.7, 0.15, 0.15, 70, 15, 15),
        (1, 1.0, 0.0, 0.0, 1, 0, 0),
        (10, 0.8, 0.1, 0.1, 8, 1, 1),
        (10, 0.5, 0.5, 0.0, 5, 5, 0),
    ],
    ids=["10-std", "100-std", "1-all-train", "10-80-10-10", "10-50-50-0"],
)
def test_split_examples_counts(n, train_f, val_f, test_f, exp_train, exp_val, exp_test) -> None:
    """Split counts match expectations across multiple dataset sizes."""
    items = _items(n)
    fractions = _fractions(train_f, val_f, test_f)

    result = split_examples(items, fractions, shuffle=False, seed=None)

    assert len(result.train) == exp_train
    assert len(result.val) == exp_val
    assert len(result.test) == exp_test


def test_split_examples_too_small_for_nonzero_val_raises() -> None:
    """Datasets too small to allocate a non-empty val split raise ``ServiceError``."""
    items = _items(3)
    fractions = _fractions(0.7, 0.15, 0.15)

    with pytest.raises(ServiceError, match="too small for a val split"):
        split_examples(items, fractions, shuffle=False, seed=None)


def test_split_examples_empty_dataset_returns_empty_splits() -> None:
    """An empty input yields three empty splits."""
    result = split_examples([], _fractions(0.7, 0.15, 0.15), shuffle=False, seed=None)

    assert result.train == []
    assert result.val == []
    assert result.test == []


def test_split_examples_all_items_accounted_for() -> None:
    """Train + val + test sum to the original dataset size."""
    items = _items(20)
    fractions = _fractions(0.7, 0.15, 0.15)

    result = split_examples(items, fractions, shuffle=False, seed=None)

    assert len(result.train) + len(result.val) + len(result.test) == len(items)


def test_split_examples_no_shuffle_preserves_order() -> None:
    """``shuffle=False`` keeps the original ordering across all three splits."""
    items = _items(10)
    fractions = _fractions(0.7, 0.15, 0.15)

    result = split_examples(items, fractions, shuffle=False, seed=None)

    combined = result.train + result.val + result.test
    assert combined == items


def test_split_examples_shuffle_changes_order() -> None:
    """``shuffle=True`` reorders the dataset while preserving membership."""
    items = _items(20)
    fractions = _fractions(0.7, 0.15, 0.15)

    result = split_examples(items, fractions, shuffle=True, seed=42)
    combined = result.train + result.val + result.test

    assert sorted(combined) == items
    assert combined != items


def test_split_examples_same_seed_is_deterministic() -> None:
    """The same seed produces identical splits across calls."""
    items = _items(30)
    fractions = _fractions(0.7, 0.15, 0.15)

    r1 = split_examples(items, fractions, shuffle=True, seed=99)
    r2 = split_examples(items, fractions, shuffle=True, seed=99)

    assert r1.train == r2.train
    assert r1.val == r2.val
    assert r1.test == r2.test


def test_split_examples_different_seeds_differ() -> None:
    """Different seeds produce different orderings (statistically certain)."""
    items = _items(30)
    fractions = _fractions(0.7, 0.15, 0.15)

    r1 = split_examples(items, fractions, shuffle=True, seed=1)
    r2 = split_examples(items, fractions, shuffle=True, seed=2)

    # It would be astronomically unlikely for both orderings to match.
    assert r1.train != r2.train


def test_split_examples_shuffle_does_not_mutate_input() -> None:
    """The caller's input list is not reordered as a side effect of shuffle."""
    items = _items(20)
    original = list(items)
    fractions = _fractions(0.7, 0.15, 0.15)

    split_examples(items, fractions, shuffle=True, seed=7)

    assert items == original


_VALID_SIG = """
import dspy
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
"""


def test_load_signature_from_code_returns_signature_class() -> None:
    """Valid source returns a ``dspy.Signature`` subclass."""
    sig = load_signature_from_code(_VALID_SIG)

    assert issubclass(sig, dspy.Signature)


def test_load_signature_from_code_syntax_error_raises_service_error() -> None:
    """A syntax error in user source raises a ``ServiceError``."""
    with pytest.raises(ServiceError, match="syntax error"):
        load_signature_from_code("def bad syntax !!!")


def test_load_signature_from_code_no_signature_raises_service_error() -> None:
    """Source without a ``dspy.Signature`` subclass raises ``ServiceError``."""
    with pytest.raises(ServiceError, match=r"must define a dspy\.Signature"):
        load_signature_from_code("x = 1")


def test_load_signature_from_code_multiple_signatures_raises_service_error() -> None:
    """Source with more than one signature raises ``ServiceError``."""
    code = """
import dspy
class A(dspy.Signature):
    q: str = dspy.InputField()
    a: str = dspy.OutputField()
class B(dspy.Signature):
    q: str = dspy.InputField()
    a: str = dspy.OutputField()
"""
    with pytest.raises(ServiceError, match="exactly one"):
        load_signature_from_code(code)


def test_load_signature_from_code_resolves_generic_annotations() -> None:
    """Dotted generic annotations resolve to real types, not ``ForwardRef``.

    Regression: ``data.py`` carries ``from __future__ import annotations``, and
    ``exec`` inherits that PEP 563 flag unless compiled with ``dont_inherit``.
    Without the fix a field typed ``typing.List[str]`` survived as an
    unresolved ``ForwardRef``, crashing GEPA's ``with_instructions`` rebuild
    with "Field types must be types, but received: ForwardRef(...)".
    """
    code = """
import dspy
import typing
class PredictDecade(dspy.Signature):
    title: str = dspy.InputField()
    subcategories: typing.List[str] = dspy.InputField()
    inventor_count: int = dspy.InputField()
    decade: str = dspy.OutputField()
"""
    sig = load_signature_from_code(code)

    assert not any(
        isinstance(f.annotation, typing.ForwardRef) for f in sig.model_fields.values()
    )
    # The loader must preserve the user's typing.List form verbatim; list[str]
    # is a distinct object that compares unequal, so UP006 must not rewrite this.
    assert sig.model_fields["subcategories"].annotation == typing.List[str]  # noqa: UP006
    sig.with_instructions("rebuilt")  # GEPA rebuild path must not raise


def test_load_metric_from_code_returns_callable() -> None:
    """A valid ``def metric`` source loads as a callable."""
    code = "def metric(example, prediction, trace=None): return 1.0"

    result = load_metric_from_code(code)

    assert callable(result)


def test_load_metric_from_code_named_metric_preferred() -> None:
    """When multiple callables exist, the one named 'metric' is selected."""
    code = """
def helper(): pass
def metric(example, prediction, trace=None): return 1.0
"""
    result = load_metric_from_code(code)

    assert result.__name__ == "metric"


def test_load_metric_from_code_no_callable_raises_service_error() -> None:
    """Source without any callable raises ``ServiceError``."""
    with pytest.raises(ServiceError, match="must define a callable"):
        load_metric_from_code("x = 42")


def test_load_metric_from_code_syntax_error_raises_service_error() -> None:
    """A syntax error in metric source raises ``ServiceError``."""
    with pytest.raises(ServiceError, match="syntax error"):
        load_metric_from_code("def !!!")


_SIMPLE_MAPPING = ColumnMapping(inputs={"question": "q"}, outputs={"answer": "a"})

_SIMPLE_ROWS = [
    {"q": "What is 1+1?", "a": "2"},
    {"q": "What is 2+2?", "a": "4"},
]


def test_rows_to_examples_happy_path_returns_correct_count() -> None:
    """Two source rows produce two examples."""
    examples = rows_to_examples(_SIMPLE_ROWS, _SIMPLE_MAPPING)

    assert len(examples) == 2


def test_rows_to_examples_example_has_expected_input_value() -> None:
    """The mapped input field carries the source-column value."""
    examples = rows_to_examples(_SIMPLE_ROWS, _SIMPLE_MAPPING)

    assert examples[0].question == "What is 1+1?"


def test_rows_to_examples_example_has_expected_output_value() -> None:
    """The mapped output field carries the source-column value."""
    examples = rows_to_examples(_SIMPLE_ROWS, _SIMPLE_MAPPING)

    assert examples[0].answer == "2"


def test_rows_to_examples_column_mapping_fanout_multi_rename() -> None:
    """A many-to-many mapping renames every column correctly."""
    mapping = ColumnMapping(inputs={"q1": "col_a", "q2": "col_b"}, outputs={"ans": "col_c"})
    rows = [{"col_a": "hello", "col_b": "world", "col_c": "hi"}]

    examples = rows_to_examples(rows, mapping)

    assert examples[0].q1 == "hello"
    assert examples[0].q2 == "world"
    assert examples[0].ans == "hi"


def test_rows_to_examples_missing_input_column_raises_service_error() -> None:
    """A row missing a mapped input column raises ``ServiceError``."""
    rows = [{"wrong_col": "value", "a": "2"}]

    with pytest.raises(ServiceError, match="Missing input column"):
        rows_to_examples(rows, _SIMPLE_MAPPING)


def test_rows_to_examples_missing_output_column_raises_service_error() -> None:
    """A row missing a mapped output column raises ``ServiceError``."""
    rows = [{"q": "value", "wrong_col": "2"}]

    with pytest.raises(ServiceError, match="Missing output column"):
        rows_to_examples(rows, _SIMPLE_MAPPING)


def test_rows_to_examples_non_dict_row_raises_service_error() -> None:
    """A non-mapping row raises ``ServiceError``."""
    rows = ["not-a-dict"]  # type: ignore[list-item]

    with pytest.raises(ServiceError, match="not a mapping"):
        rows_to_examples(rows, _SIMPLE_MAPPING)  # type: ignore[arg-type]


def test_rows_to_examples_empty_dataset_returns_empty_list() -> None:
    """An empty input list returns an empty examples list."""
    examples = rows_to_examples([], _SIMPLE_MAPPING)

    assert examples == []


def test_rows_to_examples_extra_columns_carried_but_not_in_inputs() -> None:
    """An extra column is readable on the example yet absent from ``inputs()``."""
    rows = [{"q": "What is 1+1?", "a": "2", "phase": "collecting"}]

    examples = rows_to_examples(rows, _SIMPLE_MAPPING, extra_columns={"phase"})

    assert examples[0].phase == "collecting"
    assert "phase" not in examples[0].inputs()


def test_rows_to_examples_extra_column_absent_from_row_is_skipped() -> None:
    """An extra column missing from a row is skipped without error."""
    examples = rows_to_examples(_SIMPLE_ROWS, _SIMPLE_MAPPING, extra_columns={"phase"})

    assert not hasattr(examples[0], "phase")


_SIG_CODE = """\
import dspy
class QA(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
"""


def test_extract_signature_fields_returns_correct_inputs() -> None:
    """All input fields are returned in the inputs slot."""
    sig_cls = load_signature_from_code(_SIG_CODE)

    inputs, _ = extract_signature_fields(sig_cls)

    assert "question" in inputs
    assert "context" in inputs


def test_extract_signature_fields_returns_correct_outputs() -> None:
    """All output fields are returned in the outputs slot."""
    sig_cls = load_signature_from_code(_SIG_CODE)

    _, outputs = extract_signature_fields(sig_cls)

    assert "answer" in outputs


def test_extract_signature_fields_returns_two_element_tuple() -> None:
    """The function returns a 2-tuple of ``(inputs, outputs)``."""
    sig_cls = load_signature_from_code(_SIG_CODE)

    result = extract_signature_fields(sig_cls)

    assert len(result) == 2


def test_extract_signature_fields_raises_when_no_inputs() -> None:
    """A signature with no inputs raises ``ServiceError``."""
    code = """\
import dspy
class NoIn(dspy.Signature):
    answer: str = dspy.OutputField()
"""
    sig_cls = load_signature_from_code(code)

    with pytest.raises(ServiceError, match="at least one input"):
        extract_signature_fields(sig_cls)


def test_extract_signature_fields_raises_when_no_outputs() -> None:
    """A signature with no outputs raises ``ServiceError``."""
    code = """\
import dspy
class NoOut(dspy.Signature):
    question: str = dspy.InputField()
"""
    sig_cls = load_signature_from_code(code)

    with pytest.raises(ServiceError, match="at least one"):
        extract_signature_fields(sig_cls)


def test_is_signature_field_returns_false_for_plain_string() -> None:
    """A plain string is not a DSPy field."""
    assert _is_signature_field("not-a-field", field_type="input") is False


def test_is_signature_field_returns_false_for_none() -> None:
    """``None`` is not a DSPy field."""
    assert _is_signature_field(None, field_type="output") is False


def test_is_signature_field_returns_false_for_plain_int() -> None:
    """An ``int`` is not a DSPy field."""
    assert _is_signature_field(42, field_type="input") is False


def test_is_signature_field_returns_true_for_dspy_input_field() -> None:
    """``dspy.InputField()`` is detected as an input field."""
    field = dspy.InputField()

    assert _is_signature_field(field, field_type="input") is True


def test_is_signature_field_returns_false_for_dspy_input_field_when_querying_output() -> None:
    """An input field is not classified as output."""
    field = dspy.InputField()

    assert _is_signature_field(field, field_type="output") is False


def test_is_signature_field_returns_true_for_dspy_output_field() -> None:
    """``dspy.OutputField()`` is detected as an output field."""
    field = dspy.OutputField()

    assert _is_signature_field(field, field_type="output") is True


def test_is_signature_field_detects_via_dspy_field_type_marker() -> None:
    """Detection via the ``__dspy_field_type`` instance attribute."""
    # Must use setattr to avoid Python's name-mangling of double-underscore names
    obj: object = object.__new__(type("_FakeField", (), {}))
    setattr(obj, "__dspy_field_type", "input")

    assert _is_signature_field(obj, field_type="input") is True


def test_is_signature_field_detects_via_json_schema_extra() -> None:
    """Detection via the ``json_schema_extra`` schema dict."""
    class _SchemaField:
        json_schema_extra = {"__dspy_field_type": "output"}

    assert _is_signature_field(_SchemaField(), field_type="output") is True


_IMAGE_SIG_CODE = """\
import dspy
class VisionQA(dspy.Signature):
    picture: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
"""


def test_image_input_field_names_returns_image_typed_inputs() -> None:
    """Inputs typed as ``dspy.Image`` are returned by ``image_input_field_names``."""
    sig = load_signature_from_code(_IMAGE_SIG_CODE)

    fields = image_input_field_names(sig)

    assert fields == {"picture"}


def test_image_input_field_names_empty_for_text_only_signature() -> None:
    """A text-only signature returns an empty image-field set."""
    sig = load_signature_from_code(_VALID_SIG)

    assert image_input_field_names(sig) == set()


def test_image_input_field_names_ignores_image_typed_outputs() -> None:
    """Only inputs typed as Image are returned — outputs are skipped."""
    sig = load_signature_from_code(_IMAGE_SIG_CODE)

    fields = image_input_field_names(sig)

    assert "answer" not in fields


def test_coerce_image_wraps_string_url() -> None:
    """A plain URL string is wrapped into a ``dspy.Image`` instance."""
    image = _coerce_image("https://example.com/cat.png")

    assert isinstance(image, dspy.Image)


def test_coerce_image_passes_through_existing_image_instance() -> None:
    """Already-wrapped dspy.Image instances are returned unchanged (no double-wrap)."""
    original = dspy.Image(url="https://example.com/dog.jpg")

    result = _coerce_image(original)

    assert result is original


def test_coerce_image_handles_data_uri() -> None:
    """A ``data:image/...`` URI is wrapped into a ``dspy.Image`` instance."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"

    image = _coerce_image(data_uri)

    assert isinstance(image, dspy.Image)


def test_rows_to_examples_wraps_declared_image_field() -> None:
    """Declared image fields are wrapped into ``dspy.Image`` during conversion."""
    mapping = ColumnMapping(
        inputs={"picture": "img", "question": "q"},
        outputs={"answer": "a"},
    )
    rows = [
        {"img": "https://example.com/cat.png", "q": "what?", "a": "cat"},
    ]

    examples = rows_to_examples(rows, mapping, image_input_fields={"picture"})

    assert isinstance(examples[0].picture, dspy.Image)


def test_rows_to_examples_does_not_wrap_non_image_inputs() -> None:
    """Non-image inputs stay as raw strings even when image fields are declared."""
    mapping = ColumnMapping(
        inputs={"picture": "img", "question": "q"},
        outputs={"answer": "a"},
    )
    rows = [
        {"img": "https://example.com/cat.png", "q": "what is it?", "a": "cat"},
    ]

    examples = rows_to_examples(rows, mapping, image_input_fields={"picture"})

    assert examples[0].question == "what is it?"
    assert not isinstance(examples[0].question, dspy.Image)


def test_rows_to_examples_no_image_fields_keeps_raw_strings() -> None:
    """Without ``image_input_fields``, image-like cells stay as raw strings."""
    mapping = ColumnMapping(inputs={"picture": "img"}, outputs={"answer": "a"})
    rows = [{"img": "https://example.com/cat.png", "a": "cat"}]

    examples = rows_to_examples(rows, mapping)

    assert examples[0].picture == "https://example.com/cat.png"
    assert not isinstance(examples[0].picture, dspy.Image)
