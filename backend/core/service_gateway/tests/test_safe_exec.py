"""Subprocess-isolation smoke tests for ``safe_exec``.

These tests spawn real subprocesses — no mocks, no fakes. The whole point
is to verify the boundary actually contains user-authored exec, including
its failure modes (syntax errors, timeouts, wrong return shapes).
"""

from __future__ import annotations

import pytest

from core.exceptions import ServiceError
from core.service_gateway import safe_exec
from core.service_gateway.safe_exec import (
    MetricIntrospection,
    MetricProbeResult,
    SignatureIntrospection,
    probe_metric_on_sample,
    validate_metric_code,
    validate_signature_code,
)

_VALID_SIG = """\
import dspy
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
"""

_IMAGE_SIG = """\
import dspy
class VisionQA(dspy.Signature):
    picture: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
"""

_VALID_NUMERIC_METRIC = "def metric(example, prediction, trace=None):\n    return 1.0\n"

_VALID_PREDICTION_METRIC = (
    "import dspy\n"
    "def metric(example, prediction, trace=None,"
    " pred_name=None, pred_trace=None):\n"
    "    return dspy.Prediction(score=0.5, feedback='ok')\n"
)

# Metric that asserts the picture cell really is a dspy.Image at metric time.
# Returns 1.0 only when the runtime type is Image; on a string fallback the
# isinstance check fails and the whole probe surfaces an error — exactly the
# regression we want to catch.
_IMAGE_AWARE_METRIC = (
    "import dspy\n"
    "def metric(example, prediction, trace=None):\n"
    "    assert isinstance(example.picture, dspy.Image), type(example.picture)\n"
    "    return 1.0\n"
)


class TestValidateSignatureCode:
    """Tests for ``validate_signature_code``."""

    def test_returns_fields_for_valid_signature(self) -> None:
        """A valid signature returns a ``SignatureIntrospection`` with input/output names."""
        intro = validate_signature_code(_VALID_SIG)

        assert isinstance(intro, SignatureIntrospection)
        assert intro.class_name == "QA"
        assert intro.input_fields == ["question"]
        assert intro.output_fields == ["answer"]
        assert intro.image_input_fields == []

    def test_image_input_fields_surfaced_for_dspy_image_annotation(self) -> None:
        """``dspy.Image``-annotated inputs appear in ``image_input_fields``."""
        intro = validate_signature_code(_IMAGE_SIG)

        assert intro.image_input_fields == ["picture"]
        assert "picture" in intro.input_fields
        assert "question" in intro.input_fields

    def test_syntax_error_surfaces_as_service_error(self) -> None:
        """A syntactically invalid signature raises ``ServiceError``."""
        with pytest.raises(ServiceError, match="syntax error"):
            validate_signature_code("def !!! invalid python")

    def test_no_signature_class_surfaces_as_service_error(self) -> None:
        """Source without a ``dspy.Signature`` subclass raises ``ServiceError``."""
        with pytest.raises(ServiceError, match=r"dspy\.Signature"):
            validate_signature_code("x = 1")

    def test_infinite_loop_is_terminated(self) -> None:
        """A user infinite loop is terminated by the subprocess timeout."""
        # Regression: subprocess must enforce timeout on user code, otherwise
        # a `while True` in module-level scope hangs the validator forever.
        with pytest.raises(ServiceError, match="timeout"):
            validate_signature_code("while True: pass", timeout_seconds=2.0)


class TestValidateMetricCode:
    """Tests for ``validate_metric_code``."""

    def test_returns_param_names_for_valid_metric(self) -> None:
        """A valid metric returns ``MetricIntrospection`` with the parameter names."""
        info = validate_metric_code(_VALID_PREDICTION_METRIC)

        assert isinstance(info, MetricIntrospection)
        assert info.callable_name == "metric"
        assert info.param_names == [
            "example",
            "prediction",
            "trace",
            "pred_name",
            "pred_trace",
        ]

    def test_missing_callable_surfaces_as_service_error(self) -> None:
        """Source without any callable raises ``ServiceError``."""
        with pytest.raises(ServiceError, match="metric"):
            validate_metric_code("x = 1")

    def test_syntax_error_surfaces_as_service_error(self) -> None:
        """A syntactically invalid metric raises ``ServiceError``."""
        with pytest.raises(ServiceError, match="syntax error"):
            validate_metric_code("def !!!")


class TestProbeMetricOnSample:
    """Tests for ``probe_metric_on_sample``."""

    def test_numeric_return_is_reported(self) -> None:
        """A numeric metric return is reported as ``result_kind='numeric'``."""
        probe = probe_metric_on_sample(
            metric_code=_VALID_NUMERIC_METRIC,
            example_payload={"question": "q", "answer": "a"},
            prediction_payload={"question": "q", "answer": "a"},
            input_field_names=["question"],
        )

        assert isinstance(probe, MetricProbeResult)
        assert probe.result_kind == "numeric"
        assert probe.error is None
        assert probe.result_type_name == "float"

    def test_dspy_prediction_return_is_reported(self) -> None:
        """A ``dspy.Prediction`` return is reported as ``result_kind='prediction'``."""
        probe = probe_metric_on_sample(
            metric_code=_VALID_PREDICTION_METRIC,
            example_payload={"question": "q", "answer": "a"},
            prediction_payload={"question": "q", "answer": "a"},
            input_field_names=["question"],
        )

        assert probe.result_kind == "prediction"
        assert probe.has_score_attr is True
        assert probe.error is None

    def test_metric_exception_is_caught(self) -> None:
        """A user-raised exception inside the metric is caught and surfaced as ``error``."""
        probe = probe_metric_on_sample(
            metric_code=("def metric(example, prediction, trace=None):\n    raise RuntimeError('kaboom')\n"),
            example_payload={"question": "q", "answer": "a"},
            prediction_payload={"question": "q", "answer": "a"},
            input_field_names=["question"],
        )

        assert probe.result_kind == "error"
        assert probe.error is not None
        assert "kaboom" in probe.error

    def test_broken_metric_code_surfaces_as_service_error(self) -> None:
        """A syntactically broken metric raises ``ServiceError`` from the probe entry-point."""
        with pytest.raises(ServiceError, match="syntax error"):
            probe_metric_on_sample(
                metric_code="def !!!",
                example_payload={"question": "q", "answer": "a"},
                prediction_payload={"question": "q", "answer": "a"},
                input_field_names=["question"],
            )

    def test_image_field_value_is_wrapped_into_dspy_image(self) -> None:
        """Image input cells are wrapped into ``dspy.Image`` inside the probe subprocess."""
        probe = probe_metric_on_sample(
            metric_code=_IMAGE_AWARE_METRIC,
            example_payload={
                "picture": "https://example.com/cat.png",
                "question": "what?",
                "answer": "cat",
            },
            prediction_payload={"answer": "cat"},
            input_field_names=["picture", "question"],
            image_input_fields=["picture"],
        )

        # The metric's isinstance(example.picture, dspy.Image) assertion would
        # fail with AssertionError if the wrap didn't happen — surfaced via
        # ``probe.error``. A clean numeric return proves it did.
        assert probe.error is None
        assert probe.result_kind == "numeric"

    def test_image_field_unwrapped_when_no_image_fields_declared(self) -> None:
        """Without ``image_input_fields`` the cell stays a string and the assertion fails."""
        # Without image_input_fields, the cell stays a plain string — the metric's
        # isinstance(..., dspy.Image) assertion is expected to fail and surface via probe.error.
        probe = probe_metric_on_sample(
            metric_code=_IMAGE_AWARE_METRIC,
            example_payload={
                "picture": "https://example.com/cat.png",
                "question": "what?",
                "answer": "cat",
            },
            prediction_payload={"answer": "cat"},
            input_field_names=["picture", "question"],
            # image_input_fields not passed — defaults to None
        )

        assert probe.result_kind == "error"
        assert probe.error is not None

    def test_image_input_fields_none_is_equivalent_to_empty(self) -> None:
        """Passing ``image_input_fields=None`` behaves the same as an empty list."""
        probe = probe_metric_on_sample(
            metric_code=_VALID_NUMERIC_METRIC,
            example_payload={"question": "q", "answer": "a"},
            prediction_payload={"question": "q", "answer": "a"},
            input_field_names=["question"],
            image_input_fields=None,
        )

        assert probe.result_kind == "numeric"
        assert probe.error is None


class TestBoundedCachePut:
    """The validation memo dicts stay bounded in the long-lived API process."""

    def test_evicts_oldest_at_cap(self) -> None:
        """Insertion past the cap drops the oldest key, never the newest."""
        cache: dict[str, int] = {}
        for i in range(safe_exec._VALIDATION_CACHE_MAX_ENTRIES + 10):
            safe_exec._bounded_cache_put(cache, f"code-{i}", i)

        assert len(cache) == safe_exec._VALIDATION_CACHE_MAX_ENTRIES
        assert "code-0" not in cache
        assert f"code-{safe_exec._VALIDATION_CACHE_MAX_ENTRIES + 9}" in cache

    def test_rewriting_existing_key_does_not_evict(self) -> None:
        """Updating a present key at the cap must not shrink the cache."""
        cache: dict[str, int] = {}
        for i in range(safe_exec._VALIDATION_CACHE_MAX_ENTRIES):
            safe_exec._bounded_cache_put(cache, f"code-{i}", i)
        safe_exec._bounded_cache_put(cache, "code-0", 999)

        assert len(cache) == safe_exec._VALIDATION_CACHE_MAX_ENTRIES
        assert cache["code-0"] == 999
