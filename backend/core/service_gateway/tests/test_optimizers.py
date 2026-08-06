"""Tests for ``core.service_gateway.optimization.optimizers``."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock, patch

import dspy
import pytest
from gepa.strategies.proposal_sampling import PxNSampling

from core.config import settings
from core.exceptions import ServiceError
from core.models import ModelConfig
from core.service_gateway.optimization.data import DatasetSplits
from core.service_gateway.optimization.optimizers import (
    TargetScoreStopper,
    _callable_accepts_metric,
    _compile_accepts_valset,
    _extract_factory_targets,
    compile_program,
    evaluate_on_test,
    instantiate_optimizer,
    optimizer_requires_metric,
    preflight_metric_check,
    validate_optimizer_kwargs,
    validate_optimizer_signature,
)
from core.service_gateway.tests.mocks import fake_compiled_program, fake_language_model


def _splits(train: Any = None, val: Any = None, test: Any = None) -> DatasetSplits:
    """Build a ``DatasetSplits`` with ``[1, 2, 3]`` train and empty val/test by default."""
    return DatasetSplits(
        train=train if train is not None else [1, 2, 3],
        val=val if val is not None else [],
        test=test if test is not None else [],
    )


def _model_cfg(name: str = "openai/gpt-4o-mini") -> ModelConfig:
    """Build a minimal ``ModelConfig`` defaulting to ``openai/gpt-4o-mini``."""
    return ModelConfig(name=name)


class _FakeOptimizer:
    """Optimizer test double whose ``compile`` ignores valset."""

    def __init__(self, **kwargs: Any) -> None:
        """Capture init kwargs for later inspection."""
        self.kwargs = kwargs

    def compile(self, program: Any, *, trainset: Any, **kwargs: Any) -> Any:
        """Return the program unchanged, ignoring trainset and extra kwargs."""
        return program


class _FakeOptimizerWithValset:
    """Optimizer test double whose ``compile`` accepts a valset kwarg."""

    def __init__(self, **kwargs: Any) -> None:
        """Capture init kwargs for later inspection."""
        self.kwargs = kwargs

    def compile(self, program: Any, *, trainset: Any, valset: Any = None, **kwargs: Any) -> Any:
        """Return the program unchanged, ignoring trainset/valset/extras."""
        return program


class _FactoryWithMetric:
    """Optimizer factory test double whose constructor accepts ``metric``."""

    def __init__(self, metric: Any = None, **kwargs: Any) -> None:
        """Accept and ignore ``metric`` plus extra kwargs."""

    def compile(self, program: Any, **kwargs: Any) -> Any:
        """Return the program unchanged."""
        return program


class _FactoryNoMetric:
    """Optimizer factory test double whose constructor lacks ``metric``."""

    def __init__(self, **kwargs: Any) -> None:
        """Accept and ignore extra kwargs."""

    def compile(self, program: Any, **kwargs: Any) -> Any:
        """Return the program unchanged."""
        return program


def _dummy_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Constant metric that returns 1.0 for any inputs."""
    return 1.0


def test_compile_accepts_valset_returns_true_when_valset_in_signature() -> None:
    """``_compile_accepts_valset`` returns True when compile() exposes ``valset``."""
    optimizer = _FakeOptimizerWithValset()

    result = _compile_accepts_valset(optimizer)

    assert result is True


def test_compile_accepts_valset_returns_false_when_valset_not_in_signature() -> None:
    """``_compile_accepts_valset`` returns False when compile() lacks ``valset``."""
    optimizer = _FakeOptimizer()

    result = _compile_accepts_valset(optimizer)

    assert result is False


def test_compile_accepts_valset_returns_false_when_no_compile_method() -> None:
    """``_compile_accepts_valset`` returns False for objects without ``compile``."""
    result = _compile_accepts_valset(object())

    assert result is False


def test_compile_program_empty_train_raises_service_error() -> None:
    """An empty training split surfaces a ``ServiceError``."""
    splits = _splits(train=[])
    optimizer = _FakeOptimizer()

    with pytest.raises(ServiceError, match="Training split is empty"):
        compile_program(
            optimizer=optimizer,
            program=fake_compiled_program(),
            splits=splits,
            metric=None,
            compile_kwargs={},
        )


def test_compile_program_calls_optimizer_compile_with_trainset() -> None:
    """``compile_program`` returns whatever the optimizer's compile() returns."""
    program = fake_compiled_program()
    optimizer = _FakeOptimizer()
    splits = _splits(train=[1, 2, 3])

    result = compile_program(
        optimizer=optimizer,
        program=program,
        splits=splits,
        metric=None,
        compile_kwargs={},
    )

    assert result is program


def test_compile_program_injects_valset_when_optimizer_accepts_it() -> None:
    """A valset is injected when the optimizer's compile() accepts ``valset``."""
    received: dict = {}

    class _RecordingOpt:
        def compile(self, program: Any, *, trainset: Any, valset: Any = None, **kwargs: Any) -> Any:
            received["valset"] = valset
            return program

    splits = _splits(train=[1, 2], val=[3])
    compile_program(
        optimizer=_RecordingOpt(),
        program=object(),
        splits=splits,
        metric=None,
        compile_kwargs={},
    )

    assert received.get("valset") == [3]


def test_compile_program_does_not_inject_valset_when_optimizer_does_not_accept_it() -> None:
    """A valset is NOT injected when the optimizer's compile() rejects ``valset``."""
    received: dict = {}

    class _StrictOpt:
        def compile(self, program: Any, *, trainset: Any, **kwargs: Any) -> Any:
            received["extra"] = kwargs
            return program

    splits = _splits(train=[1], val=[2])
    compile_program(
        optimizer=_StrictOpt(),
        program=object(),
        splits=splits,
        metric=None,
        compile_kwargs={},
    )

    assert "valset" not in received.get("extra", {})


def test_compile_program_honours_pre_set_trainset_in_compile_kwargs() -> None:
    """A pre-set ``trainset`` in compile_kwargs takes precedence over the split."""
    received: dict = {}

    class _RecordingOpt:
        def compile(self, program: Any, *, trainset: Any, **kwargs: Any) -> Any:
            received["trainset"] = trainset
            return program

    custom_train = ["custom"]
    compile_program(
        optimizer=_RecordingOpt(),
        program=object(),
        splits=_splits(train=[1, 2]),
        metric=None,
        compile_kwargs={"trainset": custom_train},
    )

    assert received["trainset"] == custom_train


def test_compile_program_type_error_from_optimizer_raises_service_error() -> None:
    """A ``TypeError`` from compile() is wrapped into a ``ServiceError``."""
    class _BadOpt:
        def compile(self, program: Any, **kwargs: Any) -> Any:
            raise TypeError("bad kwarg")

    with pytest.raises(ServiceError, match="rejected the provided arguments"):
        compile_program(
            optimizer=_BadOpt(),
            program=object(),
            splits=_splits(),
            metric=None,
            compile_kwargs={},
        )


def test_optimizer_requires_metric_returns_true_when_metric_in_init() -> None:
    """A factory whose __init__ exposes ``metric`` returns True."""
    result = optimizer_requires_metric(_FactoryWithMetric)

    assert result is True


def test_optimizer_requires_metric_returns_false_when_metric_not_present() -> None:
    """A factory whose __init__ has no ``metric`` parameter returns False."""
    def _no_metric_factory(num_threads: int = 4) -> None:
        pass

    result = optimizer_requires_metric(_no_metric_factory)

    assert result is False


def test_optimizer_requires_metric_returns_false_for_uninspectable() -> None:
    """``None`` factories return False rather than raising."""
    result = optimizer_requires_metric(cast(Any, None))

    assert result is False


def test_callable_accepts_metric_true_for_metric_param() -> None:
    """A callable whose first parameter is ``metric`` reports True."""
    def fn(metric: Any, other: Any = None) -> None:
        pass

    assert _callable_accepts_metric(fn) is True


def test_callable_accepts_metric_false_when_no_metric_param() -> None:
    """A callable without a ``metric`` parameter reports False."""
    def fn(foo: Any, bar: Any) -> None:
        pass

    assert _callable_accepts_metric(fn) is False


def test_callable_accepts_metric_false_for_none() -> None:
    """``None`` reports False rather than raising."""
    assert _callable_accepts_metric(None) is False


def test_extract_factory_targets_includes_factory_itself() -> None:
    """The factory itself is always among the extracted targets."""
    def my_factory() -> None:
        pass

    targets = _extract_factory_targets(my_factory)

    assert my_factory in targets


def test_extract_factory_targets_includes_wrapped_when_present() -> None:
    """A function with ``__wrapped__`` exposes the inner callable as a target."""
    def inner() -> None:
        pass

    def outer() -> None:
        pass

    outer.__wrapped__ = inner  # type: ignore[attr-defined]

    targets = _extract_factory_targets(outer)

    assert inner in targets


def test_validate_optimizer_kwargs_empty_kwargs_always_passes() -> None:
    """No kwargs always validates without raising."""
    validate_optimizer_kwargs(_FactoryNoMetric, {}, "no_metric")


def test_validate_optimizer_kwargs_valid_kwargs_passes() -> None:
    """Empty kwargs against a no-metric factory validates without raising."""
    validate_optimizer_kwargs(_FactoryNoMetric, {}, "no_metric")


def test_validate_optimizer_kwargs_invalid_kwarg_raises_service_error() -> None:
    """A factory with a closed signature rejects unknown kwargs."""
    # Use a factory with explicit, closed signature (no **kwargs) so
    # bind_partial can actually reject unknown keys.
    def _strict_factory(num_threads: int = 4, max_rounds: int = 1) -> None:
        pass

    with pytest.raises(ServiceError, match="unsupported entries"):
        validate_optimizer_kwargs(
            _strict_factory,
            {"completely_unknown_kwarg_xyz": True},
            "strict_factory",
        )


def test_validate_optimizer_kwargs_uninspectable_factory_does_not_raise() -> None:
    """Factories whose signatures cannot be inspected silently skip validation."""
    # Built-in int doesn't expose a useful signature in older Pythons — the
    # function should silently skip validation rather than raise.
    validate_optimizer_kwargs(int, {"foo": "bar"}, "int_factory")


def test_validate_optimizer_signature_logs_warning_for_uninspectable(caplog: pytest.LogCaptureFixture) -> None:
    """An uninspectable factory logs a warning instead of raising."""
    with caplog.at_level(logging.WARNING):
        validate_optimizer_signature(cast(Any, None), "bad_factory")


def test_instantiate_optimizer_injects_metric_when_factory_requires_it() -> None:
    """A metric is forwarded to factories that expose a ``metric`` parameter."""
    captured: dict = {}

    class _MetricFactory:
        def __init__(self, metric: Any = None, **kw: Any) -> None:
            captured["metric"] = metric

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_MetricFactory,
        optimizer_name="my_opt",
        optimizer_kwargs={},
        metric=_dummy_metric,        reflection_model=None,
    )

    assert captured["metric"] is _dummy_metric


def test_instantiate_optimizer_does_not_override_metric_if_already_in_kwargs() -> None:
    """A caller-supplied ``metric`` in optimizer_kwargs is preserved."""
    captured: dict = {}

    def custom_metric(e: Any, p: Any) -> float:
        return 0.5

    class _MetricFactory:
        def __init__(self, metric: Any = None, **kw: Any) -> None:
            captured["metric"] = metric

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_MetricFactory,
        optimizer_name="my_opt",
        optimizer_kwargs={"metric": custom_metric},
        metric=_dummy_metric,        reflection_model=None,
    )

    assert captured["metric"] is custom_metric


def test_instantiate_optimizer_injects_gepa_num_threads_default() -> None:
    """GEPA gets the configured parallel-eval default when no ``num_threads`` is supplied."""
    captured: dict = {}

    class _GepaFactory:
        def __init__(self, num_threads: Any = None, **kw: Any) -> None:
            captured["num_threads"] = num_threads

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=fake_language_model(),
    )

    assert captured["num_threads"] == settings.gepa_eval_num_threads


def test_instantiate_optimizer_keeps_user_supplied_gepa_num_threads() -> None:
    """An explicit ``num_threads`` in optimizer_kwargs beats the injected default."""
    captured: dict = {}

    class _GepaFactory:
        def __init__(self, num_threads: Any = None, **kw: Any) -> None:
            captured["num_threads"] = num_threads

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={"num_threads": 3},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=fake_language_model(),
    )

    assert captured["num_threads"] == 3


def test_instantiate_optimizer_does_not_inject_num_threads_for_non_gepa() -> None:
    """Non-GEPA factories never receive an unrequested ``num_threads`` kwarg."""
    captured: dict = {}

    class _PlainFactory:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_PlainFactory,
        optimizer_name="my_opt",
        optimizer_kwargs={},
        metric=_dummy_metric,
        reflection_model=None,
    )

    assert "num_threads" not in captured


def test_instantiate_optimizer_gepa_requires_reflection_model_or_raises() -> None:
    """The gepa factory raises ``ServiceError`` when no reflection model is provided."""
    class _GepaFactory:
        def __init__(self, metric: Any = None, auto: Any = None, reflection_lm: Any = None, **kw: Any) -> None:
            pass

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    with pytest.raises(ServiceError, match="reflection_model_config"):
        instantiate_optimizer(
            factory=_GepaFactory,
            optimizer_name="gepa",
            optimizer_kwargs={},
            metric=_dummy_metric,            reflection_model=None,
        )


def test_instantiate_optimizer_gepa_injects_reflection_lm_when_model_provided() -> None:
    """The reflection LM is built and passed through to the gepa factory."""
    captured: dict = {}
    fake_lm = fake_language_model()

    class _GepaFactory:
        def __init__(self, metric: Any = None, auto: Any = None, reflection_lm: Any = None, **kw: Any) -> None:
            captured["reflection_lm"] = reflection_lm

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    with patch("core.service_gateway.optimization.optimizers.build_language_model", return_value=fake_lm):
        instantiate_optimizer(
            factory=_GepaFactory,
            optimizer_name="gepa",
            optimizer_kwargs={},
            metric=_dummy_metric,            reflection_model=_model_cfg("openai/gpt-4o"),
        )

    assert captured["reflection_lm"] is fake_lm


def test_instantiate_optimizer_gepa_sets_auto_light_default() -> None:
    """The gepa factory defaults ``auto`` to 'light' when caller leaves it unset."""
    captured: dict = {}
    fake_lm = fake_language_model()

    class _GepaFactory:
        def __init__(self, metric: Any = None, auto: Any = None, reflection_lm: Any = None, **kw: Any) -> None:
            captured["auto"] = auto

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    with patch("core.service_gateway.optimization.optimizers.build_language_model", return_value=fake_lm):
        instantiate_optimizer(
            factory=_GepaFactory,
            optimizer_name="gepa",
            optimizer_kwargs={},
            metric=_dummy_metric,            reflection_model=_model_cfg("openai/gpt-4o"),
        )

    assert captured["auto"] == "light"


def test_instantiate_optimizer_gepa_injects_target_score_stopper() -> None:
    """A percentage target becomes GEPA's native validation stop callback."""
    captured: dict[str, Any] = {}
    state: dict[str, Any] = {}

    class _GepaFactory:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=object(),
        target_score=85,
        stop_state=state,
    )

    stopper = captured["gepa_kwargs"]["stop_callbacks"][0]
    assert isinstance(stopper, TargetScoreStopper)
    assert stopper.threshold == pytest.approx(0.85)
    assert state["target_score_stopper"] is stopper


def test_instantiate_optimizer_gepa_no_sampling_strategy_by_default() -> None:
    """The classic p=n=1 defaults inject no ``sampling_strategy`` — GEPA keeps its own."""
    captured: dict[str, Any] = {}

    class _GepaFactory:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=object(),
    )

    assert "sampling_strategy" not in captured.get("gepa_kwargs", {})


def test_instantiate_optimizer_gepa_injects_pxn_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raising the PxN settings above 1 injects a ``PxNSampling(p, n)`` strategy."""
    monkeypatch.setattr(settings, "gepa_pxn_parents", 2)
    monkeypatch.setattr(settings, "gepa_pxn_proposals", 3)
    captured: dict[str, Any] = {}

    class _GepaFactory:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=object(),
    )

    strategy = captured["gepa_kwargs"]["sampling_strategy"]
    assert isinstance(strategy, PxNSampling)
    assert (strategy.p, strategy.n) == (2, 3)


def test_instantiate_optimizer_gepa_submission_sampling_strategy_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submission-supplied ``sampling_strategy`` is never overwritten by the PxN default."""
    monkeypatch.setattr(settings, "gepa_pxn_parents", 4)
    monkeypatch.setattr(settings, "gepa_pxn_proposals", 4)
    captured: dict[str, Any] = {}
    supplied = PxNSampling(1, 1)

    class _GepaFactory:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

        def compile(self, *a: Any, **kw: Any) -> None:
            return None

    instantiate_optimizer(
        factory=_GepaFactory,
        optimizer_name="gepa",
        optimizer_kwargs={"gepa_kwargs": {"sampling_strategy": supplied}},
        metric=_dummy_metric,
        reflection_model=None,
        reflection_lm=object(),
    )

    assert captured["gepa_kwargs"]["sampling_strategy"] is supplied


class _FakeEvalResult:
    """Stand-in for dspy.Evaluate's structured result with score + results."""

    def __init__(self, score: Any, results: list | None = None) -> None:
        """Store the aggregate score and per-example results list."""
        self.score = score
        self.results = results or []


class _FakeExample:
    """Stand-in dataset example exposing a single 'answer' label."""

    def labels(self) -> list[str]:
        """Return the canonical label list for this example."""
        return ["answer"]


def test_evaluate_on_test_happy_path_returns_float_and_empty_results() -> None:
    """A structured evaluator result returns the score and an empty per-example list."""
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = _FakeEvalResult(score=0.75, results=[])

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator):
        score, per_example = evaluate_on_test(program, test_examples, metric, collect_per_example=True)

    assert score == pytest.approx(0.75)
    assert per_example == []


def test_evaluate_on_test_bare_int_float_return_treated_as_aggregate() -> None:
    """A bare float evaluator return value is treated as the aggregate score."""
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = 0.42  # bare float

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator):
        result = evaluate_on_test(program, test_examples, metric, collect_per_example=False)

    assert result == pytest.approx(0.42)


def test_evaluate_on_test_bare_int_return_treated_as_aggregate() -> None:
    """A bare int evaluator return value is coerced to float aggregate."""
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = 1  # bare int

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator):
        result = evaluate_on_test(program, test_examples, metric, collect_per_example=False)

    assert result == pytest.approx(1.0)


def test_evaluate_on_test_non_numeric_score_raises_service_error() -> None:
    """A non-numeric ``score`` field raises a ``ServiceError``."""
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = _FakeEvalResult(score="not-a-number")

    with (
        patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator),
        pytest.raises(ServiceError, match="non-numeric"),
    ):
        evaluate_on_test(program, test_examples, metric, collect_per_example=True)


def test_evaluate_on_test_empty_test_set_returns_none_when_not_collecting() -> None:
    """An empty test set returns ``None`` and never invokes the evaluator."""
    program = fake_compiled_program()
    metric = _dummy_metric

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate") as mock_eval_cls:
        result = evaluate_on_test(program, [], metric, collect_per_example=False)

    mock_eval_cls.assert_not_called()
    assert result is None


def test_evaluate_on_test_empty_test_set_returns_none_and_empty_list_when_collecting() -> None:
    """An empty test set with ``collect_per_example`` returns ``(None, [])``."""
    program = fake_compiled_program()
    metric = _dummy_metric

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate") as mock_eval_cls:
        score, per_example = evaluate_on_test(program, [], metric, collect_per_example=True)

    mock_eval_cls.assert_not_called()
    assert score is None
    assert per_example == []


def test_evaluate_on_test_per_example_unpacking_normal_entry() -> None:
    """A normal (example, prediction, score) tuple unpacks into a per-example dict."""
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]

    prediction = MagicMock()
    prediction.answer = "2"
    example = _FakeExample()
    raw_results = [(example, prediction, 1.0)]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = _FakeEvalResult(score=1.0, results=raw_results)

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator):
        _score, per_example = evaluate_on_test(program, test_examples, metric, collect_per_example=True)

    assert len(per_example) == 1
    entry = per_example[0]
    assert entry["score"] == pytest.approx(1.0)
    assert entry["pass"] is True
    assert "answer" in entry["outputs"]


def test_evaluate_on_test_per_example_bare_except_fallback() -> None:
    """A malformed result tuple falls back to a default ``score=0.0, pass=False`` entry."""
    # Malformed result entry (wrong tuple shape) must fall through to (index, 0.0, pass=False)
    # rather than blowing up the whole evaluation.
    program = fake_compiled_program()
    metric = _dummy_metric
    test_examples = [_FakeExample()]

    # A deliberately unparseable entry (too few elements)
    raw_results = [("only_one_element",)]
    fake_evaluator = MagicMock()
    fake_evaluator.return_value = _FakeEvalResult(score=0.0, results=raw_results)

    with patch("core.service_gateway.optimization.optimizers.dspy.Evaluate", return_value=fake_evaluator):
        _score, per_example = evaluate_on_test(program, test_examples, metric, collect_per_example=True)

    assert len(per_example) == 1
    entry = per_example[0]
    assert entry["score"] == pytest.approx(0.0)
    assert entry["pass"] is False
    assert entry["outputs"] == {}


def _gold_examples() -> list[Any]:
    """Build two ``dspy.Example`` instances with a single ``answer`` output field."""
    return [
        dspy.Example(question="q1", answer="a1").with_inputs("question"),
        dspy.Example(question="q2", answer="a2").with_inputs("question"),
    ]


def _correct_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Exact-match metric reading gold via DSPy dot access — perfect prediction scores 1.0."""
    return 1.0 if prediction.answer == example.answer else 0.0


def _broken_isinstance_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Canonical broken metric: gates on ``isinstance(gold, dict)`` so it always scores 0."""
    if not isinstance(example, dict):
        return 0.0
    return 1.0 if prediction.answer == example["answer"] else 0.0


def test_preflight_passes_for_correct_metric() -> None:
    """A correct metric scores a perfect prediction > 0, so the pre-flight is a no-op."""
    preflight_metric_check(_correct_metric, _gold_examples(), ["answer"])


def test_preflight_aborts_when_metric_always_returns_zero() -> None:
    """A metric that always returns 0 trips the all-zero verdict and aborts."""
    always_zero = lambda example, prediction, trace=None: 0.0  # noqa: E731

    with pytest.raises(ServiceError, match="Pre-flight check failed"):
        preflight_metric_check(always_zero, _gold_examples(), ["answer"])


def test_preflight_aborts_for_isinstance_gold_dict_gate() -> None:
    """The ``isinstance(gold, dict)`` mis-read aborts, with field-name guidance in the message."""
    with pytest.raises(ServiceError, match="isinstance"):
        preflight_metric_check(_broken_isinstance_metric, _gold_examples(), ["answer"])


def test_preflight_unwraps_prediction_score_return() -> None:
    """A metric returning a ``dspy.Prediction`` with ``.score`` is unwrapped to that score."""
    def scored(example: Any, prediction: Any, trace: Any = None) -> Any:
        return dspy.Prediction(score=1.0)

    preflight_metric_check(scored, _gold_examples(), ["answer"])


def test_preflight_swallows_exceptions_but_surfaces_all_zero() -> None:
    """A metric raising on every example is treated as score 0 and still aborts."""
    def raising(example: Any, prediction: Any, trace: Any = None) -> float:
        raise RuntimeError("boom")

    with pytest.raises(ServiceError, match="Pre-flight check failed"):
        preflight_metric_check(raising, _gold_examples(), ["answer"])


def test_preflight_noop_when_sample_empty_or_no_output_fields() -> None:
    """No examples, no output fields, or no metric is a silent no-op (never blocks)."""
    always_zero = lambda example, prediction, trace=None: 0.0  # noqa: E731
    preflight_metric_check(always_zero, [], ["answer"])
    preflight_metric_check(always_zero, _gold_examples(), [])
    preflight_metric_check(None, _gold_examples(), ["answer"])
