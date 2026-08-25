"""Tests for resolver helpers.

The DSPy-alias paths (``predict``, ``cot``, ``gepa``) hit real DSPy imports
when not mocked, so most cases here patch ``_load_callable`` via
``patch_loader`` to keep the suite fast and dependency-free.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from core.registry.resolvers import (
    AUTO_SIGNATURE_PATHS,
    DSPY_PREFIX,
    MODULE_ALIASES,
    OPTIMIZER_ALIASES,
    ResolverError,
    _load_callable,
    _match_module_alias,
    _match_optimizer_alias,
    _wrap_callable,
    resolve_module_factory,
    resolve_optimizer_factory,
)
from core.registry.tests.mocks import (
    REAL_MODULE_NAME,
    fake_dspy_module,
    patch_loader,
)
from core.service_gateway.react_compat import REACT_CLASS


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        ("predict", "predict"),
        ("PREDICT", "predict"),  # case-insensitive
        ("cot", "cot"),
        ("CoT", "cot"),
        ("react", "react"),
        ("ReAct", "react"),
        ("flex", "flex"),
        ("Flex", "flex"),
    ],
    ids=[
        "predict_lower",
        "predict_upper",
        "cot_lower",
        "cot_mixed",
        "react_lower",
        "react_mixed",
        "flex_lower",
        "flex_mixed",
    ],
)
def test_match_module_alias_known_names(name: str, expected_key: str) -> None:
    """Match module alias known names."""
    result = _match_module_alias(name)

    assert result is MODULE_ALIASES[expected_key]


def test_match_module_alias_unknown_returns_none() -> None:
    """Match module alias unknown returns none."""
    assert _match_module_alias("totally_unknown") is None


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        ("gepa", "gepa"),
        ("GEPA", "gepa"),
    ],
    ids=["gepa_lower", "gepa_upper"],
)
def test_match_optimizer_alias_known_names(name: str, expected_key: str) -> None:
    """Match optimizer alias known names."""
    result = _match_optimizer_alias(name)

    assert result == OPTIMIZER_ALIASES[expected_key]


def test_match_optimizer_alias_unknown_returns_none() -> None:
    """Match optimizer alias unknown returns none."""
    assert _match_optimizer_alias("unknown_optimizer") is None


def test_load_callable_stdlib_path_returns_callable() -> None:
    """Load callable stdlib path returns callable."""
    # os.path.join is a well-known stdlib callable
    target = _load_callable("os.path.join")

    assert callable(target)


def test_load_callable_bad_module_raises_resolver_error() -> None:
    """Load callable bad module raises resolver error."""
    with pytest.raises(ResolverError, match="Unable to import"):
        _load_callable("no_such_package_xyz.SomeClass")


def test_load_callable_missing_attribute_raises_resolver_error() -> None:
    """Load callable missing attribute raises resolver error."""
    with pytest.raises(ResolverError, match="has no attribute"):
        _load_callable("os.path.NO_SUCH_ATTRIBUTE_XYZ")


def test_load_callable_non_callable_attribute_raises_resolver_error() -> None:
    """Load callable non callable attribute raises resolver error."""
    # os.path.sep is a string constant, not callable
    with pytest.raises(ResolverError, match="not callable"):
        _load_callable("os.path.sep")


def test_wrap_callable_preserves_call_behavior() -> None:
    """Wrap callable preserves call behavior."""
    def add(a: int, b: int) -> int:
        """Return the sum of two integers (sample callable to wrap)."""
        return a + b

    wrapped = _wrap_callable(add)

    assert wrapped(1, 2) == 3


def test_wrap_callable_copies_signature() -> None:
    """Wrap callable copies signature."""
    def my_fn(x: int, y: str = "hello") -> None:
        """Sample callable whose signature should survive wrapping."""

    wrapped = _wrap_callable(my_fn)

    assert str(inspect.signature(wrapped)) == str(inspect.signature(my_fn))


def test_wrap_callable_returns_callable() -> None:
    """Wrap callable returns callable."""
    wrapped = _wrap_callable(len)

    assert callable(wrapped)


def test_resolve_module_factory_unknown_raises() -> None:
    """Resolve module factory unknown raises."""
    with pytest.raises(ResolverError, match="Unknown module"):
        resolve_module_factory("completely_unknown_module")


def test_resolve_module_factory_non_dspy_dotted_path_raises() -> None:
    """Resolve module factory non dspy dotted path raises."""
    # Has a dot but does NOT start with "dspy." so should raise
    with pytest.raises(ResolverError, match="Unknown module"):
        resolve_module_factory("some.other.Module")


def test_resolve_module_factory_dspy_prefix_with_bad_path_raises() -> None:
    """Resolve module factory dspy prefix with bad path raises."""
    with pytest.raises(ResolverError):
        resolve_module_factory("dspy.NonExistentClassXYZ9999")


def test_resolve_module_factory_returns_tuple_of_callable_and_bool() -> None:
    """Resolve module factory returns tuple of callable and bool."""
    # Patch _load_callable so we never actually import dspy
    with patch_loader(return_value=fake_dspy_module()):
        result = resolve_module_factory("dspy.SomeFakeClass")

    factory, auto_sig = result
    assert callable(factory)
    assert isinstance(auto_sig, bool)


def test_resolve_optimizer_factory_unknown_raises() -> None:
    """Resolve optimizer factory unknown raises."""
    with pytest.raises(ResolverError, match="Unknown optimizer"):
        resolve_optimizer_factory("no_such_optimizer")


def test_resolve_optimizer_factory_non_dspy_dotted_path_raises() -> None:
    """Resolve optimizer factory non dspy dotted path raises."""
    with pytest.raises(ResolverError, match="Unknown optimizer"):
        resolve_optimizer_factory("some.other.Optimizer")


def test_resolve_optimizer_factory_dspy_prefix_with_bad_path_raises() -> None:
    """Resolve optimizer factory dspy prefix with bad path raises."""
    with pytest.raises(ResolverError):
        resolve_optimizer_factory("dspy.teleprompt.NonExistentXYZ9999")


def test_resolve_optimizer_factory_returns_callable() -> None:
    """Resolve optimizer factory returns callable."""
    with patch_loader(return_value=fake_dspy_module()):
        result = resolve_optimizer_factory("dspy.SomeFakeOptimizer")

    assert callable(result)


def test_resolve_module_factory_alias_first_path_fails_second_succeeds() -> None:
    """Resolve module factory alias first path fails second succeeds."""
    # Fallback contract: aliases have multiple paths and the resolver MUST try the next when one fails.
    dummy = fake_dspy_module()
    call_log: list[str] = []

    def selective_load(path: str) -> Callable[..., Any]:
        """Fail the first alias path and resolve the rest, recording each attempt."""
        call_log.append(path)
        # "cot" has two paths: dspy.modules.ChainOfThought and dspy.ChainOfThought
        if path == "dspy.modules.ChainOfThought":
            raise ResolverError("first path unavailable")
        return dummy

    with patch_loader(side_effect=selective_load):
        factory, auto_sig = resolve_module_factory(REAL_MODULE_NAME)

    assert callable(factory)
    assert auto_sig is True
    # Both paths must have been attempted
    assert "dspy.modules.ChainOfThought" in call_log
    assert "dspy.ChainOfThought" in call_log


def test_resolve_module_factory_alias_all_paths_fail_raises_resolver_error() -> None:
    """Resolve module factory alias all paths fail raises resolver error."""
    def always_fail(path: str) -> Callable[..., Any]:
        """Reject every alias path so the resolver exhausts all of them."""
        raise ResolverError(f"cannot load {path}")

    with patch_loader(side_effect=always_fail), pytest.raises(ResolverError, match=REAL_MODULE_NAME):
        resolve_module_factory(REAL_MODULE_NAME)


def test_resolve_module_factory_alias_all_paths_fail_error_mentions_name() -> None:
    """Resolve module factory alias all paths fail error mentions name."""
    def always_fail(path: str) -> Callable[..., Any]:
        """Reject every alias path so the resolver raises mentioning the name."""
        raise ResolverError(f"nope: {path}")

    with patch_loader(side_effect=always_fail), pytest.raises(ResolverError) as exc_info:
        resolve_module_factory(REAL_MODULE_NAME)

    assert REAL_MODULE_NAME in str(exc_info.value)


def test_module_alias_paths_all_use_dspy_prefix() -> None:
    """Every ``MODULE_ALIASES`` path is a ``dspy.*`` dotted path."""
    for alias in MODULE_ALIASES.values():
        for path in alias.paths:
            assert path.startswith(DSPY_PREFIX), path


def test_optimizer_alias_paths_all_use_dspy_prefix() -> None:
    """Every ``OPTIMIZER_ALIASES`` value is a ``dspy.*`` dotted path."""
    for path in OPTIMIZER_ALIASES.values():
        assert path.startswith(DSPY_PREFIX), path


def test_resolve_module_factory_unaliased_dspy_path_in_auto_signature_set() -> None:
    """An un-aliased dspy path that matches ``AUTO_SIGNATURE_PATHS`` flips auto_signature on."""
    auto_path = next(iter(AUTO_SIGNATURE_PATHS))

    with patch_loader(return_value=fake_dspy_module()):
        _, auto_sig = resolve_module_factory(auto_path)

    assert auto_sig is True


def test_resolve_module_factory_unaliased_dspy_path_outside_auto_signature_set() -> None:
    """An un-aliased dspy path NOT in ``AUTO_SIGNATURE_PATHS`` defaults auto_signature off."""
    arbitrary_path = "dspy.SomeOtherClass"
    assert arbitrary_path not in AUTO_SIGNATURE_PATHS

    with patch_loader(return_value=fake_dspy_module()):
        _, auto_sig = resolve_module_factory(arbitrary_path)

    assert auto_sig is False


def test_match_module_alias_react_is_registered() -> None:
    """The ``react`` alias is registered, auto-signatured, and prefers ReActV2."""
    alias = _match_module_alias("react")

    assert alias is MODULE_ALIASES["react"]
    assert alias.auto_signature is True
    assert alias.paths[0] == "dspy.ReActV2"


def test_resolve_module_factory_react_resolves_to_reactv2() -> None:
    """``react`` resolves against real DSPy to the installed ReAct class.

    Unlike the mocked alias cases, this hits the real import. The alias prefers
    ``dspy.ReActV2`` (DSPy 3.3+) and falls back to classic ``dspy.ReAct``
    (DSPy 3.2.x); either way it lands on ``REACT_CLASS`` with auto_signature on.
    """
    factory, auto_sig = resolve_module_factory("react")

    assert auto_sig is True
    assert factory.func is REACT_CLASS
