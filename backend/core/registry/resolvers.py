"""Resolve DSPy module and optimizer factories from short aliases or dotted paths.

Used by the service gateway to translate user-facing names (``predict``,
``cot``, ``gepa``, etc.) and arbitrary ``dspy.*`` paths into callables
that build DSPy modules and optimizers at runtime.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial, update_wrapper
from types import MappingProxyType
from typing import Any

from ..constants import RESOLUTION_HINT

DSPY_PREFIX = "dspy."


@dataclass(frozen=True)
class ModuleAlias:
    """Describe how to resolve a module alias."""

    paths: tuple[str, ...]
    auto_signature: bool = False


MODULE_ALIASES: Mapping[str, ModuleAlias] = MappingProxyType(
    {
        "predict": ModuleAlias(("dspy.Predict",), auto_signature=True),
        "cot": ModuleAlias(
            (
                "dspy.modules.ChainOfThought",
                "dspy.ChainOfThought",
            ),
            auto_signature=True,
        ),
        "react": ModuleAlias(
            (
                "dspy.ReActV2",
                "dspy.ReAct",
            ),
            auto_signature=True,
        ),
        # Flex is dspy 3.3+ only; on the stable line ``dspy.Flex`` is absent
        # and this alias resolves to a ResolverError at submission time.
        "flex": ModuleAlias(("dspy.Flex",), auto_signature=True),
    }
)

OPTIMIZER_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "gepa": "dspy.teleprompt.GEPA",
    }
)

AUTO_SIGNATURE_PATHS = frozenset(
    path for alias in MODULE_ALIASES.values() if alias.auto_signature for path in alias.paths
)


class ResolverError(RuntimeError):
    """Raised when a requested DSPy asset cannot be resolved."""


def _match_module_alias(name: str) -> ModuleAlias | None:
    """Look up a module alias by case-insensitive name.

    Args:
        name: Alias to match.

    Returns:
        The matching ``ModuleAlias`` or ``None`` if no alias is registered.
    """
    return MODULE_ALIASES.get(name.lower())


def _match_optimizer_alias(name: str) -> str | None:
    """Look up an optimizer alias's dotted target by case-insensitive name.

    Args:
        name: Alias to match.

    Returns:
        The dotted import path or ``None`` if no alias is registered.
    """
    return OPTIMIZER_ALIASES.get(name.lower())


def resolve_module_factory(name: str) -> tuple[Callable[..., Any], bool]:
    """Resolve a module factory from aliases or dotted paths.

    Args:
        name: Alias (e.g. ``cot``) or fully qualified ``dspy.*`` path.

    Returns:
        ``(factory, auto_signature)`` where ``auto_signature`` is ``True``
        when the service gateway should auto-inject a compiled DSPy
        signature rather than requiring one in ``module_kwargs``.

    Raises:
        ResolverError: When ``name`` cannot be resolved to a callable.
    """

    spec = _match_module_alias(name)
    if spec is not None:
        last_error: Exception | None = None
        for path in spec.paths:
            try:
                target = _load_callable(path)
                return _wrap_callable(target), spec.auto_signature
            except ResolverError as exc:
                last_error = exc
                continue
        raise ResolverError(f"Unknown module '{name}'. {RESOLUTION_HINT}") from last_error
    if name.startswith(DSPY_PREFIX):
        target = _load_callable(name)
        auto_signature = name in AUTO_SIGNATURE_PATHS
        return _wrap_callable(target), auto_signature
    raise ResolverError(f"Unknown module '{name}'. {RESOLUTION_HINT}")


def resolve_optimizer_factory(name: str) -> Callable[..., Any]:
    """Resolve an optimizer factory from aliases or dotted paths.

    Args:
        name: Alias (e.g. ``gepa``) or fully qualified ``dspy.*`` path.

    Returns:
        A wrapped callable that forwards arguments to the resolved factory.

    Raises:
        ResolverError: When ``name`` cannot be resolved to a callable.
    """

    path = _match_optimizer_alias(name)
    if path:
        target = _load_callable(path)
        return _wrap_callable(target)
    if name.startswith(DSPY_PREFIX):
        target = _load_callable(name)
        return _wrap_callable(target)
    raise ResolverError(f"Unknown optimizer '{name}'. {RESOLUTION_HINT}")


def _load_callable(path: str) -> Callable[..., Any]:
    """Import a callable attribute specified by dotted path (e.g. ``dspy.Predict``).

    Args:
        path: Dotted import path of the attribute to load.

    Returns:
        The resolved callable.

    Raises:
        ResolverError: When the module is missing, the attribute is absent,
            or the attribute is not callable.
    """
    module_path, attribute = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Catch only ``ModuleNotFoundError`` (a subclass of ``ImportError``):
        # bad alias paths surface as ``ResolverError`` while a corrupt
        # install raising plain ``ImportError`` propagates and fails fast.
        raise ResolverError(f"Unable to import '{module_path}': {exc}") from exc
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ResolverError(f"Module '{module_path}' has no attribute '{attribute}'.") from exc
    if not callable(target):
        raise ResolverError(f"Attribute '{path}' is not callable.")
    return target


def _wrap_callable(target: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``target`` in a thin forwarder that preserves its call signature.

    Built as a ``functools.partial`` with no pre-bound arguments so the
    returned object forwards every positional and keyword argument to
    ``target`` unchanged. ``functools.update_wrapper`` copies metadata
    (``__name__``, ``__doc__``, etc.) across; a captured
    ``inspect.signature`` is pinned via ``__signature__`` so introspection
    tools display the original parameters instead of ``(*args, **kwargs)``.

    Args:
        target: Callable to wrap.

    Returns:
        A forwarding wrapper that preserves ``target``'s signature.
    """
    try:
        signature: inspect.Signature | None = inspect.signature(target)
    except (ValueError, TypeError):
        signature = None

    wrapper = partial(target)
    update_wrapper(wrapper, target, updated=())
    if signature is not None:
        # ``partial`` accepts arbitrary attributes at runtime; ``setattr``
        # avoids the static-type complaint about ``__signature__``.
        setattr(wrapper, "__signature__", signature)  # noqa: B010
    return wrapper
