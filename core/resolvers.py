from __future__ import annotations
import importlib
from dataclasses import dataclass
import inspect
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional, Tuple
from .constants import RESOLUTION_HINT

DSPY_PREFIX = "dspy."


@dataclass(frozen=True)
class ModuleAlias:
    """Describe how to resolve a module alias."""

    paths: tuple[str, ...]
    auto_signature: bool = False


MODULE_ALIASES: Dict[str, ModuleAlias] = {
    "predict": ModuleAlias(("dspy.Predict",), auto_signature=True),
    "cot": ModuleAlias(
        (
            "dspy.modules.ChainOfThought",
            "dspy.ChainOfThought",
        ),
        auto_signature=True,
    ),
}

OPTIMIZER_ALIASES: Dict[str, str] = {
    "miprov2": "dspy.teleprompt.MIPROv2",
    "gepa": "dspy.teleprompt.GEPA",
}

AUTO_SIGNATURE_PATHS = {
    path
    for alias in MODULE_ALIASES.values()
    if alias.auto_signature
    for path in alias.paths
}


class ResolverError(RuntimeError):
    """Raised when a requested DSPy asset cannot be resolved."""


def resolve_module_factory(name: str) -> Tuple[Callable[..., Any], bool]:
    """Resolve a module factory from aliases or dotted paths.

    Args:
        name: Alias or dotted path identifying a DSPy module class.

    Returns:
        Tuple[Callable[..., Any], bool]: Factory callable and flag indicating
        whether the service_gateway should auto-generate a signature when none is
        supplied via ``module_kwargs``.

    Raises:
        ResolverError: If the name cannot be resolved.
    """

    spec = _match_module_alias(name)
    if spec is not None:
        last_error: Optional[Exception] = None
        for path in spec.paths:
            try:
                target = _load_callable(path)
                return _wrap_callable(target), spec.auto_signature
            except ResolverError as exc:  # pragma: no cover - best effort fallbacks
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
        name: Alias or dotted path identifying a DSPy optimizer class.

    Returns:
        Callable[..., Any]: Factory callable that instantiates the optimizer.

    Raises:
        ResolverError: If the name cannot be resolved.
    """

    path = _match_optimizer_alias(name)
    if path:
        target = _load_callable(path)
        return _wrap_callable(target)
    if name.startswith(DSPY_PREFIX):
        target = _load_callable(name)
        return _wrap_callable(target)
    raise ResolverError(f"Unknown optimizer '{name}'. {RESOLUTION_HINT}")


def _match_module_alias(name: str) -> Optional[ModuleAlias]:
    """Return the module alias specification when available.

    Args:
        name: Requested alias.

    Returns:
        Optional[ModuleAlias]: Alias specification if defined.
    """

    return MODULE_ALIASES.get(name.lower())


def _match_optimizer_alias(name: str) -> Optional[str]:
    """Return the optimizer dotted path when alias is provided.

    Args:
        name: Requested alias.

    Returns:
        Optional[str]: Dotted path of the optimizer class.
    """

    return OPTIMIZER_ALIASES.get(name.lower())


@lru_cache(maxsize=None)
def _load_callable(path: str) -> Callable[..., Any]:
    """Import a callable attribute specified by dotted path.

    Args:
        path: Dotted path such as ``dspy.Predict``.

    Returns:
        Callable[..., Any]: Imported class or factory.

    Raises:
        ResolverError: If the attribute does not exist or is not callable.
    """

    module_path, attribute = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ResolverError(f"Unable to import '{module_path}': {exc}") from exc
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ResolverError(f"Module '{module_path}' has no attribute '{attribute}'.") from exc
    if not callable(target):
        raise ResolverError(f"Attribute '{path}' is not callable.")
    return target


def _wrap_callable(target: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a target so it behaves like a module or optimizer factory.

    Args:
        target: Callable class or function to wrap.

    Returns:
        Callable[..., Any]: Wrapper that forwards keyword arguments.
    """

    try:
        signature = inspect.signature(target)
    except (ValueError, TypeError):
        signature = None

    @wraps(target)
    def factory(*args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped callable with keyword arguments.

        Args:
            args: Positional arguments forwarded to the callable.
            kwargs: Keyword arguments forwarded to the callable.

        Returns:
            Any: Result from invoking the callable.
        """

        return target(*args, **kwargs)

    if signature is not None:
        factory.__signature__ = signature
    return factory
