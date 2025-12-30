from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping

ModuleFactory = Callable[..., "dspy.Module"]
MetricFn = Callable[..., float]
OptimizerFactory = Callable[..., Any]


class RegistryError(RuntimeError):
    """Base exception raised for registry-related issues."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a user attempts to register the same key twice."""


class UnknownRegistrationError(RegistryError):
    """Raised when requesting an entry that does not exist."""


@dataclass
class ServiceRegistry:
    """Holds user-defined factories for modules, metrics, and optimizers."""

    modules: MutableMapping[str, ModuleFactory] = field(default_factory=dict)
    metrics: MutableMapping[str, MetricFn] = field(default_factory=dict)
    optimizers: MutableMapping[str, OptimizerFactory] = field(default_factory=dict)

    def register_module(self, name: str, factory: ModuleFactory) -> None:
        """Register a module factory under a name.

        Args:
            name: Unique identifier for the DSPy module.
            factory: Callable that returns a configured ``dspy.Module``.

        Returns:
            None
        """
        self._register(self.modules, name, factory)

    def register_metric(self, name: str, metric: MetricFn) -> None:
        """Register a metric function under a name.

        Args:
            name: Unique identifier for the metric.
            metric: Callable that computes a scalar metric.

        Returns:
            None
        """
        self._register(self.metrics, name, metric)

    def register_optimizer(self, name: str, factory: OptimizerFactory) -> None:
        """Register an optimizer factory under a name.

        Args:
            name: Unique identifier for the optimizer.
            factory: Callable that returns a DSPy optimizer instance.

        Returns:
            None
        """
        self._register(self.optimizers, name, factory)

    def get_module(self, name: str) -> ModuleFactory:
        """Retrieve a module factory by name.

        Args:
            name: Registered module identifier.

        Returns:
            ModuleFactory: The stored factory callable.
        """
        return self._get(self.modules, name, kind="module")

    def get_metric(self, name: str) -> MetricFn:
        """Retrieve a metric function by name.

        Args:
            name: Registered metric identifier.

        Returns:
            MetricFn: The stored metric callable.
        """
        return self._get(self.metrics, name, kind="metric")

    def get_optimizer(self, name: str) -> OptimizerFactory:
        """Retrieve an optimizer factory by name.

        Args:
            name: Registered optimizer identifier.

        Returns:
            OptimizerFactory: The stored optimizer callable.
        """
        return self._get(self.optimizers, name, kind="optimizer")

    def snapshot(self) -> Dict[str, list[str]]:
        """Return summary information about registered assets.

        Args:
            None.

        Returns:
            Dict[str, list[str]]: Mapping from asset type to sorted names.
        """
        return {
            "modules": sorted(self.modules.keys()),
            "metrics": sorted(self.metrics.keys()),
            "optimizers": sorted(self.optimizers.keys()),
        }

    @staticmethod
    def _register(store: MutableMapping[str, Any], name: str, value: Any) -> None:
        """Store a callable while preventing duplicate names.

        Args:
            store: Internal dictionary for a registry category.
            name: Registration key.
            value: Callable or object being registered.

        Returns:
            None

        Raises:
            DuplicateRegistrationError: If the name already exists.
        """
        if name in store:
            raise DuplicateRegistrationError(f"Entry '{name}' already registered.")
        store[name] = value

    @staticmethod
    def _get(store: Mapping[str, Any], name: str, *, kind: str) -> Any:
        """Fetch a stored callable or object.

        Args:
            store: Internal dictionary for a registry category.
            name: Registration key to retrieve.
            kind: Human-readable descriptor used in error messages.

        Returns:
            Any: Stored callable or object.

        Raises:
            UnknownRegistrationError: If the key is not present.
        """
        try:
            return store[name]
        except KeyError as exc:
            raise UnknownRegistrationError(f"Unknown {kind} '{name}'.") from exc
