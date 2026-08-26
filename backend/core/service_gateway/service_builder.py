"""Factory for a fully-wired DspyService shared by the API app and the standalone worker.

The optimizer/module alias wiring used to live inline in ``create_app``; the
standalone worker entrypoint needs the identical registry (a worker whose
service can't resolve ``gepa``/``predict`` fails every job), so the wiring
lives here where both bootstrappers can reach it without importing the API.
"""

from __future__ import annotations

from ..registry import ServiceRegistry
from ..registry.resolvers import (
    MODULE_ALIASES,
    OPTIMIZER_ALIASES,
    resolve_module_factory,
    resolve_optimizer_factory,
)
from . import DspyService


def wire_registry_aliases(registry: ServiceRegistry) -> ServiceRegistry:
    """Register the alias-backed optimizer/module factories onto ``registry``.

    Skips names already registered so a pre-built registry (tests, custom
    deployments) wins.

    Args:
        registry: The registry to populate.

    Returns:
        The same registry, for chaining.
    """
    for alias in OPTIMIZER_ALIASES:
        if alias not in registry.optimizers:
            registry.register_optimizer(alias, resolve_optimizer_factory(alias))
    for alias in MODULE_ALIASES:
        if alias not in registry.modules:
            factory, _ = resolve_module_factory(alias)
            registry.register_module(alias, factory)
    return registry


def build_default_service(
    registry: ServiceRegistry | None = None,
    *,
    service_kwargs: dict | None = None,
) -> DspyService:
    """Build the standard DspyService with the alias-backed registry wiring.

    Args:
        registry: Optional pre-built registry; a fresh one is created when
            omitted. Either way the alias factories are merged in.
        service_kwargs: Optional kwargs forwarded to :class:`DspyService`.

    Returns:
        A service whose registry resolves every supported optimizer/module
        alias.
    """
    registry = wire_registry_aliases(registry or ServiceRegistry())
    return DspyService(registry, **(service_kwargs or {}))
