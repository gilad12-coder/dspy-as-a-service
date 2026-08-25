"""HTTP API layer for DSPy optimization service.

Kept intentionally slim: leaf modules under this package import from each
other freely, and an eager re-export of ``create_app`` here would force
``core.api.app`` (and its router fan-out) to load whenever any submodule
is imported, creating circular-import risk for service-gateway code that
needs only a single helper. Import
``create_app`` directly from ``core.api.app``.
"""
