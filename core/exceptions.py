from __future__ import annotations


class ServiceError(RuntimeError):
    """Raised when the service_gateway cannot fulfill a request."""
