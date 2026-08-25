"""Server-emitted product-telemetry milestones (purchases, run outcomes)."""

from .server_events import record_server_event

__all__ = ["record_server_event"]
