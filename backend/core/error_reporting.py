"""Environment-gated Sentry error reporting for API and worker processes."""

from __future__ import annotations

import os
from typing import Any

import sentry_sdk

from .config import settings

_configured = False


def _scrub_event(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    """Remove identity and request payloads before Sentry transport.

    Args:
        event: Sentry event assembled by the SDK.
        _hint: SDK capture context, intentionally unused.

    Returns:
        The event stripped of user identity and sensitive request fields.
    """
    event.pop("user", None)
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("headers", None)
        request.pop("cookies", None)
        request.pop("data", None)
        request.pop("query_string", None)
    return event


def configure_error_reporting(service_name: str) -> bool:
    """Initialize Sentry with privacy-safe, low-volume production defaults.

    Args:
        service_name: Logical process name attached as a Sentry tag.

    Returns:
        ``True`` when a DSN was configured and the SDK was initialized.
    """
    global _configured
    dsn = settings.sentry_dsn
    if dsn is None:
        _configured = False
        return False
    sentry_sdk.init(
        dsn=dsn.get_secret_value(),
        environment=settings.sentry_environment,
        release=os.getenv("RAILWAY_GIT_COMMIT_SHA"),
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        traces_sample_rate=settings.sentry_traces_sample_rate,
        before_send=_scrub_event,
    )
    sentry_sdk.set_tag("service", service_name)
    _configured = True
    return True


def capture_exception(exc: BaseException) -> None:
    """Capture an exception when Sentry is configured, otherwise do nothing.

    Args:
        exc: Exception to report.
    """
    if _configured:
        sentry_sdk.capture_exception(exc)
