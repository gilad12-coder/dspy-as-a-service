"""Local error-reporting facade for API and worker processes."""

from __future__ import annotations

import logging

logger = logging.getLogger("skynet.errors")


def configure_error_reporting(service_name: str) -> bool:
    """Keep the process-level reporting hook compatible with local logging.

    Args:
        service_name: Logical process name attached to structured log records.

    Returns:
        ``False`` because errors remain inside the deployment by default.
    """
    logger.info("Local error reporting enabled", extra={"service": service_name})
    return False


def capture_exception(exc: BaseException) -> None:
    """Write an exception and traceback to structured local logs.

    Args:
        exc: Exception to report.
    """
    logger.error(
        "Unhandled application exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
