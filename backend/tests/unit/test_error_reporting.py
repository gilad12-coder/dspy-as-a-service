"""Tests for deployment-local structured error reporting."""

from __future__ import annotations

import logging

from core import error_reporting


def test_configure_keeps_reporting_local(caplog) -> None:
    """Enable local logging without configuring external egress."""
    with caplog.at_level(logging.INFO, logger="skynet.errors"):
        configured = error_reporting.configure_error_reporting("worker")

    assert configured is False
    assert "Local error reporting enabled" in caplog.text


def test_capture_exception_writes_traceback(caplog) -> None:
    """Record exception details for on-premises debugging."""
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        with caplog.at_level(logging.ERROR, logger="skynet.errors"):
            error_reporting.capture_exception(exc)

    assert "Unhandled application exception" in caplog.text
    assert "RuntimeError: boom" in caplog.text
