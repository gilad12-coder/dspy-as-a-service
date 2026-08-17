"""Tests for the HTTP log shipper (:mod:`core.api.log_shipping`).

The network call is replaced with an in-memory recorder; no socket is ever
opened. Handlers are closed at the end of each test so their delivery threads
don't outlive the test.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from core.api import log_shipping, observability
from core.config import settings


@pytest.fixture
def captured_posts(monkeypatch) -> list[tuple[str, str, list[dict]]]:
    """Record shipped batches in-memory and configure an endpoint for the test.

    Returns:
        The list that each :func:`~core.api.log_shipping._post` call appends
        ``(url, token, batch)`` to.
    """
    posts: list[tuple[str, str, list[dict]]] = []

    def _fake_post(url, token, batch, timeout):
        posts.append((url, token, list(batch)))

    monkeypatch.setattr(log_shipping, "_post", _fake_post)
    monkeypatch.setattr(settings, "log_ship_url", "http://ingest.test/")
    monkeypatch.setattr(settings, "log_ship_token", SecretStr("src-token"))
    return posts


@pytest.fixture
def handler(captured_posts) -> Iterator[log_shipping.LogShipHandler]:
    """Build a configured handler and close it (stopping its thread) afterwards.

    Yields:
        The started handler.
    """
    built = log_shipping.build_log_ship_handler()
    assert built is not None
    yield built
    built.close()


def _record(message: str, level: int = logging.INFO, name: str = "core.api.app") -> logging.LogRecord:
    """Build a log record for the tests.

    Args:
        message: The pre-formatted message.
        level: Logging level number.
        name: Logger name.

    Returns:
        A :class:`logging.LogRecord` with no ``exc_info``.
    """
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1, msg=message, args=(), exc_info=None)


def test_build_returns_none_when_unconfigured(monkeypatch):
    """No LOG_SHIP_URL means no handler and no network."""
    monkeypatch.setattr(settings, "log_ship_url", "")
    assert log_shipping.build_log_ship_handler() is None
    assert log_shipping.log_shipping_configured() is False


def test_flush_ships_serialized_batch_with_token(handler, captured_posts):
    """Queued records go out as one JSON batch to the configured URL with the bearer token."""
    handler.emit(_record("hello", level=logging.WARNING))
    handler.emit(_record("world"))
    handler.flush()
    assert len(captured_posts) == 1
    url, token, batch = captured_posts[0]
    assert url == "http://ingest.test/"
    assert token == "src-token"
    assert [entry["message"] for entry in batch] == ["hello", "world"]
    first = batch[0]
    assert first["level"] == "WARNING"
    assert first["logger"] == "core.api.app"
    assert first["pod"] == log_shipping._POD_NAME
    assert first["dt"].endswith("+00:00")


def test_serialize_includes_traceback_and_request_id():
    """exc_info becomes an ``exception`` field; a stamped request id is carried through."""
    try:
        raise ValueError("kaboom")
    except ValueError:
        record = logging.LogRecord(
            name="core.api.app",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    record.request_id = "req-42"
    payload = log_shipping._serialize(record)
    assert payload["request_id"] == "req-42"
    assert "ValueError: kaboom" in payload["exception"]


def test_flush_with_empty_queue_posts_nothing(handler, captured_posts):
    """An empty flush never touches the network."""
    handler.flush()
    assert captured_posts == []


def test_own_records_are_skipped(handler, captured_posts):
    """Records from the shipping module itself never enter the queue (no feedback loop)."""
    handler.emit(_record("shipping failed", name=log_shipping.__name__))
    handler.flush()
    assert captured_posts == []


def test_full_queue_drops_and_reports_count(handler, captured_posts, monkeypatch):
    """Overflow records are dropped and the count leads the next batch as a WARNING."""
    monkeypatch.setattr(log_shipping, "_MAX_QUEUE_RECORDS", 2)
    small = log_shipping.LogShipHandler("http://ingest.test/", "")
    try:
        for i in range(5):
            small.emit(_record(f"m{i}"))
        small.flush()
    finally:
        small.close()
    assert len(captured_posts) == 1
    batch = captured_posts[0][2]
    assert batch[0]["level"] == "WARNING"
    assert "dropped 3 records" in batch[0]["message"]
    assert [entry["message"] for entry in batch[1:]] == ["m0", "m1"]


def test_post_failure_is_swallowed(monkeypatch):
    """A failing POST drops the batch and leaves the handler usable."""
    monkeypatch.setattr(settings, "log_ship_url", "http://ingest.test/")
    monkeypatch.setattr(settings, "log_ship_token", None)

    def _boom(*args):
        raise OSError("network down")

    monkeypatch.setattr(log_shipping, "_post", _boom)
    built = log_shipping.build_log_ship_handler()
    assert built is not None
    try:
        built.emit(_record("one"))
        built.flush()
        built.emit(_record("two"))
        built.flush()
    finally:
        built.close()


def test_configure_logging_attaches_shipper_with_context_filter(captured_posts, monkeypatch):
    """configure_logging installs the shipper behind the context filter so request ids ship."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    monkeypatch.setenv("LOG_FORMAT", "text")
    try:
        observability.configure_logging()
        shippers = [h for h in root.handlers if isinstance(h, log_shipping.LogShipHandler)]
        assert len(shippers) == 1
        shipper = shippers[0]
        token = observability._request_id_ctx.set("req-7")
        try:
            logging.getLogger("core.api.tests").info("through root")
        finally:
            observability._request_id_ctx.reset(token)
        shipper.flush()
        shipper.close()
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
    shipped = [entry for _, _, batch in captured_posts for entry in batch]
    assert any(e["message"] == "through root" and e["request_id"] == "req-7" for e in shipped)
