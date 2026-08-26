"""Ship a copy of every log record to an HTTP log aggregator.

Railway keeps container logs for a limited window and has no native log
drain, so anything older than that window is gone unless the app forwards
it somewhere itself. :class:`LogShipHandler` does that: it turns each record
into a small JSON object, buffers it, and POSTs batches to ``LOG_SHIP_URL``
with ``Authorization: Bearer LOG_SHIP_TOKEN`` from a daemon thread.

The payload is a JSON array of ``{"dt", "level", "logger", "message", "service",
"pod", "request_id"[, "exception"]}`` objects — the shape Better Stack (Logtail)
ingests as-is, and generic enough for a Vector ``http_server`` source or any
JSON-over-HTTP intake.

Everything degrades to a no-op when ``LOG_SHIP_URL`` is unset. The handler
never blocks the caller: records go into a bounded in-memory queue and are
dropped (with a running count) when the aggregator can't keep up, and a
delivery failure discards that batch and moves on. Records emitted by this
module itself are skipped so a shipping failure can never feed back into the
queue.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import socket
import threading
import traceback
import urllib.request
from datetime import UTC, datetime
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

_DELIVERY_TIMEOUT_SECONDS = 5.0
_FLUSH_INTERVAL_SECONDS = 2.0
_MAX_BATCH_RECORDS = 200
_MAX_QUEUE_RECORDS = 10_000
_MAX_MESSAGE_CHARS = 8_000
_MAX_EXCEPTION_CHARS = 16_000

# Same fallback as observability._resolve_pod_name; observability imports this
# module to install the handler, so importing it back would form a cycle.
_POD_NAME = os.environ.get("POD_NAME") or socket.gethostname()
_SERVICE_NAME = os.environ.get("RAILWAY_SERVICE_NAME") or os.environ.get("SERVICE_NAME") or ""


def log_shipping_configured() -> bool:
    """Return whether an outbound log-shipping endpoint is configured."""
    return bool(settings.log_ship_url)


def _truncate(text_value: str, limit: int) -> str:
    """Return ``text_value`` shortened to ``limit`` characters with an elision marker.

    Args:
        text_value: The text to bound.
        limit: Maximum number of characters to keep.

    Returns:
        The original text when within the limit, otherwise a truncated copy
        with a trailing note of how many characters were dropped.
    """
    if len(text_value) <= limit:
        return text_value
    dropped = len(text_value) - limit
    return f"{text_value[:limit]}… (+{dropped} more chars)"


def _serialize(record: logging.LogRecord) -> dict[str, Any]:
    """Convert a log record into the JSON object shipped to the aggregator.

    Args:
        record: The record to serialize. ``record.request_id`` is read when
            :class:`~core.api.observability._ContextFilter` has stamped it.

    Returns:
        A flat dict with ``dt`` (ISO-8601 UTC), ``level``, ``logger``,
        ``message``, ``service``, ``pod``, ``request_id`` and, for records
        carrying ``exc_info``, a formatted ``exception`` traceback.
    """
    try:
        message = record.getMessage()
    except Exception:
        message = str(record.msg)
    payload: dict[str, Any] = {
        "dt": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": _truncate(message, _MAX_MESSAGE_CHARS),
        "service": _SERVICE_NAME,
        "pod": _POD_NAME,
        "request_id": getattr(record, "request_id", "-"),
    }
    if record.exc_info:
        formatted = "".join(traceback.format_exception(*record.exc_info))
        payload["exception"] = _truncate(formatted, _MAX_EXCEPTION_CHARS)
    return payload


def _post(url: str, token: str, batch: list[dict[str, Any]], timeout: float) -> None:
    """POST one JSON batch to the aggregator.

    Args:
        url: The ingest endpoint.
        token: Bearer token; an empty string sends no ``Authorization`` header.
        batch: The serialized records.
        timeout: Socket timeout in seconds.

    Raises:
        urllib.error.URLError, OSError: On any network or protocol failure.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(batch, default=str).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


class LogShipHandler(logging.Handler):
    """Buffer log records and ship them in batches from a background thread.

    ``emit`` only enqueues; the ``log-shipper`` daemon thread drains the queue
    every :data:`_FLUSH_INTERVAL_SECONDS` in batches of at most
    :data:`_MAX_BATCH_RECORDS`. When the queue is full the record is dropped
    and counted; the count is reported as a synthetic WARNING record at the
    head of the next batch so a lossy period is visible in the aggregator
    rather than silent.
    """

    def __init__(self, url: str, token: str) -> None:
        """Create the handler and start its delivery thread.

        Args:
            url: The ingest endpoint.
            token: Bearer token for the endpoint (may be empty).
        """
        super().__init__()
        self._url = url
        self._token = token
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_MAX_QUEUE_RECORDS)
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="log-shipper", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        """Serialize ``record`` and enqueue it without blocking.

        Args:
            record: The record to ship. Records from this module are ignored.
        """
        if record.name == __name__:
            return
        try:
            self._queue.put_nowait(_serialize(record))
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1
        except Exception:
            self.handleError(record)

    def _drain(self, limit: int) -> list[dict[str, Any]]:
        """Pull up to ``limit`` queued records without waiting.

        Args:
            limit: Maximum number of records to take.

        Returns:
            The dequeued records, oldest first (possibly empty).
        """
        batch: list[dict[str, Any]] = []
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _dropped_notice(self) -> dict[str, Any] | None:
        """Return a synthetic WARNING record describing dropped logs, resetting the count.

        Returns:
            The record to prepend to the next batch, or ``None`` when nothing
            was dropped since the last notice.
        """
        with self._dropped_lock:
            dropped, self._dropped = self._dropped, 0
        if not dropped:
            return None
        return {
            "dt": datetime.now(tz=UTC).isoformat(),
            "level": "WARNING",
            "logger": __name__,
            "message": f"log shipper dropped {dropped} records (queue full)",
            "service": _SERVICE_NAME,
            "pod": _POD_NAME,
            "request_id": "-",
        }

    def _ship(self, batch: list[dict[str, Any]]) -> None:
        """Deliver one batch, swallowing failures so the thread never dies.

        Args:
            batch: The serialized records to send.
        """
        notice = self._dropped_notice()
        if notice is not None:
            batch.insert(0, notice)
        try:
            _post(self._url, self._token, batch, _DELIVERY_TIMEOUT_SECONDS)
        except Exception as exc:
            # DEBUG on purpose: this record is skipped by emit() anyway, and a
            # WARNING here would just churn stderr during an aggregator outage.
            logger.debug("Log shipping failed; dropped %d records: %s", len(batch), exc)

    def _run(self) -> None:
        """Flush loop: ship whenever a batch is ready or the interval elapses."""
        while not self._stop_event.wait(_FLUSH_INTERVAL_SECONDS):
            self.flush()
        self.flush()

    def flush(self) -> None:
        """Ship everything currently queued, in batches."""
        while True:
            batch = self._drain(_MAX_BATCH_RECORDS)
            if not batch:
                return
            self._ship(batch)

    def close(self) -> None:
        """Stop the delivery thread after a final best-effort flush."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._thread.join(timeout=_DELIVERY_TIMEOUT_SECONDS + 1.0)
        super().close()


def build_log_ship_handler() -> LogShipHandler | None:
    """Return a started :class:`LogShipHandler` when ``LOG_SHIP_URL`` is set.

    Returns:
        The handler for the caller to attach to the root logger, or ``None``
        when log shipping is not configured.
    """
    if not settings.log_ship_url:
        return None
    token = settings.log_ship_token.get_secret_value() if settings.log_ship_token else ""
    return LogShipHandler(settings.log_ship_url, token)
