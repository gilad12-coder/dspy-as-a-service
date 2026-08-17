"""Logging and metrics setup for on-prem deployments.

Three concerns live here:

* :func:`configure_logging` switches between a human-readable formatter and
  a JSON formatter based on the ``LOG_FORMAT`` env var. JSON output is what
  shipping log aggregators (Loki, Splunk, Elastic) expect; the human format
  is what developers want during local runs. Pod identity and the per-
  request id (when known) are stamped on every record so logs collected
  from many replicas stay traceable.

* :func:`install_metrics` mounts a Prometheus-compatible ``/metrics``
  endpoint via ``prometheus-fastapi-instrumentator``. The dependency is
  imported lazily so the backend still boots in environments where the
  wheel hasn't been mirrored to the local artifactory yet.

* :func:`install_request_id_middleware` adds a tiny ASGI middleware that
  reads the inbound ``X-Request-ID`` header (or generates a UUID4),
  publishes it on a :class:`contextvars.ContextVar`, and echoes it back in
  the response. The logging filter picks the contextvar up so log lines
  emitted while serving the request — including from the worker thread
  pool when it inherits the same context — share a single id.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config import settings
from ..storage.usage import purge_expired_staged_datasets
from .alerts import install_alert_log_handler
from .log_shipping import build_log_ship_handler
from .routers.agent_history import purge_stale_conversations

# python-json-logger ships its formatter under two different names depending
# on the wheel version available in the local artifactory. Prefer the new
# location, fall back to the old one, and degrade to text logging when neither
# import resolves so the backend still boots.
try:
    from pythonjsonlogger.json import JsonFormatter as _JsonFormatter
except ImportError:
    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter as _JsonFormatter  # type: ignore[no-redef]
    except ImportError:
        _JsonFormatter = None  # type: ignore[assignment]

# prometheus-fastapi-instrumentator is optional in environments where the wheel
# hasn't been mirrored — degrade ``/metrics`` to a no-op there.
try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator
except ImportError:
    _Instrumentator = None  # type: ignore[assignment]

try:
    from prometheus_client import REGISTRY as _REGISTRY
    from prometheus_client import Gauge as _Gauge
except ImportError:
    _Gauge = None  # type: ignore[assignment]
    _REGISTRY = None  # type: ignore[assignment]


def _make_gauge(name: str, documentation: str) -> Any:
    """Create a Prometheus gauge, reusing the live one when already registered.

    The module is reloaded in tests via ``importlib.reload``, which re-executes
    module-level statements. Prometheus's global registry rejects duplicate
    metric names, so a naive ``Gauge(...)`` call raises on the second import.
    This helper looks the collector up first and returns the existing one when
    present.

    Args:
        name: Prometheus metric name.
        documentation: Help text shown on ``/metrics``.

    Returns:
        The shared :class:`prometheus_client.Gauge` instance, or ``None`` when
        ``prometheus-client`` isn't installed.
    """
    if _Gauge is None:
        return None
    if _REGISTRY is not None:
        existing = _REGISTRY._names_to_collectors.get(name)
        if existing is not None:
            return existing
    return _Gauge(name, documentation)

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
QUEUE_METRICS_REFRESH_SECONDS = 5.0
# Postgres pg_try_advisory_lock takes a bigint. Each constant is a stable,
# fleet-wide key for a specific cross-replica gate.
#
# ``ORPHAN_SWEEP`` — leader election for the periodic orphan-recovery sweeper.
# ``STARTUP_WORK`` — one-pod-only gate for idempotent startup tasks (schema
# touch-ups, embedding backfill/purge) so a rolling deploy of N replicas runs
# the work once, not N times.
ORPHAN_SWEEP_LOCK_KEY = 742137000001
STARTUP_WORK_LOCK_KEY = 742137000002
STALE_CONVERSATION_SWEEP_LOCK_KEY = 742137000003
STAGED_DATASET_SWEEP_LOCK_KEY = 742137000004
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_jobs_pending_gauge = _make_gauge(
    "skynet_jobs_pending", "Number of Skynet jobs waiting to be claimed"
)
_queue_age_gauge = _make_gauge(
    "skynet_queue_age_seconds", "Age in seconds of the oldest pending Skynet job"
)


def _resolve_pod_name() -> str:
    """Return the best-effort pod identity used to enrich log records.

    Returns:
        The Kubernetes pod name from the ``POD_NAME`` env var when present,
        otherwise the OS hostname.
    """
    return os.environ.get("POD_NAME") or socket.gethostname()


_POD_NAME = _resolve_pod_name()


def get_request_id() -> str:
    """Return the request id bound to the current context.

    Returns:
        The id set by the middleware for the current request, or the
        sentinel ``"-"`` outside of any request (e.g. on startup).
    """
    return _request_id_ctx.get()


class _ContextFilter(logging.Filter):
    """Inject pod name and current request id onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``record.pod`` and ``record.request_id`` so formatters can read them.

        Args:
            record: The log record about to be formatted.

        Returns:
            ``True`` so the record continues through the handler chain.
        """
        record.pod = _POD_NAME
        record.request_id = _request_id_ctx.get()
        return True


def _build_text_handler() -> logging.Handler:
    """Build a stream handler that writes the human-readable text format.

    Returns:
        A :class:`logging.StreamHandler` with the context filter and the
        human-readable formatter installed.
    """
    handler = logging.StreamHandler()
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s pod=%(pod)s rid=%(request_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    return handler


def _build_json_handler() -> logging.Handler:
    """Build a stream handler that writes JSON-formatted records.

    Falls back to :func:`_build_text_handler` after a single warning when
    ``python-json-logger`` is not installed.

    Returns:
        A :class:`logging.StreamHandler` whose formatter renders each record
        as a JSON object with stable ``timestamp``, ``level``, ``logger``,
        ``pod``, ``request_id`` and ``message`` keys.
    """
    if _JsonFormatter is None:
        logger.warning("python-json-logger not installed; falling back to text logs")
        return _build_text_handler()
    handler = logging.StreamHandler()
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        _JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(pod)s %(request_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )
    )
    return handler


def configure_logging() -> None:
    """Configure root logging for the FastAPI backend.

    Reads two env vars:

    * ``LOG_LEVEL`` — DEBUG/INFO/WARNING/ERROR/CRITICAL (default ``INFO``).
    * ``LOG_FORMAT`` — ``json`` or ``text`` (default ``text``).

    Existing handlers (e.g. one installed by Uvicorn or pytest at import
    time) are removed so log output has a single, predictable shape. When
    ``ALERT_WEBHOOK_URL`` is set, an :class:`~core.api.alerts.AlertLogHandler`
    is also attached so records at or above ``ALERT_MIN_LEVEL`` are forwarded
    to the chat webhook. When ``LOG_SHIP_URL`` is set, a
    :class:`~core.api.log_shipping.LogShipHandler` ships a JSON copy of every
    admitted record to that endpoint.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "text").lower()

    handler = _build_json_handler() if log_format == "json" else _build_text_handler()

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    install_alert_log_handler(root)
    shipper = build_log_ship_handler()
    if shipper is not None:
        shipper.addFilter(_ContextFilter())
        root.addHandler(shipper)


def install_metrics(app: FastAPI) -> None:
    """Mount a Prometheus ``/metrics`` endpoint on ``app`` when the dep is present.

    Falls back to a single warning and leaves the app untouched when
    ``prometheus-fastapi-instrumentator`` is not installed, so the backend
    keeps booting in environments where the dep wheel hasn't been mirrored.

    Args:
        app: The FastAPI application to instrument. The ``/metrics`` route
            is registered in-place.
    """
    if _Instrumentator is None:
        logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")
        return
    _Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class QueueMetricsRefresher:
    """Refresh cached queue-depth gauges from the job store."""

    def __init__(self, job_store: Any, interval_seconds: float = QUEUE_METRICS_REFRESH_SECONDS) -> None:
        """Initialize the refresher.

        Args:
            job_store: Store exposing ``get_queue_metrics()``.
            interval_seconds: Seconds between DB reads.
        """
        self._job_store = job_store
        self._interval_seconds = max(1.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background refresh loop after one immediate refresh."""
        if _jobs_pending_gauge is None or _queue_age_gauge is None:
            logger.warning("prometheus-client not installed; queue gauges disabled")
            return
        self.refresh_once()
        self._thread = threading.Thread(target=self._run, name="queue-metrics-refresher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background refresh loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def refresh_once(self) -> None:
        """Read queue metrics once and update the cached gauges."""
        if _jobs_pending_gauge is None or _queue_age_gauge is None:
            return
        try:
            pending_count, queue_age_seconds = self._job_store.get_queue_metrics()
        except AttributeError:
            pending_count = int(self._job_store.count_jobs(status="pending"))
            queue_age_seconds = 0.0
        except Exception:
            logger.warning("Queue metrics refresh failed", exc_info=True)
            return
        _jobs_pending_gauge.set(pending_count)
        _queue_age_gauge.set(queue_age_seconds)

    def _run(self) -> None:
        """Run the refresh loop until stopped."""
        while not self._stop_event.wait(self._interval_seconds):
            self.refresh_once()


def start_queue_metrics_refresher(job_store: Any) -> QueueMetricsRefresher:
    """Start and return the queue metrics refresher for ``job_store``.

    Args:
        job_store: Store used to read pending queue metrics.

    Returns:
        The started refresher; callers should invoke ``stop()`` during shutdown.
    """
    refresher = QueueMetricsRefresher(job_store)
    refresher.start()
    return refresher


@contextmanager
def advisory_lock(engine: Any, key: int) -> Iterator[bool]:
    """Best-effort Postgres advisory lock for cross-replica work serialization.

    On PostgreSQL, attempts ``pg_try_advisory_lock(key)`` on a dedicated
    connection and yields whether the lock was acquired. The lock is released
    via ``pg_advisory_unlock`` on context exit. The caller is expected to gate
    its work on the yielded boolean — peer pods that lost the race should
    treat the leader's run as authoritative and skip.

    On non-PostgreSQL dialects (tests / SQLite) the function is a no-op that
    yields ``True`` so single-process callers run the work unconditionally.

    Args:
        engine: SQLAlchemy engine used to source the lock-holding connection.
        key: Stable ``bigint`` lock identifier; see the module-level constants.

    Yields:
        ``True`` when this replica won the lock (and should do the work),
        ``False`` when a peer already holds it.
    """
    if engine is None or engine.dialect.name != "postgresql":
        yield True
        return
    with engine.connect() as conn:
        acquired = bool(
            conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
        )
        conn.commit()
        try:
            yield acquired
        finally:
            if acquired:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                conn.commit()


class OrphanRecoverySweeper:
    """Periodically re-queue jobs whose worker lease expired.

    The startup-only sweep in :func:`backend.core.api.app.lifespan` only fires
    once. If a worker dies mid-job after the fleet has booted, the lease ticks
    down but nothing scans for the dropped row until the next restart. This
    sweeper closes that gap by running ``recover_orphaned_jobs`` on a schedule
    in every backend replica. Leader election uses a Postgres advisory lock so
    only one replica per tick actually performs the work — peers fall through
    cheaply when ``pg_try_advisory_lock`` returns false.

    On non-PostgreSQL dialects (tests / SQLite) the lock check is skipped and
    the sweep runs unconditionally; tests run single-threaded so there is no
    race to mediate.
    """

    def __init__(self, job_store: Any, interval_seconds: float | None = None) -> None:
        """Initialize the sweeper.

        Args:
            job_store: Store exposing ``recover_orphaned_jobs()`` and ``engine``.
            interval_seconds: Override for the polling interval; defaults to
                ``settings.orphan_sweep_interval_seconds``.
        """
        self._job_store = job_store
        resolved = interval_seconds if interval_seconds is not None else settings.orphan_sweep_interval_seconds
        self._interval_seconds = max(5.0, float(resolved))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background sweep loop."""
        self._thread = threading.Thread(target=self._run, name="orphan-recovery-sweeper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sweep loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sweep_once(self) -> int:
        """Acquire the leader-election lock and run one orphan-recovery pass.

        Returns:
            The number of orphans handled by the sweep, or ``0`` if this
            replica did not win the advisory lock (peer is sweeping).
        """
        engine = getattr(self._job_store, "engine", None)
        if engine is None or engine.dialect.name != "postgresql":
            try:
                return int(self._job_store.recover_orphaned_jobs())
            except Exception:
                logger.warning("Orphan recovery sweep failed", exc_info=True)
                return 0
        try:
            with engine.begin() as conn:
                acquired = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)"),
                    {"k": ORPHAN_SWEEP_LOCK_KEY},
                ).scalar()
                if not acquired:
                    return 0
                return int(self._job_store.recover_orphaned_jobs())
        except Exception:
            logger.warning("Orphan recovery sweep failed", exc_info=True)
            return 0

    def _run(self) -> None:
        """Run the sweep loop until stopped."""
        while not self._stop_event.wait(self._interval_seconds):
            self.sweep_once()


def start_orphan_recovery_sweeper(job_store: Any) -> OrphanRecoverySweeper:
    """Start and return the orphan-recovery sweeper for ``job_store``.

    Args:
        job_store: Store used to recover orphaned jobs.

    Returns:
        The started sweeper; callers should invoke ``stop()`` during shutdown.
    """
    sweeper = OrphanRecoverySweeper(job_store)
    sweeper.start()
    return sweeper


class StaleConversationSweeper:
    """Periodically delete unpinned agent conversations past the staleness threshold.

    The thread polls at ``settings.stale_conversation_sweep_interval_seconds``
    (default 24 h). The actual purge is a small bounded ``DELETE`` so a long
    interval is fine — the goal is "users don't have to manage stale threads",
    not "react in seconds." Leader election uses a Postgres advisory lock so
    only one replica per tick actually deletes; peers fall through cheaply.

    On non-PostgreSQL dialects (tests / SQLite) the lock check is skipped and
    the sweep runs unconditionally.
    """

    def __init__(self, engine: Any, interval_seconds: float | None = None) -> None:
        """Initialize the sweeper.

        Args:
            engine: SQLAlchemy engine the conversations table is bound to.
            interval_seconds: Override for the polling interval; defaults to
                ``settings.stale_conversation_sweep_interval_seconds``.
        """
        self._engine = engine
        resolved = (
            interval_seconds
            if interval_seconds is not None
            else settings.stale_conversation_sweep_interval_seconds
        )
        self._interval_seconds = max(60.0, float(resolved))
        self._threshold_days = int(settings.stale_conversation_threshold_days)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background sweep loop."""
        self._thread = threading.Thread(
            target=self._run, name="stale-conversation-sweeper", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sweep loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sweep_once(self) -> int:
        """Acquire the leader-election lock and run one stale-conversation pass.

        Returns:
            The number of conversations deleted by the sweep, or ``0`` if this
            replica did not win the advisory lock (peer is sweeping).
        """
        if self._engine is None:
            return 0
        if self._engine.dialect.name != "postgresql":
            try:
                return purge_stale_conversations(
                    self._engine, threshold_days=self._threshold_days
                )
            except Exception:
                logger.warning("Stale conversation sweep failed", exc_info=True)
                return 0
        try:
            with self._engine.begin() as conn:
                acquired = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)"),
                    {"k": STALE_CONVERSATION_SWEEP_LOCK_KEY},
                ).scalar()
                if not acquired:
                    return 0
            return purge_stale_conversations(
                self._engine, threshold_days=self._threshold_days
            )
        except Exception:
            logger.warning("Stale conversation sweep failed", exc_info=True)
            return 0

    def _run(self) -> None:
        """Run the sweep loop until stopped."""
        # Run once immediately so a fresh process drains accumulated staleness
        # without waiting a full interval (default 24 h is too long for that).
        self.sweep_once()
        while not self._stop_event.wait(self._interval_seconds):
            self.sweep_once()


def start_stale_conversation_sweeper(engine: Any) -> StaleConversationSweeper:
    """Start and return the stale-conversation sweeper bound to ``engine``.

    Args:
        engine: SQLAlchemy engine the conversations table is bound to.

    Returns:
        The started sweeper; callers should invoke ``stop()`` during shutdown.
    """
    sweeper = StaleConversationSweeper(engine)
    sweeper.start()
    return sweeper


class StagedDatasetSweeper:
    """Periodically delete abandoned wizard staged-dataset rows past their TTL.

    A staged row parks a wizard upload server-side; a successful submit evicts
    it, but a wizard abandoned before submit leaves the row orphaned. The thread
    polls at ``settings.staged_dataset_sweep_interval_seconds`` (default 5 min)
    and deletes rows older than ``settings.staged_dataset_ttl_seconds`` (default
    10 min). The human submit posts rows inline and an agent submit-by-id
    consumes a row within seconds of staging, so the TTL never races a live
    submit. Leader election uses a Postgres advisory lock so only one replica
    per tick deletes; peers fall through cheaply.

    On non-PostgreSQL dialects (tests / SQLite) the lock check is skipped and
    the sweep runs unconditionally.
    """

    def __init__(self, engine: Any, interval_seconds: float | None = None) -> None:
        """Initialize the sweeper.

        Args:
            engine: SQLAlchemy engine the ``agent_staged_datasets`` table is on.
            interval_seconds: Override for the polling interval; defaults to
                ``settings.staged_dataset_sweep_interval_seconds``.
        """
        self._engine = engine
        resolved = (
            interval_seconds
            if interval_seconds is not None
            else settings.staged_dataset_sweep_interval_seconds
        )
        self._interval_seconds = max(60.0, float(resolved))
        self._ttl_seconds = int(settings.staged_dataset_ttl_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background sweep loop."""
        self._thread = threading.Thread(
            target=self._run, name="staged-dataset-sweeper", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sweep loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sweep_once(self) -> int:
        """Acquire the leader-election lock and run one staged-dataset TTL pass.

        Returns:
            The number of staged rows deleted by the sweep, or ``0`` if this
            replica did not win the advisory lock (peer is sweeping).
        """
        if self._engine is None:
            return 0
        if self._engine.dialect.name != "postgresql":
            try:
                return purge_expired_staged_datasets(self._engine, max_age_seconds=self._ttl_seconds)
            except Exception:
                logger.warning("Staged dataset sweep failed", exc_info=True)
                return 0
        try:
            with self._engine.begin() as conn:
                acquired = conn.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)"),
                    {"k": STAGED_DATASET_SWEEP_LOCK_KEY},
                ).scalar()
                if not acquired:
                    return 0
            return purge_expired_staged_datasets(self._engine, max_age_seconds=self._ttl_seconds)
        except Exception:
            logger.warning("Staged dataset sweep failed", exc_info=True)
            return 0

    def _run(self) -> None:
        """Run the sweep loop until stopped."""
        # Run once immediately so a fresh process drains rows abandoned while it
        # was down without waiting a full interval.
        self.sweep_once()
        while not self._stop_event.wait(self._interval_seconds):
            self.sweep_once()


def start_staged_dataset_sweeper(engine: Any) -> StagedDatasetSweeper:
    """Start and return the staged-dataset TTL sweeper bound to ``engine``.

    Args:
        engine: SQLAlchemy engine the ``agent_staged_datasets`` table is on.

    Returns:
        The started sweeper; callers should invoke ``stop()`` during shutdown.
    """
    sweeper = StagedDatasetSweeper(engine)
    sweeper.start()
    return sweeper


class EventLoopLagMonitor:
    """Diagnostic: measure asyncio event-loop scheduling lag.

    A coroutine that ``await``\\s only on IO returns control to the loop
    promptly; a coroutine that runs sync CPU between awaits holds the loop and
    delays every other task. This monitor schedules itself every
    ``interval_seconds`` and records the gap between when the next tick was
    scheduled and when it actually fired. Sustained gaps above
    ``threshold_ms`` indicate blocking — the kind that multi-worker uvicorn
    would parallelize at the cost of breaking in-process MCP sessions.

    Gated by ``settings.event_loop_lag_monitor_enabled``; off by default so
    prod pods don't run the extra task. Intended as a short diagnostic run,
    not a permanent metric.
    """

    def __init__(
        self,
        interval_seconds: float = 0.05,
        threshold_ms: float | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            interval_seconds: How often to wake and sample. 50 ms is short
                enough to catch sub-second blocks without measurable cost.
            threshold_ms: Override for the warn threshold; defaults to
                ``settings.event_loop_lag_threshold_ms``.
        """
        self._interval = float(interval_seconds)
        resolved = threshold_ms if threshold_ms is not None else settings.event_loop_lag_threshold_ms
        self._threshold = float(resolved) / 1000.0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Schedule the monitor task on the current running loop."""
        self._task = asyncio.create_task(self._run(), name="event-loop-lag-monitor")

    def stop(self) -> None:
        """Cancel the monitor task; safe to call when not running."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        """Sample loop lag forever; warn when a sample exceeds the threshold."""
        loop = asyncio.get_running_loop()
        expected = loop.time() + self._interval
        try:
            while True:
                await asyncio.sleep(self._interval)
                now = loop.time()
                lag = now - expected
                if lag > self._threshold:
                    logger.warning(
                        "event loop lag: %.0fms (a handler is doing sync CPU work)",
                        lag * 1000,
                    )
                expected = now + self._interval
        except asyncio.CancelledError:
            return


def start_event_loop_lag_monitor() -> EventLoopLagMonitor:
    """Start and return an event-loop lag monitor on the current loop.

    Returns:
        The started monitor; callers should invoke ``stop()`` during shutdown.
    """
    monitor = EventLoopLagMonitor()
    monitor.start()
    return monitor


class RequestIDMiddleware:
    """Bind a stable request id to every request for log correlation.

    Pure ASGI (not ``BaseHTTPMiddleware``) so it never buffers the response
    body. ``BaseHTTPMiddleware`` re-emits responses through an internal memory
    stream, which corrupts long-lived ``StreamingResponse``/SSE framing when the
    client disconnects — uvicorn/h11 then raises ``Too much data for declared
    Content-Length``. Header-only work belongs in pure ASGI middleware.
    """

    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER) -> None:
        """Wrap ``app`` so each request has a request id bound on a ContextVar.

        Args:
            app: The downstream ASGI application.
            header_name: Inbound and outbound header name carrying the id.
        """
        self.app = app
        self._header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Mint/propagate the request id and stamp it on the response start.

        Non-HTTP scopes (lifespan, websocket) pass straight through unstamped.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get(self._header_name)
        request_id = incoming or uuid.uuid4().hex
        token = _request_id_ctx.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[self._header_name] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id_ctx.reset(token)


def install_request_id_middleware(app: FastAPI) -> None:
    """Install :class:`RequestIDMiddleware` on ``app``.

    Args:
        app: The FastAPI application to instrument.
    """
    app.add_middleware(RequestIDMiddleware)
