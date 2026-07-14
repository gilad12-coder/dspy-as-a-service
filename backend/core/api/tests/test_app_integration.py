"""Integration tests for ``create_app()``.

Covers lifespan startup, ``/health``, ``/queue``, cache-control middleware, the
SIGTERM handler, and ``/openapi.public.json``. These tests use the real
``create_app()`` factory but stub out every external dependency (Postgres, the
background worker, the DSPy service) so the suite runs without any I/O.
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ...config import settings as app_settings
from ...models.constants import HEALTH_STATUS_OK
from ...registry.core import ServiceRegistry
from ..app import _SCALAR_PUBLIC_PATHS, create_app


def _make_mock_worker(*, threads_alive: bool = True, stale_seconds: float | None = 0.0):
    """Build a ``MagicMock`` worker pre-configured for /health and /queue.

    Args:
        threads_alive: Value returned by ``threads_alive``.
        stale_seconds: Value returned by ``seconds_since_last_activity``.

    Returns:
        A ``MagicMock`` with the worker interface stubbed out.
    """
    w = MagicMock()
    w.threads_alive.return_value = threads_alive
    w.seconds_since_last_activity.return_value = stale_seconds
    w.queue_size.return_value = 2
    w.active_jobs.return_value = 1
    w.thread_count.return_value = 2
    return w


def _make_mock_job_store(*, pending_ids: list[str] | None = None):
    """Build a ``MagicMock`` job store covering the lifespan recovery paths.

    Args:
        pending_ids: Identifiers returned by ``recover_pending_jobs``.

    Returns:
        A ``MagicMock`` job store with recovery hooks stubbed out.
    """
    js = MagicMock()
    js.recover_orphaned_jobs.return_value = 0
    js.recover_pending_jobs.return_value = pending_ids or []
    return js


@pytest.fixture
def mock_worker():
    """Provide a default healthy worker mock for app integration tests.

    Returns:
        The mock built by ``_make_mock_worker``.
    """
    return _make_mock_worker()


@pytest.fixture
def mock_job_store():
    """Provide a default job store mock for app integration tests.

    Returns:
        The mock built by ``_make_mock_job_store``.
    """
    return _make_mock_job_store()


@pytest.fixture
def app_client(mock_worker, mock_job_store):
    """Build a ``TestClient`` around the real app with patched dependencies.

    Args:
        mock_worker: Worker mock injected via ``get_worker``.
        mock_job_store: Job store mock injected via ``get_job_store``.

    Yields:
        A 3-tuple of ``(client, worker, job_store)`` so individual tests can
        adjust the underlying mocks before issuing requests.
    """
    registry = ServiceRegistry()

    with (
        patch("core.api.app.get_job_store", return_value=mock_job_store),
        patch("core.api.app.get_worker", return_value=mock_worker),
        patch("core.api.app.DspyService"),  # prevent real DSPy init
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client, mock_worker, mock_job_store


def test_lifespan_calls_recover_orphaned_jobs(mock_worker, mock_job_store):
    """Lifespan startup invokes the orphan-recovery hook on the job store."""
    registry = ServiceRegistry()
    with (
        patch("core.api.app.get_job_store", return_value=mock_job_store),
        patch("core.api.app.get_worker", return_value=mock_worker),
        patch("core.api.app.DspyService"),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False):
            mock_job_store.recover_orphaned_jobs.assert_called_once()


def test_lifespan_calls_get_worker(mock_worker, mock_job_store):
    """Lifespan startup obtains the worker exactly once via ``get_worker``."""
    registry = ServiceRegistry()
    with (
        patch("core.api.app.get_job_store", return_value=mock_job_store),
        patch("core.api.app.get_worker", return_value=mock_worker) as gw_mock,
        patch("core.api.app.DspyService"),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False):
            gw_mock.assert_called_once()


def test_lifespan_requeues_pending_jobs(mock_worker):
    """Pending jobs surfaced by the store are forwarded to the worker."""
    pending = ["job-aaa", "job-bbb"]
    mock_js = _make_mock_job_store(pending_ids=pending)
    registry = ServiceRegistry()
    with (
        patch("core.api.app.get_job_store", return_value=mock_js),
        patch("core.api.app.get_worker", return_value=mock_worker) as gw_mock,
        patch("core.api.app.DspyService"),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False):
            _, kwargs = gw_mock.call_args
            assert kwargs.get("pending_optimization_ids") == pending


def test_health_live_worker_returns_200(app_client):
    """A live, recently-active worker yields a 200 from ``/health``."""
    client, mock_worker, _ = app_client
    mock_worker.threads_alive.return_value = True
    mock_worker.seconds_since_last_activity.return_value = 0.0

    resp = client.get("/health")

    assert resp.status_code == 200


def test_health_live_worker_returns_expected_shape(app_client):
    """``/health`` returns the standard status and registered-assets payload."""
    client, mock_worker, _ = app_client
    mock_worker.threads_alive.return_value = True
    mock_worker.seconds_since_last_activity.return_value = 0.0

    body = client.get("/health").json()

    assert body["status"] == HEALTH_STATUS_OK
    assert "registered_assets" in body


def test_health_stale_worker_returns_503(app_client):
    """An idle worker past the staleness threshold returns 503."""
    client, mock_worker, _ = app_client
    mock_worker.threads_alive.return_value = True
    # 601 s > default threshold of 600 s
    mock_worker.seconds_since_last_activity.return_value = 601.0

    resp = client.get("/health")

    assert resp.status_code == 503


def test_health_dead_worker_returns_503(app_client):
    """A worker reporting ``threads_alive=False`` returns 503."""
    client, mock_worker, _ = app_client
    mock_worker.threads_alive.return_value = False

    resp = client.get("/health")

    assert resp.status_code == 503


def test_health_no_worker_returns_503():
    """An app started without a worker reports unhealthy."""
    registry = ServiceRegistry()
    mock_js = _make_mock_job_store()
    null_worker = MagicMock()
    null_worker.threads_alive.return_value = False
    null_worker.seconds_since_last_activity.return_value = None

    with (
        patch("core.api.app.get_job_store", return_value=mock_js),
        patch("core.api.app.get_worker", return_value=null_worker),
        patch("core.api.app.DspyService"),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False) as client:
            resp = client.get("/health")
    assert resp.status_code == 503


def test_queue_with_worker_returns_200(app_client):
    """``/queue`` reports 200 whenever a worker is wired in."""
    client, _, _ = app_client
    resp = client.get("/queue")
    assert resp.status_code == 200


def test_queue_with_worker_returns_correct_shape(app_client):
    """``/queue`` propagates the worker's queue and thread counters."""
    client, mock_worker, _ = app_client
    mock_worker.queue_size.return_value = 3
    mock_worker.active_jobs.return_value = 1
    mock_worker.thread_count.return_value = 2
    mock_worker.threads_alive.return_value = True

    body = client.get("/queue").json()

    assert body["pending_jobs"] == 3
    assert body["active_jobs"] == 1
    assert body["worker_threads"] == 2
    assert body["workers_alive"] is True


def test_queue_without_worker_returns_zeros():
    """Without a worker, ``/queue`` returns zeroed counters and 200."""
    registry = ServiceRegistry()
    mock_js = _make_mock_job_store()

    with (
        patch("core.api.app.get_job_store", return_value=mock_js),
        patch("core.api.app.get_worker", return_value=None),
        patch("core.api.app.DspyService"),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False) as client:
            resp = client.get("/queue")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pending_jobs"] == 0
    assert body["workers_alive"] is False


def test_cache_headers_health_no_cache(app_client):
    """``/health`` responses must declare ``no-cache``."""
    client, _, _ = app_client
    resp = client.get("/health")
    cc = resp.headers.get("cache-control", "")
    assert "no-cache" in cc


def test_cache_headers_queue_private(app_client):
    """``/queue`` declares a short private cache window."""
    client, _, _ = app_client
    resp = client.get("/queue")
    cc = resp.headers.get("cache-control", "")
    assert "private" in cc
    assert "max-age=5" in cc


def test_cache_headers_models_public(app_client):
    """``/models`` declares the long public cache window."""
    client, _, _ = app_client
    resp = client.get("/models")
    cc = resp.headers.get("cache-control", "")
    assert "public" in cc
    assert "max-age=300" in cc


def test_cache_headers_health_differs_from_models(app_client):
    """Per-route cache policies are not collapsed into a single value."""
    client, mock_worker, _ = app_client
    mock_worker.threads_alive.return_value = True
    mock_worker.seconds_since_last_activity.return_value = 0.0

    health_cc = client.get("/health").headers.get("cache-control", "")
    models_cc = client.get("/models").headers.get("cache-control", "")

    assert health_cc != models_cc


def test_sigterm_handler_calls_worker_stop() -> None:
    """SIGTERM triggers a clean worker shutdown via ``worker.stop``."""
    registry = ServiceRegistry()
    mock_js = _make_mock_job_store()
    mock_w = _make_mock_worker()

    captured_handler: dict = {}

    def fake_signal(signum, handler):
        """Capture the SIGTERM handler so the test can invoke it manually."""
        if signum == signal.SIGTERM:
            captured_handler["fn"] = handler

    with (
        patch("core.api.app.get_job_store", return_value=mock_js),
        patch("core.api.app.get_worker", return_value=mock_w),
        patch("core.api.app.DspyService"),
        patch("core.api.app.signal.signal", side_effect=fake_signal),
        patch("core.api.app.threading.current_thread", return_value=__import__("threading").main_thread()),
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False):
            # Handler is registered during lifespan startup
            assert "fn" in captured_handler, "SIGTERM handler was never registered"
            # Invoke it directly as if the OS sent SIGTERM
            captured_handler["fn"](signal.SIGTERM, None)
            mock_w.stop.assert_called()


def test_openapi_public_returns_200(app_client):
    """The public OpenAPI spec endpoint serves a 200."""
    client, _, _ = app_client
    resp = client.get("/openapi.public.json")
    assert resp.status_code == 200


def test_openapi_public_returns_valid_json(app_client):
    """The public OpenAPI spec endpoint returns a JSON document with paths."""
    client, _, _ = app_client
    resp = client.get("/openapi.public.json")
    body = resp.json()
    assert isinstance(body, dict)
    assert "paths" in body


def test_openapi_public_only_exposes_allowlisted_paths(app_client):
    """Every path in the public spec is on the public allowlist."""
    client, _, _ = app_client
    paths = client.get("/openapi.public.json").json().get("paths", {})
    leaked = set(paths.keys()) - _SCALAR_PUBLIC_PATHS
    assert not leaked, f"Non-public paths leaked into public spec: {sorted(leaked)}"


def test_openapi_public_removes_empty_tag_groups(app_client) -> None:
    """Tags with no remaining operations are pruned from the public spec."""
    client, _, _ = app_client
    body = client.get("/openapi.public.json").json()
    paths = body.get("paths", {})
    tags_in_paths: set[str] = set()
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for op in methods.values():
            if isinstance(op, dict):
                for tag in op.get("tags", []):
                    tags_in_paths.add(tag)

    for tag_obj in body.get("tags", []):
        tag_name = tag_obj.get("name")
        assert tag_name in tags_in_paths, f"Tag {tag_name!r} is still in spec['tags'] but has no operations"


def test_openapi_public_retains_public_paths(app_client):
    """Public-facing paths remain in the public spec after filtering."""
    client, _, _ = app_client
    paths = client.get("/openapi.public.json").json().get("paths", {})
    assert "/health" in paths
    assert "/queue" in paths


def test_openapi_public_declares_bearer_security(app_client) -> None:
    """The public spec declares a bearer security scheme and a global requirement."""
    client, _, _ = app_client
    body = client.get("/openapi.public.json").json()
    schemes = body.get("components", {}).get("securitySchemes", {})
    assert any(s.get("type") == "http" and s.get("scheme") == "bearer" for s in schemes.values())
    assert body.get("security")


def test_openapi_public_strips_internal_agent_tag(app_client) -> None:
    """The internal ``agent`` MCP marker never leaks into the public spec."""
    client, _, _ = app_client
    body = client.get("/openapi.public.json").json()
    assert all(t.get("name") != "agent" for t in body.get("tags", []))
    for methods in body.get("paths", {}).values():
        if not isinstance(methods, dict):
            continue
        for op in methods.values():
            if isinstance(op, dict):
                assert "agent" not in op.get("tags", [])


@pytest.fixture
def worker_disabled_client(mock_job_store, monkeypatch):
    """Build a ``TestClient`` for an API-only pod (``WORKER_ENABLED=0``).

    The in-process worker is never constructed, so ``/health`` and ``/queue``
    must answer from the store alone.

    Yields:
        A 2-tuple of ``(client, job_store)``.
    """
    monkeypatch.setattr(app_settings, "worker_enabled", False)
    mock_job_store.get_queue_metrics.return_value = (0, 0.0)
    mock_job_store.count_jobs_by_status.return_value = {}
    registry = ServiceRegistry()
    with (
        patch("core.api.app.get_job_store", return_value=mock_job_store),
        patch("core.api.app.DspyService"),  # prevent real DSPy init
    ):
        application = create_app(registry=registry)
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client, mock_job_store


def test_health_worker_disabled_returns_200(worker_disabled_client):
    """An API-only pod is healthy without an in-process worker."""
    client, _ = worker_disabled_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == HEALTH_STATUS_OK


def test_queue_worker_disabled_reports_store_counts(worker_disabled_client):
    """Queue status comes from the shared store, and a fresh queue reads alive."""
    client, store = worker_disabled_client
    store.get_queue_metrics.return_value = (2, 30.0)
    store.count_jobs_by_status.return_value = {"running": 1, "validating": 1, "pending": 2}
    body = client.get("/queue").json()
    assert body["pending_jobs"] == 2
    assert body["active_jobs"] == 2
    assert body["worker_threads"] == 0
    assert body["workers_alive"] is True


def test_queue_worker_disabled_flags_stalled_queue(worker_disabled_client):
    """Pending work nobody claims past the stale threshold reads as not alive."""
    client, store = worker_disabled_client
    store.get_queue_metrics.return_value = (3, 100_000.0)
    body = client.get("/queue").json()
    assert body["workers_alive"] is False
