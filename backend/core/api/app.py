"""FastAPI application factory for Skynet.

This module now contains *only* the app-lifecycle concerns:

* DI wiring (``ServiceRegistry`` + ``DspyService`` + ``job_store``)
* Background worker lifespan + SIGTERM handling
* CORS middleware + cache headers
* Consistent JSON error response shape
* Two "infra" endpoints that reference the worker directly: ``/health``
  and ``/queue``
* Scalar API reference mounted at ``/scalar`` with air-gapped static assets

Every other route has been extracted into ``backend/core/api/routers/``
and wired up via ``app.include_router``. See ``AGENTS.md`` for the
extraction rules.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

try:
    from scalar_fastapi import AgentScalarConfig, DocumentDownloadType, get_scalar_api_reference
except ImportError:  # Optional dep: tests/CI can run without the Scalar docs UI installed.
    AgentScalarConfig = None  # type: ignore[assignment, misc]
    DocumentDownloadType = None  # type: ignore[assignment, misc]
    get_scalar_api_reference = None  # type: ignore[assignment]

from ..config import settings
from ..exceptions import AppError
from ..models import HEALTH_STATUS_OK, HealthResponse, QueueStatusResponse
from ..registry import ServiceRegistry
from ..registry.resolvers import (
    MODULE_ALIASES,
    OPTIMIZER_ALIASES,
    resolve_module_factory,
    resolve_optimizer_factory,
)
from ..service_gateway import DspyService
from ..service_gateway.embedding_pipeline import (
    backfill_missing_conversation_embeddings,
    backfill_missing_embeddings,
    purge_orphan_conversation_embeddings,
    purge_orphan_embeddings,
)
from ..storage import get_job_store
from ..worker.engine import BackgroundWorker, get_worker
from .directory_client import build_directory_client
from .errors import DomainError
from .mcp_mount import mount_mcp_on_app
from .model_catalog import prewarm_catalog
from .observability import (
    STARTUP_WORK_LOCK_KEY,
    advisory_lock,
    get_request_id,
    install_metrics,
    install_request_id_middleware,
    start_event_loop_lag_monitor,
    start_orphan_recovery_sweeper,
    start_queue_metrics_refresher,
    start_staged_dataset_sweeper,
    start_stale_conversation_sweeper,
)
from .routers.accounts import create_accounts_router
from .routers.admin import create_admin_router
from .routers.agent_history import create_agent_history_router
from .routers.analytics import create_analytics_router
from .routers.api_tokens import create_api_tokens_router
from .routers.code_agent import create_code_agent_router
from .routers.code_validation import create_code_validation_router
from .routers.dashboard import create_dashboard_router
from .routers.dataset_library import create_dataset_library_router
from .routers.dataset_share import create_dataset_share_router
from .routers.datasets import create_datasets_router
from .routers.generalist_agent import create_generalist_agent_router
from .routers.models import create_models_router
from .routers.optimizations import create_optimizations_router
from .routers.optimizations_meta import create_optimizations_meta_router
from .routers.registry import create_registry_router
from .routers.serve import create_serve_router
from .routers.share import create_share_router
from .routers.submissions import create_submissions_router
from .routers.usage import create_usage_router
from .routers.wizard import create_wizard_router

logger = logging.getLogger(__name__)

_SCALAR_STATIC_DIR = Path(__file__).parent / "static" / "scalar"

# Allowlist of routes shown in the *public* Scalar API reference. Every
# route is still live and still appears in /openapi.json — the filter is
# applied only to the copy served at /openapi.public.json, which is what
# Scalar fetches. The cut is aimed at developers integrating with Skynet:
# discover the available optimizers/modules/models, submit a job, poll
# status, fetch the artifact, run inference. Anything not on this list is
# treated as internal (dashboard plumbing, SSE streams, analytics
# aggregations, per-pair readers, admin tooling) and auto-hidden from the
# docs without any further action.
_SCALAR_PUBLIC_PATHS = frozenset(
    {
        # Service health
        "/health",
        "/queue",
        # Discover valid optimizers, modules, and models for a submission
        "/registry",
        "/models",
        # Submit work
        "/run",
        "/grid-search",
        # Browse + read results
        "/optimizations",
        "/optimizations/{optimization_id}",
        "/optimizations/{optimization_id}/summary",
        "/optimizations/{optimization_id}/logs",
        "/optimizations/{optimization_id}/payload",
        "/optimizations/{optimization_id}/artifact",
        "/optimizations/{optimization_id}/grid-result",
        # Live progress
        "/optimizations/{optimization_id}/stream",
        # Lifecycle
        "/optimizations/{optimization_id}/cancel",
        "/optimizations/{optimization_id}/clone",
        "/optimizations/{optimization_id}/retry",
        "/optimizations/{optimization_id}/resume",
        # Inference on a finished optimization
        "/serve/{optimization_id}",
        "/serve/{optimization_id}/info",
    }
)

# Custom CSS for Scalar that (a) hides UI chrome we don't want in an
# air-gapped deployment, (b) renders the Skynet logo next to the API
# title, and (c) retints the reference in the same warm beige/brown
# palette as the main app (see frontend/src/app/globals.css light mode).
_SCALAR_CUSTOM_CSS = """
/* Hide Generate/Connect MCP sidebar block — no-op in air-gapped mode */
.scalar-mcp-layer { display: none !important; }

/* Hide all toolbar popover buttons (Share, Deploy, Configure, Developer Tools).
   HeadlessUI IDs are dynamic, so match the stable prefix. */
[id^="headlessui-popover-button"] { display: none !important; }

/* Hide the version + OAS chips and the "Skynet" h1 from the intro */
.introduction-section .badge,
.introduction-section .section-header-label {
  display: none !important;
}

/* Sidebar: hide "Introduction" link (visible at page top anyway)
   and the auto-generated "Models" schemas section (duplicate of the
   tagged Models group — devs see schemas inline per endpoint). */
[data-sidebar-id="api-1/description/introduction"],
[data-sidebar-id="api-1/models"] {
  display: none !important;
}

/* Animate sidebar group expand/collapse — chevron rotates and
   child list slides open with height transition. */
.group\/group-button > button .size-4 {
  transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1);
}
.group\/group-button > button[aria-expanded="true"] .size-4 {
  transform: rotate(90deg);
}
.group\/group-button + ul,
.group\/group-button + div {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 250ms cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
}
.group\/group-button > button[aria-expanded="true"] ~ ul,
.group\/group-button > button[aria-expanded="true"] ~ div {
  grid-template-rows: 1fr;
}
.group\/group-button + ul > *,
.group\/group-button + div > * {
  min-height: 0;
}

/* Hide "Powered by Scalar" footer and dark mode toggle in sidebar */
.t-doc__sidebar [href*="scalar.com"],
.t-doc__sidebar [href*="scalar.com"] ~ button,
.t-doc__header [href*="scalar.com"],
.t-doc__header [href*="scalar.com"] ~ button {
  display: none !important;
}

/* Remove the empty client-libraries / server-url bar above the
   description so the intro paragraph sits higher on the page. */
.introduction-section .custom-scroll {
  display: none !important;
}
.introduction-section .section-column {
  padding-top: 0 !important;
}
.introduction-section {
  padding-top: 16px !important;
  gap: 0 !important;
}

/* ── Toolbar brand button ─────────────────────────────────────────
   The brand button sits in the top-left of the toolbar. When the
   sidebar is hidden, hovering the button fades the SKYNET wordmark
   out and reveals a sidebar-toggle icon in the same spot — click to
   open. When the sidebar is open, the brand button is hidden entirely
   (the wordmark + close button move into the sidebar header instead).
   This mirrors chatgpt.com's sidebar UX. */
.api-reference-toolbar {
  position: sticky !important;
  top: 0;
  z-index: 40;
}

.skynet-toolbar-brand {
  position: absolute;
  left: 16px;
  top: 50%;
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 32px;
  padding: 4px 8px;
  border: 0;
  background: transparent;
  color: #3D2E22;
  cursor: pointer;
  border-radius: 8px;
  transition: background-color 140ms ease;
  overflow: visible;
}
.skynet-toolbar-brand:hover { background: #f0ebe4; }
.skynet-toolbar-brand:focus-visible {
  outline: 2px solid #3D2E22;
  outline-offset: 2px;
}

.skynet-toolbar-brand .skynet-toolbar-icon {
  flex-shrink: 0;
  display: block;
}
.skynet-toolbar-brand .skynet-wordmark {
  flex-shrink: 0;
}
.skynet-wordmark { color: #3D2E22; user-select: none; }
.skynet-wordmark svg { overflow: visible; }

/* When the sidebar is open, the brand moves into the sidebar header */
html[data-skynet-sidebar="visible"] .skynet-toolbar-brand {
  display: none;
}

/* ── Sidebar header (only shown while sidebar is visible) ─────────── */
.skynet-sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 10px 10px 16px;
  border-bottom: 1px solid #ddd6cc;
  flex-shrink: 0;
}
html[data-skynet-sidebar="hidden"] .skynet-sidebar-header {
  display: none;
}

.skynet-sidebar-home {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 8px;
  color: #3D2E22;
  text-decoration: none;
  transition: background-color 120ms ease;
}
.skynet-sidebar-home:hover { background: #ede7dd; }
.skynet-sidebar-home:focus-visible {
  outline: 2px solid #3D2E22;
  outline-offset: 2px;
}

.skynet-sidebar-close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  border: 0;
  background: transparent;
  color: #3D2E22;
  cursor: pointer;
  border-radius: 8px;
  transition: background-color 120ms ease;
}
.skynet-sidebar-close:hover { background: #ede7dd; }
.skynet-sidebar-close:focus-visible {
  outline: 2px solid #3D2E22;
  outline-offset: 2px;
}

/* ── Animated collapsible sidebar (desktop only) ───────────────────
   Only override Scalar's grid on viewports wide enough for the
   sidebar. Below 1024px Scalar's own responsive layout takes over. */
@media (min-width: 1024px) {
  .references-layout {
    grid-template-columns: 288px minmax(0, 1fr) !important;
    transition: grid-template-columns 320ms cubic-bezier(0.4, 0, 0.2, 1);
  }
  .t-doc__sidebar {
    min-width: 0 !important;
    transition:
      transform 320ms cubic-bezier(0.4, 0, 0.2, 1),
      opacity 220ms ease;
  }
  html[data-skynet-sidebar="hidden"] .references-layout {
    grid-template-columns: 0px minmax(0, 1fr) !important;
    overflow: hidden;
  }
  html[data-skynet-sidebar="hidden"] .t-doc__sidebar {
    transform: translateX(-100%);
    opacity: 0;
    pointer-events: none;
    width: 0 !important;
    min-width: 0 !important;
    overflow: hidden;
  }
}

/* Let the content fill the full width on wide viewports */
.scalar-api-reference {
  --refs-content-max-width: none !important;
}

/* Skynet warm-beige light theme to match the main app */
.light-mode {
  --scalar-background-1: #faf8f5;
  --scalar-background-2: #f5f1ec;
  --scalar-background-3: #f0ebe4;
  --scalar-background-accent: #ede7dd;
  --scalar-background-card: #ffffff;
  --scalar-color-1: #1c1612;
  --scalar-color-2: #5c4f42;
  --scalar-color-3: #8c7a6b;
  --scalar-color-accent: #3d2e22;
  --scalar-border-color: #ddd6cc;
}
.light-mode .t-doc__sidebar {
  --scalar-sidebar-background-1: #f5f1ec;
  --scalar-sidebar-item-hover-background: #ede7dd;
  --scalar-sidebar-item-active-background: #e3dcd0;
  --scalar-sidebar-border-color: #ddd6cc;
  --scalar-sidebar-color-1: #1c1612;
  --scalar-sidebar-color-2: #8c7a6b;
  --scalar-sidebar-color-active: #3d2e22;
  --scalar-sidebar-search-background: #ffffff;
  --scalar-sidebar-search-border-color: #ddd6cc;
  --scalar-sidebar-search--color: #8c7a6b;
}

/* ── Mobile sidebar slide animation ────────────────────────────────
   Scalar's mobile menu lives inside .t-doc__header (NOT .t-doc__sidebar).
   When the hamburger is tapped, the header expands to full screen.
   We animate that expansion with a slide-in from the left. The desktop
   .t-doc__sidebar stays hidden on mobile (Tailwind's `hidden` class). */
@media (max-width: 1023px) {
  .skynet-toolbar-brand { display: none !important; }

  .t-doc__header {
    transform-origin: left top;
    transition:
      transform 320ms cubic-bezier(0.4, 0, 0.2, 1),
      opacity 200ms ease;
  }

  /* When the mobile menu is closed, the header is just the toolbar bar */
  .references-layout:not(.references-sidebar-mobile-open) .t-doc__header {
    /* no extra styles — Scalar handles the collapsed state */
  }

  /* When open, slide in from left */
  .references-sidebar-mobile-open .t-doc__header {
    animation: skynet-slide-in 320ms cubic-bezier(0.4, 0, 0.2, 1) both;
  }
}

@keyframes skynet-slide-in {
  from {
    transform: translateX(-40px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
@media (max-width: 1023px) and (min-width: 769px) {
  .section-container,
  .endpoint-container {
    padding-inline: 16px !important;
  }
}
@media (max-width: 768px) {
  .section-container,
  .endpoint-container {
    padding-inline: 12px !important;
  }
  .api-reference-toolbar {
    padding-inline: 8px !important;
  }
}

/* ── Cross-platform viewport adaptation ──────────────────────────
   Reserve scrollbar width on Windows (classic scrollbars) to prevent
   layout shift. On macOS overlay scrollbars this is a no-op. */
.scalar-api-reference {
  scrollbar-gutter: stable;
}

/* Fluid padding for narrow desktop windows (split-screen, snapped) */
@media (max-width: 600px) {
  .section-container,
  .endpoint-container {
    padding-inline: 8px !important;
  }
  /* Prevent code blocks from overflowing */
  pre, code {
    font-size: 12px !important;
    word-break: break-all;
  }
}

/* Sensible content width on ultrawides so lines stay readable */
@media (min-width: 1800px) {
  .scalar-api-reference {
    --refs-content-max-width: 1600px !important;
  }
}
"""

_OPENAPI_TAGS = [
    {"name": "Optimizations", "description": "Submit, list, inspect, and manage DSPy optimization jobs."},
    {"name": "Inference", "description": "Run inference against an optimized program."},
    {"name": "Analytics", "description": "Aggregate stats across jobs, optimizers, and models."},
    {"name": "Models", "description": "Model catalog and provider discovery."},
    {"name": "Registry", "description": "Optimizers and modules you can reference by name in a submission."},
    {"name": "Datasets", "description": "Profile uploaded datasets and recommend split plans."},
    {"name": "Code Validation", "description": "Format and validate user-supplied Python code."},
    {"name": "System", "description": "Health and queue status for readiness probes."},
]


def _graceful_shutdown_handler(
    worker: BackgroundWorker | None,
    original_handler: Any,
    signum: int,
    frame: FrameType | None,
) -> None:
    """Stop the background worker on SIGTERM, then chain to the original handler.

    Any prior handler that is neither ``SIG_DFL`` nor ``SIG_IGN`` is invoked
    so an outer process supervisor can still respond.

    Args:
        worker: The background worker to stop, or ``None`` when not yet started.
        original_handler: The previously installed SIGTERM handler.
        signum: The signal number forwarded by the runtime.
        frame: The current stack frame at signal delivery, or ``None``.
    """
    logger.info("SIGTERM received, stopping worker gracefully")
    if worker:
        worker.stop()
    if callable(original_handler) and original_handler not in (signal.SIG_DFL, signal.SIG_IGN):
        original_handler(signum, frame)


def _format_validation_loc(loc: Iterable[Any]) -> str:
    """Flatten a pydantic ``loc`` tuple into a dotted path.

    Translates e.g. ``("body", "items", 0, "name")`` → ``"items[0].name"``.
    The synthetic ``"body"`` / ``"__root__"`` entries pydantic emits at the
    top of request-body errors are stripped so the path matches the user's
    actual schema.

    Args:
        loc: Iterable of path segments (strings and ints) from pydantic's error.

    Returns:
        A dotted path string, or ``"body"`` when only synthetic prefixes were present.
    """
    parts: list[str] = []
    for entry in loc:
        if entry in {"body", "__root__"}:
            continue
        if isinstance(entry, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{entry}]"
            else:
                parts.append(f"[{entry}]")
        else:
            parts.append(str(entry))
    return ".".join(parts) if parts else "body"


def _cache_control_for(path: str, query_string: bytes) -> str | None:
    """Return the ``Cache-Control`` value for a GET path, or ``None`` to skip.

    Args:
        path: Request path.
        query_string: Raw query-string bytes (used to detect a ``status`` filter
            on the job list, which must not be cached).

    Returns:
        The ``Cache-Control`` header value, or ``None`` when the path is not
        cached.
    """
    if path == "/models":
        return "public, max-age=300, stale-while-revalidate=600"
    if path == "/health":
        return "no-cache, max-age=0"
    if path == "/queue":
        return "private, max-age=5"
    if path == "/optimizations" and b"status" not in query_string:
        return "private, max-age=2, stale-while-revalidate=10"
    return None


class CacheControlMiddleware:
    """Attach ``Cache-Control`` to selected GET endpoints.

    Pure ASGI rather than ``@app.middleware("http")`` (which is
    ``BaseHTTPMiddleware``) so it never buffers response bodies and stays out of
    the way of SSE/streaming endpoints — see :class:`RequestIDMiddleware` for the
    framing-corruption this avoids.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app`` to stamp cache headers on matching GET responses.

        Args:
            app: The downstream ASGI application.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stamp ``Cache-Control`` on matching GET responses; pass others through.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] != "http" or scope["method"] != "GET":
            await self.app(scope, receive, send)
            return
        cache_value = _cache_control_for(scope["path"], scope.get("query_string", b""))
        if cache_value is None:
            await self.app(scope, receive, send)
            return

        async def send_with_cache(message: Message) -> None:
            """Stamp the resolved ``Cache-Control`` value on the response start message."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = cache_value
            await send(message)

        await self.app(scope, receive, send_with_cache)


class _SSEFriendlyGZipResponder(GZipResponder):
    """GZip responder that forwards ``text/event-stream`` bodies verbatim.

    Starlette's stock responder compresses streaming bodies chunk-by-chunk,
    which would buffer every SSE flush. This bypasses compression entirely once
    the response declares an event-stream content type.
    """

    def __init__(self, app: ASGIApp, minimum_size: int, compresslevel: int = 9) -> None:
        """Track whether the in-flight response opted out of compression.

        Args:
            app: The downstream ASGI application.
            minimum_size: Smallest body (bytes) still worth compressing.
            compresslevel: zlib level forwarded to the gzip stream.
        """
        super().__init__(app, minimum_size, compresslevel=compresslevel)
        self._passthrough = False

    async def send_with_gzip(self, message: Message) -> None:
        """Forward SSE responses raw; defer everything else to the gzip path.

        Args:
            message: The outgoing ASGI response message.
        """
        if message["type"] == "http.response.start":
            content_type = Headers(raw=message["headers"]).get("content-type", "")
            self._passthrough = content_type.startswith("text/event-stream")
        if self._passthrough:
            await self.send(message)
            return
        await super().send_with_gzip(message)


class SSEFriendlyGZipMiddleware(GZipMiddleware):
    """``GZipMiddleware`` variant that never buffers ``text/event-stream``.

    Compresses the large, highly repetitive trajectory/log JSON the read
    endpoints return while leaving the dashboard / per-job / agent SSE streams
    flushing incrementally and uncompressed.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Gzip compressible HTTP responses; pass SSE and non-HTTP through.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.
        """
        if scope["type"] == "http" and "gzip" in Headers(scope=scope).get("Accept-Encoding", ""):
            responder = _SSEFriendlyGZipResponder(self.app, self.minimum_size, compresslevel=self.compresslevel)
            await responder(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app(
    registry: ServiceRegistry | None = None,
    *,
    service: DspyService | None = None,
    service_kwargs: dict | None = None,
) -> FastAPI:
    """Assemble and return the fully-configured FastAPI application.

    Args:
        registry: Optional pre-built :class:`ServiceRegistry`; a fresh one is
            created when omitted.
        service: Optional pre-built :class:`DspyService`; constructed from
            ``registry`` and ``service_kwargs`` when omitted.
        service_kwargs: Optional kwargs forwarded to :class:`DspyService`
            when building the default instance.

    Returns:
        The fully wired :class:`FastAPI` application.
    """
    registry = registry or ServiceRegistry()
    # Mirror the alias-backed factories from the resolver into the registry
    # so ``get_registry_snapshot`` reflects the actually-supported names
    # rather than empty lists. Skipped on names already registered so a
    # pre-built ``registry`` (tests, custom deployments) wins.
    for _alias in OPTIMIZER_ALIASES:
        if _alias not in registry.optimizers:
            registry.register_optimizer(_alias, resolve_optimizer_factory(_alias))
    for _alias in MODULE_ALIASES:
        if _alias not in registry.modules:
            _factory, _ = resolve_module_factory(_alias)
            registry.register_module(_alias, _factory)
    service = service or DspyService(
        registry,
        **(service_kwargs or {}),
    )

    job_store = get_job_store()

    worker: BackgroundWorker | None = None
    queue_metrics_refresher = None
    orphan_sweeper = None
    stale_conversation_sweeper = None
    staged_dataset_sweeper = None
    loop_lag_monitor = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Recover orphan jobs, start the background worker, and install a SIGTERM handler.

        Args:
            app: The FastAPI application whose lifespan is being managed.

        Yields:
            ``None`` once the worker is running; stops the worker on exit.
        """
        nonlocal \
            loop_lag_monitor, \
            orphan_sweeper, \
            queue_metrics_refresher, \
            staged_dataset_sweeper, \
            stale_conversation_sweeper, \
            worker
        # Reclaim jobs whose worker lease has expired. Under multi-pod scaling
        # this only fails rows whose ``lease_expires_at`` is in the past — a
        # peer pod's in-flight job is not orphaned and is left alone. The
        # periodic sweeper (below) covers steady-state; this one-shot keeps
        # boot latency low when a single replica restarts.
        job_store.recover_orphaned_jobs()
        # ``recover_pending_jobs`` is no longer required for correctness because
        # any pod can claim a pending row via ``claim_next_job`` on its next
        # tick, but we still pass the IDs as a same-pod hint so a fresh restart
        # resumes work without waiting a full poll interval.
        pending_ids = job_store.recover_pending_jobs()
        worker = get_worker(job_store, service=service, pending_optimization_ids=pending_ids)
        queue_metrics_refresher = start_queue_metrics_refresher(job_store)
        # Without a periodic sweeper, a pod that dies mid-job leaves its
        # leases ticking down with no peer scanning for them — the row is
        # only requeued at the next pod restart. The sweeper is gated by a
        # Postgres advisory lock so only one replica per tick does the work.
        orphan_sweeper = start_orphan_recovery_sweeper(job_store)
        stale_conversation_sweeper = start_stale_conversation_sweeper(getattr(job_store, "engine", None))
        staged_dataset_sweeper = start_staged_dataset_sweeper(getattr(job_store, "engine", None))
        if settings.event_loop_lag_monitor_enabled:
            loop_lag_monitor = start_event_loop_lag_monitor()
            logger.info("Event-loop lag monitor enabled (threshold %.0fms)", settings.event_loop_lag_threshold_ms)
        if pending_ids:
            logger.info("Re-queued %d pending jobs from previous run (local hint)", len(pending_ids))
        logger.info("Background worker started")

        # Probe every provider's /v1/models endpoint in a background thread
        # so the catalog is hot by the time the first agent turn arrives.
        # Without this the first ``list_models_for_agent`` request waits
        # ~15-20s for the cold-cache parallel probe to settle.
        prewarm_catalog()

        # Embedding maintenance below is idempotent but redundantly expensive
        # across a rolling deploy. The advisory lock makes one replica the
        # leader so the work runs once per restart, not once per pod. Peers
        # skip the block and trust the leader's writes (visible in the
        # shared DB before they begin serving traffic).
        engine = getattr(job_store, "engine", None)
        with advisory_lock(engine, STARTUP_WORK_LOCK_KEY) as is_startup_leader:
            if is_startup_leader:
                # The explore search vector is embedded on a daemon thread
                # when a job succeeds; a crashed thread (LLM creds, API blip)
                # leaves the row missing forever and the job silently drops
                # out of search. A startup backfill drains the gap; the orphan
                # purge cleans up rows whose job was deleted before the
                # cascade into job_embeddings completed.
                try:
                    purge_orphan_embeddings(job_store)
                    queued = backfill_missing_embeddings(job_store)
                    if queued:
                        logger.info("Embedding backfill queued for %d job(s)", queued)
                except Exception as exc:
                    logger.warning("Embedding backfill scan failed: %s", exc)
                if engine is not None:
                    try:
                        purge_orphan_conversation_embeddings(engine)
                        conv_queued = backfill_missing_conversation_embeddings(engine)
                        if conv_queued:
                            logger.info(
                                "Conversation embedding backfill queued for %d row(s)",
                                conv_queued,
                            )
                    except Exception as exc:
                        logger.warning("Conversation embedding backfill scan failed: %s", exc)
            else:
                logger.info("Embedding maintenance skipped — peer replica is leader")

        # SIGTERM handler can only be registered on the main interpreter
        # thread. ``threading.current_thread()`` lets us detect when the
        # lifespan is running inside a worker thread (e.g. uvicorn reload).
        can_register_signal = threading.current_thread() is threading.main_thread()
        original_handler = signal.getsignal(signal.SIGTERM) if can_register_signal else None

        if can_register_signal:
            signal.signal(
                signal.SIGTERM,
                partial(_graceful_shutdown_handler, worker, original_handler),
            )

        try:
            yield
        finally:
            if can_register_signal and original_handler is not None:
                signal.signal(signal.SIGTERM, original_handler)
            if worker:
                worker.stop()
                logger.info("Background worker stopped")
            if queue_metrics_refresher:
                queue_metrics_refresher.stop()
            if orphan_sweeper:
                orphan_sweeper.stop()
            if stale_conversation_sweeper:
                stale_conversation_sweeper.stop()
            if staged_dataset_sweeper:
                staged_dataset_sweeper.stop()
            if loop_lag_monitor:
                loop_lag_monitor.stop()

    app = FastAPI(
        title="Skynet",
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
        servers=[{"url": "/", "description": "This server"}],
        # Disable FastAPI's built-in Swagger UI and ReDoc — Scalar at
        # /scalar replaces both. /openapi.json stays enabled so Scalar
        # and any other consumers can still fetch the spec.
        docs_url=None,
        redoc_url=None,
        description=(
            "Backend API for **Skynet** — a platform for submitting, "
            "comparing, and serving optimized language-model "
            "programs. Submit optimization runs or grid searches, stream "
            "live progress and metrics, inspect per-example baselines vs. "
            "optimized predictions, and run inference against the final "
            "compiled program. All traffic stays on your local network; "
            "this interactive reference is served directly from the "
            "running backend and works fully offline."
        ),
    )

    # Mount before middleware so the instrumentator wraps every request.
    install_metrics(app)
    install_request_id_middleware(app)

    app.mount(
        "/scalar-static",
        StaticFiles(directory=_SCALAR_STATIC_DIR),
        name="scalar-static",
    )

    @app.get("/openapi.public.json", include_in_schema=False)
    async def public_openapi() -> JSONResponse:
        """Filtered copy of /openapi.json restricted to the public dev surface.

        Returns:
            A :class:`JSONResponse` containing the trimmed OpenAPI document.
        """
        base = app.openapi()
        spec = deepcopy(base)
        paths = spec.get("paths", {})
        for path in list(paths.keys()):
            if path not in _SCALAR_PUBLIC_PATHS:
                del paths[path]
        used_tags: set[str] = set()
        for methods in paths.values():
            if not isinstance(methods, dict):
                continue
            for op in methods.values():
                if not isinstance(op, dict):
                    continue
                # ``agent`` is an internal MCP-projection marker, not a public
                # docs group; strip it so each endpoint renders once under its
                # real section instead of also appearing in an undocumented
                # ``agent`` group.
                if "tags" in op:
                    op["tags"] = [t for t in op["tags"] if t != "agent"]
                for tag in op.get("tags", []):
                    used_tags.add(tag)
        if "tags" in spec:
            spec["tags"] = [t for t in spec["tags"] if t.get("name") in used_tags]
        # Declare bearer auth on the public surface so the Scalar reference shows
        # an Authorize button and Try-It sends the token, instead of every call
        # returning 401 against an otherwise-unauthenticated-looking spec.
        spec.setdefault("components", {})["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        spec["security"] = [{"bearerAuth": []}]
        return JSONResponse(spec)

    @app.get("/scalar", include_in_schema=False)
    async def scalar_docs() -> HTMLResponse:
        """Render the Scalar API reference HTML, patched with the animated wordmark script.

        Returns:
            An :class:`HTMLResponse` with the patched Scalar reference page.

        Raises:
            HTTPException: 503 when ``scalar-fastapi`` is not installed.
        """
        if get_scalar_api_reference is None:
            raise HTTPException(status_code=503, detail="scalar-fastapi is not installed")
        base = get_scalar_api_reference(
            openapi_url="/openapi.public.json",
            title=f"{app.title} API",
            scalar_favicon_url="/scalar-static/favicon.svg",
            scalar_js_url="/scalar-static/standalone.js",
            agent=AgentScalarConfig(disabled=True),
            telemetry=False,
            document_download_type=DocumentDownloadType.NONE,
            persist_auth=True,
            default_open_all_tags=False,
            dark_mode=False,
            custom_css=_SCALAR_CUSTOM_CSS,
        )
        # Inject the animated wordmark script. scalar-fastapi only exposes
        # custom_css, so we append our own <script> tag just before </body>.
        html = bytes(base.body).decode("utf-8")
        script_tag = '<script src="/scalar-static/wordmark.js" defer></script>'
        html = html.replace("</body>", f"{script_tag}</body>")
        return HTMLResponse(content=html)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Error envelope follows RFC 9457 (Problem Details) with stable English
    # ``detail`` strings and a ``code`` extension member for client-side i18n.
    # Legacy ``error`` field is retained for backwards compatibility.
    status_to_error_type = {
        400: "validation_error",
        404: "not_found",
        409: "conflict",
        422: "invalid_request",
        500: "internal_error",
        503: "service_unavailable",
    }

    def _problem_response(
        request: Request,
        *,
        status: int,
        error_type: str,
        detail: object,
        code: str | None = None,
        params: dict[str, Any] | None = None,
        title: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        """Build an RFC 9457 ``application/problem+json`` response.

        Args:
            request: The incoming HTTP request used to populate ``instance``.
            status: HTTP status code to return.
            error_type: Legacy ``error`` slug retained for backward compatibility.
            detail: Stable English string (or list, for validation issues).
            code: Optional i18n key identifying the problem class.
            params: Optional substitution params attached to ``code``.
            title: Optional human-readable title; derived from ``code`` when omitted.
            headers: Optional response headers (e.g. ``WWW-Authenticate``).

        Returns:
            A :class:`JSONResponse` with media type ``application/problem+json``.
        """
        body: dict[str, Any] = {
            "type": f"https://errors.skynet.app/{code}" if code else "about:blank",
            "title": title
            or (
                code.replace("_", " ").replace(".", " ").capitalize()
                if code
                else error_type.replace("_", " ").capitalize()
            ),
            "status": status,
            "detail": detail,
            "instance": request.url.path,
            "trace_id": get_request_id(),
            "error": error_type,
        }
        if code:
            body["code"] = code
            body["params"] = params or {}
        return JSONResponse(
            status_code=status,
            content=body,
            headers=headers,
            media_type="application/problem+json",
        )

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Serialize ``AppError`` instances to the RFC 9457 problem envelope.

        Args:
            request: The incoming HTTP request.
            exc: The raised :class:`AppError` to render.

        Returns:
            A :class:`JSONResponse` with the shared error envelope.
        """
        return _problem_response(
            request,
            status=exc.status_code,
            error_type=exc.error_code.lower(),
            detail=exc.message,
            code=getattr(exc, "code", None),
            params=getattr(exc, "params", None),
        )

    @app.exception_handler(HTTPException)
    async def _http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Serialize Starlette ``HTTPException`` into the RFC 9457 problem envelope.

        Args:
            request: The incoming HTTP request.
            exc: The raised :class:`HTTPException` to render.

        Returns:
            A :class:`JSONResponse` with the shared error envelope.
        """
        return _problem_response(
            request,
            status=exc.status_code,
            error_type=status_to_error_type.get(exc.status_code, "error"),
            detail=exc.detail,
            code=getattr(exc, "code", None),
            params=getattr(exc, "params", None),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Convert FastAPI ``RequestValidationError`` into a list of ``{field, message, type}`` issues.

        Args:
            request: The incoming HTTP request.
            exc: The raised :class:`RequestValidationError`.

        Returns:
            A 422 :class:`JSONResponse` with one entry per failed field.
        """
        issues = [
            {
                "field": _format_validation_loc(error.get("loc", [])),
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "validation_error"),
            }
            for error in exc.errors()
        ]
        return _problem_response(
            request,
            status=422,
            error_type="invalid_request",
            detail=issues,
            title="Invalid request",
        )

    @app.exception_handler(Exception)
    async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler that logs unhandled exceptions and returns a 500 envelope.

        Args:
            request: The incoming HTTP request.
            exc: The unhandled exception.

        Returns:
            A 500 :class:`JSONResponse` with a generic ``internal_error`` envelope.
        """
        logger.error(
            "Unhandled exception in %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return _problem_response(
            request,
            status=500,
            error_type="internal_error",
            detail="An internal server error occurred. Please contact support.",
            title="Internal server error",
        )

    worker_stale_threshold = float(os.getenv("WORKER_STALE_THRESHOLD", "600"))

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Liveness and readiness probe",
    )
    def healthcheck() -> HealthResponse:
        """Return registered-asset inventory and worker health for liveness/readiness probes.

        Returns:
            A :class:`HealthResponse` snapshot of the registered assets.

        Raises:
            DomainError: 503 when workers are dead or stuck longer than
                ``WORKER_STALE_THRESHOLD`` seconds (default 600).
        """
        if worker is None or not worker.threads_alive():
            logger.error("Health check failed: worker threads are not alive")
            raise DomainError("health.workers_dead", status=503)

        stale_seconds = worker.seconds_since_last_activity()
        if stale_seconds is not None and stale_seconds > worker_stale_threshold:
            stack_dump = worker.dump_thread_stacks()
            logger.error(
                "Health check failed: workers stuck for %.0fs. Thread stacks:\n%s",
                stale_seconds,
                stack_dump,
            )
            raise DomainError(
                "health.workers_stuck",
                status=503,
                seconds=f"{stale_seconds:.0f}",
            )

        snapshot = registry.snapshot()
        logger.debug("Health check requested; registered assets: %s", snapshot)
        return HealthResponse(
            status=HEALTH_STATUS_OK,
            registered_assets=snapshot,
            vector_search_enabled=getattr(job_store, "vector_search_enabled", None),
        )

    app.add_middleware(CacheControlMiddleware)
    # Outermost so it compresses the fully-assembled response body. Skips
    # text/event-stream to keep SSE streams flushing incrementally.
    app.add_middleware(SSEFriendlyGZipMiddleware, minimum_size=1024)

    @app.get(
        "/queue",
        response_model=QueueStatusResponse,
        tags=["System"],
        summary="Current worker queue depth and health",
    )
    def get_queue_status() -> QueueStatusResponse:
        """Return pending/active job counts and worker-thread health.

        All counts are zero before the lifespan context starts.

        Returns:
            A :class:`QueueStatusResponse` describing the current worker queue.
        """
        if worker is None:
            return QueueStatusResponse(
                pending_jobs=0,
                active_jobs=0,
                worker_threads=0,
                workers_alive=False,
            )

        return QueueStatusResponse(
            pending_jobs=worker.queue_size(),
            active_jobs=worker.active_jobs(),
            worker_threads=worker.thread_count(),
            workers_alive=worker.threads_alive(),
        )

    # Expose the store on app.state so the auth dependency can resolve
    # personal access tokens against the api_tokens table at request time —
    # the only request-reachable handle to the store (routers otherwise get it
    # via closure, which a module-level dependency can't reach).
    app.state.job_store = job_store
    app.include_router(create_models_router(), tags=["Models"])
    app.include_router(
        create_admin_router(job_store=job_store, directory_client=build_directory_client()),
        tags=["Admin"],
    )
    app.include_router(create_registry_router(registry=registry), tags=["Registry"])
    app.include_router(create_code_validation_router(), tags=["Code Validation"])
    app.include_router(create_code_agent_router(), tags=["Code Validation"])
    app.include_router(create_generalist_agent_router(job_store=job_store), tags=["Optimizations"])
    app.include_router(create_agent_history_router(job_store=job_store), tags=["Optimizations"])
    app.include_router(create_api_tokens_router(job_store=job_store), tags=["Settings"])
    app.include_router(create_accounts_router(job_store=job_store), tags=["Auth"])
    app.include_router(create_datasets_router(job_store=job_store), tags=["Datasets"])
    app.include_router(create_dataset_library_router(job_store=job_store), tags=["Datasets"])
    app.include_router(create_dataset_share_router(job_store=job_store), tags=["Datasets"])
    app.include_router(create_usage_router(job_store=job_store), tags=["Settings"])
    app.include_router(create_wizard_router(), tags=["Wizard"])
    app.include_router(
        create_submissions_router(service=service, job_store=job_store),
        tags=["Optimizations"],
    )
    app.include_router(create_analytics_router(job_store=job_store), tags=["Analytics"])
    app.include_router(
        create_optimizations_router(job_store=job_store, get_worker_ref=lambda: worker),
        tags=["Optimizations"],
    )
    app.include_router(create_serve_router(job_store=job_store), tags=["Inference"])
    app.include_router(create_share_router(job_store=job_store), tags=["Optimizations"])
    app.include_router(
        create_dashboard_router(job_store=job_store),
        tags=["Dashboard"],
    )
    app.include_router(
        create_optimizations_meta_router(job_store=job_store),
        tags=["Optimizations"],
    )

    # Mount FastMCP AFTER routers so from_fastapi() sees the full route
    # table. Only endpoints tagged "agent" are projected into tools; every
    # other REST route is unaffected. The sub-app handles Streamable HTTP
    # at POST /mcp. Tolerated import error so the backend still boots in
    # environments where the dep hasn't been reinstalled yet.
    try:
        mount_mcp_on_app(app)
    except ImportError as exc:
        logger.warning("FastMCP not available, skipping /mcp mount: %s", exc)

    return app
