"""Cross-optimization read routes: list, counts, and sidebar. [MIXED]

Public dev surface (in ``_SCALAR_PUBLIC_PATHS``):
- ``GET /optimizations`` — list jobs.

Internal (dashboard plumbing, hidden from public docs):
- ``GET /optimizations/counts`` — sidebar badge totals.
- ``GET /optimizations/sidebar`` — left-rail listing for the dashboard.
- ``GET /optimizations/shared-with-me`` — runs others shared with the caller.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ....constants import (
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_TOTAL_PAIRS,
    PAYLOAD_OVERVIEW_USERNAME,
)
from ....models import (
    OptimizationCountsResponse,
    PaginatedJobsResponse,
)
from ...auth import AuthenticatedUser, get_authenticated_user, is_admin
from ...converters import parse_overview, parse_timestamp
from ...errors import DomainError
from ...response_limits import AGENT_DEFAULT_LIST, AGENT_MAX_LIST, clamp_limit
from .._helpers import build_summary, grant_roles_for, resumable_id_flags
from ..constants import VALID_OPTIMIZATION_TYPES, VALID_STATUSES
from .schemas import SidebarJobItem, SidebarJobsResponse

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def _scope_username(current_user: AuthenticatedUser, requested: str | None) -> str | None:
    """Resolve the effective ``username`` filter for a list/count/sidebar query.

    Non-admins are forced to their own username regardless of what they
    pass. Admins can either filter to a single user (when ``requested`` is
    given) or see everyone (``None``).

    Args:
        current_user: Authenticated caller.
        requested: Raw ``username`` query parameter from the request.

    Returns:
        The username filter to forward to the job store.
    """
    if is_admin(current_user):
        if requested is None or not requested.strip():
            return None
        return requested.strip().lower()
    return current_user.username


def register_listing_routes(router: APIRouter, *, job_store) -> None:
    """Register cross-optimization read routes on ``router``.

    Args:
        router: The router to attach the listing routes to.
        job_store: Job-store the routes read from.
    """

    @router.get(
        "/optimizations",
        response_model=PaginatedJobsResponse,
        summary="List optimizations with filtering and pagination",
        tags=["agent"],
    )
    def list_jobs(
        current_user: AuthenticatedUserDep,
        status: str | None = Query(
            default=None,
            description="Exact-match status filter: pending, validating, running, success, failed, cancelled",
        ),
        username: str | None = Query(default=None, description="Only include optimizations submitted by this user"),
        optimization_type: str | None = Query(
            default=None, description="'run' (single optimization) or 'grid_search' (model-pair sweep)"
        ),
        limit: int = Query(
            default=AGENT_DEFAULT_LIST,
            ge=1,
            le=AGENT_MAX_LIST,
            description=(
                f"Page size (default {AGENT_DEFAULT_LIST}, ceiling {AGENT_MAX_LIST}). "
                "Use the default unless the user explicitly asks for fewer; "
                "paginate with offset."
            ),
        ),
        offset: int = Query(
            default=0,
            ge=0,
            description="Number of optimizations to skip before returning; combine with limit for stable pagination",
        ),
        include_shared: bool = Query(
            default=False,
            description=(
                "Also include runs shared with the caller via a member grant "
                "(Drive-style sharing). Non-admins only; admins already see all."
            ),
        ),
    ) -> PaginatedJobsResponse:
        """Return a page of optimizations ordered by ``created_at`` descending.

        Filters combine with AND. Non-admins are silently scoped to their own
        username regardless of any ``username`` query they pass; admins can
        omit it for a network-wide view or pin to a specific user. When
        ``include_shared`` is set, a non-admin's page is the union of their
        owned runs and runs shared to them, and each shared row carries the
        caller's grant ``role`` (owned rows leave ``role`` null).
        ``limit`` is clamped to keep agent responses context-safe; UI callers
        that genuinely need larger pages hit the dedicated
        ``/optimizations/sidebar`` route.

        Args:
            current_user: Authenticated caller resolved from the bearer token.
            status: Optional exact-match status filter.
            username: Optional submitter filter (admin-only).
            optimization_type: Optional ``"run"`` / ``"grid_search"`` filter.
            limit: Page size (clamped to ``AGENT_MAX_LIST``).
            offset: Number of optimizations to skip.
            include_shared: Union in runs shared with the caller (non-admin).

        Returns:
            A ``PaginatedJobsResponse`` carrying summaries and pagination
            counters.

        Raises:
            DomainError: 422 when ``status`` or ``optimization_type`` is
                outside its closed list.
        """
        if status is not None and status not in VALID_STATUSES:
            raise DomainError(
                "filter.invalid_status",
                status=422,
                value=status,
                allowed=sorted(VALID_STATUSES),
            )
        if optimization_type is not None and optimization_type not in VALID_OPTIMIZATION_TYPES:
            raise DomainError(
                "filter.invalid_optimization_type",
                status=422,
                value=optimization_type,
                allowed=sorted(VALID_OPTIMIZATION_TYPES),
            )
        resolved_limit = clamp_limit(limit)
        use_shared = (
            include_shared
            and not is_admin(current_user)
            and bool(current_user.username)
            and hasattr(job_store, "list_jobs_visible_to")
        )
        if use_shared:
            caller = current_user.username
            total = job_store.count_jobs_visible_to(caller, status=status, optimization_type=optimization_type)
            rows = job_store.list_jobs_visible_to(
                caller,
                status=status,
                optimization_type=optimization_type,
                limit=resolved_limit,
                offset=offset,
            )
        else:
            scoped_username = _scope_username(current_user, username)
            total = job_store.count_jobs(status=status, username=scoped_username, optimization_type=optimization_type)
            rows = job_store.list_jobs(
                status=status,
                username=scoped_username,
                optimization_type=optimization_type,
                limit=resolved_limit,
                offset=offset,
            )
        items = [build_summary(job_data) for job_data in rows]
        resumable_ids = resumable_id_flags(job_store, rows)
        for summary in items:
            summary.resumable = summary.optimization_id in resumable_ids
        if use_shared:
            caller_norm = current_user.username.strip().lower()
            roles = grant_roles_for(job_store, [s.optimization_id for s in items], caller_norm)
            for s in items:
                s.role = roles.get(s.optimization_id)
        return PaginatedJobsResponse(items=items, total=total, limit=resolved_limit, offset=offset)

    @router.get(
        "/optimizations/counts",
        response_model=OptimizationCountsResponse,
        summary="Aggregate optimization counts grouped by status",
        tags=["agent"],
    )
    def get_optimization_counts(
        current_user: AuthenticatedUserDep,
        username: str | None = Query(default=None, description="Restrict counts to a single user"),
        include_shared: bool = Query(
            default=False,
            description="Fold runs shared with the caller into the tally and report them under `shared` (non-admin).",
        ),
    ) -> OptimizationCountsResponse:
        """Return backend row counts grouped by status for dashboard stat cards.

        Non-admins are silently scoped to their own username; admins can omit
        ``username`` for a network-wide tally or pin to a specific user. When
        ``include_shared`` is set, a non-admin's tally is the union of owned
        and shared-with-them runs, and ``shared`` reports how many of the
        visible runs came from a grant (powering the "shared" stat card).

        Args:
            current_user: Authenticated caller resolved from the bearer token.
            username: Optional submitter filter (admin-only).
            include_shared: Union in runs shared with the caller (non-admin).

        Returns:
            An ``OptimizationCountsResponse`` keyed by status, with ``shared``
            set when ``include_shared`` is requested.
        """
        use_shared = (
            include_shared
            and not is_admin(current_user)
            and bool(current_user.username)
            and hasattr(job_store, "count_jobs_visible_to")
        )
        if use_shared:
            caller = current_user.username
            # Single GROUP BY round trip when the store supports it; the
            # per-status fan-out (7 COUNT queries) remains as the fallback for
            # stores that don't.
            if hasattr(job_store, "count_jobs_by_status_visible_to"):
                buckets = job_store.count_jobs_by_status_visible_to(caller)
                return OptimizationCountsResponse(
                    total=sum(buckets.values()),
                    pending=buckets.get("pending", 0),
                    validating=buckets.get("validating", 0),
                    running=buckets.get("running", 0),
                    success=buckets.get("success", 0),
                    failed=buckets.get("failed", 0),
                    cancelled=buckets.get("cancelled", 0),
                    shared=job_store.count_jobs_shared_with(caller),
                )
            return OptimizationCountsResponse(
                total=job_store.count_jobs_visible_to(caller),
                pending=job_store.count_jobs_visible_to(caller, status="pending"),
                validating=job_store.count_jobs_visible_to(caller, status="validating"),
                running=job_store.count_jobs_visible_to(caller, status="running"),
                success=job_store.count_jobs_visible_to(caller, status="success"),
                failed=job_store.count_jobs_visible_to(caller, status="failed"),
                cancelled=job_store.count_jobs_visible_to(caller, status="cancelled"),
                shared=job_store.count_jobs_shared_with(caller),
            )
        scoped_username = _scope_username(current_user, username)
        if hasattr(job_store, "count_jobs_by_status"):
            buckets = job_store.count_jobs_by_status(username=scoped_username)
            return OptimizationCountsResponse(
                total=sum(buckets.values()),
                pending=buckets.get("pending", 0),
                validating=buckets.get("validating", 0),
                running=buckets.get("running", 0),
                success=buckets.get("success", 0),
                failed=buckets.get("failed", 0),
                cancelled=buckets.get("cancelled", 0),
            )
        total = job_store.count_jobs(username=scoped_username)
        return OptimizationCountsResponse(
            total=total,
            pending=job_store.count_jobs(status="pending", username=scoped_username),
            validating=job_store.count_jobs(status="validating", username=scoped_username),
            running=job_store.count_jobs(status="running", username=scoped_username),
            success=job_store.count_jobs(status="success", username=scoped_username),
            failed=job_store.count_jobs(status="failed", username=scoped_username),
            cancelled=job_store.count_jobs(status="cancelled", username=scoped_username),
        )

    @router.get(
        "/optimizations/sidebar",
        response_model=SidebarJobsResponse,
        summary="Compact optimization list tuned for sidebar navigation",
    )
    def list_jobs_sidebar(
        current_user: AuthenticatedUserDep,
        username: str | None = Query(default=None, description="Restrict the list to a single user's optimizations"),
        limit: int = Query(
            default=50,
            ge=1,
            le=200,
            description="Page size; capped at 200 because the sidebar only renders a finite slice",
        ),
        offset: int = Query(default=0, ge=0, description="Number of optimizations to skip before the returned slice"),
    ) -> SidebarJobsResponse:
        """Return minimal per-optimization fields for the sidebar navigation list.

        No result payload, metrics, logs, or progress. Newest-first; pin
        reordering is client-side. Non-admins are silently scoped to their
        own username; admins see everyone unless they pass ``username``.

        Args:
            current_user: Authenticated caller resolved from the bearer token.
            username: Optional submitter filter (admin-only).
            limit: Page size (clamped to 200).
            offset: Number of optimizations to skip.

        Returns:
            A ``SidebarJobsResponse`` carrying compact items and total count.
        """
        scoped_username = _scope_username(current_user, username)
        total = job_store.count_jobs(username=scoped_username)
        rows = job_store.list_jobs(username=scoped_username, limit=limit, offset=offset)
        resumable_ids = resumable_id_flags(job_store, rows)
        items = []
        for row in rows:
            overview = parse_overview(row)
            items.append(
                SidebarJobItem(
                    optimization_id=row["optimization_id"],
                    status=row.get("status", "pending"),
                    name=overview.get(PAYLOAD_OVERVIEW_NAME),
                    module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME),
                    optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
                    model_name=overview.get(PAYLOAD_OVERVIEW_MODEL_NAME),
                    username=overview.get(PAYLOAD_OVERVIEW_USERNAME),
                    created_at=parse_timestamp(row.get("created_at")),
                    pinned=bool(overview.get("pinned", False)),
                    optimization_type=overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE),
                    total_pairs=overview.get(PAYLOAD_OVERVIEW_TOTAL_PAIRS),
                    resumable=row["optimization_id"] in resumable_ids,
                )
            )
        return SidebarJobsResponse(items=items, total=total)

    @router.get(
        "/optimizations/shared-with-me",
        response_model=SidebarJobsResponse,
        summary="Compact list of optimizations shared with the caller",
    )
    def list_jobs_shared_with_me(
        current_user: AuthenticatedUserDep,
        limit: int = Query(
            default=50,
            ge=1,
            le=200,
            description="Page size; capped at 200 because the sidebar only renders a finite slice",
        ),
        offset: int = Query(default=0, ge=0, description="Number of optimizations to skip before the returned slice"),
    ) -> SidebarJobsResponse:
        """Return runs another user invited the caller to (Drive-style sharing).

        Mirrors ``/optimizations/sidebar`` but lists optimizations the caller
        holds a member grant on (not their own). Each item carries the caller's
        ``role`` (``viewer``/``editor``/``owner``) so the UI can badge and gate
        actions. Stores without a grant-bearing engine return an empty list.

        Args:
            current_user: Authenticated caller resolved from the bearer token.
            limit: Page size (clamped to 200).
            offset: Number of optimizations to skip.

        Returns:
            A ``SidebarJobsResponse`` of shared items with per-item ``role``.
        """
        username = current_user.username.strip().lower()
        if not hasattr(job_store, "list_jobs_shared_with"):
            return SidebarJobsResponse(items=[], total=0)
        total = job_store.count_jobs_shared_with(username)
        rows = job_store.list_jobs_shared_with(username, limit=limit, offset=offset)
        roles = grant_roles_for(job_store, [row["optimization_id"] for row in rows], username)
        resumable_ids = resumable_id_flags(job_store, rows)
        items = []
        for row in rows:
            overview = parse_overview(row)
            items.append(
                SidebarJobItem(
                    optimization_id=row["optimization_id"],
                    status=row.get("status", "pending"),
                    name=overview.get(PAYLOAD_OVERVIEW_NAME),
                    module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME),
                    optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
                    model_name=overview.get(PAYLOAD_OVERVIEW_MODEL_NAME),
                    username=overview.get(PAYLOAD_OVERVIEW_USERNAME),
                    created_at=parse_timestamp(row.get("created_at")),
                    pinned=bool(overview.get("pinned", False)),
                    optimization_type=overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE),
                    total_pairs=overview.get(PAYLOAD_OVERVIEW_TOTAL_PAIRS),
                    resumable=row["optimization_id"] in resumable_ids,
                    role=roles.get(row["optimization_id"]),
                )
            )
        return SidebarJobsResponse(items=items, total=total)
