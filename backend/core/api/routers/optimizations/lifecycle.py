"""Lifecycle mutations: clone, retry, cancel, bulk-cancel, bulk-pin. [MIXED]

Public dev surface (in ``_SCALAR_PUBLIC_PATHS``):
- ``POST /optimizations/{id}/cancel``
- ``POST /optimizations/{id}/clone``
- ``POST /optimizations/{id}/retry``
- ``POST /optimizations/{id}/resume``

Internal (dashboard plumbing, hidden from public docs):
- ``POST /optimizations/bulk-cancel``
- ``POST /optimizations/bulk-pin``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ....config import settings
from ....constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    OPTIMIZATION_TYPE_RUN,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
)
from ....i18n import CANCELLATION_REASON, CLONE_NAME_PREFIX, PAUSE_REASON, RETRY_NAME_PREFIX
from ....models import (
    BulkCancelRequest,
    BulkCancelResponse,
    BulkCancelSkipped,
    JobCancelResponse,
    OptimizationStatus,
    OptimizationSubmissionResponse,
)
from ....storage.usage import json_byte_size
from ...auth import AuthenticatedUser, get_authenticated_user
from ...converters import parse_overview, status_to_job_status
from ...errors import DomainError
from ...sharing_access import ShareRole
from .._helpers import (
    enforce_storage_quota,
    filter_ids_at_least,
    is_pausable,
    load_job_for_user,
    require_role_at_least,
)
from ..constants import TERMINAL_STATUSES
from ._local import bulk_set_flag, clone_payload, persist_and_enqueue
from .schemas import (
    BulkMetadataRequest,
    BulkMetadataResponse,
    CloneJobRequest,
    CloneJobResponse,
)

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def _set_terminal_if_active(job_store, optimization_id: str, expected: tuple[str, ...], **fields: Any) -> bool:
    """Compare-and-set a lifecycle status, falling back to last-writer-wins.

    Uses the store's ``update_job_if_status`` CAS when available so a pause /
    cancel can't clobber a status the worker moved the row to in the race window
    after the handler's status pre-check. Stores without the CAS (test fakes)
    keep the prior unconditional write.

    Args:
        job_store: The active job store.
        optimization_id: ID of the job to update.
        expected: Statuses the row must currently hold for the write to apply.
        **fields: Column values to write.

    Returns:
        ``True`` when the write applied (or the store has no CAS); ``False`` only
        when the CAS found the row already past ``expected``.
    """
    cas = getattr(job_store, "update_job_if_status", None)
    if cas is None:
        job_store.update_job(optimization_id, **fields)
        return True
    return cas(optimization_id, expected, **fields)


def register_lifecycle_routes(
    router: APIRouter,
    *,
    job_store,
    get_worker_ref: Callable[[], Any],
) -> None:
    """Register mutation routes that change state or queue new work.

    Args:
        router: The router to attach the lifecycle routes to.
        job_store: Job-store the routes read from / mutate.
        get_worker_ref: Zero-arg callable returning the active worker (or
            ``None`` when no worker is bound) — wrapped so tests can inject.
    """

    @router.post(
        "/optimizations/{optimization_id}/cancel",
        response_model=JobCancelResponse,
        status_code=200,
        summary="Cancel a pending or running optimization",
        tags=["agent"],
    )
    def cancel_job(optimization_id: str, current_user: AuthenticatedUserDep) -> JobCancelResponse:
        """Cooperatively cancel an active optimization.

        Flips status to ``cancelled`` immediately; the worker stops between
        DSPy calls. One-way — no uncancel.

        Args:
            optimization_id: Optimization id to cancel.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` confirming the cancellation.

        Raises:
            DomainError: 404 if unknown or inaccessible, 403 if the caller's
                share role is below ``editor``, 409 if already terminal.
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)

        status = status_to_job_status(job_data.get("status", "pending"))
        if status in TERMINAL_STATUSES:
            raise DomainError(
                "optimization.already_terminal",
                status=409,
                params={"status": status.value},
            )

        worker = get_worker_ref()
        if worker:
            worker.cancel_job(optimization_id)

        now = datetime.now(UTC).isoformat()
        if not _set_terminal_if_active(
            job_store,
            optimization_id,
            ("pending", "validating", "running"),
            status="cancelled",
            message=CANCELLATION_REASON,
            completed_at=now,
        ):
            # The worker reached a terminal status in the window after the
            # pre-check above; reflect the row's actual status.
            current = status_to_job_status(job_store.get_job(optimization_id).get("status", "pending"))
            raise DomainError("optimization.already_terminal", status=409, params={"status": current.value})
        logger.info("Optimization %s (%s) cancelled", optimization_id, status.value)
        return JobCancelResponse(optimization_id=optimization_id, status="cancelled")

    @router.post(
        "/optimizations/{optimization_id}/pause",
        response_model=JobCancelResponse,
        status_code=200,
        summary="Pause a running optimization, keeping its checkpoint for resume",
        tags=["agent"],
    )
    def pause_job(optimization_id: str, current_user: AuthenticatedUserDep) -> JobCancelResponse:
        """Suspend a running optimization at its last checkpoint so it can be resumed.

        A manual pause is intentional, not a failure: it flips status to
        ``paused`` (distinct from ``cancelled``) and signals the worker, which
        stops between DSPy calls and persists the GEPA checkpoint on its way out —
        the same cooperative path cancel uses. Valid only while the run is
        actively ``running`` and already has a saved checkpoint, so a pause is
        always resumable; pausing before the first checkpoint is rejected. Resume
        from ``paused`` does not count against the attempt cap.

        Args:
            optimization_id: Optimization id to pause.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` echoing the id and its new ``paused`` status.

        Raises:
            DomainError: 404 if unknown or inaccessible, 403 if the caller's
                share role is below ``editor``, 409 if not running or without a
                checkpoint to resume from.
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)

        status = status_to_job_status(job_data.get("status", "pending"))
        if status != OptimizationStatus.running:
            raise DomainError(
                "optimization.pause_wrong_status",
                status=409,
                params={"status": status.value},
            )
        if not is_pausable(job_store, job_data):
            raise DomainError("optimization.pause_not_pausable", status=409)

        worker = get_worker_ref()
        if worker:
            worker.cancel_job(optimization_id)

        now = datetime.now(UTC).isoformat()
        if not _set_terminal_if_active(
            job_store, optimization_id, ("running",), status="paused", message=PAUSE_REASON, completed_at=now
        ):
            # The worker finished in the window after the running pre-check; the
            # run is no longer pausable.
            current = status_to_job_status(job_store.get_job(optimization_id).get("status", "pending"))
            raise DomainError("optimization.pause_wrong_status", status=409, params={"status": current.value})
        logger.info("Optimization %s paused", optimization_id)
        return JobCancelResponse(optimization_id=optimization_id, status="paused")

    @router.post(
        "/optimizations/bulk-cancel",
        response_model=BulkCancelResponse,
        status_code=200,
        summary="Cancel many running or pending optimizations in a single request",
        tags=["agent"],
    )
    def bulk_cancel_jobs(body: BulkCancelRequest, current_user: AuthenticatedUserDep) -> BulkCancelResponse:
        """Cancel a batch of non-terminal optimizations and report per-id outcomes.

        Same semantics as single-ID ``POST /optimizations/{id}/cancel``: flips
        status to ``cancelled`` immediately, worker stops between DSPy calls,
        one-way. Duplicate ids in the request are deduplicated. IDs that don't
        exist, are already terminal, or are not owned by the caller (when
        non-admin) are reported in ``skipped`` with the reason ``not_found``.

        Args:
            body: The bulk-cancel request body.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``BulkCancelResponse`` listing cancelled and skipped ids.
        """
        cancelled: list[str] = []
        skipped: list[BulkCancelSkipped] = []
        seen: set[str] = set()
        ordered_unique: list[str] = []
        for optimization_id in body.optimization_ids:
            if optimization_id in seen:
                continue
            seen.add(optimization_id)
            ordered_unique.append(optimization_id)

        if not ordered_unique:
            return BulkCancelResponse(cancelled=cancelled, skipped=skipped)

        allowed, denied = filter_ids_at_least(job_store, ordered_unique, current_user, ShareRole.editor)
        skipped.extend(
            BulkCancelSkipped(optimization_id=optimization_id, reason="not_found") for optimization_id in denied
        )

        if not allowed:
            return BulkCancelResponse(cancelled=cancelled, skipped=skipped)

        status_by_id = job_store.get_jobs_status_by_ids(allowed)

        cancellable: list[str] = []
        for optimization_id in allowed:
            raw_status = status_by_id.get(optimization_id)
            if raw_status is None:
                skipped.append(BulkCancelSkipped(optimization_id=optimization_id, reason="not_found"))
                continue
            status = status_to_job_status(raw_status)
            if status in TERMINAL_STATUSES:
                skipped.append(BulkCancelSkipped(optimization_id=optimization_id, reason=status.value))
                continue
            cancellable.append(optimization_id)

        if cancellable:
            worker = get_worker_ref()
            now = datetime.now(UTC).isoformat()
            for optimization_id in cancellable:
                try:
                    if worker:
                        worker.cancel_job(optimization_id)
                    job_store.update_job(
                        optimization_id,
                        status="cancelled",
                        message=CANCELLATION_REASON,
                        completed_at=now,
                    )
                except Exception as exc:
                    logger.exception("Bulk cancel failed for %s", optimization_id)
                    skipped.append(
                        BulkCancelSkipped(
                            optimization_id=optimization_id,
                            reason=f"error: {exc}",
                        )
                    )
                else:
                    cancelled.append(optimization_id)

        logger.info(
            "Bulk cancel: %d cancelled, %d skipped (requested %d)",
            len(cancelled),
            len(skipped),
            len(body.optimization_ids),
        )
        return BulkCancelResponse(cancelled=cancelled, skipped=skipped)

    @router.post(
        "/optimizations/{optimization_id}/clone",
        response_model=CloneJobResponse,
        status_code=201,
        summary="Duplicate an optimization and queue N copies",
        tags=["agent"],
    )
    def clone_job(optimization_id: str, req: CloneJobRequest, current_user: AuthenticatedUserDep) -> CloneJobResponse:
        """Clone a finished or active optimization into ``count`` fresh runs.

        Reads the stored payload, assigns new ids and seeds, and enqueues each
        copy. Each clone's display name is prefixed with ``req.name_prefix`` or
        ``CLONE_NAME_PREFIX``. The clone is owned by the authenticated caller
        regardless of who owned the source; quota is checked against the
        caller. Respects the per-user quota.

        Args:
            optimization_id: Source optimization id.
            req: Clone request with ``count`` and optional ``name_prefix``.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``CloneJobResponse`` listing every newly enqueued copy.

        Raises:
            DomainError: 404 (unknown / inaccessible source), 409 (no
                payload / quota / saved payload no longer resubmittable).
        """
        job_data = load_job_for_user(job_store, optimization_id, current_user)

        source_payload = job_data.get("payload")
        if not source_payload or not isinstance(source_payload, dict):
            raise DomainError("optimization.clone_no_payload", status=409)

        overview = parse_overview(job_data)
        optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)
        prefix = (req.name_prefix or CLONE_NAME_PREFIX).strip()
        source_name = overview.get(PAYLOAD_OVERVIEW_NAME) or optimization_id[:8]
        cloned_payload_seed: dict[str, Any] = {**source_payload, "username": current_user.username}

        enforce_storage_quota(
            job_store,
            current_user.username,
            incoming_bytes=json_byte_size(cloned_payload_seed) * req.count,
        )

        created: list[OptimizationSubmissionResponse] = []
        for i in range(req.count):
            suffix = f" ({i + 1})" if req.count > 1 else ""
            cloned_name = f"{prefix} {source_name}{suffix}".strip()
            new_id, payload = clone_payload(
                cloned_payload_seed,
                optimization_type=optimization_type,
                new_name=cloned_name,
            )
            created.append(persist_and_enqueue(job_store, new_id, payload, optimization_type=optimization_type))

        logger.info("Cloned optimization %s into %d copies", optimization_id, len(created))
        return CloneJobResponse(source_optimization_id=optimization_id, created=created)

    @router.post(
        "/optimizations/{optimization_id}/retry",
        response_model=OptimizationSubmissionResponse,
        status_code=201,
        summary="Re-run a failed or cancelled optimization with the same configuration",
        tags=["agent"],
    )
    def retry_job(optimization_id: str, current_user: AuthenticatedUserDep) -> OptimizationSubmissionResponse:
        """Re-run a failed or cancelled optimization using the original payload.

        Only valid when the source optimization is in a terminal non-success
        state. The new run's name is prefixed with ``RETRY_NAME_PREFIX``.
        The retry is owned by the authenticated caller regardless of who
        owned the source; quota is checked against the caller.

        Args:
            optimization_id: Source optimization id to retry.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            An ``OptimizationSubmissionResponse`` for the new run.

        Raises:
            DomainError: 404 (unknown / inaccessible source), 403 (caller's
                share role below ``editor``), 409 (wrong status / no payload —
                use clone instead / saved payload no longer resubmittable).
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)

        status = status_to_job_status(job_data.get("status", "pending"))
        if status not in {OptimizationStatus.failed, OptimizationStatus.cancelled}:
            raise DomainError(
                "optimization.retry_wrong_status",
                status=409,
                params={"status": status.value},
            )

        source_payload = job_data.get("payload")
        if not source_payload or not isinstance(source_payload, dict):
            raise DomainError("optimization.retry_no_payload", status=409)

        overview = parse_overview(job_data)
        optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)
        source_name = overview.get(PAYLOAD_OVERVIEW_NAME) or optimization_id[:8]
        retry_name = f"{RETRY_NAME_PREFIX} {source_name}".strip()
        retry_payload_seed: dict[str, Any] = {**source_payload, "username": current_user.username}
        enforce_storage_quota(
            job_store,
            current_user.username,
            incoming_bytes=json_byte_size(retry_payload_seed),
        )
        new_id, payload = clone_payload(retry_payload_seed, optimization_type=optimization_type, new_name=retry_name)
        response = persist_and_enqueue(job_store, new_id, payload, optimization_type=optimization_type)
        logger.info("Retried optimization %s as %s", optimization_id, new_id)
        return response

    @router.post(
        "/optimizations/{optimization_id}/resume",
        response_model=JobCancelResponse,
        status_code=202,
        summary="Resume an optimization that stopped mid-run from its last checkpoint",
        tags=["agent"],
    )
    def resume_job(optimization_id: str, current_user: AuthenticatedUserDep) -> JobCancelResponse:
        """Resume a mid-run optimization in place from its saved GEPA checkpoint.

        Valid only when the run stopped after producing optimizer state — a
        terminal ``failed``/``cancelled`` or manually ``paused`` status with a
        saved checkpoint. Unlike retry/clone this creates no new run: the existing
        row is flipped back to ``pending`` with its original id, seed and budget,
        and a worker continues GEPA from the last completed iteration with no
        budget double-spend. A ``failed``/``cancelled`` resume shares the attempt
        cap with automatic pod-failure recovery; a manual ``paused`` resume is
        exempt (it neither consumes an attempt nor is bounded by the cap).

        Args:
            optimization_id: The optimization to resume.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` echoing the id and its new ``pending`` status.

        Raises:
            DomainError: 404 (unknown / inaccessible), 403 (caller's share role
                below ``editor``), 409 (not mid-run / no checkpoint / attempts
                exhausted).
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)

        status = status_to_job_status(job_data.get("status", "pending"))
        if status not in {
            OptimizationStatus.failed,
            OptimizationStatus.cancelled,
            OptimizationStatus.paused,
        }:
            raise DomainError(
                "optimization.resume_wrong_status",
                status=409,
                params={"status": status.value},
            )

        has_checkpoint = getattr(job_store, "has_gepa_checkpoint", None)
        has_pairs = getattr(job_store, "has_grid_pair_results", None)
        resumable_state = (callable(has_checkpoint) and has_checkpoint(optimization_id)) or (
            callable(has_pairs) and has_pairs(optimization_id)
        )
        if not resumable_state:
            raise DomainError("optimization.resume_not_resumable", status=409)

        # A manual pause/resume is user-driven, not failure recovery: it neither
        # consumes an attempt nor is bounded by the cap. Only failed/cancelled
        # resumes share ``job_max_attempts`` with automatic pod-failure recovery.
        is_paused = status == OptimizationStatus.paused
        if not is_paused:
            attempts = int(job_data.get("attempts") or 0)
            if attempts >= settings.job_max_attempts:
                raise DomainError(
                    "optimization.resume_exhausted",
                    status=409,
                    params={"attempts": attempts},
                )

        new_attempt = job_store.requeue_for_resume(optimization_id, bump_attempts=not is_paused)
        if new_attempt is None:
            raise DomainError("optimization.not_found", status=404)
        logger.info("Resumed optimization %s in place (attempt %s)", optimization_id, new_attempt)
        return JobCancelResponse(optimization_id=optimization_id, status=OptimizationStatus.pending.value)

    @router.post(
        "/optimizations/{optimization_id}/restart",
        response_model=JobCancelResponse,
        status_code=202,
        summary="Re-run a failed or cancelled optimization from scratch under the same id",
        tags=["agent"],
    )
    def restart_job(optimization_id: str, current_user: AuthenticatedUserDep) -> JobCancelResponse:
        """Reset a terminal optimization in place and re-run it from scratch.

        Unlike ``retry`` — which clones the payload into a brand-new id — this
        keeps the same id, name and URL: the existing row is wiped of its prior
        attempt's logs, progress, result and checkpoint, flipped back to
        ``pending`` and re-queued so a worker runs the same configuration from
        iteration zero. Unlike ``resume`` it does not continue from a checkpoint.
        Shares ``retry``'s precondition — valid only for a terminal
        ``failed``/``cancelled`` run.

        Args:
            optimization_id: The optimization to reset and re-run.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` echoing the id and its new ``pending`` status.

        Raises:
            DomainError: 404 (unknown / inaccessible), 403 (caller's share role
                below ``editor``), 409 (not in a failed/cancelled state).
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)

        status = status_to_job_status(job_data.get("status", "pending"))
        if status not in {OptimizationStatus.failed, OptimizationStatus.cancelled}:
            raise DomainError(
                "optimization.retry_wrong_status",
                status=409,
                params={"status": status.value},
            )

        if not job_store.requeue_for_rerun(optimization_id):
            raise DomainError("optimization.not_found", status=404)
        logger.info("Restarted optimization %s in place from scratch", optimization_id)
        return JobCancelResponse(optimization_id=optimization_id, status=OptimizationStatus.pending.value)

    def _rerun_grid_pair(
        optimization_id: str, pair_index: int, current_user: AuthenticatedUser, *, resume: bool
    ) -> JobCancelResponse:
        """Re-queue a terminal grid to re-run only ``pair_index``, keeping the others.

        Treats one grid pair like a single run: every *other* pair is seeded from
        the stored result so the worker keeps it, and the grid is re-queued in
        place to run only the target — fresh (``resume=False``) or from its saved
        checkpoint (``resume=True``). This is unbounded by ``job_max_attempts``
        (a targeted user action), works regardless of the grid's overall status,
        and merges the target's new result back into the grid result.

        Args:
            optimization_id: The grid optimization id.
            pair_index: The pair to re-run.
            current_user: Authenticated caller resolved from the bearer token.
            resume: Continue the pair from its checkpoint instead of re-running it.

        Returns:
            A ``JobCancelResponse`` echoing the id and its new ``pending`` status.

        Raises:
            DomainError: 404 (unknown id / not a grid / no result / missing pair),
                403 (share role below ``editor``), 409 (grid not terminal, or
                resume with no checkpoint for the pair).
        """
        job_data, _role = require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)
        overview = parse_overview(job_data)
        if overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE) != OPTIMIZATION_TYPE_GRID_SEARCH:
            raise DomainError("grid_search.not_a_grid_search", status=404)
        status = status_to_job_status(job_data.get("status", "pending"))
        if status not in TERMINAL_STATUSES:
            raise DomainError("optimization.pair_not_rerunnable", status=409, params={"status": status.value})

        result_data = job_data.get("result")
        pair_results = result_data.get("pair_results") if isinstance(result_data, dict) else None
        if not pair_results:
            raise DomainError("grid_search.no_result_to_modify", status=404)
        by_index = {pr.get("pair_index"): pr for pr in pair_results if isinstance(pr, dict)}
        if pair_index not in by_index:
            raise DomainError("grid_search.pair_position_missing", status=404, params={"pair_index": pair_index})

        if resume:
            getter = getattr(job_store, "get_gepa_checkpoint", None)
            checkpoint = getter(optimization_id, pair_index) if callable(getter) else None
            if checkpoint is None:
                raise DomainError("optimization.pair_not_resumable", status=409)

        # Seed every other pair so the runner keeps it and re-runs only the target.
        job_store.delete_grid_pair_results(optimization_id)
        for idx, pr in by_index.items():
            if idx != pair_index:
                job_store.save_grid_pair_result(optimization_id, idx, pr)
        if not resume:
            job_store.delete_gepa_checkpoint(optimization_id, pair_index)

        requeued = job_store.requeue_for_resume(optimization_id, bump_attempts=False)
        if requeued is None:
            raise DomainError("optimization.not_found", status=404)
        logger.info("Re-running grid pair %s of %s (resume=%s)", pair_index, optimization_id, resume)
        return JobCancelResponse(optimization_id=optimization_id, status=OptimizationStatus.pending.value)

    @router.post(
        "/optimizations/{optimization_id}/pair/{pair_index}/restart",
        response_model=JobCancelResponse,
        status_code=202,
        summary="Re-run one failed grid-search pair from scratch, keeping the others",
    )
    def restart_grid_pair(
        optimization_id: str, pair_index: int, current_user: AuthenticatedUserDep
    ) -> JobCancelResponse:
        """Re-run a single grid pair fresh — like a single run's Restart, scoped to one pair.

        Every other pair's result is kept; only this pair re-runs from scratch.
        Works even on a grid that succeeded overall, so a failed pair never costs
        you the good ones.

        Args:
            optimization_id: The grid optimization id.
            pair_index: The pair to re-run from scratch.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` for the re-queued grid.
        """
        return _rerun_grid_pair(optimization_id, pair_index, current_user, resume=False)

    @router.post(
        "/optimizations/{optimization_id}/pair/{pair_index}/resume",
        response_model=JobCancelResponse,
        status_code=202,
        summary="Resume one grid-search pair from its checkpoint, keeping the others",
    )
    def resume_grid_pair(
        optimization_id: str, pair_index: int, current_user: AuthenticatedUserDep
    ) -> JobCancelResponse:
        """Resume a single grid pair from its checkpoint — like a single run's Resume, per pair.

        Every other pair's result is kept; only this pair continues mid-GEPA from
        its saved checkpoint. 409 if the pair has no checkpoint (restart it instead).

        Args:
            optimization_id: The grid optimization id.
            pair_index: The pair to resume.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``JobCancelResponse`` for the re-queued grid.
        """
        return _rerun_grid_pair(optimization_id, pair_index, current_user, resume=True)

    @router.post(
        "/optimizations/bulk-pin",
        response_model=BulkMetadataResponse,
        summary="Pin or unpin many optimizations in one call",
        tags=["agent"],
    )
    def bulk_pin_jobs(req: BulkMetadataRequest, current_user: AuthenticatedUserDep) -> BulkMetadataResponse:
        """Pin or unpin up to 100 optimizations in a single call.

        Non-admin callers may only flag their own optimizations; ids they
        don't own are surfaced as ``not_found`` skips.

        Args:
            req: Bulk-metadata request with ``optimization_ids`` and ``value``.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``BulkMetadataResponse`` listing successful and skipped ids.
        """
        return bulk_set_flag(job_store, req, flag="pinned", user=current_user)
