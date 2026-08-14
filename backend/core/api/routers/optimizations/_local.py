"""Private helpers shared between the optimizations route modules."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError

from ....constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODEL_SETTINGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_TOTAL_PAIRS,
    PAYLOAD_OVERVIEW_USERNAME,
)
from ....i18n import t
from ....models import (
    GridSearchRequest,
    OptimizationStatus,
    OptimizationSubmissionResponse,
    RunRequest,
)
from ....models.common import OptimizationType
from ....notifications import notify_job_started
from ....worker.engine import get_worker
from ...auth import AuthenticatedUser
from ...converters import parse_overview
from ...errors import DomainError
from ...sharing_access import ShareRole
from .._helpers import (
    compute_task_fingerprint,
    filter_ids_at_least,
    get_job_no_payload,
    stable_seed,
    strip_api_key,
)
from .schemas import BulkMetadataRequest, BulkMetadataResponse, BulkMetadataSkipped

logger = logging.getLogger(__name__)

DASHBOARD_POLL_SECONDS = 3
JOB_STREAM_POLL_SECONDS = 2


async def stream_dashboard_snapshots(
    job_store,
    *,
    owner_filter: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield live dashboard snapshots as ``{event, data}`` dicts.

    Polls every pending/validating/running optimization every
    ``DASHBOARD_POLL_SECONDS`` seconds and yields one default (``message``)
    event per tick. When no optimizations are active, emits a terminal
    ``idle`` event and closes.

    Args:
        job_store: Job-store used to enumerate active optimizations.
        owner_filter: When set, restrict the stream to jobs owned by this
            username; ``None`` means show every active optimization
            (admin-scoped behaviour).

    Yields:
        ``{"event": "message"|"idle", "data": ...}`` dicts.
    """
    loop = asyncio.get_running_loop()

    def _fetch(status: str) -> list[dict]:
        """List up to 100 jobs in the given status from ``job_store``.

        Args:
            status: The job-store status bucket to enumerate.

        Returns:
            Raw ``job_store`` rows for the requested status (capped at 100).
        """
        return job_store.list_jobs(status=status, limit=100, username=owner_filter)

    while True:
        active_rows: list[dict] = []
        for s in ("pending", "validating", "running"):
            rows = await loop.run_in_executor(None, _fetch, s)
            active_rows.extend(rows)

        summaries = []
        for row in active_rows:
            overview = parse_overview(row)
            summaries.append(
                {
                    "optimization_id": row["optimization_id"],
                    "status": row.get("status", "pending"),
                    "name": overview.get(PAYLOAD_OVERVIEW_NAME),
                    "latest_metrics": row.get("latest_metrics", {}),
                    # list_jobs already folds these in via two aggregate queries
                    # (see RemoteDBJobStore._rows_with_counts), so the previous
                    # per-row COUNT round trips (2 per active job, every tick)
                    # were pure redundancy.
                    "log_count": row.get("log_count", 0),
                    "progress_count": row.get("progress_count", 0),
                }
            )

        yield {"event": "message", "data": {"active_jobs": summaries, "active_count": len(summaries)}}

        if len(summaries) == 0:
            yield {"event": "idle", "data": {"active_count": 0}}
            return

        await asyncio.sleep(DASHBOARD_POLL_SECONDS)


async def stream_job_updates(job_store, optimization_id: str) -> AsyncIterator[dict[str, Any]]:
    """Yield live status updates for a single optimization.

    Polls the job every ``JOB_STREAM_POLL_SECONDS`` seconds and yields a
    default (``message``) snapshot event per tick. Emits a terminal ``done``
    event once the job reaches a terminal status, or an ``error`` event if
    the job vanishes mid-stream.

    Args:
        job_store: Job-store used to read the job snapshot.
        optimization_id: Optimization id to follow.

    Yields:
        ``{"event": "message"|"done"|"error", "data": ...}`` dicts.
    """
    loop = asyncio.get_running_loop()
    terminal = {"success", "failed", "cancelled"}
    while True:
        # Narrow projection: the loop polls every few seconds and only needs
        # status/message/latest_metrics, never the multi-MB payload that
        # get_job materializes. get_job_status_fields raises KeyError (not None)
        # when the row is gone, so the not-found path is an except, not a guard.
        try:
            raw = await loop.run_in_executor(None, job_store.get_job_status_fields, optimization_id)
        except KeyError:
            yield {"event": "error", "data": {"error": "Optimization not found"}}
            return

        status = raw.get("status", "pending")
        log_count, progress_count = await asyncio.gather(
            loop.run_in_executor(None, job_store.get_log_count, optimization_id),
            loop.run_in_executor(None, job_store.get_progress_count, optimization_id),
        )
        yield {
            "event": "message",
            "data": {
                "optimization_id": optimization_id,
                "status": status,
                "message": raw.get("message"),
                "latest_metrics": raw.get("latest_metrics", {}),
                "log_count": log_count,
                "progress_count": progress_count,
            },
        }

        if status in terminal:
            yield {"event": "done", "data": {"status": status}}
            return

        await asyncio.sleep(JOB_STREAM_POLL_SECONDS)


def clone_payload(
    source_payload: dict,
    *,
    optimization_type: str,
    new_name: str | None,
) -> tuple[str, Any]:
    """Return ``(new_optimization_id, payload_model)`` from a stored payload dict.

    Re-parses the stored payload into the original Pydantic request model,
    assigns a fresh UUID and seed, and overrides the display name if given.

    Args:
        source_payload: Stored submission payload dict.
        optimization_type: ``"run"`` or ``"grid_search"``.
        new_name: Optional override for the cloned optimization's display name.

    Returns:
        ``(new_id, payload_model)`` where ``payload_model`` is a parsed
        ``RunRequest`` or ``GridSearchRequest`` ready to be enqueued.

    Raises:
        DomainError: 409 when the stored payload no longer validates against
            its request model (schema drift from when it was stored). This is a
            state condition on the saved resource, not a server fault — surface
            it as a conflict the caller can act on, not an opaque 500.
    """
    copy = dict(source_payload)
    if new_name is not None:
        copy["name"] = new_name
    request_cls: type[GridSearchRequest] | type[RunRequest] = (
        GridSearchRequest if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH else RunRequest
    )
    try:
        payload = request_cls.model_validate(copy)
    except ValidationError as exc:
        raise DomainError(
            "optimization.cannot_resubmit_payload",
            status=409,
            error=exc.errors()[0]["msg"],
        ) from exc
    new_id = str(uuid4())
    # Derive the seed from the task fingerprint so clones and resubmissions reproduce
    # the original data split. hash() is per-process salted (PYTHONHASHSEED), while
    # stable_seed(...) stays byte-stable across workers.
    fingerprint = compute_task_fingerprint(payload.signature_code, payload.metric_code, payload.dataset)
    payload.seed = stable_seed(fingerprint)
    return new_id, payload


def persist_and_enqueue(
    job_store,
    new_id: str,
    payload: Any,
    *,
    optimization_type: str,
) -> OptimizationSubmissionResponse:
    """Persist a cloned/retried payload and hand it to the worker.

    Writes the overview dict the same way ``/run`` and ``/grid-search`` do,
    then hands the payload to the background worker and fires the
    job-started notification.

    Args:
        job_store: Job-store the new row is written to.
        new_id: Pre-allocated optimization id for the new run.
        payload: Parsed ``RunRequest`` or ``GridSearchRequest``.
        optimization_type: ``"run"`` or ``"grid_search"``.

    Returns:
        An ``OptimizationSubmissionResponse`` describing the enqueued run.
    """
    fingerprint = compute_task_fingerprint(payload.signature_code, payload.metric_code, payload.dataset)
    is_grid = optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH

    overview: dict[str, Any] = {
        PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: optimization_type,
        PAYLOAD_OVERVIEW_NAME: payload.name,
        "description": payload.description,
        PAYLOAD_OVERVIEW_USERNAME: payload.username,
        PAYLOAD_OVERVIEW_MODULE_NAME: payload.module_name,
        "module_kwargs": dict(payload.module_kwargs),
        PAYLOAD_OVERVIEW_OPTIMIZER_NAME: payload.optimizer_name,
        "column_mapping": payload.column_mapping.model_dump(),
        "dataset_rows": len(payload.dataset),
        "dataset_filename": payload.dataset_filename,
        "split_fractions": payload.split_fractions.model_dump(),
        "shuffle": payload.shuffle,
        "seed": payload.seed,
        "optimizer_kwargs": dict(payload.optimizer_kwargs),
        "compile_kwargs": dict(payload.compile_kwargs),
        "task_fingerprint": fingerprint,
    }
    if is_grid:
        overview[PAYLOAD_OVERVIEW_TOTAL_PAIRS] = len(payload.generation_models) * len(payload.reflection_models)
        overview["generation_models"] = [m.model_dump() for m in payload.generation_models]
        overview["reflection_models"] = [m.model_dump() for m in payload.reflection_models]
    else:
        overview[PAYLOAD_OVERVIEW_MODEL_NAME] = payload.model_settings.normalized_identifier()
        overview[PAYLOAD_OVERVIEW_MODEL_SETTINGS] = strip_api_key(payload.model_settings.model_dump())

    job_store.create_job(new_id)
    job_store.set_payload_overview(new_id, overview)

    worker = get_worker(job_store, service=None)
    worker.submit_job(new_id, payload)

    notify_job_started(
        optimization_id=new_id,
        username=payload.username,
        optimization_type=optimization_type,
        optimizer_name=payload.optimizer_name,
        module_name=payload.module_name,
        model_name=(
            t(
                "optimization.pairs_label",
                count=overview.get(PAYLOAD_OVERVIEW_TOTAL_PAIRS, 0),
            )
            if is_grid
            else payload.model_settings.normalized_identifier()
        ),
    )
    return OptimizationSubmissionResponse(
        optimization_id=new_id,
        optimization_type=cast(OptimizationType, optimization_type),
        status=OptimizationStatus.pending,
        created_at=datetime.now(UTC),
        name=payload.name,
        username=payload.username,
        module_name=payload.module_name,
        optimizer_name=payload.optimizer_name,
    )


def bulk_set_flag(
    job_store,
    req: BulkMetadataRequest,
    *,
    flag: str,
    user: AuthenticatedUser,
    minimum: ShareRole = ShareRole.editor,
) -> BulkMetadataResponse:
    """Shared bulk-metadata update used by ``/bulk-pin``.

    Walks ``req.optimization_ids``, setting ``overview[flag]`` to ``req.value``
    on each row the caller may edit. Duplicate ids are collapsed; unknown ids —
    and ids the caller's share role can't reach ``minimum`` on — are surfaced in
    ``skipped`` with ``reason="not_found"`` rather than raising. Members can
    pin/flag a shared run when their grant is at least ``minimum`` (editor).

    Args:
        job_store: Job-store the flag is written to.
        req: Bulk-metadata request body.
        flag: Overview key to toggle (e.g. ``"pinned"``).
        user: Authenticated caller.
        minimum: Lowest share role permitted to edit each row (pin is editor+).

    Returns:
        A ``BulkMetadataResponse`` listing successful and skipped ids.
    """
    updated: list[str] = []
    skipped: list[BulkMetadataSkipped] = []
    seen: set[str] = set()
    ordered: list[str] = []
    for oid in req.optimization_ids:
        if oid in seen:
            continue
        seen.add(oid)
        ordered.append(oid)
    allowed, denied = filter_ids_at_least(job_store, ordered, user, minimum)
    skipped.extend(
        BulkMetadataSkipped(optimization_id=oid, reason="not_found") for oid in denied
    )
    for oid in allowed:
        try:
            job_data = get_job_no_payload(job_store, oid)
        except KeyError:
            skipped.append(BulkMetadataSkipped(optimization_id=oid, reason="not_found"))
            continue
        overview = parse_overview(job_data)
        overview[flag] = bool(req.value)
        job_store.set_payload_overview(oid, overview)
        updated.append(oid)
    return BulkMetadataResponse(updated=updated, skipped=skipped)


def remap_test_indices(
    results: list[dict[str, Any]],
    test_indices: list[int],
) -> list[dict[str, Any]]:
    """Translate sequential test-split indices back to global dataset indices.

    The test-split evaluation records each result with the index inside the
    test split (0..len(test_indices)-1). For display we want the row's
    original position in the full dataset so the UI can line results up
    with the user's source data.

    Args:
        results: Test-split records with sequential ``index`` keys.
        test_indices: Mapping from sequential index to global dataset index.

    Returns:
        A new list of records with ``index`` rewritten to the global value.
    """
    remapped: list[dict[str, Any]] = []
    for record in results:
        seq_idx = record.get("index", 0)
        global_idx = test_indices[seq_idx] if seq_idx < len(test_indices) else seq_idx
        remapped.append({**record, "index": global_idx})
    return remapped
