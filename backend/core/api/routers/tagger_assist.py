"""AI co-tagging routes for tagger sessions. [INTERNAL]

Drives the tagger's assist modes on top of the persisted session rows:
the dataset interview (rubric distillation), batched label predictions for
calibration and review rounds, pre-run credit estimates, and the bulk
auto-tag job.

The bulk job is a ``tagging_autotag`` row in the shared jobs table, claimed
by the DB-lease background worker (any pod) and executed by
:mod:`core.worker.tagging_job`, which persists progress onto the session row
itself (``assist.autotag``); the client simply polls the status route. The
job only ever processes rows that have no final label yet, which makes a
rerun after cancel, crash or orphan recovery a plain resume. The status
route reconciles the session-row mirror against the job row, so a job that
died between mirror writes still reports honestly.

Ownership is enforced on every route by comparing the authenticated principal
to the row's ``username``. Hidden from the public Scalar reference.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...constants import (
    OPTIMIZATION_TYPE_TAGGING,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_USERNAME,
)
from ...service_gateway import tagging
from ...storage.models import TaggingSessionModel
from ...worker.tagging_job import TaggingAutotagPayload, untagged_rows
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..model_catalog import get_catalog_cached
from ._helpers import sse_from_events

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

MAX_PREDICT_ROWS = 50
MAX_INTERVIEW_TURNS = 40

# Job-row statuses that mean the worker fleet still owns the job.
_ACTIVE_JOB_STATUSES = ("pending", "validating", "running")


class InterviewRequest(BaseModel):
    """One interview exchange: the transcript so far, oldest turn first."""

    turns: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_INTERVIEW_TURNS)
    locale: str | None = Field(
        default=None,
        description="UI locale code; the assistant replies in that language.",
    )


# A single pickable answer offered for a closed interview question. The UI
# always adds its own free-text option, so this never carries an "other".
class InterviewOption(BaseModel):
    label: str
    description: str = ""


class InterviewResponse(BaseModel):
    """The assistant's next interview turn."""

    message: str
    options: list[InterviewOption] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    task_override: dict[str, Any] = Field(default_factory=dict)
    done: bool
    model: str | None = None


class PredictRequest(BaseModel):
    """Rows to predict labels for (calibration and review batches)."""

    row_ids: list[str] = Field(min_length=1, max_length=MAX_PREDICT_ROWS)


class PredictResponse(BaseModel):
    """Per-row predictions plus the credit cost of the calls made."""

    predictions: dict[str, dict[str, Any]]
    credits: int


class EstimateResponse(BaseModel):
    """Credit estimate for auto-tagging every currently-unlabeled row."""

    rows: int
    model: str
    credits_low: int
    credits_high: int


class AutotagStartResponse(BaseModel):
    """Ack that the bulk auto-tag job started."""

    total: int


class AutotagStatusResponse(BaseModel):
    """Bulk-job progress as persisted on the session row."""

    status: str
    total: int
    done: int
    credits_spent: int = 0
    live: bool = Field(
        description="False when the session claims 'running' but the worker fleet "
        "no longer owns an active job for it — the client should offer a resume."
    )


def _load_owned(session: Session, session_id: str, username: str) -> TaggingSessionModel:
    """Load a session row and enforce ownership.

    Args:
        session: An open SQLAlchemy session.
        session_id: UUID of the tagger session.
        username: The authenticated caller.

    Returns:
        The loaded row.

    Raises:
        DomainError: 404 when unknown, 403 when the caller does not own it.
    """
    row = session.get(TaggingSessionModel, session_id)
    if row is None:
        raise DomainError("tagger.session.not_found", status=404)
    if row.username != username:
        raise DomainError("tagger.session.forbidden", status=403)
    return row


def _effective_config(row: TaggingSessionModel) -> dict[str, Any]:
    """Return the session config with the interview's task refinements applied.

    The ``config`` column is immutable by design; the interview stores its
    refined question/prompt — and, for provisional-mode sessions, the inferred
    answer style itself — in ``assist.taskOverride`` and every LLM surface
    reads through :func:`tagging.effective_task_config`.

    Args:
        row: The loaded session row.
    """
    return tagging.effective_task_config(
        cast("dict[str, Any]", row.config),
        cast("dict[str, Any]", row.assist) or {},
    )


def _require_known_model(assist: dict[str, Any]) -> None:
    """Reject a session whose chosen tagging model is not in the catalog.

    The client only offers catalog models, but ``assist`` is a free-form JSON
    column any API caller can write — and tagging spends platform credits, so
    the curated catalog stays the boundary of what a session may run on.

    Args:
        assist: The session's assist state (may carry ``model``).

    Raises:
        DomainError: 422 when a chosen model is not in the curated catalog.
    """
    model = str((assist or {}).get("model") or "").strip()
    if not model:
        return
    if all(entry.value != model for entry in get_catalog_cached().models):
        raise DomainError("tagger.assist.unknown_model", status=422)


def _interview_config(row: TaggingSessionModel) -> dict[str, Any]:
    """Return effective config plus the assist mode used by the interviewer.

    Args:
        row: The loaded tagging session.

    Returns:
        A transient config carrying the selected assist mode.
    """
    config = _effective_config(row)
    assist = cast("dict[str, Any]", row.assist) or {}
    config["_assist_mode"] = assist.get("mode")
    return config


def _job_status(job_store: Any, job_id: str) -> str | None:
    """Read a job row's status, tolerating a missing row or store hiccup.

    Args:
        job_store: Job store to read from.
        job_id: The job row's id.

    Returns:
        The status string, or ``None`` when it cannot be read.
    """
    try:
        return job_store.get_job_status_fields(job_id).get("status")
    except Exception:
        return None


def create_tagger_assist_router(*, job_store, get_worker_ref: Callable[[], Any]) -> APIRouter:
    """Build the AI co-tagging router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.
        get_worker_ref: Zero-arg callable returning the background worker (or
            ``None`` before startup completes) — bulk auto-tag jobs are
            submitted through it.

    Returns:
        A FastAPI ``APIRouter`` exposing the interview / predict / estimate /
        autotag routes under ``/tagging-sessions/{id}/assist``.
    """
    router = APIRouter()

    @router.post(
        "/tagging-sessions/{session_id}/assist/interview",
        response_model=InterviewResponse,
        summary="Run one dataset-interview turn",
    )
    def assist_interview(session_id: str, req: InterviewRequest, user: AuthenticatedUserDep) -> InterviewResponse:
        """Return the assistant's next interview question or the final rubric.

        Args:
            session_id: UUID of the tagger session.
            req: Transcript so far plus the caller's UI locale.
            user: Authenticated caller; must own the session.

        Returns:
            The assistant turn; ``rubric`` is populated once ``done`` is true.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            config = _interview_config(row)
            columns = cast("list[str]", row.columns)
            data = cast("list[dict[str, Any]]", row.data)
        try:
            turn = tagging.interview_turn(config, columns, data, req.turns, req.locale)
        except Exception as exc:
            logger.exception("interview turn failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        return InterviewResponse(**turn)

    @router.post(
        "/tagging-sessions/{session_id}/assist/interview/stream",
        summary="Run one dataset-interview turn as an SSE stream",
    )
    async def assist_interview_stream(
        session_id: str, req: InterviewRequest, user: AuthenticatedUserDep
    ) -> StreamingResponse:
        """Stream one interview turn with the generalist agent's event shape.

        Events: ``reasoning_patch`` (provider thinking tokens),
        ``message_patch`` (reply deltas), ``message_end`` (the reply is fully
        streamed; options and rubric are still generating), ``message_reset``
        (a failed attempt is being retried or leaked structure was dropped;
        the client drops any partial reply), a terminal ``interview_done``
        carrying the parsed turn, and ``error`` on failure.

        Args:
            session_id: UUID of the tagger session.
            req: Transcript so far plus the caller's UI locale.
            user: Authenticated caller; must own the session.

        Returns:
            A ``text/event-stream`` response.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            config = _interview_config(row)
            columns = cast("list[str]", row.columns)
            data = cast("list[dict[str, Any]]", row.data)

        async def source() -> Any:
            """Relay engine events, translating failures into an error event."""
            try:
                async for event in tagging.interview_turn_stream(config, columns, data, req.turns, req.locale):
                    yield event
            except Exception:
                logger.exception("interview stream failed for session %s", session_id)
                yield {"event": "error", "data": {"code": "tagger.assist.llm_failed"}}

        return StreamingResponse(
            sse_from_events(source()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/tagging-sessions/{session_id}/assist/predict",
        response_model=PredictResponse,
        summary="Predict labels for specific rows (calibration / review)",
    )
    def assist_predict(session_id: str, req: PredictRequest, user: AuthenticatedUserDep) -> PredictResponse:
        """Predict labels for the requested rows using the stored rubric.

        Few-shot examples are compiled from the session's human labels,
        excluding the requested rows themselves, so calibration predictions
        never see the answer they are guessed against.

        Args:
            session_id: UUID of the tagger session.
            req: The row ids to predict (capped at ``MAX_PREDICT_ROWS``).
            user: Authenticated caller; must own the session.

        Returns:
            The ``{row_id: {value, confidence, reason}}`` map and the credit
            cost of the calls made.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            config = _effective_config(row)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        _require_known_model(assist)
        wanted = set(req.row_ids)
        rows = [r for r in data if str(r.get("id")) in wanted]
        if not rows:
            raise DomainError("tagger.assist.rows_not_found", status=404)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist, exclude_ids=wanted)
        instructions = tagging.compile_instructions(config, rubric, examples)
        try:
            predictions, credits = tagging.predict_rows(config, instructions, rows)
        except Exception as exc:
            logger.exception("prediction failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        return PredictResponse(predictions=predictions, credits=credits)

    @router.post(
        "/tagging-sessions/{session_id}/assist/estimate",
        response_model=EstimateResponse,
        summary="Estimate the credit cost of auto-tagging the remaining rows",
    )
    def assist_estimate(session_id: str, user: AuthenticatedUserDep) -> EstimateResponse:
        """Estimate credits for tagging every currently-unlabeled row.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            Row count, model id and a low/high credit range.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            config = _effective_config(row)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        _require_known_model(assist)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        instructions = tagging.compile_instructions(config, rubric, examples)
        pending = untagged_rows(data, annotations)
        return EstimateResponse(
            **tagging.estimate_credits_for_rows(
                instructions, pending, model=cast("str | None", config.get("model"))
            )
        )

    @router.post(
        "/tagging-sessions/{session_id}/assist/autotag",
        response_model=AutotagStartResponse,
        status_code=202,
        summary="Start (or resume) the bulk auto-tag job",
    )
    def assist_autotag_start(session_id: str, user: AuthenticatedUserDep) -> AutotagStartResponse:
        """Submit the bulk auto-tag job to the background worker fleet.

        Only rows without a final label are processed, so calling this after
        a cancel, failure or restart simply resumes where the job stopped.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            The number of rows the job will tag.

        Raises:
            DomainError: 409 when a job is already active for this session,
                422 when every row is already labeled, 503 when the worker is
                not available yet.
        """
        worker = get_worker_ref()
        if worker is None:
            raise DomainError("tagger.assist.worker_unavailable", status=503)
        job_id = str(uuid4())
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            pending = untagged_rows(data, annotations)
            if not pending:
                raise DomainError("tagger.assist.nothing_to_tag", status=422)
            state = dict(cast("dict[str, Any]", row.assist) or {})
            _require_known_model(state)
            prior = dict(state.get("autotag") or {})
            prior_job = str(prior.get("job_id") or "")
            if (
                prior.get("status") == "running"
                and prior_job
                and _job_status(job_store, prior_job) in _ACTIVE_JOB_STATUSES
            ):
                raise DomainError("tagger.assist.autotag_running", status=409)
            state["autotag"] = {
                "status": "running",
                "total": len(pending),
                "done": 0,
                "credits_spent": int(prior.get("credits_spent", 0)),
                "job_id": job_id,
            }
            row.assist = cast(Any, state)
            row.phase = cast(Any, "autotagging")
            row.updated_at = cast(Any, datetime.now(UTC))
            session_name = cast(str, row.name)
            db.commit()
        # The session row must show "running" before the job can be claimed:
        # the worker reads it as its first act.
        job_store.create_job(job_id, username=user.username)
        job_store.set_payload_overview(
            job_id,
            {
                PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_TAGGING,
                PAYLOAD_OVERVIEW_USERNAME: user.username,
                PAYLOAD_OVERVIEW_NAME: f"Auto-tagging · {session_name}"[:200],
            },
        )
        worker.submit_job(job_id, TaggingAutotagPayload(session_id=session_id, username=user.username))
        return AutotagStartResponse(total=len(pending))

    @router.get(
        "/tagging-sessions/{session_id}/assist/autotag",
        response_model=AutotagStatusResponse,
        summary="Poll bulk auto-tag progress",
    )
    def assist_autotag_status(session_id: str, user: AuthenticatedUserDep) -> AutotagStatusResponse:
        """Report the bulk job's progress, reconciled against the job row.

        The session row's ``assist.autotag`` mirror is the primary source;
        when it claims ``running``, the job row decides ``live`` — and a job
        that reached a terminal state without updating the mirror (e.g. it
        exhausted its orphan-recovery attempts) is reported with that
        terminal state instead of a phantom ``running``.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            Status, done/total counters, credits spent, and ``live``.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            state = cast("dict[str, Any]", row.assist) or {}
        autotag = state.get("autotag") or {}
        status = str(autotag.get("status", "done"))
        live = False
        job_id = str(autotag.get("job_id") or "")
        if status == "running" and job_id:
            job_status = _job_status(job_store, job_id)
            live = job_status in _ACTIVE_JOB_STATUSES
            if job_status == "failed":
                status = "failed"
            elif job_status in ("cancelled", "paused"):
                status = "canceled"
        return AutotagStatusResponse(
            status=status,
            total=int(autotag.get("total", 0)),
            done=int(autotag.get("done", 0)),
            credits_spent=int(autotag.get("credits_spent", 0)),
            live=live if status == "running" else False,
        )

    @router.delete(
        "/tagging-sessions/{session_id}/assist/autotag",
        summary="Cancel the running bulk auto-tag job",
    )
    def assist_autotag_cancel(session_id: str, user: AuthenticatedUserDep) -> dict[str, Any]:
        """Request cooperative cancellation of the running bulk job.

        Labels written so far are kept; the job stops at the next batch
        boundary. The job-row status is flipped to ``cancelled`` with a CAS so
        a pod other than the one running the job observes it through its
        status poll (cross-pod cancel), and the local worker — when it is the
        one processing — is signalled directly.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            ``{"cancelled": bool}`` — false when no active job was found.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            state = cast("dict[str, Any]", row.assist) or {}
        job_id = str((state.get("autotag") or {}).get("job_id") or "")
        if not job_id:
            return {"cancelled": False}
        worker = get_worker_ref()
        locally_cancelled = bool(worker.cancel_job(job_id)) if worker is not None else False
        cas = getattr(job_store, "update_job_if_status", None)
        store_cancelled = False
        if cas is not None:
            try:
                store_cancelled = bool(
                    cas(
                        job_id,
                        _ACTIVE_JOB_STATUSES,
                        status="cancelled",
                        completed_at=datetime.now(UTC).isoformat(),
                    )
                )
            except KeyError:
                store_cancelled = False
        return {"cancelled": locally_cancelled or store_cancelled}

    return router
