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

Access is resolved through :mod:`core.api.tagging_session_access`: mutating
assist routes need ``editor`` on the session, read-only ones (estimate, autotag
status) need ``viewer``. Hidden from the public Scalar reference.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...billing import ProviderKeyVault, resolve_byok_model_config
from ...billing.metering import meter_llm_run
from ...constants import (
    OPTIMIZATION_TYPE_TAGGING,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_USERNAME,
    TOKEN_SOURCE_BYOK,
    TOKEN_SOURCE_MANAGED,
)
from ...models import ModelConfig
from ...service_gateway import tagging
from ...storage.models import TaggingSessionModel
from ...worker.tagging_job import TaggingAutotagPayload, untagged_rows
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..model_catalog import get_catalog_cached
from ..model_router import route_menu_model
from ..sharing_access import ShareRole
from ..tagging_session_access import require_role
from ._helpers import enforce_llm_credits, sse_from_events, stream_with_llm_metering

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

MAX_PREDICT_ROWS = 50
MAX_INTERVIEW_TURNS = 40

# Job-row statuses that mean the worker fleet still owns the job.
_ACTIVE_JOB_STATUSES = ("pending", "validating", "running")


# One transcript turn as the client replays it. Clients echo their stored
# turns, which carry extra bookkeeping fields (``model``, ``servedModel`` —
# possibly null); parsing into this model drops them so only role/content
# reach the interview engine.
class InterviewTurn(BaseModel):
    role: str
    content: str


class InterviewRequest(BaseModel):
    """One interview exchange: the transcript so far, oldest turn first."""

    turns: list[InterviewTurn] = Field(default_factory=list, max_length=MAX_INTERVIEW_TURNS)
    locale: str | None = Field(
        default=None,
        description="UI locale code; the assistant replies in that language.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "LiteLLM id of the catalog model conducting the interview (the "
            "composer's model menu). Absent routes automatically (balanced "
            "tier); the sentinel 'auto:intelligent' routes to a frontier-"
            "quality model."
        ),
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Explicit reasoning-effort level for the chosen model; absent "
            "keeps the model's default."
        ),
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


def _load_for_role(
    session: Session,
    session_id: str,
    user: AuthenticatedUser,
    minimum: ShareRole = ShareRole.editor,
) -> TaggingSessionModel:
    """Load a session row and enforce a minimum effective role on it.

    Args:
        session: An open SQLAlchemy session.
        session_id: UUID of the tagger session.
        user: The authenticated caller.
        minimum: Lowest tier the route requires (mutating assist routes need
            ``editor``; read-only ones pass ``viewer``).

    Returns:
        The loaded row.

    Raises:
        DomainError: 404 when unknown or inaccessible, 403 when the caller's
            tier is below ``minimum``.
    """
    row = session.get(TaggingSessionModel, session_id)
    if row is None:
        raise DomainError("tagger.session.not_found", status=404)
    require_role(session, session_id, user, minimum)
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
    model_config = tagging.assist_model_config(assist)
    if model_config.token_source == TOKEN_SOURCE_BYOK:
        return
    model = str((assist or {}).get("model") or "").strip()
    if not model:
        return
    if all(entry.value != model for entry in get_catalog_cached().models):
        raise DomainError("tagger.assist.unknown_model", status=422)


def _resolve_assist_model(job_store: Any, username: str, assist: dict[str, Any]) -> ModelConfig:
    """Validate and resolve the session's tagging model for execution.

    Args:
        job_store: Job store whose engine backs the encrypted provider vault.
        username: Account that owns the selected BYOK connection.
        assist: Persisted session assist state.

    Returns:
        The managed model config or a BYOK copy carrying its decrypted runtime
        connection only in memory.

    Raises:
        DomainError: 400 when BYOK lacks a verified matching connection, or
            422 when a managed model is outside the curated catalog.
    """
    _require_known_model(assist)
    model_config = tagging.assist_model_config(assist)
    if model_config.token_source != TOKEN_SOURCE_BYOK:
        return model_config
    engine = getattr(job_store, "engine", None)
    if engine is None:
        raise DomainError("billing.byok_missing_connection", status=400, provider="")
    try:
        return resolve_byok_model_config(
            model_config,
            username=username,
            vault=ProviderKeyVault(engine=engine),
        )
    except ValueError as exc:
        raise DomainError("billing.byok_missing_connection", status=400, provider=str(exc)) from exc


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
            ``None`` when API and worker processes are split) — used as a
            local queue hint and for cooperative cancellation.

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
        enforce_llm_credits(job_store, user.username)
        with Session(job_store.engine) as db:
            row = _load_for_role(db, session_id, user)
            config = _interview_config(row)
            columns = cast("list[str]", row.columns)
            data = cast("list[dict[str, Any]]", row.data)
        model, lm_extra_body = route_menu_model(req.model, session_id=session_id)
        usage_sink: list = []
        try:
            turn = tagging.interview_turn(
                config,
                columns,
                data,
                [t.model_dump() for t in req.turns],
                req.locale,
                model=model,
                reasoning_effort=req.reasoning_effort,
                lm_extra_body=lm_extra_body,
                usage_sink=usage_sink,
            )
        except Exception as exc:
            logger.exception("interview turn failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        finally:
            # A failed turn's retries still consumed tokens; bill what ran.
            meter_llm_run(
                job_store.engine, user.username, usage_sink, description="Tagging interview"
            )
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
        await asyncio.to_thread(enforce_llm_credits, job_store, user.username)
        with Session(job_store.engine) as db:
            row = _load_for_role(db, session_id, user)
            config = _interview_config(row)
            columns = cast("list[str]", row.columns)
            data = cast("list[dict[str, Any]]", row.data)
        model, lm_extra_body = route_menu_model(req.model, session_id=session_id)
        usage_sink: list = []

        async def source() -> Any:
            """Relay engine events, translating failures into an error event."""
            try:
                async for event in tagging.interview_turn_stream(
                    config,
                    columns,
                    data,
                    [t.model_dump() for t in req.turns],
                    req.locale,
                    model=model,
                    reasoning_effort=req.reasoning_effort,
                    lm_extra_body=lm_extra_body,
                    usage_sink=usage_sink,
                ):
                    yield event
            except Exception:
                logger.exception("interview stream failed for session %s", session_id)
                yield {"event": "error", "data": {"code": "tagger.assist.llm_failed"}}

        metered = stream_with_llm_metering(
            source(),
            job_store=job_store,
            username=user.username,
            description="Tagging interview",
            usage_sink=usage_sink,
        )
        return StreamingResponse(
            sse_from_events(metered),
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
            row = _load_for_role(db, session_id, user)
            config = _effective_config(row)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        model_config = _resolve_assist_model(job_store, user.username, assist)
        wanted = set(req.row_ids)
        rows = [r for r in data if str(r.get("id")) in wanted]
        if not rows:
            raise DomainError("tagger.assist.rows_not_found", status=404)
        enforce_llm_credits(job_store, user.username)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist, exclude_ids=wanted)
        instructions = tagging.compile_instructions(config, rubric, examples)
        usage_sink: list = []
        try:
            predictions, credits = tagging.predict_rows(
                config,
                instructions,
                rows,
                usage_sink=usage_sink,
                model_config=model_config,
            )
        except Exception as exc:
            logger.exception("prediction failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        finally:
            # A failed batch's completed calls still consumed tokens; bill what ran.
            meter_llm_run(
                job_store.engine,
                user.username,
                usage_sink,
                description="Tagging predictions",
                token_source=model_config.token_source or TOKEN_SOURCE_MANAGED,
            )
        return PredictResponse(predictions=predictions, credits=credits)

    @router.post(
        "/tagging-sessions/{session_id}/assist/predict/stream",
        summary="Predict labels for specific rows as an SSE stream",
    )
    async def assist_predict_stream(
        session_id: str, req: PredictRequest, user: AuthenticatedUserDep
    ) -> StreamingResponse:
        """Stream per-row label predictions as the model produces them.

        The streaming twin of the predict route — same few-shot compilation
        and row lookup — emitting a ``prediction`` event per row the moment
        its label lands, a terminal ``predict_done`` with the merged map and
        credit cost, and ``error`` on total failure.

        Args:
            session_id: UUID of the tagger session.
            req: The row ids to predict (capped at ``MAX_PREDICT_ROWS``).
            user: Authenticated caller; must own the session.

        Returns:
            A ``text/event-stream`` response.
        """
        await asyncio.to_thread(enforce_llm_credits, job_store, user.username)
        with Session(job_store.engine) as db:
            row = _load_for_role(db, session_id, user)
            config = _effective_config(row)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        model_config = _resolve_assist_model(job_store, user.username, assist)
        wanted = set(req.row_ids)
        rows = [r for r in data if str(r.get("id")) in wanted]
        if not rows:
            raise DomainError("tagger.assist.rows_not_found", status=404)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist, exclude_ids=wanted)
        instructions = tagging.compile_instructions(config, rubric, examples)
        usage_sink: list = []

        async def source() -> Any:
            """Relay engine events, translating failures into an error event."""
            try:
                async for event in tagging.predict_rows_stream(
                    config,
                    instructions,
                    rows,
                    usage_sink=usage_sink,
                    model_config=model_config,
                ):
                    yield event
            except Exception:
                logger.exception("prediction stream failed for session %s", session_id)
                yield {"event": "error", "data": {"code": "tagger.assist.llm_failed"}}

        metered = stream_with_llm_metering(
            source(),
            job_store=job_store,
            username=user.username,
            description="Tagging predictions",
            usage_sink=usage_sink,
            token_source=model_config.token_source or TOKEN_SOURCE_MANAGED,
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/tagging-sessions/{session_id}/assist/estimate",
        response_model=EstimateResponse,
        summary="Estimate the credit cost of auto-tagging the remaining rows",
    )
    def assist_estimate(session_id: str, user: AuthenticatedUserDep) -> EstimateResponse:
        """Estimate credits for tagging every currently-unlabeled row.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; needs at least ``viewer`` access.

        Returns:
            Row count, model id and a low/high credit range.
        """
        with Session(job_store.engine) as db:
            row = _load_for_role(db, session_id, user, ShareRole.viewer)
            config = _effective_config(row)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        model_config = _resolve_assist_model(job_store, user.username, assist)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        instructions = tagging.compile_instructions(config, rubric, examples)
        pending = untagged_rows(data, annotations)
        return EstimateResponse(
            **tagging.estimate_credits_for_rows(
                instructions,
                pending,
                model=model_config.name,
                token_source=model_config.token_source or TOKEN_SOURCE_MANAGED,
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
                or 422 when every row is already labeled.
        """
        worker = get_worker_ref()
        enforce_llm_credits(job_store, user.username)
        job_id = str(uuid4())
        with Session(job_store.engine) as db:
            row = _load_for_role(db, session_id, user)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            pending = untagged_rows(data, annotations)
            if not pending:
                raise DomainError("tagger.assist.nothing_to_tag", status=422)
            state = dict(cast("dict[str, Any]", row.assist) or {})
            _resolve_assist_model(job_store, user.username, state)
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
        payload = TaggingAutotagPayload(session_id=session_id, username=user.username)
        if worker is None:
            job_store.update_job(job_id, payload=payload.model_dump(mode="json", by_alias=True))
        else:
            worker.submit_job(job_id, payload)
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
            row = _load_for_role(db, session_id, user, ShareRole.viewer)
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
            row = _load_for_role(db, session_id, user)
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
