"""AI co-tagging routes for tagger sessions. [INTERNAL]

Drives the tagger's assist modes on top of the persisted session rows:
the dataset interview (rubric distillation), batched label predictions for
calibration and review rounds, reflective rubric refinement ("deep
optimize"), pre-run credit estimates, and the bulk auto-tag job.

The bulk job runs as an in-process background thread that persists its
progress onto the session row itself (``assist.autotag``), so the client
simply polls the status route; the job only ever processes rows that have no
final label yet, which makes a restart after a crash or cancel a plain
resume. Single-pod semantics: liveness is tracked in a module-level registry,
and a row that claims ``running`` without a live local thread is reported
with ``live=false`` so the client can offer a resume.

Ownership is enforced on every route by comparing the authenticated principal
to the row's ``username``. Hidden from the public Scalar reference.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...service_gateway import tagging
from ...storage.models import TaggingSessionModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from .tagging_sessions import _count_tagged

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

MAX_PREDICT_ROWS = 50
MAX_INTERVIEW_TURNS = 40

# Cancel events for bulk jobs running in this process, keyed by session id.
# Presence means a live local thread; the event is the cooperative kill switch.
_AUTOTAG_JOBS: dict[str, threading.Event] = {}
_AUTOTAG_LOCK = threading.Lock()


class InterviewRequest(BaseModel):
    """One interview exchange: the transcript so far, oldest turn first."""

    turns: list[dict[str, str]] = Field(default_factory=list, max_length=MAX_INTERVIEW_TURNS)
    locale: str | None = Field(
        default=None,
        description="UI locale code; the assistant replies in that language.",
    )


class InterviewResponse(BaseModel):
    """The assistant's next interview turn."""

    message: str
    quick_replies: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    done: bool


class PredictRequest(BaseModel):
    """Rows to predict labels for (calibration and review batches)."""

    row_ids: list[str] = Field(min_length=1, max_length=MAX_PREDICT_ROWS)


class PredictResponse(BaseModel):
    """Per-row predictions plus the credit cost of the calls made."""

    predictions: dict[str, dict[str, Any]]
    credits: int


class OptimizeRequest(BaseModel):
    """Trigger a reflective rubric rewrite from the labels so far."""

    locale: str | None = Field(
        default=None,
        description="UI locale code; the rubric is written in that language.",
    )


class OptimizeResponse(BaseModel):
    """The improved rubric."""

    rubric: list[str]


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
        description="False when the row claims 'running' but no local thread exists "
        "(e.g. after a server restart) — the client should offer a resume."
    )


def _is_labeled(value: Any) -> bool:
    """Return True when an annotation value counts as a final label.

    Args:
        value: One entry of the ``{row_id: annotation}`` map.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


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


def _untagged_rows(data: list[dict[str, Any]], annotations: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the rows that carry no final label yet.

    Args:
        data: The session's full row payload.
        annotations: The ``{row_id: value}`` final-label map.
    """
    return [row for row in data if not _is_labeled(annotations.get(str(row.get("id"))))]


def _run_autotag(engine: Any, session_id: str, cancel: threading.Event) -> None:
    """Bulk-tag every unlabeled row, persisting progress onto the session row.

    Runs on a background thread. Each completed batch is merged into the row
    under a ``FOR UPDATE`` read so concurrent batch writers never lose
    updates. Terminal states: ``done`` (phase flips to ``complete``),
    ``canceled`` (labels written so far are kept), ``failed``.

    Args:
        engine: SQLAlchemy engine of the job store.
        session_id: UUID of the tagger session being tagged.
        cancel: Cooperative cancellation event set by the DELETE route.
    """
    try:
        with Session(engine) as db:
            row = db.get(TaggingSessionModel, session_id)
            if row is None:
                return
            config = cast("dict[str, Any]", row.config)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = dict(cast("dict[str, Any]", row.annotations))
            assist = dict(cast("dict[str, Any]", row.assist) or {})
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        instructions = tagging.compile_instructions(config, rubric, examples)
        pending = _untagged_rows(data, annotations)

        def persist_batch(batch: dict[str, dict[str, Any]]) -> None:
            """Merge one completed batch into the session row."""
            with Session(engine) as db:
                fresh = (
                    db.query(TaggingSessionModel)
                    .filter(TaggingSessionModel.id == session_id)
                    .with_for_update()
                    .one_or_none()
                )
                if fresh is None:
                    return
                anns = dict(cast("dict[str, Any]", fresh.annotations))
                state = dict(cast("dict[str, Any]", fresh.assist) or {})
                predictions = dict(state.get("predictions") or {})
                provenance = dict(state.get("provenance") or {})
                autotag = dict(state.get("autotag") or {})
                for row_id, pred in batch.items():
                    # A label the human placed while the job ran wins.
                    if _is_labeled(anns.get(row_id)):
                        continue
                    anns[row_id] = pred["value"]
                    predictions[row_id] = pred
                    provenance[row_id] = "ai_auto"
                autotag["done"] = int(autotag.get("done", 0)) + len(batch)
                state.update(
                    {"predictions": predictions, "provenance": provenance, "autotag": autotag}
                )
                fresh.annotations = cast(Any, anns)
                fresh.assist = cast(Any, state)
                fresh.tagged_count = cast(Any, _count_tagged(anns))
                fresh.updated_at = cast(Any, datetime.now(UTC))
                db.commit()

        _, credits = tagging.predict_rows(
            config, instructions, pending, on_batch=persist_batch, cancel=cancel
        )
        final_status = "canceled" if cancel.is_set() else "done"
    except Exception:
        logger.exception("auto-tag job failed for session %s", session_id)
        final_status = "failed"
        credits = 0
    with _AUTOTAG_LOCK:
        _AUTOTAG_JOBS.pop(session_id, None)
    try:
        with Session(engine) as db:
            row = db.get(TaggingSessionModel, session_id)
            if row is None:
                return
            state = dict(cast("dict[str, Any]", row.assist) or {})
            autotag = dict(state.get("autotag") or {})
            autotag["status"] = final_status
            autotag["credits_spent"] = int(autotag.get("credits_spent", 0)) + credits
            state["autotag"] = autotag
            row.assist = cast(Any, state)
            if final_status == "done":
                row.phase = cast(Any, "complete")
            row.updated_at = cast(Any, datetime.now(UTC))
            db.commit()
    except Exception:
        logger.exception("failed to persist auto-tag terminal state for %s", session_id)


def create_tagger_assist_router(*, job_store) -> APIRouter:
    """Build the AI co-tagging router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.

    Returns:
        A FastAPI ``APIRouter`` exposing the interview / predict / optimize /
        estimate / autotag routes under ``/tagging-sessions/{id}/assist``.
    """
    router = APIRouter()

    @router.post(
        "/tagging-sessions/{session_id}/assist/interview",
        response_model=InterviewResponse,
        summary="Run one dataset-interview turn",
    )
    def assist_interview(
        session_id: str, req: InterviewRequest, user: AuthenticatedUserDep
    ) -> InterviewResponse:
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
            config = cast("dict[str, Any]", row.config)
            columns = cast("list[str]", row.columns)
            data = cast("list[dict[str, Any]]", row.data)
        try:
            turn = tagging.interview_turn(config, columns, data, req.turns, req.locale)
        except Exception as exc:
            logger.exception("interview turn failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        return InterviewResponse(**turn)

    @router.post(
        "/tagging-sessions/{session_id}/assist/predict",
        response_model=PredictResponse,
        summary="Predict labels for specific rows (calibration / review)",
    )
    def assist_predict(
        session_id: str, req: PredictRequest, user: AuthenticatedUserDep
    ) -> PredictResponse:
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
            config = cast("dict[str, Any]", row.config)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
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
        "/tagging-sessions/{session_id}/assist/optimize",
        response_model=OptimizeResponse,
        summary="Reflectively rewrite the rubric from labels and corrections",
    )
    def assist_optimize(
        session_id: str, req: OptimizeRequest, user: AuthenticatedUserDep
    ) -> OptimizeResponse:
        """Return an improved rubric derived from every human label so far.

        Args:
            session_id: UUID of the tagger session.
            req: The caller's UI locale (rubric language).
            user: Authenticated caller; must own the session.

        Returns:
            The rewritten rubric (the original when the model output is
            unusable).
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            config = cast("dict[str, Any]", row.config)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        if not examples:
            raise DomainError("tagger.assist.no_examples", status=422)
        try:
            improved = tagging.refine_rubric(config, rubric, examples, req.locale)
        except Exception as exc:
            logger.exception("rubric optimize failed for session %s", session_id)
            raise DomainError("tagger.assist.llm_failed", status=502) from exc
        return OptimizeResponse(rubric=improved)

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
            config = cast("dict[str, Any]", row.config)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            assist = cast("dict[str, Any]", row.assist) or {}
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        instructions = tagging.compile_instructions(config, rubric, examples)
        pending = _untagged_rows(data, annotations)
        return EstimateResponse(**tagging.estimate_credits_for_rows(instructions, pending))

    @router.post(
        "/tagging-sessions/{session_id}/assist/autotag",
        response_model=AutotagStartResponse,
        status_code=202,
        summary="Start (or resume) the bulk auto-tag job",
    )
    def assist_autotag_start(
        session_id: str, user: AuthenticatedUserDep
    ) -> AutotagStartResponse:
        """Spawn the background bulk-tagging thread for this session.

        Only rows without a final label are processed, so calling this after
        a cancel, failure or restart simply resumes where the job stopped.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            The number of rows the job will tag.

        Raises:
            DomainError: 409 when a job is already running for this session,
                422 when every row is already labeled.
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            data = cast("list[dict[str, Any]]", row.data)
            annotations = cast("dict[str, Any]", row.annotations)
            pending = _untagged_rows(data, annotations)
            if not pending:
                raise DomainError("tagger.assist.nothing_to_tag", status=422)
            with _AUTOTAG_LOCK:
                if session_id in _AUTOTAG_JOBS:
                    raise DomainError("tagger.assist.autotag_running", status=409)
                cancel = threading.Event()
                _AUTOTAG_JOBS[session_id] = cancel
            state = dict(cast("dict[str, Any]", row.assist) or {})
            prior = dict(state.get("autotag") or {})
            state["autotag"] = {
                "status": "running",
                "total": len(pending),
                "done": 0,
                "credits_spent": int(prior.get("credits_spent", 0)),
            }
            row.assist = cast(Any, state)
            row.phase = cast(Any, "autotagging")
            row.updated_at = cast(Any, datetime.now(UTC))
            db.commit()
        thread = threading.Thread(
            target=_run_autotag,
            args=(job_store.engine, session_id, cancel),
            name=f"tagger-autotag-{session_id[:8]}",
            daemon=True,
        )
        thread.start()
        return AutotagStartResponse(total=len(pending))

    @router.get(
        "/tagging-sessions/{session_id}/assist/autotag",
        response_model=AutotagStatusResponse,
        summary="Poll bulk auto-tag progress",
    )
    def assist_autotag_status(
        session_id: str, user: AuthenticatedUserDep
    ) -> AutotagStatusResponse:
        """Report the bulk job's persisted progress plus local liveness.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            Status, done/total counters, credits spent, and ``live`` (false
            when the row claims running but no local thread exists).
        """
        with Session(job_store.engine) as db:
            row = _load_owned(db, session_id, user.username)
            state = cast("dict[str, Any]", row.assist) or {}
        autotag = state.get("autotag") or {}
        status = str(autotag.get("status", "done"))
        with _AUTOTAG_LOCK:
            live = session_id in _AUTOTAG_JOBS
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

        Labels written so far are kept; the job flips its status to
        ``canceled`` at the next batch boundary.

        Args:
            session_id: UUID of the tagger session.
            user: Authenticated caller; must own the session.

        Returns:
            ``{"cancelled": bool}`` — false when no local job was running.
        """
        with Session(job_store.engine) as db:
            _load_owned(db, session_id, user.username)
        with _AUTOTAG_LOCK:
            cancel = _AUTOTAG_JOBS.get(session_id)
        if cancel is None:
            return {"cancelled": False}
        cancel.set()
        return {"cancelled": True}

    return router
