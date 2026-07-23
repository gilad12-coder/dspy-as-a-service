"""Bulk auto-tagging as a DB-lease worker job. [INTERNAL]

Runs the tagger's "tag the rest" phase inside the background worker so any pod
can claim it, instead of the original in-process thread. The job's payload is
tiny (``{session_id, username}``); all real state lives on the tagging session
row itself — per-batch label/provenance merges and the ``assist.autotag``
progress mirror the frontend polls. Because only rows without a final label
are ever processed, a rerun after cancel, crash or orphan recovery is a plain
resume.

Unlike optimization runs there is no subprocess: the LLM batch loop runs in
the worker thread, so this module owns the responsibilities the subprocess
drain loop normally covers — renewing the claim lease via the injected
``heartbeat`` and watching both the in-memory cancel event and the persisted
job status (cross-pod cancel) through a small monitor thread.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..billing.metering import meter_llm_run
from ..service_gateway import tagging
from ..storage.models import TaggingSessionModel

logger = logging.getLogger(__name__)

# Cadence of the lease/cancel monitor; well inside the 60s claim lease.
MONITOR_TICK_SECONDS = 5.0


class TaggingAutotagPayload(BaseModel):
    """Worker payload for a bulk auto-tag job — everything else is on the session row."""

    session_id: str
    username: str


def is_labeled(value: Any) -> bool:
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


def untagged_rows(data: list[dict[str, Any]], annotations: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the rows that carry no final label yet.

    Args:
        data: The session's full row payload.
        annotations: The ``{row_id: value}`` final-label map.
    """
    return [row for row in data if not is_labeled(annotations.get(str(row.get("id"))))]


def _write_terminal(engine: Any, session_id: str, status: str, credits: int) -> None:
    """Write the session row's terminal autotag state (best-effort).

    Args:
        engine: SQLAlchemy engine of the job store.
        session_id: UUID of the tagger session.
        status: Terminal autotag status (``done`` / ``canceled`` / ``failed``).
        credits: Credits spent by this run, added to any prior spend.
    """
    try:
        with Session(engine) as db:
            row = db.get(TaggingSessionModel, session_id)
            if row is None:
                return
            state = dict(cast("dict[str, Any]", row.assist) or {})
            autotag = dict(state.get("autotag") or {})
            autotag["status"] = status
            autotag["credits_spent"] = int(autotag.get("credits_spent", 0)) + credits
            state["autotag"] = autotag
            row.assist = cast(Any, state)
            if status == "done":
                row.phase = cast(Any, "complete")
            row.updated_at = cast(Any, datetime.now(UTC))
            db.commit()
    except Exception:
        logger.exception("failed to persist auto-tag terminal state for %s", session_id)


def run_autotag_job(
    job_store: Any,
    optimization_id: str,
    session_id: str,
    *,
    username: str = "",
    cancel_event: threading.Event,
    heartbeat: Any,
) -> dict[str, Any]:
    """Tag every unlabeled row of a session, persisting progress onto its row.

    Each completed batch is merged into the session row under a ``FOR UPDATE``
    read so concurrent batch writers never lose updates, and a human label
    placed while the job runs always wins. A monitor thread renews the worker's
    claim lease and folds cross-pod cancellation (persisted job status) into
    the local stop signal; a stolen lease also lands here because the worker's
    heartbeat sets ``cancel_event`` itself.

    Args:
        job_store: Job store whose engine backs the session rows and whose
            status read drives cross-pod cancel.
        optimization_id: The job row's id (for the cross-pod status poll).
        session_id: UUID of the tagger session being tagged.
        username: Account the run's LM usage is debited to (the job
            initiator, from the worker payload); empty skips billing.
        cancel_event: The worker's per-job cooperative cancel flag.
        heartbeat: Zero-arg callable renewing the claim lease; called every
            monitor tick.

    Returns:
        ``{"status": "done", "rows_tagged", "credits_spent"}`` on completion;
        ``{"status": "cancelled"}`` when the user cancelled; ``{"status":
        "aborted"}`` when the claim was lost to a peer pod (no terminal state
        is written — the peer owns the session now).

    Raises:
        Exception: Any batch-loop failure, after the session row is marked
            ``failed`` — the worker's generic handler marks the job row.
    """
    engine = job_store.engine
    stop = threading.Event()
    done = threading.Event()

    def monitor() -> None:
        """Renew the lease and fold every cancel source into ``stop``."""
        while not done.wait(MONITOR_TICK_SECONDS):
            try:
                heartbeat()
            except Exception:
                logger.exception("autotag heartbeat failed for %s", optimization_id)
            if cancel_event.is_set():
                stop.set()
                return
            try:
                status = job_store.get_job_status_fields(optimization_id).get("status")
            except Exception:
                continue
            if status in ("cancelled", "paused"):
                stop.set()
                return

    monitor_thread = threading.Thread(
        target=monitor, name=f"tagger-autotag-monitor-{session_id[:8]}", daemon=True
    )
    monitor_thread.start()

    credits = 0
    usage_sink: list = []
    try:
        with Session(engine) as db:
            row = db.get(TaggingSessionModel, session_id)
            if row is None:
                raise ValueError(f"tagging session {session_id} not found")
            config = dict(cast("dict[str, Any]", row.config))
            data = cast("list[dict[str, Any]]", row.data)
            annotations = dict(cast("dict[str, Any]", row.annotations))
            assist = dict(cast("dict[str, Any]", row.assist) or {})
        # The interview's task refinements — including the inferred answer
        # style and categories — live beside the immutable config.
        config = tagging.effective_task_config(config, assist)
        rubric = [str(r) for r in assist.get("rubric") or []]
        examples = tagging.select_examples(config, data, annotations, assist)
        instructions = tagging.compile_instructions(config, rubric, examples)
        pending = untagged_rows(data, annotations)

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
                    if is_labeled(anns.get(row_id)):
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
                fresh.tagged_count = cast(Any, _count_labeled(anns))
                fresh.updated_at = cast(Any, datetime.now(UTC))
                db.commit()

        tagged, credits = tagging.predict_rows(
            config, instructions, pending, on_batch=persist_batch, cancel=stop, usage_sink=usage_sink
        )
    except Exception:
        done.set()
        _write_terminal(engine, session_id, "failed", credits)
        raise
    finally:
        done.set()
        # Every exit path — success, cancel, failure, even a lease-loss abort —
        # debits the LM calls this pod actually made; a resumed job re-tags only
        # still-unlabeled rows, so reruns never double-bill.
        meter_llm_run(engine, username, usage_sink, description="Auto-tagging")

    if stop.is_set():
        if cancel_event.is_set() and not _job_cancelled(job_store, optimization_id):
            # The cancel came from the worker's own lease-loss self-cancel, not
            # the user: a peer pod owns the session now, so write nothing.
            logger.warning("autotag %s abandoned after lease loss", optimization_id)
            return {"status": "aborted"}
        _write_terminal(engine, session_id, "canceled", credits)
        return {"status": "cancelled"}

    _write_terminal(engine, session_id, "done", credits)
    return {"status": "done", "rows_tagged": len(tagged), "credits_spent": credits}


def _job_cancelled(job_store: Any, optimization_id: str) -> bool:
    """Whether the job row carries a user-driven terminal cancel/pause status.

    Args:
        job_store: Job store to read the status from.
        optimization_id: The job row's id.
    """
    try:
        status = job_store.get_job_status_fields(optimization_id).get("status")
    except Exception:
        return False
    return status in ("cancelled", "paused")


def _count_labeled(annotations: dict[str, Any]) -> int:
    """Count rows carrying a present, non-empty final label.

    Args:
        annotations: The ``{row_id: value}`` final-label map.
    """
    return sum(1 for value in annotations.values() if is_labeled(value))
