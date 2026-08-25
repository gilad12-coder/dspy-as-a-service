"""Subprocess-side entry point for DSPy optimization jobs.

Houses the code that runs inside the child process forked/spawned by
``BackgroundWorker`` — the log handler, the event-queue helpers, and
the ``_run_service_in_subprocess`` function itself. Kept in its own
module so ``engine.py`` can stay focused on the parent-side worker
lifecycle.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from datetime import UTC, datetime
from functools import partial
from typing import Any

import dspy

from ..config import settings
from ..constants import OPTIMIZATION_TYPE_GRID_SEARCH, OPTIMIZATION_TYPE_RUN
from ..models import GridSearchRequest, RunRequest
from ..registry import ServiceRegistry
from ..service_gateway import DspyService
from ..service_gateway.language_models import activate_job_lm_budget
from ..service_gateway.optimization.llm_error import (
    LlmErrorCapture,
    enrich_error_message,
    reset_llm_error,
)
from ..service_gateway.react_compat import configure_native_tool_calling
from .constants import EVENT_ERROR, EVENT_LOG, EVENT_PROGRESS, EVENT_RESULT
from .log_handler import get_current_pair_index

# Populated by the parent via ``set_fork_service`` before forking so
# child processes can reuse the same registry-backed service.
_FORK_SERVICE: DspyService | None = None


def set_fork_service(service: DspyService | None) -> None:
    """Store a service instance that child processes may reuse after fork (or clear with None).

    Args:
        service: The service to share with forked children, or ``None`` to clear.
    """
    global _FORK_SERVICE
    _FORK_SERVICE = service


def safe_queue_put(event_queue: Any, event: dict[str, Any]) -> None:
    """Put an event onto a multiprocessing queue, swallowing any exception so the child keeps running.

    Args:
        event_queue: The shared queue to write to.
        event: Pickleable payload to enqueue.
    """
    with contextlib.suppress(Exception):
        event_queue.put(event)


def _emit_progress_event(event_queue: Any, message: str, metrics: dict[str, Any]) -> None:
    """Forward one optimizer progress event to the parent process.

    Shape expected by the parent's ``_drain_subprocess_events``: an
    ``EVENT_PROGRESS`` dict carrying the human-readable ``event`` label and
    a ``metrics`` mapping (empty if the caller passed ``None``).

    Args:
        event_queue: The shared queue used to talk to the parent.
        message: Human-readable phase label.
        metrics: Metric snapshot for this phase (may be empty/None).
    """
    safe_queue_put(
        event_queue,
        {
            "type": EVENT_PROGRESS,
            "event": message,
            "metrics": metrics or {},
        },
    )


class SubprocessLogHandler(logging.Handler):
    """Forward DSPy log records from the child process to the parent via the event queue.

    Attached to the ``dspy`` logger inside ``run_service_in_subprocess`` and removed
    in the ``finally`` block so it does not persist across calls.  Each record is
    serialised into an ``EVENT_LOG`` dict and placed on the shared queue; errors
    from the queue are suppressed via ``safe_queue_put``.
    """

    def __init__(self, event_queue: Any) -> None:
        """Initialize the handler with the shared multiprocessing queue.

        Args:
            event_queue: The shared queue to which ``EVENT_LOG`` events go.
        """
        super().__init__()
        self._event_queue = event_queue

    def emit(self, record: logging.LogRecord) -> None:
        """Format and forward a log record to the parent process via the event queue.

        Args:
            record: The log record produced by the underlying logger.
        """
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        safe_queue_put(
            self._event_queue,
            {
                "type": EVENT_LOG,
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "pair_index": get_current_pair_index(),
            },
        )


def run_service_in_subprocess(
    payload_dict: dict[str, Any],
    artifact_id: str,
    event_queue: Any,
    start_method: str,
) -> None:
    """Execute a DSPy optimization in the child process, streaming events back to the parent.

    This is the ``target`` function passed to ``mp.Process``.  It emits four event
    types onto ``event_queue``:

    - ``EVENT_LOG`` — every record from the ``dspy`` logger and the
      ``core.service_gateway.optimization`` logger at DEBUG or above, forwarded
      via ``SubprocessLogHandler``.
    - ``EVENT_PROGRESS`` — one event per ``progress_callback`` invocation from the
      optimizer, carrying ``event`` (phase name) and ``metrics`` (dict).
    - ``EVENT_RESULT`` — emitted once on success, containing the serialised
      ``RunResponse`` or ``GridSearchResponse``.
    - ``EVENT_ERROR`` — emitted on any ``BaseException``, containing the error
      string and formatted traceback.

    When ``start_method == "fork"`` and ``_FORK_SERVICE`` is set, the pre-built
    service from the parent is reused directly; otherwise a fresh ``DspyService``
    is constructed.

    Args:
        payload_dict: Submission payload plus the ``_optimization_type`` key and,
            for resumable runs, the ``_gepa_log_dir`` path the worker owns.
        artifact_id: ID used to attach progress and logs to the right job.
        event_queue: Shared multiprocessing queue back to the parent.
        start_method: The active multiprocessing start method (e.g. ``"fork"``).
    """
    # Disable dspy caching entirely for this child. The in-memory cache pins
    # every response in a process-wide LRU (1M-entry cap — unbounded in
    # practice) for its whole multi-hour lifetime, and the disk cache routes
    # through diskcache, whose pickle-backed store has an unpatched
    # deserialization flaw (GHSA-w8v5-vhqr-4h9v — no fixed release exists).
    # Dropping the disk cache keeps that path out of the worker, trading away
    # retry/resume dedup. suppress() so an incompatible dspy build can't fail
    # the job.
    with contextlib.suppress(Exception):
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    # Route ReActV2 tools through the provider's native function-calling API
    # when REACT_NATIVE_TOOL_CALLING is set, matching the serve process so
    # optimization rollouts and serving share one tool protocol. No-op when
    # off; suppress() so an incompatible dspy build can't fail the job.
    with contextlib.suppress(Exception):
        configure_native_tool_calling()
    # Cap this child's in-flight LM calls: grid pair threads x GEPA eval
    # threads multiply, and without a ceiling a single job can spray the
    # provider hard enough to trip rate limits and fail the run.
    activate_job_lm_budget(settings.job_lm_max_concurrency)
    service = _FORK_SERVICE if start_method == "fork" and _FORK_SERVICE is not None else DspyService(ServiceRegistry())
    log_handler = SubprocessLogHandler(event_queue)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(message)s"))

    # Forward DSPy's own records AND our optimization-harness records (rollout
    # heartbeats, phase logs) so both surface in job_logs at DEBUG+. The Logs
    # tab defaults to an INFO+ view and only reveals DEBUG when the operator
    # opts into the "verbose" verbosity level, so capturing DEBUG here is what
    # makes that opt-in possible. The app logger sits outside the ``dspy`` tree,
    # so it needs its own hookup.
    forwarded_loggers = (
        logging.getLogger("dspy"),
        logging.getLogger("core.service_gateway.optimization"),
    )
    saved_levels = [lg.level for lg in forwarded_loggers]
    for lg in forwarded_loggers:
        if lg.level == 0 or lg.level > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
        lg.addHandler(log_handler)

    # DSPy hides the real per-worker failure behind a generic "Execution
    # cancelled" message; this handler keeps the logged cause so EVENT_ERROR
    # can report what actually went wrong (e.g. an LLM billing/quota error).
    dspy_logger = forwarded_loggers[0]
    error_capture = LlmErrorCapture()
    reset_llm_error()
    dspy_logger.addHandler(error_capture)

    try:
        payload_dict = dict(payload_dict)
        optimization_type = payload_dict.pop("_optimization_type", OPTIMIZATION_TYPE_RUN)
        gepa_log_dir_path = payload_dict.pop("_gepa_log_dir", None)
        completed_pairs = payload_dict.pop("_completed_pairs", None)
        # Distributed grid pair: this child holds one pair of a larger grid and
        # must report the pair's GLOBAL index / the grid's real pair count.
        pair_index_base = int(payload_dict.pop("_pair_index_base", 0) or 0)
        grid_total_pairs = payload_dict.pop("_grid_total_pairs", None)
        progress_callback = partial(_emit_progress_event, event_queue)

        if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH:
            if not hasattr(service, "run_grid_search"):
                service = DspyService(ServiceRegistry())
            grid_payload = GridSearchRequest.model_validate(payload_dict)
            result: Any = service.run_grid_search(
                grid_payload,
                artifact_id=artifact_id,
                progress_callback=progress_callback,
                gepa_log_dir_path=gepa_log_dir_path,
                completed_pairs=completed_pairs,
                pair_index_base=pair_index_base,
                total_pairs_override=int(grid_total_pairs) if grid_total_pairs is not None else None,
            )
        else:
            run_payload = RunRequest.model_validate(payload_dict)
            result = service.run(
                run_payload,
                artifact_id=artifact_id,
                progress_callback=progress_callback,
                gepa_log_dir_path=gepa_log_dir_path,
            )
        safe_queue_put(
            event_queue,
            {
                "type": EVENT_RESULT,
                "result": result.model_dump(mode="json"),
            },
        )
    # subprocess isolation boundary: report any failure (including
    # SystemExit and KeyboardInterrupt) to the parent before the child exits.
    except BaseException as exc:
        safe_queue_put(
            event_queue,
            {
                "type": EVENT_ERROR,
                "error": enrich_error_message(str(exc)),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        dspy_logger.removeHandler(error_capture)
        reset_llm_error()
        for lg, saved_level in zip(forwarded_loggers, saved_levels, strict=True):
            lg.removeHandler(log_handler)
            lg.setLevel(saved_level)
