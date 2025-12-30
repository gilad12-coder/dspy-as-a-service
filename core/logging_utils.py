import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Union
from .jobs import JobManager, RedisJobStore

_ITERATION_SCORE_RE = re.compile(
    r"Iteration (?P<iteration>\d+): Selected program (?P<program>\d+) score: (?P<score>[0-9.]+)"
)
_AVERAGE_METRIC_RE = re.compile(
    r"Average Metric: (?P<numerator>[0-9.]+) / (?P<denominator>[0-9.]+) \((?P<percent>[0-9.]+)%\)"
)
_PERFECT_SUBSAMPLE_RE = re.compile(r"Iteration (?P<iteration>\d+): All subsample scores perfect")
_NO_MUTATION_RE = re.compile(
    r"Iteration (?P<iteration>\d+): Reflective mutation did not propose a new candidate"
)


class JobLogHandler(logging.Handler):
    """Route DSPy log records into the job manager for later inspection."""

    def __init__(self, job_id: str, jobs: Union[JobManager, RedisJobStore]) -> None:
        """Initialize the handler with job context.

        Args:
            job_id: Identifier for the job receiving log entries.
            jobs: Job manager or Redis store responsible for persisting logs.

        Returns:
            None
        """

        super().__init__()
        self._job_id = job_id
        self._jobs = jobs

    def emit(self, record: logging.LogRecord) -> None:
        """Persist log records on the associated job.

        Args:
            record: Log record emitted by DSPy or optimizer components.

        Returns:
            None
        """

        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        self._jobs.append_log(
            self._job_id,
            level=record.levelname,
            logger_name=record.name,
            message=message,
            timestamp=timestamp,
        )
        for event_name, metrics in _extract_progress_from_log(message):
            # RedisJobStore.record_progress doesn't have update_job_message parameter
            if isinstance(self._jobs, JobManager):
                self._jobs.record_progress(
                    self._job_id,
                    event_name,
                    metrics,
                    update_job_message=False,
                )
            else:
                self._jobs.record_progress(
                    self._job_id,
                    event_name,
                    metrics,
                )


def _extract_progress_from_log(message: str) -> Iterable[Tuple[str, dict[str, Any]]]:
    """Parse well-known DSPy log lines into structured telemetry.

    Args:
        message: Raw log line emitted by DSPy or optimizer code.

    Returns:
        Iterable[Tuple[str, dict[str, Any]]]: Zero or more derived progress events.
    """

    events: list[Tuple[str, dict[str, Any]]] = []
    if match := _ITERATION_SCORE_RE.search(message):
        events.append(
            (
                "optimizer_iteration",
                {
                    "iteration": int(match.group("iteration")),
                    "program": int(match.group("program")),
                    "score": float(match.group("score")),
                },
            )
        )
    if match := _AVERAGE_METRIC_RE.search(message):
        numerator = float(match.group("numerator"))
        denominator = float(match.group("denominator"))
        percent = float(match.group("percent"))
        events.append(
            (
                "average_metric_snapshot",
                {
                    "value": numerator,
                    "maximum": denominator,
                    "percent": percent,
                },
            )
        )
    if match := _PERFECT_SUBSAMPLE_RE.search(message):
        events.append(
            (
                "optimizer_iteration_perfect",
                {
                    "iteration": int(match.group("iteration")),
                    "perfect_subsamples": True,
                },
            )
        )
    if match := _NO_MUTATION_RE.search(message):
        events.append(
            (
                "optimizer_reflection_idle",
                {
                    "iteration": int(match.group("iteration")),
                    "mutation_proposed": False,
                },
            )
        )
    return events
