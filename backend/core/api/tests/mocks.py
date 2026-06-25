"""Centralized mock/fake builders for ``core.api.tests``.

Domain fakes that were previously scattered inline across individual test
files live here. Test-specific values (e.g. a particular metric pair that
the test is asserting against) stay inline in the test — only the generic
"give me any valid fake X" belongs here.

Example:
    >>> from core.api.tests.mocks import (
    ...     fake_background_worker,
    ...     real_grid_response_dict,
    ...     make_artifact,
    ...     make_run_result,
    ...     make_grid_job,
    ...     REAL_OPTIMIZATION_ID,
    ...     REAL_USERNAME,
    ... )
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from ...models import (
    GridSearchResponse,
    OptimizedPredictor,
    PairResult,
    ProgramArtifact,
    RunResponse,
    SplitCounts,
)

_FIXTURES_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


@cache
def _read_fixture_text(name: str) -> str:
    """Read and cache the raw text of a fixture file.

    Args:
        name: Path to the fixture relative to the fixtures root.

    Returns:
        The file contents as a string.
    """
    return (_FIXTURES_ROOT / name).read_text()


def load_fixture(name: str) -> Any:
    """Return a freshly-parsed JSON payload from a fixture file.

    The result is decoded from the cached fixture text on every call so it
    remains safe to mutate between calls.

    Args:
        name: Path to the fixture file relative to the fixtures root.

    Returns:
        The parsed JSON content (typically a ``dict`` or ``list``).
    """
    return json.loads(_read_fixture_text(name))


# These come from jobs/success_single_gepa.detail.json
REAL_OPTIMIZATION_ID: str = load_fixture("jobs/success_single_gepa.detail.json")["optimization_id"]
REAL_USERNAME: str = load_fixture("jobs/success_single_gepa.detail.json")["username"]


def real_grid_response_dict() -> dict:
    """Return the ``grid_result`` block from the recorded grid-search fixture.

    Returns:
        A fresh dict that can be mutated by the caller.
    """
    return load_fixture("jobs/success_grid.detail.json")["grid_result"]


def make_artifact(
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
) -> ProgramArtifact:
    """Build a ``ProgramArtifact`` for serve-route tests.

    Pass an empty list for ``input_fields`` to drive the 400 "missing inputs"
    response; an empty ``output_fields`` triggers the 409 "missing outputs"
    path. Passing ``None`` for both returns the fixture-backed artifact.

    Args:
        input_fields: Optional override for the optimized prompt's input fields.
        output_fields: Optional override for the optimized prompt's output fields.

    Returns:
        A ``ProgramArtifact`` ready to be embedded in a fake job result.
    """
    if input_fields is None and output_fields is None:
        raw = load_fixture("jobs/success_grid.detail.json")
        return ProgramArtifact.model_validate(raw["grid_result"]["pair_results"][0]["program_artifact"])
    prompt = OptimizedPredictor(
        predictor_name="predict",
        instructions="Be helpful.",
        input_fields=input_fields or [],
        output_fields=output_fields or [],
    )
    return ProgramArtifact(program_pickle_base64="AAAA", optimized_prompt=prompt)


def make_run_result(artifact: ProgramArtifact) -> RunResponse:
    """Build a fully-populated ``RunResponse`` wrapping the given artifact.

    Args:
        artifact: The program artifact to embed in the result.

    Returns:
        A ``RunResponse`` parsed from the recorded single-run fixture, with
        its ``program_artifact`` swapped for the supplied one.
    """
    raw = dict(load_fixture("jobs/success_single_gepa.detail.json")["result"])
    raw["program_artifact"] = artifact.model_dump()
    return RunResponse.model_validate(raw)


def make_grid_job(
    opt_id: str,
    pair_index: int = 0,
    pair_error: str | None = None,
    artifact: ProgramArtifact | None = None,
    status: str = "success",
) -> dict:
    """Build a fake grid-search job document.

    Returns the recorded fixture when both ``artifact`` and ``pair_error`` are
    ``None``; otherwise hand-builds a minimal one-pair grid result.

    Args:
        opt_id: Optimization identifier to embed in the document.
        pair_index: Pair index to assign to the synthesised pair result.
        pair_error: Optional error message for the synthesised pair.
        artifact: Optional artifact for the synthesised pair; defaults to the
            fixture artifact when omitted.
        status: Job-level status string.

    Returns:
        A dict shaped like a ``JobStore`` row for grid-search jobs.
    """
    if artifact is None and pair_error is None:
        raw = load_fixture("jobs/success_grid.detail.json")
        return {
            "optimization_id": opt_id,
            "status": status,
            "payload_overview": {
                "optimization_type": "grid_search",
                "model_name": "openai/gpt-4o-mini",
                "module_name": raw["module_name"],
                "optimizer_name": raw["optimizer_name"],
            },
            "result": raw["grid_result"],
        }
    if artifact is None:
        artifact = make_artifact()
    pair = PairResult(
        pair_index=pair_index,
        generation_model="openai/gpt-4o-mini",
        reflection_model="openai/gpt-4o",
        program_artifact=artifact,
        error=pair_error,
    )
    grid = GridSearchResponse(
        module_name="MyModule",
        optimizer_name="GEPA",
        metric_name="accuracy",
        split_counts=SplitCounts(train=10, val=3, test=3),
        total_pairs=1,
        completed_pairs=1,
        pair_results=[pair],
        best_pair=pair,
    )
    return {
        "optimization_id": opt_id,
        "status": status,
        "payload_overview": {
            "optimization_type": "grid_search",
            "model_name": "openai/gpt-4o-mini",
            "module_name": "MyModule",
            "optimizer_name": "gepa",
        },
        "result": grid.model_dump(),
    }


class _BaseFakeJobStore:
    """In-memory stand-in for the production job store.

    Implements only the methods that the routers under test reach for, with no
    persistence and no concurrency guarantees.
    """

    def __init__(self) -> None:
        """Initialise empty job, log, progress, and staged-dataset dictionaries."""
        self._jobs: dict[str, dict] = {}
        self._logs: dict[str, list] = {}
        self._progress: dict[str, list] = {}
        self._staged: dict[str, dict] = {}
        self._checkpoints: dict[str, dict[int, dict]] = {}
        self._grid_pair_results: dict[str, dict[int, dict]] = {}

    def stage_dataset(self, username: str, dataset_filename: str, rows: list[dict]) -> str:
        """Persist staged rows in memory and return their opaque id.

        Args:
            username: Owner of the staged copy.
            dataset_filename: Original filename (kept for parity with prod).
            rows: Non-empty parsed rows.

        Returns:
            The staged-dataset id.

        Raises:
            ValueError: When ``rows`` is empty.
        """
        if not rows:
            raise ValueError("staged dataset rows must be non-empty")
        staged_id = uuid4().hex
        self._staged[staged_id] = {
            "username": username,
            "dataset_filename": dataset_filename,
            "rows": list(rows),
        }
        return staged_id

    def get_staged_dataset(self, staged_dataset_id: str, username: str) -> list[dict] | None:
        """Return staged rows scoped to ``username``, or None when unmatched.

        Args:
            staged_dataset_id: Id from :meth:`stage_dataset`.
            username: Authenticated caller; guards cross-user reads.

        Returns:
            The persisted rows, or None when missing or owned by another user.
        """
        entry = self._staged.get(staged_dataset_id)
        if entry is None or entry["username"] != username:
            return None
        return list(entry["rows"])

    def seed_job(self, optimization_id: str, **fields: Any) -> dict:
        """Insert a job row, with sensible defaults for any fields not given.

        Args:
            optimization_id: Identifier for the synthesised job.
            **fields: Overrides applied on top of the default success row.

        Returns:
            The stored job row.
        """
        now = datetime.now(UTC).isoformat()
        job = {
            "optimization_id": optimization_id,
            "status": "success",
            "created_at": now,
            "started_at": now,
            "completed_at": now,
            "payload_overview": {},
            "payload": {},
            "result": None,
            "latest_metrics": {},
            "message": None,
            **fields,
        }
        self._jobs[optimization_id] = job
        return job

    def update_job(self, optimization_id: str, **fields: Any) -> None:
        """Apply field overrides to an existing job.

        Args:
            optimization_id: Identifier of the job to mutate.
            **fields: Fields to merge into the stored row.
        """
        self._jobs[optimization_id].update(fields)

    def delete_job(self, optimization_id: str) -> None:
        """Remove a job and its associated logs and progress events.

        Args:
            optimization_id: Identifier of the job to delete.
        """
        self._jobs.pop(optimization_id, None)
        self._logs.pop(optimization_id, None)
        self._progress.pop(optimization_id, None)
        self._checkpoints.pop(optimization_id, None)
        self._grid_pair_results.pop(optimization_id, None)

    def save_gepa_checkpoint(self, optimization_id: str, data: bytes, iteration: int, pair_index: int = -1) -> None:
        """Store GEPA checkpoint bytes in memory (test seam for resume tests)."""
        self._checkpoints.setdefault(optimization_id, {})[pair_index] = {"data": data, "iteration": iteration}

    def get_gepa_checkpoint(self, optimization_id: str, pair_index: int = -1):
        """Return one run/pair's in-memory checkpoint as a record, or ``None``."""
        row = self._checkpoints.get(optimization_id, {}).get(pair_index)
        if row is None:
            return None
        return SimpleNamespace(
            optimization_id=optimization_id,
            pair_index=pair_index,
            data=row["data"],
            iteration=row["iteration"],
            stored_bytes=len(row["data"]),
        )

    def list_gepa_checkpoints(self, optimization_id: str):
        """Return every stored checkpoint record for a job."""
        return [
            self.get_gepa_checkpoint(optimization_id, pair_index)
            for pair_index in self._checkpoints.get(optimization_id, {})
        ]

    def delete_gepa_checkpoint(self, optimization_id: str, pair_index: int = -1) -> None:
        """Drop one run/pair's in-memory checkpoint."""
        self._checkpoints.get(optimization_id, {}).pop(pair_index, None)

    def delete_all_gepa_checkpoints(self, optimization_id: str) -> None:
        """Drop every checkpoint for a job."""
        self._checkpoints.pop(optimization_id, None)

    def has_gepa_checkpoint(self, optimization_id: str) -> bool:
        """Return whether any checkpoint exists for the job (single run or any pair)."""
        return bool(self._checkpoints.get(optimization_id))

    def save_grid_pair_result(self, optimization_id: str, pair_index: int, result: dict) -> None:
        """Store one completed grid pair's result in memory."""
        self._grid_pair_results.setdefault(optimization_id, {})[pair_index] = result

    def get_grid_pair_results(self, optimization_id: str) -> dict:
        """Return ``{pair_index: result}`` for the grid's completed pairs."""
        return dict(self._grid_pair_results.get(optimization_id, {}))

    def delete_grid_pair_results(self, optimization_id: str) -> None:
        """Drop every stored pair result for a grid."""
        self._grid_pair_results.pop(optimization_id, None)

    def has_grid_pair_results(self, optimization_id: str) -> bool:
        """Return whether the grid has any completed-pair result stored."""
        return bool(self._grid_pair_results.get(optimization_id))

    def requeue_for_resume(self, optimization_id: str, *, bump_attempts: bool = True):
        """Flip a job back to ``pending`` in place, optionally bumping its attempt count."""
        job = self._jobs.get(optimization_id)
        if job is None:
            return None
        current = int(job.get("attempts") or 0)
        next_attempt = current + 1 if bump_attempts else current
        job["attempts"] = next_attempt
        job["status"] = "pending"
        start_raw = job.get("started_at")
        if start_raw:
            start = datetime.fromisoformat(start_raw)
            end = datetime.fromisoformat(job.get("completed_at") or datetime.now(UTC).isoformat())
            start = start if start.tzinfo else start.replace(tzinfo=UTC)
            end = end if end.tzinfo else end.replace(tzinfo=UTC)
            job["accumulated_runtime_seconds"] = float(
                job.get("accumulated_runtime_seconds") or 0.0
            ) + max(0.0, (end - start).total_seconds())
        job["completed_at"] = None
        job["started_at"] = None
        job["message"] = "Resuming" if bump_attempts else "Re-running grid pair"
        return next_attempt

    def requeue_for_rerun(self, optimization_id: str) -> bool:
        """Reset a job in place for a fresh re-run, clearing its prior artefacts."""
        job = self._jobs.get(optimization_id)
        if job is None:
            return False
        self._logs.pop(optimization_id, None)
        self._progress.pop(optimization_id, None)
        self._checkpoints.pop(optimization_id, None)
        self._grid_pair_results.pop(optimization_id, None)
        job["status"] = "pending"
        job["started_at"] = None
        job["completed_at"] = None
        job["message"] = None
        job["latest_metrics"] = {}
        job["result"] = None
        job["attempts"] = 0
        job["accumulated_runtime_seconds"] = 0.0
        return True

    def get_job(self, optimization_id: str) -> dict:
        """Return a copy of the stored job row.

        Args:
            optimization_id: Identifier of the job to fetch.

        Returns:
            A shallow copy of the job row.

        Raises:
            KeyError: If no job with the given identifier exists.
        """
        if optimization_id not in self._jobs:
            raise KeyError(optimization_id)
        return dict(self._jobs[optimization_id])

    def job_exists(self, optimization_id: str) -> bool:
        """Report whether a job with the given identifier is stored.

        Args:
            optimization_id: Identifier to look up.

        Returns:
            ``True`` if the job is present, ``False`` otherwise.
        """
        return optimization_id in self._jobs

    def list_jobs(self, **kwargs: Any) -> list[dict]:
        """Return jobs, optionally filtered and paginated.

        Args:
            **kwargs: Optional ``status``, ``username``, ``optimization_type``,
                ``limit``, and ``offset`` filters.

        Returns:
            The matching jobs as a list of rows.
        """
        rows = list(self._jobs.values())
        status = kwargs.get("status")
        username = kwargs.get("username")
        optimization_type = kwargs.get("optimization_type")
        if status:
            rows = [r for r in rows if r.get("status") == status]
        if username:
            rows = [r for r in rows if r.get("payload_overview", {}).get("username") == username]
        if optimization_type:
            rows = [r for r in rows if r.get("payload_overview", {}).get("job_type") == optimization_type]
        limit = kwargs.get("limit", len(rows))
        offset = kwargs.get("offset", 0)
        return rows[offset : offset + limit]

    def count_jobs(self, **kwargs: Any) -> int:
        """Count jobs matching the supplied filters.

        Args:
            **kwargs: Same filters accepted by ``list_jobs``; pagination
                parameters are ignored.

        Returns:
            The number of matching jobs.
        """
        k = dict(kwargs)
        k.pop("limit", None)
        k.pop("offset", None)
        return len(self.list_jobs(limit=10**9, offset=0, **k))

    def get_logs(self, optimization_id: str, **kwargs: Any) -> list:
        """Return log rows for a job, optionally filtered and paginated.

        Args:
            optimization_id: Identifier of the job whose logs are returned.
            **kwargs: Optional ``level``, ``limit``, and ``offset`` filters.

        Returns:
            The matching log rows.
        """
        rows = list(self._logs.get(optimization_id, []))
        level = kwargs.get("level")
        if level:
            rows = [r for r in rows if r.get("level") == level]
        limit = kwargs.get("limit")
        offset = kwargs.get("offset", 0)
        rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def get_log_count(self, optimization_id: str) -> int:
        """Return the number of stored log rows for a job.

        Args:
            optimization_id: Identifier of the job to inspect.

        Returns:
            The log count, ``0`` if the job has no logs.
        """
        return len(self._logs.get(optimization_id, []))

    def get_progress_events(self, optimization_id: str, *, since: int = 0) -> list:
        """Return the stored progress events for a job, skipping ``since`` leading rows.

        Args:
            optimization_id: Identifier of the job to inspect.
            since: Number of leading events to skip for delta fetches.

        Returns:
            The progress event list from offset ``since`` onward.
        """
        return list(self._progress.get(optimization_id, []))[since:]

    def get_progress_count(self, optimization_id: str) -> int:
        """Return the number of stored progress events for a job.

        Args:
            optimization_id: Identifier of the job to inspect.

        Returns:
            The progress event count, ``0`` if the job has no events.
        """
        return len(self._progress.get(optimization_id, []))

    def set_payload_overview(self, optimization_id: str, overview: dict) -> None:
        """Replace the payload overview stored on a job.

        Args:
            optimization_id: Identifier of the job to update.
            overview: New overview to store.
        """
        self._jobs[optimization_id]["payload_overview"] = dict(overview)

    def seed_raw(
        self,
        optimization_id: str,
        *,
        job: dict,
        logs: list | None = None,
        progress: list | None = None,
    ) -> None:
        """Seed an entire job row, plus optional logs and progress events.

        Args:
            optimization_id: Identifier of the synthesised job.
            job: The full job row to store.
            logs: Optional log rows to associate with the job.
            progress: Optional progress events to associate with the job.
        """
        self._jobs[optimization_id] = job
        self._logs[optimization_id] = list(logs or [])
        self._progress[optimization_id] = list(progress or [])

    def delete_jobs(self, optimization_ids: list[str]) -> None:
        """Delete a batch of jobs by identifier.

        Args:
            optimization_ids: Identifiers of the jobs to delete.
        """
        for oid in optimization_ids:
            self.delete_job(oid)

    def get_jobs_status_by_ids(self, optimization_ids: list[str]) -> dict[str, str | None]:
        """Return the status of each requested job.

        Args:
            optimization_ids: Identifiers to look up.

        Returns:
            A mapping from identifier to status; missing jobs map to ``None``.
        """
        return {oid: self._jobs[oid]["status"] if oid in self._jobs else None for oid in optimization_ids}


class FakeJobStore(_BaseFakeJobStore):
    """Concrete in-memory job store used by router unit tests."""


def fake_background_worker() -> MagicMock:
    """Build a ``MagicMock`` mimicking a healthy idle background worker.

    Returns:
        A pre-configured ``MagicMock`` exposing the methods the API expects.
    """
    w = MagicMock()
    w.threads_alive.return_value = True
    w.seconds_since_last_activity.return_value = 0.0
    w.queue_size.return_value = 0
    w.active_jobs.return_value = 0
    w.thread_count.return_value = 2
    w.submit_job = MagicMock()
    w.cancel = MagicMock()
    w.pending_ids = []
    w.status.return_value = "idle"
    return w
