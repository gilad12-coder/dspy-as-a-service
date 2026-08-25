"""Convert raw job store dicts to API response model fields."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from ..constants import (
    OPTIMIZATION_TYPE_RUN,
    PAYLOAD_OVERVIEW_COLUMN_MAPPING,
    PAYLOAD_OVERVIEW_COMPOSITION,
    PAYLOAD_OVERVIEW_DATASET_ROWS,
    PAYLOAD_OVERVIEW_DESCRIPTION,
    PAYLOAD_OVERVIEW_GENERATION_MODELS,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODEL_SETTINGS,
    PAYLOAD_OVERVIEW_MODULE_KWARGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_REFLECTION_MODEL,
    PAYLOAD_OVERVIEW_REFLECTION_MODELS,
    PAYLOAD_OVERVIEW_SOURCE_DATASET_ID,
    PAYLOAD_OVERVIEW_TASK_FINGERPRINT,
    PAYLOAD_OVERVIEW_TASK_MODEL,
    PAYLOAD_OVERVIEW_TOTAL_PAIRS,
    PAYLOAD_OVERVIEW_USERNAME,
    TQDM_REMAINING_KEY,
)
from ..models import OptimizationStatus

logger = logging.getLogger(__name__)


def status_to_job_status(status: str) -> OptimizationStatus:
    """Convert a raw status string to an OptimizationStatus enum, defaulting to pending.

    Args:
        status: Raw status string (e.g. ``"running"``).

    Returns:
        The matching enum member, or ``OptimizationStatus.pending`` for unknown values.
    """
    try:
        return OptimizationStatus(status)
    except ValueError:
        return OptimizationStatus.pending


def parse_timestamp(val: Any) -> datetime | None:
    """Parse a timestamp value into a timezone-aware datetime.

    Args:
        val: Raw timestamp value; accepts ``None``, empty string, ``datetime``, or ISO 8601 string.

    Returns:
        A ``datetime`` parsed from the input, or ``None`` for empty / unparseable values.
    """
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return None
    return None


def _seconds_to_hhmmss(seconds: float) -> str:
    """Format a non-negative number of seconds as ``HH:MM:SS``.

    Args:
        seconds: Total seconds; fractional values are truncated.

    Returns:
        Zero-padded ``HH:MM:SS`` string.
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _compute_elapsed_raw(
    created_at: datetime,
    started_at: datetime | None,
    completed_at: datetime | None,
) -> float | None:
    """Return elapsed seconds between the job reference point and now or completion.

    Uses ``started_at`` as the reference if available; falls back to ``created_at``.

    Args:
        created_at: When the job row was created.
        started_at: When the worker actually picked the job up; ``None`` if not yet running.
        completed_at: When the job ended; ``None`` while still running.

    Returns:
        Elapsed seconds as a non-negative float, or ``None`` when the job has not started yet.
    """
    ref = started_at or created_at
    if completed_at is not None:
        return max(0.0, (completed_at - ref).total_seconds())
    if started_at is not None:
        now = datetime.now(UTC)
        ref_utc = started_at if started_at.tzinfo else started_at.replace(tzinfo=UTC)
        return max(0.0, (now - ref_utc).total_seconds())
    return None


def compute_elapsed(
    created_at: datetime,
    started_at: datetime | None,
    completed_at: datetime | None,
    accumulated_seconds: float = 0.0,
) -> tuple[str | None, float | None]:
    """Return a formatted elapsed string and raw seconds for a job.

    Args:
        created_at: When the job row was created.
        started_at: When the worker actually picked the job up; ``None`` if not yet running.
        completed_at: When the job ended; ``None`` while still running.
        accumulated_seconds: Net active seconds from earlier resumed legs, added to
            the current leg so the total is cumulative compute across resumes (the
            paused gap between legs is excluded). ``0`` for a never-resumed run.

    Returns:
        A ``(HH:MM:SS, seconds)`` tuple, or ``(None, None)`` when the job has neither
        started its current leg nor banked any time from a prior leg.
    """
    seconds = _compute_elapsed_raw(created_at, started_at, completed_at)
    if seconds is None and not accumulated_seconds:
        return None, None
    total = (accumulated_seconds or 0.0) + (seconds or 0.0)
    return _seconds_to_hhmmss(total), round(total, 2)


def parse_overview(job_data: dict) -> dict:
    """Extract and deserialize the ``payload_overview`` field from a job dict.

    Handles both pre-parsed dicts and JSON strings stored by older job rows.

    Args:
        job_data: Raw job dict from the job store.

    Returns:
        The parsed overview dict, or an empty dict when the field is missing or unparseable.
    """
    overview = job_data.get("payload_overview", {})
    if isinstance(overview, str):
        try:
            overview = json.loads(overview)
        except json.JSONDecodeError:
            logger.debug("Failed to parse payload_overview as JSON, using empty dict")
            overview = {}
    return overview


def job_owner(job_data: dict) -> str | None:
    """Return the lowercased owner username for a job row, or ``None`` if absent.

    Lives here (beside :func:`parse_overview`) rather than in ``routers._helpers``
    so the access layer (:mod:`core.api.sharing_access`) can resolve ownership
    without importing the heavyweight ``_helpers`` module — keeping the share /
    grant resolver a leaf and breaking what would otherwise be an import cycle.

    Args:
        job_data: Raw job row from the job store.

    Returns:
        The job owner's lowercased username, or ``None`` when neither the
        payload nor the overview carry one.
    """
    payload = job_data.get("payload")
    if isinstance(payload, dict):
        raw = payload.get("username")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    overview = parse_overview(job_data)
    raw = overview.get(PAYLOAD_OVERVIEW_USERNAME)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return None


def extract_estimated_remaining(job_data: dict) -> str | None:
    """Read the tqdm-derived remaining-time metric and format it as ``HH:MM:SS``.

    Args:
        job_data: Raw job dict; the ``latest_metrics`` field is consulted.

    Returns:
        ``HH:MM:SS`` string, or ``None`` when no usable remaining-time value is present.
    """
    metrics = job_data.get("latest_metrics") or {}
    val = metrics.get(TQDM_REMAINING_KEY)
    if isinstance(val, (int, float)) and val >= 0:
        return _seconds_to_hhmmss(val)
    return None


def overview_to_base_fields(overview: dict) -> dict:
    """Map payload-overview keys to the flat field names expected by response models.

    Args:
        overview: Parsed payload-overview dict.

    Returns:
        Dict keyed by response-model field names with values pulled from the overview.
    """
    return {
        "optimization_type": overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN),
        "name": overview.get(PAYLOAD_OVERVIEW_NAME),
        "description": overview.get(PAYLOAD_OVERVIEW_DESCRIPTION),
        "pinned": overview.get("pinned", False),
        "username": overview.get(PAYLOAD_OVERVIEW_USERNAME),
        "module_name": overview.get(PAYLOAD_OVERVIEW_MODULE_NAME),
        "composition": overview.get(PAYLOAD_OVERVIEW_COMPOSITION),
        "module_kwargs": overview.get(PAYLOAD_OVERVIEW_MODULE_KWARGS, {}),
        "optimizer_name": overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
        "column_mapping": overview.get(PAYLOAD_OVERVIEW_COLUMN_MAPPING),
        "dataset_rows": overview.get(PAYLOAD_OVERVIEW_DATASET_ROWS),
        "source_dataset_id": overview.get(PAYLOAD_OVERVIEW_SOURCE_DATASET_ID),
        "model_name": overview.get(PAYLOAD_OVERVIEW_MODEL_NAME),
        "model_settings": overview.get(PAYLOAD_OVERVIEW_MODEL_SETTINGS),
        "reflection_model_name": overview.get(PAYLOAD_OVERVIEW_REFLECTION_MODEL),
        "task_model_name": overview.get(PAYLOAD_OVERVIEW_TASK_MODEL),
        "total_pairs": overview.get(PAYLOAD_OVERVIEW_TOTAL_PAIRS),
        "generation_models": overview.get(PAYLOAD_OVERVIEW_GENERATION_MODELS),
        "reflection_models": overview.get(PAYLOAD_OVERVIEW_REFLECTION_MODELS),
        "task_fingerprint": overview.get(PAYLOAD_OVERVIEW_TASK_FINGERPRINT),
    }
