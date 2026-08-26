"""Tests for the converters that adapt stored job rows into API DTOs."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ...constants import COMPOSITION_WORKFLOW, PAYLOAD_OVERVIEW_COMPOSITION, TQDM_REMAINING_KEY
from ...models import OptimizationStatus
from ..converters import (
    compute_elapsed,
    extract_estimated_remaining,
    overview_to_base_fields,
    parse_overview,
    parse_timestamp,
    status_to_job_status,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", OptimizationStatus.pending),
        ("validating", OptimizationStatus.validating),
        ("running", OptimizationStatus.running),
        ("success", OptimizationStatus.success),
        ("failed", OptimizationStatus.failed),
        ("cancelled", OptimizationStatus.cancelled),
    ],
    ids=["pending", "validating", "running", "success", "failed", "cancelled"],
)
def test_status_to_job_status_maps_valid_values(raw: str, expected: OptimizationStatus) -> None:
    """Every recognised status string maps to its enum member."""
    assert status_to_job_status(raw) == expected


def test_status_to_job_status_unknown_falls_back_to_pending() -> None:
    """Unknown status strings fall back to ``pending``."""
    assert status_to_job_status("nonsense") == OptimizationStatus.pending


def test_status_to_job_status_empty_string_falls_back_to_pending() -> None:
    """Empty status strings fall back to ``pending``."""
    assert status_to_job_status("") == OptimizationStatus.pending


def test_parse_timestamp_returns_none_for_none() -> None:
    """``None`` input yields ``None`` output."""
    assert parse_timestamp(None) is None


def test_parse_timestamp_returns_none_for_empty_string() -> None:
    """An empty string yields ``None``."""
    assert parse_timestamp("") is None


def test_parse_timestamp_passthrough_for_datetime() -> None:
    """An already-parsed ``datetime`` is returned unchanged."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    assert parse_timestamp(dt) is dt


def test_parse_timestamp_parses_iso_string_with_offset() -> None:
    """ISO strings with explicit offsets parse to a ``datetime``."""
    result = parse_timestamp("2024-01-15T12:00:00+00:00")
    assert isinstance(result, datetime)
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 15


def test_parse_timestamp_parses_z_suffix() -> None:
    """ISO strings ending in ``Z`` parse correctly."""
    result = parse_timestamp("2024-06-01T08:30:00Z")
    assert isinstance(result, datetime)
    assert result.hour == 8
    assert result.minute == 30


def test_parse_timestamp_returns_none_for_invalid_string() -> None:
    """Strings that aren't ISO timestamps return ``None``."""
    assert parse_timestamp("not-a-date") is None


def test_parse_timestamp_returns_none_for_non_string_non_datetime() -> None:
    """Unsupported input types return ``None``."""
    assert parse_timestamp(42) is None


def _ts(y: int, m: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> datetime:
    """Build a UTC ``datetime`` for use in elapsed-time tests.

    Args:
        y: Year component.
        m: Month component.
        d: Day component.
        h: Hour component (defaults to 0).
        mi: Minute component (defaults to 0).
        s: Second component (defaults to 0).

    Returns:
        A timezone-aware UTC ``datetime``.
    """
    return datetime(y, m, d, h, mi, s, tzinfo=UTC)


def test_compute_elapsed_not_started_returns_none_pair() -> None:
    """A job that has not started yet has no elapsed time."""
    created = _ts(2024, 1, 1, 10, 0, 0)
    elapsed_str, elapsed_secs = compute_elapsed(created, started_at=None, completed_at=None)
    assert elapsed_str is None
    assert elapsed_secs is None


def test_compute_elapsed_completed_job_is_exact() -> None:
    """A completed job's elapsed time is the difference between start and end."""
    created = _ts(2024, 1, 1, 10, 0, 0)
    started = _ts(2024, 1, 1, 10, 0, 0)
    completed = _ts(2024, 1, 1, 10, 1, 5)  # 65 seconds

    elapsed_str, elapsed_secs = compute_elapsed(created, started, completed)

    assert elapsed_str == "00:01:05"
    assert elapsed_secs == 65.0


def test_compute_elapsed_formats_hours_minutes_seconds() -> None:
    """Elapsed time formats as ``HH:MM:SS`` even past the one-hour mark."""
    created = _ts(2024, 1, 1, 0, 0, 0)
    started = _ts(2024, 1, 1, 0, 0, 0)
    completed = _ts(2024, 1, 1, 2, 3, 7)  # 2h 3m 7s = 7387s

    elapsed_str, _ = compute_elapsed(created, started, completed)

    assert elapsed_str == "02:03:07"


def test_compute_elapsed_clamps_to_zero_for_completed_before_started() -> None:
    """Clock-skew negatives are clamped to zero."""
    created = _ts(2024, 1, 1, 10, 0, 0)
    started = _ts(2024, 1, 1, 10, 0, 5)
    completed = _ts(2024, 1, 1, 10, 0, 0)  # before started

    elapsed_str, elapsed_secs = compute_elapsed(created, started, completed)

    assert elapsed_secs == 0.0
    assert elapsed_str == "00:00:00"


def test_compute_elapsed_running_job_uses_wall_clock_and_is_positive() -> None:
    """A running job uses wall-clock now to produce a non-negative duration."""
    created = _ts(2024, 1, 1, 10, 0, 0)
    started = datetime.now(UTC)  # just started
    elapsed_str, elapsed_secs = compute_elapsed(created, started, completed_at=None)

    assert elapsed_str is not None
    assert elapsed_secs is not None
    assert elapsed_secs >= 0.0


def test_compute_elapsed_folds_accumulated_into_current_leg() -> None:
    """A resumed run sums banked prior-leg time with the current leg's duration."""
    created = _ts(2024, 1, 1, 10, 0, 0)
    started = _ts(2024, 1, 1, 12, 0, 0)
    completed = _ts(2024, 1, 1, 12, 1, 5)  # current leg = 65s

    elapsed_str, elapsed_secs = compute_elapsed(created, started, completed, accumulated_seconds=100.0)

    assert elapsed_secs == 165.0
    assert elapsed_str == "00:02:45"


def test_compute_elapsed_reports_banked_time_before_resumed_leg_starts() -> None:
    """Between the resume and the worker re-stamping started_at, elapsed shows banked time only.

    The paused gap (now - created) is never counted: with started_at cleared the
    current leg contributes nothing and only the accumulated prior legs remain.
    """
    created = _ts(2024, 1, 1, 10, 0, 0)

    elapsed_str, elapsed_secs = compute_elapsed(
        created, started_at=None, completed_at=None, accumulated_seconds=180.0
    )

    assert elapsed_secs == 180.0
    assert elapsed_str == "00:03:00"


def test_parse_overview_returns_dict_when_already_dict() -> None:
    """An overview already stored as a dict is returned as-is."""
    job = {"payload_overview": {"job_type": "run"}}
    assert parse_overview(job) == {"job_type": "run"}


def test_parse_overview_parses_json_string() -> None:
    """An overview stored as a JSON string is parsed into a dict."""
    job = {"payload_overview": json.dumps({"job_type": "grid_search", "username": "alice"})}
    result = parse_overview(job)
    assert result["job_type"] == "grid_search"
    assert result["username"] == "alice"


def test_parse_overview_invalid_json_returns_empty_dict() -> None:
    """Malformed overview JSON falls back to an empty dict."""
    job = {"payload_overview": "not-json{{"}
    assert parse_overview(job) == {}


def test_parse_overview_missing_key_returns_empty_dict() -> None:
    """A job row without a ``payload_overview`` field yields an empty dict."""
    assert parse_overview({}) == {}


def test_extract_estimated_remaining_returns_formatted_seconds() -> None:
    """Integer seconds remaining are formatted as ``HH:MM:SS``."""
    job = {"latest_metrics": {TQDM_REMAINING_KEY: 125}}  # 2m 5s
    assert extract_estimated_remaining(job) == "00:02:05"


def test_extract_estimated_remaining_float_input() -> None:
    """Float seconds remaining are also formatted as ``HH:MM:SS``."""
    job = {"latest_metrics": {TQDM_REMAINING_KEY: 3661.9}}  # 1h 1m 1s
    assert extract_estimated_remaining(job) == "01:01:01"


def test_extract_estimated_remaining_zero_returns_formatted_zero() -> None:
    """Zero seconds remaining renders as ``00:00:00`` rather than ``None``."""
    job = {"latest_metrics": {TQDM_REMAINING_KEY: 0}}
    assert extract_estimated_remaining(job) == "00:00:00"


def test_extract_estimated_remaining_missing_key_returns_none() -> None:
    """A missing ``remaining`` key yields ``None``."""
    assert extract_estimated_remaining({"latest_metrics": {}}) is None


def test_extract_estimated_remaining_none_metrics_returns_none() -> None:
    """A ``None`` metrics blob yields ``None``."""
    assert extract_estimated_remaining({"latest_metrics": None}) is None


def test_extract_estimated_remaining_negative_value_returns_none() -> None:
    """Negative remaining values are treated as missing data."""
    job = {"latest_metrics": {TQDM_REMAINING_KEY: -5}}
    assert extract_estimated_remaining(job) is None


def test_overview_to_base_fields_defaults_type_to_run() -> None:
    """``optimization_type`` defaults to ``run`` when no key is present."""
    fields = overview_to_base_fields({})
    assert fields["optimization_type"] == "run"


def test_overview_to_base_fields_picks_up_job_type() -> None:
    """``optimization_type`` is propagated when present in the overview."""
    fields = overview_to_base_fields({"optimization_type": "grid_search"})
    assert fields["optimization_type"] == "grid_search"


def test_overview_to_base_fields_pinned_default_false() -> None:
    """``pinned`` defaults to ``False``."""
    fields = overview_to_base_fields({})
    assert fields["pinned"] is False


def test_overview_to_base_fields_passes_through_model_name() -> None:
    """``model_name`` is propagated unchanged."""
    fields = overview_to_base_fields({"model_name": "gpt-4o-mini"})
    assert fields["model_name"] == "gpt-4o-mini"


def test_overview_to_base_fields_passes_through_composition() -> None:
    """``composition`` is propagated when present in the overview."""
    fields = overview_to_base_fields({PAYLOAD_OVERVIEW_COMPOSITION: COMPOSITION_WORKFLOW})
    assert fields["composition"] == COMPOSITION_WORKFLOW


def test_overview_to_base_fields_composition_defaults_to_none() -> None:
    """``composition`` is ``None`` for legacy rows submitted before the field existed."""
    fields = overview_to_base_fields({})
    assert fields["composition"] is None


def test_overview_to_base_fields_module_kwargs_defaults_to_empty_dict() -> None:
    """``module_kwargs`` defaults to an empty dict."""
    fields = overview_to_base_fields({})
    assert fields["module_kwargs"] == {}


def test_overview_to_base_fields_only_optimization_type_key() -> None:
    """A bare ``optimization_type`` key drives the resulting type field."""
    fields = overview_to_base_fields({"optimization_type": "grid_search"})
    assert fields["optimization_type"] == "grid_search"


def test_overview_to_base_fields_legacy_job_type_key_is_ignored() -> None:
    """Legacy 'job_type' is intentionally NOT read.

    The overview is normalised to 'optimization_type' before storage. Default
    'run' is returned when only the legacy key is present.
    """
    fields = overview_to_base_fields({"job_type": "grid_search"})
    assert fields["optimization_type"] == "run"


def test_overview_to_base_fields_both_keys_optimization_type_wins() -> None:
    """When both keys exist, ``optimization_type`` takes precedence."""
    fields = overview_to_base_fields({"optimization_type": "grid_search", "job_type": "run"})
    assert fields["optimization_type"] == "grid_search"


def test_overview_to_base_fields_neither_key_defaults_to_run() -> None:
    """An empty overview defaults to ``run``."""
    fields = overview_to_base_fields({})
    assert fields["optimization_type"] == "run"
