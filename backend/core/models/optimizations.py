"""Outbound payloads for GET /optimizations/* endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .artifacts import ProgramArtifact
from .common import ColumnMapping, Composition, OptimizationStatus, OptimizationType, SplitFractions
from .results import GridSearchResponse, RunResponse
from .telemetry import JobLogEntry, ProgressEvent


class _JobResponseBase(BaseModel):
    """Shared fields across optimization response endpoints."""

    optimization_id: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    message: str | None = None
    name: str | None = None
    description: str | None = None
    pinned: bool = False

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed: str | None = None
    elapsed_seconds: float | None = None
    estimated_remaining: str | None = None

    username: str | None = None
    module_name: str | None = None
    # "single" (one atomic module) vs "workflow" (a DAG of module nodes). Derived
    # at submit from module_name; null on rows submitted before this field existed.
    composition: Composition | None = None
    module_kwargs: dict[str, Any] = Field(default_factory=dict)
    optimizer_name: str | None = None
    column_mapping: ColumnMapping | None = None
    dataset_rows: int | None = None

    # The run's persisted footprint (payload + result JSON) in bytes, mirrored
    # from ``jobs.stored_bytes``. Powers the per-run size shown in the storage
    # observability surfaces; 0 until the row's size has been accounted.
    stored_bytes: int = 0

    # Id of the personal-library dataset this run was submitted from (by-reference
    # submit); null for inline/staged submits. Powers the live optimization→dataset
    # link in the UI.
    source_dataset_id: str | None = None

    # True when this run stopped mid-optimization with a saved GEPA checkpoint and
    # can be resumed in place (terminal failed/cancelled, checkpoint present,
    # attempts below the cap). Drives the Resume-vs-Restart affordance; defaults
    # False so non-resumable runs and legacy rows render exactly as before.
    resumable: bool = False

    # True while this run is actively ``running`` AND already has a saved checkpoint,
    # so a manual Pause control is offered and is guaranteed resumable. Computed on the
    # live detail-status response only; defaults False on summary/list rows.
    pausable: bool = False

    latest_metrics: dict[str, Any] = Field(default_factory=dict)

    model_name: str | None = Field(default=None, description="Single-run model name; null for grid searches.")
    model_settings: dict[str, Any] | None = Field(
        default=None, description="Single-run model settings; null for grid searches."
    )
    reflection_model_name: str | None = Field(
        default=None, description="Single-run reflection model name; null for grid searches."
    )
    task_model_name: str | None = Field(
        default=None, description="Single-run task model name; null for grid searches."
    )

    total_pairs: int | None = Field(default=None, description="Grid-search pair count; null for single runs.")
    completed_pairs: int | None = Field(
        default=None, description="Grid-search pairs that finished; null for single runs."
    )
    failed_pairs: int | None = Field(
        default=None, description="Grid-search pairs that errored; null for single runs."
    )
    generation_models: list[Any] | None = Field(
        default=None, description="Grid-search generation model list; null for single runs."
    )
    reflection_models: list[Any] | None = Field(
        default=None, description="Grid-search reflection model list; null for single runs."
    )

    @field_validator("column_mapping", mode="before")
    @classmethod
    def _coerce_legacy_column_mapping(cls, value: Any) -> Any:
        """Tolerate legacy list-shaped column mappings on persisted rows.

        Older optimizations persisted ``column_mapping.inputs``/``outputs`` as
        bare lists of column names instead of ``{field: column}`` dicts. The
        response model declares them as dicts, so a single legacy row would
        fail validation and take down the whole listing (one bad row → the
        entire ``list_jobs`` response 500s). Coerce each list to an identity
        ``{name: name}`` mapping so the row still renders; leave well-formed
        dicts and ``None`` untouched.

        Args:
            value: The raw ``column_mapping`` value pulled from the persisted row.

        Returns:
            The value with any list-shaped ``inputs``/``outputs`` normalised to
            identity dicts; returned unchanged otherwise.
        """
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for side in ("inputs", "outputs"):
            mapping = normalized.get(side)
            if isinstance(mapping, list):
                normalized[side] = {str(name): str(name) for name in mapping}
        return normalized


class OptimizationStatusResponse(_JobResponseBase):
    """Full optimization detail returned by GET /optimizations/{optimization_id}."""

    progress_events: list[ProgressEvent] = Field(default_factory=list)
    logs: list[JobLogEntry] = Field(default_factory=list)
    # Start index of the returned progress_events / logs slices within the full
    # server-side stream. 0 means the slice is the complete stream (the default,
    # and the only value pre-cursor clients ever see); a positive value marks a
    # delta tail the client must splice onto the rows it already holds. See the
    # since_progress / since_log query params on GET /optimizations/{id}.
    progress_offset: int = 0
    logs_offset: int = 0
    result: RunResponse | None = None
    grid_result: GridSearchResponse | None = None
    # Grid pair indices with a saved checkpoint: a failed pair listed here offers
    # Resume (continue mid-GEPA), one not listed offers Restart (re-run fresh).
    # Empty for single runs and for grids with no in-flight pair state.
    grid_resumable_pairs: list[int] = Field(default_factory=list)
    effective_role: str | None = Field(
        default=None,
        description=(
            "The caller's share role on this run ('viewer'/'editor'/'owner') "
            "when reached via a member grant; null for the owner's own view "
            "(the owner is implicitly 'owner')."
        ),
    )


class OptimizationSummaryResponse(_JobResponseBase):
    """Lightweight dashboard view of an optimization."""

    split_fractions: SplitFractions | None = None
    shuffle: bool | None = None
    seed: int | None = None
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    progress_count: int = 0
    log_count: int = 0

    baseline_test_metric: float | None = Field(
        default=None,
        description="Baseline test metric; for grid searches this is the best pair's value.",
    )
    optimized_test_metric: float | None = Field(
        default=None,
        description="Optimized test metric; for grid searches this is the best pair's value.",
    )
    metric_improvement: float | None = Field(
        default=None,
        description="Metric improvement; for grid searches this is the best pair's value.",
    )

    best_pair_label: str | None = None

    # AI-authored task summary from job_embeddings.summary_text. Populated
    # at read-time via a side lookup so the dashboard "mine" corpus can
    # render the same one-liner the public /explore corpus shows. Null
    # when no embedding row exists yet (queued/failed/not-yet-embedded).
    summary_text: str | None = None

    # Caller's share role (viewer/editor/owner) when this run was reached
    # via a member grant rather than ownership; null for runs the caller
    # owns. Attached by the listing route, not by build_summary.
    role: str | None = None


class PaginatedJobsResponse(BaseModel):
    """Paginated wrapper for optimization listings."""

    items: list[OptimizationSummaryResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class OptimizationCountsResponse(BaseModel):
    """Aggregate counts by status for dashboard stat cards."""

    total: int = 0
    pending: int = 0
    validating: int = 0
    running: int = 0
    success: int = 0
    failed: int = 0
    cancelled: int = 0
    # Runs shared with the caller via a member grant (not owned). Powers the
    # control panel's "shared" stat card; 0 for admins and solo users.
    shared: int = 0


class JobCancelResponse(BaseModel):
    """Response payload for the cancel endpoint."""

    optimization_id: str
    status: str


class JobDeleteResponse(BaseModel):
    """Response payload for the delete endpoint."""

    optimization_id: str
    deleted: bool


class BulkDeleteSkipped(BaseModel):
    """One entry in the ``skipped`` list of a bulk-delete response."""

    optimization_id: str
    reason: str


class BulkDeleteRequest(BaseModel):
    """Request payload for the bulk-delete endpoint."""

    optimization_ids: list[str] = Field(default_factory=list)


class BulkDeleteResponse(BaseModel):
    """Response payload for the bulk-delete endpoint."""

    deleted: list[str] = Field(default_factory=list)
    skipped: list[BulkDeleteSkipped] = Field(default_factory=list)


class BulkCancelSkipped(BaseModel):
    """One entry in the ``skipped`` list of a bulk-cancel response."""

    optimization_id: str
    reason: str


class BulkCancelRequest(BaseModel):
    """Request payload for the bulk-cancel endpoint."""

    optimization_ids: list[str] = Field(default_factory=list)


class BulkCancelResponse(BaseModel):
    """Response payload for the bulk-cancel endpoint."""

    cancelled: list[str] = Field(default_factory=list)
    skipped: list[BulkCancelSkipped] = Field(default_factory=list)


class OptimizationPayloadResponse(BaseModel):
    """Response payload for the payload retrieval endpoint."""

    optimization_id: str
    optimization_type: OptimizationType
    payload: dict[str, Any]


class ProgramArtifactResponse(BaseModel):
    """Response payload for the artifact retrieval endpoint."""

    program_artifact: ProgramArtifact | None
