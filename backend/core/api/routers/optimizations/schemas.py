"""Pydantic request and response models local to the optimizations routes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from ....i18n import CLONE_NAME_PREFIX
from ....models import OptimizationSubmissionResponse


class SidebarJobItem(BaseModel):
    """Compact per-optimization entry for the sidebar navigation list."""

    optimization_id: str
    status: str
    name: str | None = None
    module_name: str | None = None
    optimizer_name: str | None = None
    model_name: str | None = None
    username: str | None = None
    created_at: datetime | None = None
    pinned: bool = False
    optimization_type: str | None = None
    total_pairs: int | None = None
    # True when this run stopped mid-optimization with a saved checkpoint and can
    # be resumed in place; drives the sidebar Resume-vs-Restart menu item.
    resumable: bool = False
    # Caller's share role on this run ("viewer"/"editor"/"owner"); only set on
    # the "shared with me" listing, None for the caller's own optimizations.
    role: str | None = None


class SidebarJobsResponse(BaseModel):
    """Paginated response for the sidebar optimization list."""

    items: list[SidebarJobItem]
    total: int


class CloneJobRequest(BaseModel):
    """Request body for cloning an optimization."""

    count: int = Field(default=1, ge=1, le=5, description="Number of copies to create (1–5).")
    name_prefix: str | None = Field(
        default=None,
        max_length=100,
        description=f"Prefix prepended to each clone's name. Defaults to '{CLONE_NAME_PREFIX}'.",
    )


class CloneJobResponse(BaseModel):
    """List of newly-created clones plus the source id for reference."""

    source_optimization_id: str
    created: list[OptimizationSubmissionResponse]


class BulkMetadataRequest(BaseModel):
    """Request body for bulk pin."""

    optimization_ids: list[str] = Field(min_length=1, max_length=100)
    value: bool = Field(description="Target state — true to pin, false to clear.")


class BulkMetadataSkipped(BaseModel):
    """An optimization that was omitted from a bulk update, with a human-readable reason."""

    optimization_id: str
    reason: str


class BulkMetadataResponse(BaseModel):
    """Response for POST /optimizations/bulk-pin."""

    updated: list[str]
    skipped: list[BulkMetadataSkipped]
