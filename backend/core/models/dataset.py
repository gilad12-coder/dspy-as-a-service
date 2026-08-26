"""Models for dataset profiling and automatic split-plan recommendations.

Descriptive shapes returned by ``POST /datasets/profile``. The profiler
produces a ``DatasetProfile`` summarizing an uploaded dataset, and the
planner turns that profile into a ``SplitPlan`` the submit wizard renders
as the "we'll split it like this" card. Users can accept the plan or
override it on final submission.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import ColumnMapping, SplitCounts, SplitFractions


class TargetColumnProfile(BaseModel):
    """Summary of the output column chosen as the primary target."""

    name: str
    kind: str = Field(description="One of 'categorical', 'numeric', 'freeform'.")
    unique_values: int = Field(ge=0)
    class_histogram: dict[str, int] = Field(
        default_factory=dict,
        description="Class → count map; empty for non-categorical targets.",
    )


class InputColumnProfile(BaseModel):
    """Summary of an input column, surfacing its detected modality."""

    name: str
    kind: str = Field(description="One of 'text', 'image'.")


class DatasetProfile(BaseModel):
    """Structural summary of an uploaded dataset."""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    target: TargetColumnProfile | None = Field(
        default=None,
        description=(
            "Primary target the planner reasons about. The first categorical "
            "column among the outputs (or the first output if none are "
            "categorical), kept for convenience and backward compatibility."
        ),
    )
    targets: list[TargetColumnProfile] = Field(
        default_factory=list,
        description="Profile for every output column, in mapping order.",
    )
    inputs: list[InputColumnProfile] = Field(
        default_factory=list,
        description=(
            "Profile for every input column, in mapping order. Surfaces the "
            "detected modality so the wizard can render an Image badge and "
            "the signature generator can emit dspy.Image typed fields."
        ),
    )
    duplicate_count: int = Field(default=0, ge=0)


class SplitPlan(BaseModel):
    """Recommended split configuration for a profiled dataset."""

    fractions: SplitFractions
    shuffle: bool
    seed: int = Field(ge=0)
    counts: SplitCounts
    rationale: list[str] = Field(
        default_factory=list,
        description="Ordered list of short Hebrew bullets explaining each decision.",
    )


class ProfileDatasetRequest(BaseModel):
    """Inbound body for ``POST /datasets/profile``."""

    dataset: list[dict[str, Any]] = Field(default_factory=list)
    column_mapping: ColumnMapping
    staged_dataset_id: str | None = Field(
        default=None,
        description=(
            "Opaque id from ``POST /datasets/stage-for-agent``. Pass instead of "
            "an inline ``dataset`` to profile staged rows; the backend rehydrates them."
        ),
    )
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reproducibility; a random one is chosen if omitted.",
    )


class ProfileDatasetResponse(BaseModel):
    """Outbound body for ``POST /datasets/profile``."""

    profile: DatasetProfile
    plan: SplitPlan


class ValidateDatasetRequest(BaseModel):
    """Inbound body for ``POST /datasets/validate``.

    Mirrors the ``ValidateCode`` preflight: a small payload the submit
    wizard sends before advancing past the dataset step so the user can't
    enqueue a job whose chosen split is unrunnable.
    """

    row_count: int = Field(ge=0)
    fractions: SplitFractions


class ValidateDatasetResponse(BaseModel):
    """Outbound body for ``POST /datasets/validate``.

    Shape matches :class:`ValidateCodeResponse` so the wizard can treat
    both preflights identically.
    """

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
