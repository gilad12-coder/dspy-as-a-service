"""Foundational Pydantic models used across every domain.

Split out of the old ``backend/core/models.py`` per AGENTS.md. Contains
the primitive building blocks (mappings, model configs, split specs,
status enum) that every other model file depends on. Cross-file string
constants live in :mod:`core.models.constants`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

OptimizationType = Literal["run", "grid_search"]
Composition = Literal["single", "workflow"]


class ColumnMapping(BaseModel):
    """Describe how dataframe columns map onto DSPy signature fields.

    ``inputs`` / ``outputs`` are plural mappings — ``{signature_field: column_name}``
    — because a single signature can have multiple input or output fields. This
    is distinct from the ``ServeResponse.input_fields`` / ``output_fields`` lists,
    which live at a different layer (inference response) and carry only field
    *names*, not column bindings. The two shapes are intentionally different.
    """

    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_non_empty(self) -> ColumnMapping:
        """Reject empty inputs, shared signature fields, or shared dataset columns.

        Returns:
            The validated mapping instance.

        Raises:
            ValueError: When ``inputs`` is empty, when a signature field name
                appears in both ``inputs`` and ``outputs``, or when ``inputs``
                and ``outputs`` reuse the same dataset column.
        """
        if not self.inputs:
            raise ValueError("At least one input column must be specified.")
        shared_fields = self.inputs.keys() & self.outputs.keys()
        if shared_fields:
            raise ValueError(
                f"Signature fields cannot be both input and output: {sorted(shared_fields)}"
            )
        shared_columns = set(self.inputs.values()) & set(self.outputs.values())
        if shared_columns:
            raise ValueError(
                f"Input and output column mappings must not reuse the same columns: {sorted(shared_columns)}"
            )
        return self


class ModelConfig(BaseModel):
    """Configuration block for language-model/backbone selection."""

    name: str
    base_url: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)

    def normalized_identifier(self) -> str:
        """Return the LiteLLM identifier with any leading or trailing slashes stripped.

        Returns:
            ``name`` with any leading or trailing ``/`` characters stripped.
        """
        return self.name.strip("/")


class SplitFractions(BaseModel):
    """Train/val/test fraction spec."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _validate(self) -> SplitFractions:
        """Reject negative fractions or fractions that do not sum to 1.0.

        Returns:
            The validated fractions instance.

        Raises:
            ValueError: When any fraction is negative, or when ``train``,
                ``val`` and ``test`` do not sum to 1.0.
        """
        parts = [self.train, self.val, self.test]
        if any(part < 0 for part in parts):
            raise ValueError("Split fractions must be non-negative.")
        total = sum(parts)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0.")
        return self


class SplitCounts(BaseModel):
    """Container for the number of examples in each dataset split."""

    train: int
    val: int
    test: int


class OptimizationStatus(StrEnum):
    """Enumerate background job states."""

    pending = "pending"
    validating = "validating"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"
    paused = "paused"


class BulkDeleteByIdsRequest(BaseModel):
    """Request body for the id-keyed bulk-delete endpoints (datasets, staged uploads, chats)."""

    ids: list[str] = Field(default_factory=list)


class BulkDeleteByIdsSkipped(BaseModel):
    """One entry in the ``skipped`` list of an id-keyed bulk-delete response."""

    id: str
    reason: str


class BulkDeleteByIdsResponse(BaseModel):
    """Response payload for the id-keyed bulk-delete endpoints."""

    deleted: list[str] = Field(default_factory=list)
    skipped: list[BulkDeleteByIdsSkipped] = Field(default_factory=list)
