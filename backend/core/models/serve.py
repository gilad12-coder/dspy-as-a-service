"""Request/response models for the /serve/* inference endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .common import ModelConfig


class ServeRequest(BaseModel):
    """Request payload for running inference on an optimized program."""

    inputs: dict[str, Any] = Field(..., description="Input field values matching the program's signature.")
    model_config_override: ModelConfig | None = Field(
        default=None,
        description="Optional model config override. Uses the original optimization model if omitted.",
    )

    @model_validator(mode="after")
    def _ensure_inputs(self) -> ServeRequest:
        """Reject inference requests with no input fields.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When ``inputs`` is empty.
        """
        if not self.inputs:
            raise ValueError("At least one input field is required.")
        return self


class WorkflowNodeTrace(BaseModel):
    """One node's execution record from a workflow inference or dry run."""

    node_id: str
    kind: str
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


class ServeResponse(BaseModel):
    """Response payload from program inference.

    ``input_fields`` / ``output_fields`` are lists of signature field *names*.
    They are NOT the same shape as ``ColumnMapping.inputs`` / ``outputs``, which
    are ``{field_name: column_name}`` dicts used at the submission layer. The
    naming differs on purpose: here we are echoing the servable program's
    signature, not binding dataset columns.
    """

    optimization_id: str
    outputs: dict[str, Any]
    input_fields: list[str]
    output_fields: list[str]
    model_used: str
    node_traces: list[WorkflowNodeTrace] | None = Field(
        default=None,
        description="Per-node execution trace, present only for workflow runs.",
    )


class ServeInfoResponse(BaseModel):
    """Metadata about a servable program (no inference call)."""

    optimization_id: str
    module_name: str
    optimizer_name: str
    model_name: str
    input_fields: list[str]
    output_fields: list[str]
    instructions: str | None = None
    demo_count: int = 0
    sample_inputs: dict[str, str] = Field(default_factory=dict)
