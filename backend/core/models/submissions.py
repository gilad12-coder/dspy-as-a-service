"""Inbound payloads for POST /run and POST /grid-search plus the initial ack."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ColumnMapping, ModelConfig, OptimizationStatus, OptimizationType, SplitFractions
from .serve import WorkflowNodeTrace
from .workflow import WORKFLOW_MODULE_NAME, WorkflowSpec


# Where a react run sources its tool roster: a live MCP endpoint or a snapshot
# carried alongside the dataset.
class ToolSource(BaseModel):
    kind: Literal["live_mcp", "dataset_snapshot"]
    mcp_url: str | None = None
    mcp_auth_header: str | None = None
    tool_filter: list[str] | None = None


class _OptimizationRequestBase(BaseModel):
    """Shared fields for all optimization submissions."""

    # ``model_config`` here is the Pydantic class-config attr; ``RunRequest``
    # additionally exposes a wire-aliased field whose alias is also
    # ``model_config`` (the OpenAPI property name the frontend sends), so the
    # same identifier intentionally serves two purposes — class config here
    # vs. field alias on the subclass.
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, description="User-defined display name for this optimization.")
    description: str | None = Field(
        default=None, max_length=280, description="Short description of the optimization goal (max 280 characters)."
    )
    username: str | None = Field(
        default=None,
        description=(
            "Submitter identity. Optional on the wire — the API always overwrites it from the authenticated session, "
            "so clients (including MCP tool callers) can omit it."
        ),
    )
    module_name: str
    module_kwargs: dict[str, Any] = Field(default_factory=dict)
    signature_code: str | None = Field(
        default=None,
        description=(
            "The single dspy.Signature the module wraps. Required for every module except "
            "'workflow', whose per-node signatures live inside the workflow spec instead."
        ),
    )
    workflow: WorkflowSpec | None = Field(
        default=None,
        description=(
            "Workflow graph spec. Required when module_name is 'workflow', forbidden otherwise. "
            "The graph's input-anchor fields are the run's input ports (covered by "
            "column_mapping.inputs) and its output-anchor fields are the final outputs the "
            "metric scores (covered by column_mapping.outputs)."
        ),
    )
    metric_code: str | None = None
    optimizer_name: str
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    # max_length matches the staging cap (StageDatasetForAgentRequest) — an
    # uncapped inline list lets a single submit balloon the shared API process.
    dataset: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=200_000,
        description=(
            "Inline dataset rows. Optional when ``staged_dataset_id`` is provided — the server then loads the rows "
            "from the staged copy. Exactly one of ``dataset`` or ``staged_dataset_id`` must be present."
        ),
    )
    staged_dataset_id: str | None = Field(
        default=None,
        description=(
            "Opaque id returned by ``POST /datasets/stage-for-agent``. Used by agent-driven submits so the model "
            "does not have to inline tens of thousands of dataset rows into its tool arguments."
        ),
    )
    source_dataset_id: str | None = Field(
        default=None,
        description=(
            "Id of a saved personal-library dataset to run by reference. The server resolves the caller's access, "
            "loads the rows onto ``dataset``, and records the link from the optimization back to the dataset. "
            "Mutually exclusive with ``dataset`` and ``staged_dataset_id``."
        ),
    )
    column_mapping: ColumnMapping
    column_order: list[str] | None = Field(
        default=None,
        description=(
            "Dataset column names in the order the user arranged them at submit time. "
            "Persisted as an array because JSONB does not preserve object key order — a "
            "clone reads this back to restore the original column order in the UI."
        ),
    )
    split_fractions: SplitFractions = Field(default_factory=SplitFractions)
    shuffle: bool = True
    seed: int | None = None
    dataset_filename: str | None = Field(default=None, description="Original dataset file name.")
    is_private: bool = Field(
        default=False,
        description="When true, the optimization is excluded from the public explore page.",
    )
    token_source: Literal["managed", "byok"] = Field(
        default="managed",
        description=(
            "How the run's tokens are billed: 'managed' (Skynet credits — any model is runnable, "
            "and the run's cost ceiling is capped at the account's spendable credits so it can't "
            "overspend) or 'byok' (the user's own provider key — billed directly, no credits). "
            "Threaded from the wizard so the credit gate and the 'No lift, no charge' guarantee "
            "are enforced server-side, not advisory."
        ),
    )
    max_cost_credits: int | None = Field(
        default=None,
        ge=1,
        description=(
            "User-set per-job spend ceiling, in credits. A DSPy optimizer's token use is not "
            "linear (bootstrapping, compile steps, validation loops), so the wizard shows a "
            "projected bracket rather than a tight estimate and lets the user cap the run here. "
            "The run is hard-stopped server-side once its accumulated credit cost reaches this "
            "cap; a stopped run fails and is never billed. "
            "Omit (null) for no ceiling."
        ),
    )
    estimated_credits_low: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Low end of the projected credit bracket the wizard showed at submit. Persisted so "
            "the post-run proof moment can reconcile the estimate against the actual charge. "
            "Carries the chargeable bracket for the run's token_source (managed: full per-model "
            "cost; byok: platform fee). Advisory only — never gates or bills. Omit (null) when "
            "no estimate was computed."
        ),
    )
    estimated_credits_high: int | None = Field(
        default=None,
        ge=0,
        description=(
            "High end of the projected credit bracket (see estimated_credits_low). Seeds the "
            "post-run estimate-vs-actual reconciliation. Advisory only."
        ),
    )

    @model_validator(mode="after")
    def _ensure_dataset(self) -> _OptimizationRequestBase:
        """Require exactly one dataset source: inline rows, a staged id, or a library id.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When more than one of ``dataset``, ``staged_dataset_id``,
                and ``source_dataset_id`` is supplied, or when none is.
        """
        provided = sum((bool(self.dataset), bool(self.staged_dataset_id), bool(self.source_dataset_id)))
        if provided > 1:
            raise ValueError("Provide exactly one of dataset, staged_dataset_id, or source_dataset_id.")
        if provided == 0:
            raise ValueError(
                "Dataset must contain at least one row, or staged_dataset_id / source_dataset_id must be provided."
            )
        return self

    @model_validator(mode="after")
    def _ensure_module_shape(self) -> _OptimizationRequestBase:
        """Pair ``module_name`` with the matching program definition.

        Workflow runs carry their signatures per node inside ``workflow``;
        every other module wraps exactly one top-level ``signature_code``.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When a workflow run lacks ``workflow``, a non-workflow
                run lacks ``signature_code``, or ``workflow`` is supplied for a
                non-workflow module.
        """
        if self.module_name.lower() == WORKFLOW_MODULE_NAME:
            if self.workflow is None:
                raise ValueError("workflow is required when module_name is 'workflow'.")
        else:
            if self.workflow is not None:
                raise ValueError("workflow is only valid when module_name is 'workflow'.")
            if not self.signature_code:
                raise ValueError("signature_code is required.")
        return self


class RunRequest(_OptimizationRequestBase):
    """Payload for the /run endpoint."""

    model_settings: ModelConfig = Field(alias="model_config")
    reflection_model_settings: ModelConfig | None = Field(default=None, alias="reflection_model_config")
    task_model_settings: ModelConfig | None = Field(default=None, alias="task_model_config")
    tool_source: ToolSource | None = None

    @model_validator(mode="after")
    def _require_metric_code(self) -> RunRequest:
        """Re-require ``metric_code`` for every run, including react.

        ``metric_code`` is declared optional on the base only so the field can be
        shared; every run must supply it. React is a generic module that scores
        rollouts with the same standard ``(gold, pred, trace, pred_name,
        pred_trace)`` metric the predict/cot path uses.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When ``metric_code`` is missing.
        """
        if self.metric_code is None:
            raise ValueError("metric_code is required.")
        return self

    @model_validator(mode="after")
    def _require_tool_source_for_workflow_tools(self) -> RunRequest:
        """Require a run-level tool roster when the graph contains tool-using nodes.

        Workflow react and mcp nodes resolve their tools from the run's
        single ``tool_source`` (optionally narrowed per node), mirroring how
        a top-level react run sources its roster.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When the workflow has react/mcp nodes but no
                ``tool_source`` is supplied.
        """
        if self.workflow is not None and self.tool_source is None:
            tool_users = [
                node.id
                for node in self.workflow.nodes
                if node.kind == "mcp" or (node.kind == "signature" and node.module_name == "react")
            ]
            if tool_users:
                raise ValueError(f"tool_source is required — these workflow nodes use tools: {tool_users}.")
        return self


class GridSearchRequest(_OptimizationRequestBase):
    """Payload for the /grid-search endpoint — sweep over model pairs."""

    generation_models: list[ModelConfig] = Field(default_factory=list)
    reflection_models: list[ModelConfig] = Field(default_factory=list)
    use_all_available_generation_models: bool = Field(
        default=False,
        description=(
            "Populate generation_models from every available model in the catalog. "
            "When true, generation_models may be omitted and is replaced server-side."
        ),
    )
    use_all_available_reflection_models: bool = Field(
        default=False,
        description=(
            "Populate reflection_models from every available model in the catalog. "
            "When true, reflection_models may be omitted and is replaced server-side."
        ),
    )

    @model_validator(mode="after")
    def _validate_model_lists(self) -> GridSearchRequest:
        """Reject requests missing required model lists.

        Each side (``generation_models``, ``reflection_models``) must either be
        non-empty or be marked for server-side expansion via its matching
        ``use_all_available_*`` flag.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When ``metric_code`` is missing, when ``generation_models``
                is empty and ``use_all_available_generation_models`` is false, or
                when ``reflection_models`` is empty and
                ``use_all_available_reflection_models`` is false.
        """
        if self.module_name.lower() == WORKFLOW_MODULE_NAME:
            raise ValueError("Grid search does not support workflow modules yet — submit via /run.")
        if self.metric_code is None:
            raise ValueError("metric_code is required.")
        if not self.use_all_available_generation_models and not self.generation_models:
            raise ValueError("At least one generation model is required.")
        if not self.use_all_available_reflection_models and not self.reflection_models:
            raise ValueError("At least one reflection model is required.")
        return self


class WorkflowDryRunRequest(BaseModel):
    """Request payload for POST /workflows/dry-run — one unoptimized test execution."""

    model_config = ConfigDict(populate_by_name=True)

    workflow: WorkflowSpec
    inputs: dict[str, Any] = Field(description="Values for the workflow's input-anchor fields.")
    model_settings: ModelConfig = Field(alias="model_config")
    tool_source: ToolSource | None = None

    @model_validator(mode="after")
    def _require_tool_source_for_tools(self) -> WorkflowDryRunRequest:
        """Require a tool roster when the graph contains tool-using nodes.

        Returns:
            The validated request instance.

        Raises:
            ValueError: When react/mcp nodes exist but no ``tool_source``.
        """
        if self.tool_source is None:
            tool_users = [
                node.id
                for node in self.workflow.nodes
                if node.kind == "mcp" or (node.kind == "signature" and node.module_name == "react")
            ]
            if tool_users:
                raise ValueError(f"tool_source is required — these workflow nodes use tools: {tool_users}.")
        return self


class WorkflowDryRunResponse(BaseModel):
    """Result of a workflow dry run.

    A node failure is an expected outcome the canvas renders, not an HTTP
    error: the response carries the failing node id, the error, and every
    trace collected up to (and including) the failure.
    """

    outputs: dict[str, Any] | None = None
    node_traces: list[WorkflowNodeTrace] = Field(default_factory=list)
    model_used: str
    error: str | None = None
    failed_node_id: str | None = None


class OptimizationSubmissionResponse(BaseModel):
    """Immediate response to POST /run or POST /grid-search."""

    optimization_id: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    created_at: datetime
    name: str | None = None
    description: str | None = None
    username: str
    module_name: str
    optimizer_name: str
