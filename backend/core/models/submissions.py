"""Inbound payloads for POST /run and POST /grid-search plus the initial ack."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ColumnMapping, ModelConfig, OptimizationStatus, OptimizationType, SplitFractions


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
    signature_code: str
    metric_code: str | None = None
    optimizer_name: str
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)
    dataset: list[dict[str, Any]] | None = Field(
        default=None,
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
        if self.metric_code is None:
            raise ValueError("metric_code is required.")
        if not self.use_all_available_generation_models and not self.generation_models:
            raise ValueError("At least one generation model is required.")
        if not self.use_all_available_reflection_models and not self.reflection_models:
            raise ValueError("At least one reflection model is required.")
        return self


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
