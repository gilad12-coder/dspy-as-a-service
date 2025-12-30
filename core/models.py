from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

HEALTH_STATUS_OK = "ok"


class ColumnMapping(BaseModel):
    """Describe how dataframe columns map onto DSPy signature fields."""

    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_non_empty(self) -> "ColumnMapping":
        """Validate that mappings include inputs and no shared columns.

        Args:
            self: The ``ColumnMapping`` instance being validated.

        Returns:
            ColumnMapping: Validated mapping.

        Raises:
            ValueError: If inputs are missing or columns overlap.
        """
        if not self.inputs:
            raise ValueError("At least one input column must be specified.")
        shared = set(self.inputs.values()) & set(self.outputs.values())
        if shared:
            raise ValueError(
                "Input and output column mappings must not reuse the same columns:"
                f" {sorted(shared)}"
            )
        return self


class ModelConfig(BaseModel):
    """Configuration block for language-model/backbone selection."""

    name: str
    base_url: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def normalized_identifier(self) -> str:
        """Return the Litellm identifier (deprecated: identical to ``name``).

        Args:
            None.

        Returns:
            str: Normalized model identifier preferred by LiteLLM.
        """

        return self.name.strip("/")


class SplitFractions(BaseModel):
    """Train/val/test fraction spec."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _validate(self) -> "SplitFractions":
        """Verify that split fractions are non-negative and sum to one.

        Args:
            self: The ``SplitFractions`` instance being validated.

        Returns:
            SplitFractions: Validated fraction set.

        Raises:
            ValueError: If constraints are violated.
        """
        parts = [self.train, self.val, self.test]
        if any(part < 0 for part in parts):
            raise ValueError("Split fractions must be non-negative.")
        total = sum(parts)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0.")
        return self


class RunRequest(BaseModel):
    """Primary payload for the /run endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    module_name: str
    module_kwargs: Dict[str, Any] = Field(default_factory=dict)
    signature_code: str
    metric_code: str
    optimizer_name: str
    optimizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: Dict[str, Any] = Field(default_factory=dict)
    dataset: List[Dict[str, Any]]
    column_mapping: ColumnMapping
    split_fractions: SplitFractions = Field(default_factory=SplitFractions)
    shuffle: bool = True
    seed: Optional[int] = None
    model_settings: ModelConfig = Field(alias="model_config")
    reflection_model_settings: Optional[ModelConfig] = Field(
        default=None, alias="reflection_model_config"
    )
    prompt_model_settings: Optional[ModelConfig] = Field(
        default=None, alias="prompt_model_config"
    )
    task_model_settings: Optional[ModelConfig] = Field(
        default=None, alias="task_model_config"
    )

    @model_validator(mode="after")
    def _ensure_dataset(self) -> "RunRequest":
        """Ensure that at least one data row is provided.

        Args:
            self: The ``RunRequest`` instance being validated.

        Returns:
            RunRequest: Validated request.

        Raises:
            ValueError: If the dataset is empty.
        """
        if not self.dataset:
            raise ValueError("Dataset must contain at least one row.")
        return self


class SplitCounts(BaseModel):
    """Container for the number of examples in each dataset split.

    Attributes:
        train: Number of training examples.
        val: Number of validation examples.
        test: Number of test examples.
    """

    train: int
    val: int
    test: int


class ProgressEvent(BaseModel):
    """Structured telemetry emitted while an optimization job runs."""

    timestamp: datetime
    event: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class JobLogEntry(BaseModel):
    """Log line captured from DSPy/optimizer loggers."""

    timestamp: datetime
    level: str
    logger: str
    message: str


class ProgramArtifact(BaseModel):
    """Serializable payload that carries the optimized DSPy program files."""

    path: Optional[str] = Field(
        default=None,
        description="Absolute path on the server where the artifact lives.",
    )
    program_pickle_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded contents of the saved program.pkl file.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="metadata.json contents already parsed into a dict.",
    )


class RunResponse(BaseModel):
    """Response payload containing optimization results and the compiled program.

    Attributes:
        module_name: Name of the DSPy module that was optimized.
        optimizer_name: Name of the optimizer used for compilation.
        metric_name: Name of the evaluation metric function.
        split_counts: Number of examples in train/val/test splits.
        baseline_test_metric: Score on test set before optimization.
        optimized_test_metric: Score on test set after optimization.
        optimization_metadata: Extra metadata from the optimizer.
        details: Additional run details and diagnostics.
        program_artifact_path: Server path to the saved artifact (deprecated).
        program_artifact: Serialized program artifact with base64 pickle.
        runtime_seconds: Total optimization runtime in seconds.
        run_log: Captured log entries from the optimization run.
    """

    module_name: str
    optimizer_name: str
    metric_name: Optional[str]
    split_counts: SplitCounts
    baseline_test_metric: Optional[float] = None
    optimized_test_metric: Optional[float] = None
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)
    program_artifact_path: Optional[str] = None
    program_artifact: Optional[ProgramArtifact] = None
    runtime_seconds: Optional[float] = None
    run_log: List[JobLogEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response payload for the health check endpoint.

    Attributes:
        status: Health status string, typically "ok".
        registered_assets: Dictionary of registered modules, metrics, and optimizers.
    """

    status: str = Field(default=HEALTH_STATUS_OK)
    registered_assets: Dict[str, List[str]]


class JobStatus(str, Enum):
    """Enumerate background job states."""

    pending = "pending"
    validating = "validating"
    running = "running"
    success = "success"
    failed = "failed"


class JobSubmissionResponse(BaseModel):
    """Response payload for job submission requests."""

    job_id: str
    status: JobStatus
    estimated_total_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    """Response payload returned by the job-inspection endpoint."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    message: Optional[str]
    latest_metrics: Dict[str, Any] = Field(default_factory=dict)
    progress_events: List[ProgressEvent] = Field(default_factory=list)
    logs: List[JobLogEntry] = Field(default_factory=list)
    estimated_seconds_remaining: Optional[float]
    result: Optional[RunResponse] = None


class JobSummaryResponse(BaseModel):
    """Aggregated view of a job with coarse progress information."""

    job_id: str
    status: JobStatus
    message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    elapsed_seconds: float
    estimated_seconds_remaining: Optional[float]
    module_name: Optional[str] = None
    optimizer_name: Optional[str] = None
    dataset_rows: Optional[int] = None
    split_fractions: Optional[SplitFractions] = None
    shuffle: Optional[bool] = None
    seed: Optional[int] = None
    optimizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    compile_kwargs: Dict[str, Any] = Field(default_factory=dict)
    latest_metrics: Dict[str, Any] = Field(default_factory=dict)
class ProgramArtifactResponse(BaseModel):
    """Response payload for the artifact retrieval endpoint.

    Attributes:
        program_artifact_path: Server path to artifact (deprecated, always None).
        program_artifact: Serialized artifact containing base64-encoded program pickle.
    """

    program_artifact_path: Optional[str]
    program_artifact: Optional[ProgramArtifact]
