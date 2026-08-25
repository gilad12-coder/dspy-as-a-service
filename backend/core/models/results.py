"""Outbound result payloads for single optimization runs and grid searches."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .artifacts import ProgramArtifact
from .common import SplitCounts
from .telemetry import JobLogEntry


# Per-(LM, stage) cell of the LM activity matrix. Pydantic class docstrings
# are part of the OpenAPI contract — see AGENTS.md "Pydantic class
# docstrings" — so this annotation lives in a comment, not in the class body.
class LMStageStats(BaseModel):
    calls: int = 0
    avg_response_time_ms: float | None = None


# Two-LM × N-stage matrix returned alongside RunResponse / PairResult.
# Inner dicts are keyed by stage name ("baseline" / "training" /
# "evaluation"); missing keys mean "no calls in that stage". The wire
# shape is stable — the frontend renders rows in a fixed order.
class LMActivity(BaseModel):
    generation: dict[str, LMStageStats] = Field(default_factory=dict)
    reflection: dict[str, LMStageStats] = Field(default_factory=dict)


# Per-model measured token usage (input/output split) stamped onto a run result
# so the billing worker charges per-model and the UI reconciles the pre-run
# estimate against real per-model spend. Pydantic class docstrings are part of
# the OpenAPI contract — see AGENTS.md — so this annotation lives in a comment.
class ModelTokenUsage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class RunResponse(BaseModel):
    """Result of a single optimization run."""

    module_name: str
    optimizer_name: str
    metric_name: str | None
    split_counts: SplitCounts
    baseline_test_metric: float | None = None
    optimized_test_metric: float | None = None
    metric_improvement: float | None = None
    optimization_metadata: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    program_artifact_path: str | None = None
    program_artifact: ProgramArtifact | None = None
    runtime_seconds: float | None = None
    num_lm_calls: int | None = None
    total_tokens: int | None = None
    # Per-model input/output split behind ``total_tokens`` — the basis the billing
    # worker charges from and the UI reconciles the estimate against.
    usage_by_model: list[ModelTokenUsage] = Field(default_factory=list)
    avg_response_time_ms: float | None = None
    lm_activity: LMActivity | None = None
    run_log: list[JobLogEntry] = Field(default_factory=list)
    baseline_test_results: list[dict[str, Any]] = Field(default_factory=list)
    optimized_test_results: list[dict[str, Any]] = Field(default_factory=list)
    # Named scores the metric logged via log_metrics (precision/recall-style
    # components), macro-averaged over the test rows that logged each name.
    # Empty when the metric never logs.
    baseline_logged_metrics: dict[str, float] = Field(default_factory=dict)
    optimized_logged_metrics: dict[str, float] = Field(default_factory=dict)


class PairResult(BaseModel):
    """Result of a single (generation, reflection) model pair run."""

    pair_index: int
    generation_model: str
    reflection_model: str
    generation_reasoning_effort: str | None = None
    reflection_reasoning_effort: str | None = None
    baseline_test_metric: float | None = None
    optimized_test_metric: float | None = None
    metric_improvement: float | None = None
    target_score: float | None = None
    target_score_reached: bool | None = None
    stop_reason: str | None = None
    runtime_seconds: float | None = None
    num_lm_calls: int | None = None
    total_tokens: int | None = None
    usage_by_model: list[ModelTokenUsage] = Field(default_factory=list)
    avg_response_time_ms: float | None = None
    lm_activity: LMActivity | None = None
    program_artifact: ProgramArtifact | None = None
    error: str | None = None
    baseline_test_results: list[dict[str, Any]] = Field(default_factory=list)
    optimized_test_results: list[dict[str, Any]] = Field(default_factory=list)
    # Same macro-averaged log_metrics aggregates as on RunResponse, per pair.
    baseline_logged_metrics: dict[str, float] = Field(default_factory=dict)
    optimized_logged_metrics: dict[str, float] = Field(default_factory=dict)


class GridSearchResponse(BaseModel):
    """Result of a grid search over model pairs.

    Contains a leaderboard of per-pair scores and highlights the best config.
    """

    module_name: str
    optimizer_name: str
    metric_name: str | None = None
    split_counts: SplitCounts
    total_pairs: int
    completed_pairs: int = 0
    failed_pairs: int = 0
    pair_results: list[PairResult] = Field(default_factory=list)
    best_pair: PairResult | None = None
    runtime_seconds: float | None = None
    total_tokens: int | None = None
    # Per-model usage summed across all pairs — the basis the worker charges the
    # whole grid from (each pair priced on its own gen/refl models).
    usage_by_model: list[ModelTokenUsage] = Field(default_factory=list)
