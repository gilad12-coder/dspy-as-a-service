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


# Baseline-vs-optimized scores on the slice the "No lift, no charge" guarantee
# is adjudicated against. ``basis`` is "test" when a held-out test split was
# reserved (the strongest, unbiased proof) or "val" when the dataset was too
# small for a test split and the guarantee falls back to the valset gain. Pydantic
# class docstrings are part of the OpenAPI contract — see AGENTS.md — so this
# annotation lives in a comment, not the class body.
class GuaranteeBasis(BaseModel):
    basis: str
    baseline: float | None = None
    optimized: float | None = None


class RunResponse(BaseModel):
    """Result of a single optimization run."""

    module_name: str
    optimizer_name: str
    metric_name: str | None
    split_counts: SplitCounts
    baseline_test_metric: float | None = None
    optimized_test_metric: float | None = None
    metric_improvement: float | None = None
    # Scores the guarantee is judged on (test split, or valset fallback). None on
    # a run that produced no comparable baseline/optimized pair.
    guarantee: GuaranteeBasis | None = None
    optimization_metadata: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    program_artifact_path: str | None = None
    program_artifact: ProgramArtifact | None = None
    runtime_seconds: float | None = None
    num_lm_calls: int | None = None
    total_tokens: int | None = None
    avg_response_time_ms: float | None = None
    lm_activity: LMActivity | None = None
    run_log: list[JobLogEntry] = Field(default_factory=list)
    baseline_test_results: list[dict[str, Any]] = Field(default_factory=list)
    optimized_test_results: list[dict[str, Any]] = Field(default_factory=list)


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
    runtime_seconds: float | None = None
    num_lm_calls: int | None = None
    total_tokens: int | None = None
    avg_response_time_ms: float | None = None
    lm_activity: LMActivity | None = None
    program_artifact: ProgramArtifact | None = None
    error: str | None = None
    baseline_test_results: list[dict[str, Any]] = Field(default_factory=list)
    optimized_test_results: list[dict[str, Any]] = Field(default_factory=list)


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
    # Scores the guarantee is judged on — the best pair's baseline/optimized on
    # the test split (or valset fallback). None when no pair produced a pair.
    guarantee: GuaranteeBasis | None = None
