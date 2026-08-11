"""Tests for dataset profile, plan, and request/response models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models.common import ColumnMapping, SplitCounts, SplitFractions
from core.models.dataset import (
    DatasetProfile,
    InputColumnProfile,
    ProfileDatasetRequest,
    ProfileDatasetResponse,
    SplitPlan,
    TargetColumnProfile,
)


def test_target_column_profile_required_fields() -> None:
    """Verify TargetColumnProfile requires name/kind/unique_values and defaults histogram."""
    t = TargetColumnProfile(name="answer", kind="categorical", unique_values=3)

    assert t.name == "answer"
    assert t.kind == "categorical"
    assert t.unique_values == 3
    assert t.class_histogram == {}


def test_target_column_profile_unique_values_must_be_non_negative() -> None:
    """Verify TargetColumnProfile rejects a negative unique_values count."""
    with pytest.raises(ValidationError):
        TargetColumnProfile(name="answer", kind="numeric", unique_values=-1)


def test_target_column_profile_with_class_histogram() -> None:
    """Verify TargetColumnProfile persists a populated class histogram."""
    t = TargetColumnProfile(
        name="answer",
        kind="categorical",
        unique_values=2,
        class_histogram={"yes": 7, "no": 3},
    )

    assert t.class_histogram == {"yes": 7, "no": 3}


def test_input_column_profile_persists_kind() -> None:
    """Verify InputColumnProfile persists name and detected modality kind."""
    p = InputColumnProfile(name="image_url", kind="image")

    assert p.name == "image_url"
    assert p.kind == "image"


def test_dataset_profile_minimal_construction() -> None:
    """Verify DatasetProfile defaults targets/inputs and accepts row/column counts."""
    profile = DatasetProfile(row_count=10, column_count=2)

    assert profile.row_count == 10
    assert profile.column_count == 2
    assert profile.target is None
    assert profile.targets == []
    assert profile.inputs == []
    assert profile.duplicate_count == 0


def test_dataset_profile_rejects_negative_counts() -> None:
    """Verify DatasetProfile rejects negative row_count or column_count."""
    with pytest.raises(ValidationError):
        DatasetProfile(row_count=-1, column_count=2)
    with pytest.raises(ValidationError):
        DatasetProfile(row_count=10, column_count=-1)


def test_dataset_profile_with_targets_and_inputs() -> None:
    """Verify DatasetProfile stores nested target and input column profiles."""
    profile = DatasetProfile(
        row_count=100,
        column_count=3,
        target=TargetColumnProfile(name="label", kind="categorical", unique_values=2),
        targets=[TargetColumnProfile(name="label", kind="categorical", unique_values=2)],
        inputs=[InputColumnProfile(name="text", kind="text")],
        duplicate_count=4,
    )

    assert profile.target is not None
    assert profile.target.name == "label"
    assert len(profile.targets) == 1
    assert profile.inputs[0].kind == "text"
    assert profile.duplicate_count == 4


def test_split_plan_round_trip() -> None:
    """Verify SplitPlan persists fractions, shuffle flag, seed, counts, and rationale."""
    plan = SplitPlan(
        fractions=SplitFractions(),
        shuffle=True,
        seed=42,
        counts=SplitCounts(train=70, val=15, test=15),
        rationale=["bullet a", "bullet b"],
    )

    assert plan.shuffle is True
    assert plan.seed == 42
    assert plan.counts.train == 70
    assert plan.rationale == ["bullet a", "bullet b"]


def test_split_plan_rejects_negative_seed() -> None:
    """Verify SplitPlan rejects a negative seed."""
    with pytest.raises(ValidationError):
        SplitPlan(
            fractions=SplitFractions(),
            shuffle=True,
            seed=-1,
            counts=SplitCounts(train=70, val=15, test=15),
        )


def test_profile_dataset_request_persists_seed() -> None:
    """Verify ProfileDatasetRequest persists optional seed and required dataset/mapping."""
    req = ProfileDatasetRequest(
        dataset=[{"q": "a", "a": "b"}],
        column_mapping=ColumnMapping(inputs={"q": "q"}, outputs={"a": "a"}),
        seed=7,
    )

    assert len(req.dataset) == 1
    assert req.seed == 7


def test_profile_dataset_request_seed_defaults_none() -> None:
    """Verify ProfileDatasetRequest defaults seed to None."""
    req = ProfileDatasetRequest(
        dataset=[{"q": "a", "a": "b"}],
        column_mapping=ColumnMapping(inputs={"q": "q"}, outputs={"a": "a"}),
    )

    assert req.seed is None


def test_profile_dataset_response_round_trip() -> None:
    """Verify ProfileDatasetResponse stores the nested profile and plan."""
    resp = ProfileDatasetResponse(
        profile=DatasetProfile(row_count=5, column_count=2),
        plan=SplitPlan(
            fractions=SplitFractions(),
            shuffle=False,
            seed=0,
            counts=SplitCounts(train=4, val=0, test=1),
        ),
    )

    assert resp.profile.row_count == 5
    assert resp.plan.shuffle is False
    assert resp.plan.counts.train == 4
