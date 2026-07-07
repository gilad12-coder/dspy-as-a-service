"""Tests for the AI co-tagging engine's pure helpers.

Covers label normalization across the three annotation modes, defensive JSON
parsing of model output, few-shot example selection (corrections-first,
exclusions, provenance filtering), instruction compilation, and the credit
estimator — everything that runs without an LM.
"""

from __future__ import annotations

from ..tagging import (
    MAX_EXAMPLES,
    _parse_json,
    compile_instructions,
    estimate_credits_for_rows,
    normalize_label,
    select_examples,
    summarize_dataset,
    task_description,
)

_BINARY = {"mode": "binary", "inputColumns": ["text"], "question": "Positive?"}
_MULTI = {
    "mode": "multiclass",
    "inputColumns": ["text"],
    "categories": [
        {"id": "cat1", "label": "Billing"},
        {"id": "cat2", "label": "Support"},
    ],
}
_FREE = {"mode": "freetext", "inputColumns": ["text"], "prompt": "Extract the city."}


def test_normalize_binary_maps_variants() -> None:
    """Binary labels normalize yes/no spellings and reject junk."""
    assert normalize_label(_BINARY, "Yes") == "yes"
    assert normalize_label(_BINARY, "NO") == "no"
    assert normalize_label(_BINARY, "true") == "yes"
    assert normalize_label(_BINARY, "כן") == "yes"
    assert normalize_label(_BINARY, "maybe") is None


def test_normalize_multiclass_maps_names_to_ids() -> None:
    """Category names map to ids case-insensitively; unknowns are dropped."""
    assert normalize_label(_MULTI, ["billing", "Support"]) == ["cat1", "cat2"]
    assert normalize_label(_MULTI, "Billing") == ["cat1"]
    assert normalize_label(_MULTI, ["cat2"]) == ["cat2"]
    assert normalize_label(_MULTI, ["nonsense"]) is None


def test_normalize_freetext_strips_and_rejects_empty() -> None:
    """Freetext labels are stripped strings; empty means unmappable."""
    assert normalize_label(_FREE, "  Paris ") == "Paris"
    assert normalize_label(_FREE, "") is None
    assert normalize_label(_FREE, None) is None


def test_parse_json_tolerates_fences_and_prose() -> None:
    """Model JSON survives code fences and surrounding prose."""
    assert _parse_json('```json\n[1, 2]\n```', None) == [1, 2]
    assert _parse_json('Here you go: {"a": 1} — done.', None) == {"a": 1}
    assert _parse_json("not json at all", "fallback") == "fallback"


def test_select_examples_prioritizes_corrections_and_excludes() -> None:
    """Corrections rank first; excluded and auto-tagged rows never appear."""
    data = [{"id": i, "text": f"row {i}"} for i in range(1, 6)]
    annotations = {"1": "yes", "2": "no", "3": "yes", "4": "no"}
    assist = {
        "predictions": {"2": {"value": "yes", "confidence": 0.9}},
        "provenance": {"1": "human", "2": "human", "3": "ai_confirmed", "4": "ai_auto"},
    }
    examples = select_examples(_BINARY, data, annotations, assist, exclude_ids={"3"})
    texts = [e["text"] for e in examples]
    assert texts[0] == "row 2"
    assert examples[0]["corrected_from"] == "yes"
    assert "row 3" not in texts
    assert "row 4" not in texts
    assert len(examples) <= MAX_EXAMPLES


def test_compile_instructions_carries_task_rubric_examples() -> None:
    """The compiled prompt contains the task, every rubric rule and examples."""
    examples = [{"text": "great", "label": "yes"}]
    prompt = compile_instructions(_BINARY, ["Sarcasm counts as negative."], examples)
    assert "Positive?" in prompt
    assert "Sarcasm counts as negative." in prompt
    assert "great" in prompt
    assert task_description(_MULTI).count("Billing") == 1


def test_summarize_dataset_samples_rows() -> None:
    """The summary carries the row count and a bounded sample."""
    data = [{"id": i, "text": f"row {i}"} for i in range(100)]
    summary = summarize_dataset(_BINARY, ["text"], data)
    assert '"row_count": 100' in summary
    assert "row 0" in summary


def test_estimate_scales_with_rows_and_handles_empty() -> None:
    """More rows cost more; zero rows cost nothing."""
    rows = [{"id": i, "text": "x" * 400} for i in range(50)]
    small = estimate_credits_for_rows("instructions", rows[:10])
    large = estimate_credits_for_rows("instructions", rows)
    assert small["rows"] == 10
    assert large["credits_high"] >= large["credits_low"] >= small["credits_low"] >= 0
    empty = estimate_credits_for_rows("instructions", [])
    assert empty == {
        "rows": 0,
        "model": empty["model"],
        "credits_low": 0,
        "credits_high": 0,
    }
