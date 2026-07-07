"""Tests for the AI co-tagging engine's pure helpers.

Covers label normalization across the three annotation modes, defensive JSON
parsing of model output, few-shot example selection (corrections-first,
exclusions, provenance filtering), instruction compilation, the credit
estimator, and the deep-optimize builders — the generated signature/metric
sources are exec'd and exercised the way the run loader would, so a payload
that would fail at run time fails here first.
"""

from __future__ import annotations

from types import SimpleNamespace

import dspy

from ..tagging import (
    MAX_EXAMPLES,
    _deep_optimize_metric_code,
    _deep_optimize_signature_code,
    _parse_json,
    build_deep_optimize_dataset,
    build_deep_optimize_request,
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


def _exec_metric(code: str):
    """Exec generated metric source the way the run loader does.

    Args:
        code: The metric module source.

    Returns:
        The ``metric`` callable from the exec'd namespace.
    """
    namespace: dict = {"dspy": dspy}
    exec(code, namespace)
    return namespace["metric"]


def test_deep_optimize_dataset_flattens_labels_and_skips_auto() -> None:
    """The trainset carries string labels and only human-vetted rows."""
    data = [{"id": i, "text": f"row {i}"} for i in range(1, 5)]
    annotations = {"1": ["cat1", "cat2"], "2": ["cat2"], "3": ["cat1"]}
    assist = {"predictions": {}, "provenance": {"1": "human", "2": "ai_confirmed", "3": "ai_auto"}}
    rows = build_deep_optimize_dataset(_MULTI, data, annotations, assist)
    by_text = {r["text"]: r["label"] for r in rows}
    assert by_text == {"row 1": "Billing; Support", "row 2": "Support"}


def test_deep_optimize_signature_is_single_and_sanitized() -> None:
    """The generated source defines exactly one Signature, docstring intact."""
    code = _deep_optimize_signature_code(_BINARY, ['Rule with """triple quotes"""'])
    namespace: dict = {"dspy": dspy}
    exec(code, namespace)
    classes = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, dspy.Signature)
    ]
    assert len(classes) == 1
    signature = classes[0]
    assert "Positive?" in (signature.__doc__ or "")
    assert '"""' not in (signature.__doc__ or "")
    assert set(signature.input_fields) == {"text"}
    assert set(signature.output_fields) == {"label"}


def test_deep_optimize_metrics_score_correctly() -> None:
    """Each mode's generated metric scores agreement and disagreement."""
    binary = _exec_metric(_deep_optimize_metric_code(_BINARY))
    assert binary(SimpleNamespace(label="yes"), SimpleNamespace(label="Yes ")).score == 1.0
    miss = binary(SimpleNamespace(label="yes"), SimpleNamespace(label="no"))
    assert miss.score == 0.0
    assert "yes" in miss.feedback

    multi = _exec_metric(_deep_optimize_metric_code(_MULTI))
    assert (
        multi(
            SimpleNamespace(label="Billing; Support"),
            SimpleNamespace(label="support;  billing"),
        ).score
        == 1.0
    )
    wrong = multi(SimpleNamespace(label="Billing"), SimpleNamespace(label="Support"))
    assert wrong.score == 0.0
    assert "billing" in wrong.feedback

    free = _exec_metric(_deep_optimize_metric_code(_FREE))
    assert (
        free(SimpleNamespace(label="Paris, France"), SimpleNamespace(label="paris france")).score
        == 1.0
    )
    assert free(SimpleNamespace(label="Paris"), SimpleNamespace(label="London")).score == 0.0


def test_deep_optimize_request_builds_valid_payload() -> None:
    """The assembled RunRequest passes model validation with GEPA wiring."""
    dataset = [{"text": f"row {i}", "label": "yes" if i % 2 else "no"} for i in range(12)]
    payload = build_deep_optimize_request(
        _BINARY, ["Sarcasm counts."], dataset, name="Deep optimize · test", username="alice"
    )
    assert payload.optimizer_name == "gepa"
    assert payload.optimizer_kwargs == {"auto": "light"}
    assert payload.reflection_model_settings is not None
    assert payload.column_mapping.inputs == {"text": "text"}
    assert payload.column_mapping.outputs == {"label": "label"}
    # Parity with wizard submissions: no forced privacy, default visibility.
    assert payload.is_private is False
    assert len(payload.dataset) == 12
    assert "Sarcasm counts." in payload.signature_code


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
