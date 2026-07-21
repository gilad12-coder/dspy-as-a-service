"""Tests for the AI co-tagging engine's pure helpers.

Covers label normalization across the three annotation modes, defensive JSON
parsing of model output, few-shot example selection (corrections-first,
exclusions, provenance filtering), instruction compilation, and the credit
estimator.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from .. import tagging
from ..tagging import (
    MAX_EXAMPLES,
    InterviewTurnSig,
    _MessageLeakGuard,
    _parse_interview_prediction,
    _parse_json,
    assist_model_name,
    compile_instructions,
    effective_task_config,
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
    assert _parse_json("```json\n[1, 2]\n```", None) == [1, 2]
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


def test_missing_task_definition_interviews_in_both_assist_modes() -> None:
    """Autopilot automates the tagging, not the task definition — both modes ask."""
    base = {"mode": "binary", "inputColumns": ["text"]}
    copilot = task_description({**base, "_assist_mode": "copilot"})
    autopilot = task_description({**base, "_assist_mode": "autopilot"})
    assert "first question must ask" in copilot
    assert autopilot == copilot
    multiclass = task_description({"mode": "multiclass", "_assist_mode": "copilot"})
    extraction = task_description({"mode": "freetext", "_assist_mode": "autopilot"})
    assert "which categories" in multiclass
    assert "must ask the user" in extraction


def test_interview_normalizes_inferred_multiclass_categories() -> None:
    """Convert inferred category names into stable tagger category records."""
    pred = SimpleNamespace(
        done="true",
        message="Ready",
        options_json="[]",
        rubric_json='["Apply every matching category."]',
        task_config_json='{"categories": ["Billing", "Support", "Billing"]}',
    )
    turn = _parse_interview_prediction(pred, 1, {"mode": "multiclass"})
    assert turn["task_override"] == {
        "categories": [
            {"id": "cat1", "label": "Billing"},
            {"id": "cat2", "label": "Support"},
        ]
    }


def test_provisional_mode_interviews_in_both_assist_modes() -> None:
    """A provisional-mode config leaves the answer style to the interview."""
    base = {"mode": "freetext", "modeProvisional": True, "inputColumns": ["text"]}
    copilot = task_description({**base, "_assist_mode": "copilot"})
    autopilot = task_description({**base, "_assist_mode": "autopilot"})
    assert "You decide the answer style" in copilot
    assert "2-4 concrete task directions" in copilot
    assert '"mode"' in copilot
    assert autopilot == copilot


def test_provisional_override_carries_inferred_mode() -> None:
    """Finalize payloads on provisional sessions emit the inferred mode."""
    provisional = {"mode": "freetext", "modeProvisional": True}
    pred = SimpleNamespace(
        done="true",
        message="Ready",
        options_json="[]",
        rubric_json='["Answer the question."]',
        task_config_json='{"mode": "binary", "question": "Is it spam?"}',
    )
    turn = _parse_interview_prediction(pred, 1, provisional)
    assert turn["task_override"] == {"mode": "binary", "question": "Is it spam?"}
    pred.task_config_json = '{"categories": ["Billing", "Support"]}'
    turn = _parse_interview_prediction(pred, 1, provisional)
    assert turn["task_override"]["mode"] == "multiclass"
    assert [c["label"] for c in turn["task_override"]["categories"]] == ["Billing", "Support"]
    pred.task_config_json = '{"mode": "multiclass", "categories": ["Only one"]}'
    assert _parse_interview_prediction(pred, 1, provisional)["task_override"] == {}
    pred.task_config_json = '{"mode": "freetext"}'
    assert _parse_interview_prediction(pred, 1, provisional)["task_override"] == {"mode": "freetext"}


def test_interview_title_rides_the_final_turn_only() -> None:
    """The proposed session title is stripped and gated on ``done``."""
    pred = SimpleNamespace(
        done="true",
        message="Ready",
        options_json="[]",
        rubric_json='["Rule."]',
        task_config_json="{}",
        session_title='  "Routing support tickets"  ',
    )
    turn = _parse_interview_prediction(pred, 1, _FREE)
    assert turn["title"] == "Routing support tickets"
    pred.done = "false"
    assert _parse_interview_prediction(pred, 1, _FREE)["title"] == ""


def test_message_leak_guard_passes_clean_prose() -> None:
    """A prose reply streams through unchanged, flushing buffered whitespace."""
    guard = _MessageLeakGuard()
    assert guard.feed("\n") == ("", False)
    assert guard.feed("What should") == ("\nWhat should", False)
    assert guard.feed(" a row count as?") == (" a row count as?", False)


def test_message_leak_guard_mutes_structured_openings() -> None:
    """A reply that opens like the raw payload is never forwarded."""
    for opening in ('{"message": "hi",', "[[ ## message ## ]]", "```json"):
        guard = _MessageLeakGuard()
        assert guard.feed(opening) == ("", False)
        assert guard.feed(' "options_json": []}') == ("", False)


def test_message_leak_guard_resets_on_midstream_drift() -> None:
    """Prose that drifts into payload fields resets the client once, then mutes."""
    guard = _MessageLeakGuard()
    assert guard.feed("Which direction fits?") == ("Which direction fits?", False)
    assert guard.feed('\n"options_json": [{"label"') == ("", True)
    assert guard.feed(': "Topic"}]') == ("", False)


def test_interview_signature_streams_done_before_payload_fields() -> None:
    """``done`` precedes the slow payload fields so the stream can hint early.

    The interview stream emits ``turn_hint`` from the streamed ``done`` field;
    that only works while ``done`` is generated before options/rubric/task.
    """
    fields = list(InterviewTurnSig.output_fields)
    assert fields.index("done") < fields.index("options_json")
    assert fields.index("done") < fields.index("rubric_json")
    assert fields.index("done") < fields.index("task_config_json")
    assert fields.index("done") < fields.index("session_title")


def test_summarize_dataset_samples_rows() -> None:
    """The summary carries the row count and a bounded sample."""
    data = [{"id": i, "text": f"row {i}"} for i in range(100)]
    summary = summarize_dataset(_BINARY, ["text"], data)
    assert '"row_count": 100' in summary
    assert "row 0" in summary


def test_summarize_dataset_marks_unselected_columns_excluded() -> None:
    """Columns outside the input selection surface as excluded, not available."""
    data = [{"id": 1, "text": "What is 2+2?", "question": "What is 2+2?", "answer": "4"}]
    config = {"mode": "freetext", "inputColumns": ["question"], "modeProvisional": True}
    summary = json.loads(summarize_dataset(config, ["question", "answer"], data))
    assert summary["input_columns"] == ["question"]
    assert summary["excluded_columns"] == ["answer"]
    assert "all_columns" not in summary


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


def test_estimate_prices_on_chosen_model() -> None:
    """A chosen tagging model rides the estimate; blank falls back to default."""
    rows = [{"id": 1, "text": "x" * 400}]
    chosen = estimate_credits_for_rows("instructions", rows, model="openai/gpt-test")
    assert chosen["model"] == "openai/gpt-test"
    fallback = estimate_credits_for_rows("instructions", rows, model="  ")
    assert fallback["model"] == assist_model_name()


def test_effective_task_config_lifts_chosen_model() -> None:
    """``assist.model`` merges into the effective config; blank stays absent."""
    merged = effective_task_config(_BINARY, {"model": " openai/gpt-test "})
    assert merged["model"] == "openai/gpt-test"
    assert "model" not in effective_task_config(_BINARY, {"model": "  "})
    assert "model" not in effective_task_config(_BINARY, {})


def test_build_assist_lm_honors_model_override(monkeypatch) -> None:
    """The chosen model reaches the LM builder; empty falls back to default."""
    captured: list[str] = []

    def fake_build(config, disable_cache):
        """Record the requested model name instead of building an LM."""
        captured.append(config.name)
        return "lm"

    monkeypatch.setattr(tagging, "build_language_model", fake_build)
    monkeypatch.setattr(tagging, "apply_model_reasoning_config", lambda config: config)
    tagging._build_assist_lm("openai/gpt-test")
    tagging._build_assist_lm(None)
    assert captured == ["openai/gpt-test", assist_model_name()]
