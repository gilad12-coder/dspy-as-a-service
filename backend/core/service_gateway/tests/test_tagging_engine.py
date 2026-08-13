"""Tests for the AI co-tagging engine's pure helpers.

Covers label normalization across the three annotation modes, defensive JSON
parsing of model output, few-shot example selection (corrections-first,
exclusions, provenance filtering), instruction compilation, and the credit
estimator.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from .. import tagging
from ..tagging import (
    MAX_EXAMPLES,
    InterviewTurnSig,
    _MessageLeakGuard,
    _parse_interview_prediction,
    _parse_json,
    _StreamedArrayItems,
    assist_model_config,
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
    """Binary labels normalize yes/no spellings to 1/0 and reject junk."""
    assert normalize_label(_BINARY, "Yes") == "1"
    assert normalize_label(_BINARY, "NO") == "0"
    assert normalize_label(_BINARY, "true") == "1"
    assert normalize_label(_BINARY, "כן") == "1"
    assert normalize_label(_BINARY, "1") == "1"
    assert normalize_label(_BINARY, "0") == "0"
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
    # session_title is the last output field, so minimax's malformed terminal
    # marker lands in its tail (seen in the wild as "… [[ ## completed ## ]").
    pred.session_title = "Route support tickets [[ ## completed ## ]"
    assert _parse_interview_prediction(pred, 1, _FREE)["title"] == "Route support tickets"
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


def test_streamed_array_items_emits_objects_per_chunk() -> None:
    """Objects surface the moment their closing brace arrives, across chunk splits."""
    scanner = _StreamedArrayItems()
    assert scanner.feed('[{"id": "1", "label"') == []
    assert scanner.feed(': "yes"}, {"id": "2",') == [{"id": "1", "label": "yes"}]
    assert scanner.feed(' "label": "no"}]') == [{"id": "2", "label": "no"}]


def test_streamed_array_items_ignores_braces_inside_strings() -> None:
    """Braces and escaped quotes inside string values never skew the balance."""
    scanner = _StreamedArrayItems()
    items = scanner.feed('[{"id": "1", "reason": "brace } and quote \\" inside"}]')
    assert items == [{"id": "1", "reason": 'brace } and quote " inside'}]


def test_streamed_array_items_survives_fences_and_nesting() -> None:
    """Fences and prose around the array are ignored; nested objects stay whole."""
    scanner = _StreamedArrayItems()
    items = scanner.feed('```json\n[{"id": "1", "extra": {"a": 1}}]\n```')
    assert items == [{"id": "1", "extra": {"a": 1}}]


def test_predict_rows_stream_merges_batches_into_terminal_event(monkeypatch) -> None:
    """Per-row events from every batch relay through; the terminal map merges them."""

    async def fake_drive(*, lm, instructions, batch, config, queue, semaphore) -> None:
        """Emit one canned prediction event per batch row."""
        for row in batch:
            await queue.put(
                {
                    "event": "prediction",
                    "data": {"id": str(row["id"]), "prediction": {"value": "1", "confidence": 0.9, "reason": ""}},
                }
            )

    monkeypatch.setattr(tagging, "_drive_predict_batch", fake_drive)
    monkeypatch.setattr(tagging, "_build_assist_lm", lambda *a, **k: SimpleNamespace(history=[]))
    monkeypatch.setattr(tagging, "usage_by_model_from_history", lambda lm: {})
    rows = [{"id": i, "text": f"row {i}"} for i in range(tagging.BATCH_SIZE + 2)]

    async def run() -> list[dict]:
        """Collect the full event stream."""
        return [event async for event in tagging.predict_rows_stream(_BINARY, "instructions", rows)]

    events = asyncio.run(run())
    assert events[-1]["event"] == "predict_done"
    assert all(e["event"] == "prediction" for e in events[:-1])
    assert set(events[-1]["data"]["predictions"]) == {str(i) for i in range(tagging.BATCH_SIZE + 2)}
    assert events[-1]["data"]["credits"] == 0


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


def test_estimate_applies_byok_platform_fee() -> None:
    """A BYOK estimate charges only the platform-fee share of the same usage."""
    rows = [{"id": i, "text": "x" * 4000} for i in range(100)]
    managed = estimate_credits_for_rows("instructions", rows, model="openai/gpt-test", token_source="managed")
    byok = estimate_credits_for_rows("instructions", rows, model="openai/gpt-test", token_source="byok")
    assert 0 < byok["credits_low"] < managed["credits_low"]


def test_effective_task_config_lifts_chosen_model() -> None:
    """``assist.model`` merges into the effective config; blank stays absent."""
    merged = effective_task_config(_BINARY, {"model": " openai/gpt-test "})
    assert merged["model"] == "openai/gpt-test"
    assert "model" not in effective_task_config(_BINARY, {"model": "  "})
    assert "model" not in effective_task_config(_BINARY, {})


def test_effective_task_config_lifts_model_params_with_model() -> None:
    """``assist.modelParams`` rides along the chosen model, never alone."""
    params = {"temperature": 0.2, "max_tokens": 2048}
    merged = effective_task_config(_BINARY, {"model": "openai/gpt-test", "modelParams": params})
    assert merged["modelParams"] == params
    assert "modelParams" not in effective_task_config(_BINARY, {"modelParams": params})
    assert "modelParams" not in effective_task_config(
        _BINARY, {"model": "openai/gpt-test", "modelParams": "junk"}
    )


def test_sanitize_model_params_bounds_and_filters() -> None:
    """Sampling knobs are clamped; connection fields and junk never pass."""
    out = tagging._sanitize_model_params(
        {
            "temperature": 5,
            "top_p": -1,
            "max_tokens": 512.0,
            "base_url": "http://evil",
            "token_source": "byok",
            "byok_provider": " openrouter ",
            "extra": {"reasoning_effort": "High", "api_key": "sk-leak"},
        }
    )
    assert out == {
        "temperature": 2.0,
        "top_p": 0.0,
        "max_tokens": 512,
        "token_source": "byok",
        "byok_provider": "openrouter",
        "extra": {"reasoning_effort": "high"},
    }
    assert tagging._sanitize_model_params(None) == {}
    assert tagging._sanitize_model_params("junk") == {}
    assert tagging._sanitize_model_params(
        {"temperature": "hot", "max_tokens": 0, "extra": {"reasoning_effort": "extreme"}}
    ) == {}


def test_assist_model_config_preserves_source_without_inline_connection() -> None:
    """Tagging keeps BYOK selectors while rejecting persisted connection fields."""
    config = assist_model_config(
        {
            "model": " openrouter/openai/gpt-test ",
            "modelParams": {
                "token_source": "byok",
                "byok_provider": "openrouter",
                "base_url": "https://untrusted.example",
                "extra": {"api_key": "sk-inline", "reasoning_effort": "high"},
            },
        }
    )
    assert config.name == "openrouter/openai/gpt-test"
    assert config.token_source == "byok"
    assert config.byok_provider == "openrouter"
    assert config.base_url is None
    assert config.extra == {"reasoning_effort": "high"}


def test_build_assist_lm_honors_model_override(monkeypatch) -> None:
    """The chosen model and params reach the LM builder; empty falls back."""
    captured: list[tuple[str, float | None, int | None]] = []

    def fake_build(config, disable_cache):
        """Record the requested model config instead of building an LM."""
        captured.append((config.name, config.temperature, config.max_tokens))
        return "lm"

    monkeypatch.setattr(tagging, "build_language_model", fake_build)
    monkeypatch.setattr(tagging, "apply_model_reasoning_config", lambda config: config)
    tagging._build_assist_lm("openai/gpt-test", {"temperature": 0.1, "max_tokens": 2048})
    tagging._build_assist_lm(None)
    assert captured == [
        ("openai/gpt-test", 0.1, 2048),
        (assist_model_name(), None, None),
    ]


def test_build_assist_lm_merges_lm_extra_body(monkeypatch) -> None:
    """Auto-router extras land in the LM config without clobbering params."""
    captured: list[dict] = []

    def fake_build(config, disable_cache):
        """Record the config extras instead of building an LM."""
        captured.append(config.extra)
        return "lm"

    monkeypatch.setattr(tagging, "build_language_model", fake_build)
    monkeypatch.setattr(tagging, "apply_model_reasoning_config", lambda config: config)
    tagging._build_assist_lm(
        "openrouter/openrouter/auto-beta",
        {"extra": {"reasoning_effort": "high"}},
        lm_extra_body={"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 5}]},
    )
    assert captured == [
        {
            "reasoning_effort": "high",
            "extra_body": {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 5}]},
        }
    ]
