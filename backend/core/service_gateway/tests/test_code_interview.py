"""Tests for the Signature & Metric interview engine.

Pure-function coverage of the turn parser and input assembly — the streaming
path is exercised through the router test with the engine monkeypatched, so
no LLM or network is touched here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from core.service_gateway.agents.code_interview import (
    MAX_INTERVIEW_QUESTIONS,
    _interview_inputs,
    _parse_interview_prediction,
)


def _pred(**fields: str) -> SimpleNamespace:
    """Build a prediction-shaped object with the given output fields."""
    return SimpleNamespace(**fields)


def test_parse_turn_in_progress() -> None:
    """A mid-interview turn keeps quick replies and withholds the brief."""
    turn = _parse_interview_prediction(
        _pred(
            message="How strict should matching be?",
            quick_replies_json='["Exact", "Lenient"]',
            brief_json='["premature directive"]',
            done="false",
        ),
        asked=1,
    )
    assert turn["done"] is False
    assert turn["quick_replies"] == ["Exact", "Lenient"]
    assert turn["brief"] == []
    assert turn["message"] == "How strict should matching be?"
    assert turn["model"]


def test_parse_turn_done_emits_brief_and_drops_quick_replies() -> None:
    """A finished turn carries the cleaned brief and no quick replies."""
    turn = _parse_interview_prediction(
        _pred(
            message="Wrapping up.",
            quick_replies_json='["Yes"]',
            brief_json='["Outputs must be lowercase.", "  ", "Score partial matches at 0.5."]',
            done="true",
        ),
        asked=3,
    )
    assert turn["done"] is True
    assert turn["quick_replies"] == []
    assert turn["brief"] == ["Outputs must be lowercase.", "Score partial matches at 0.5."]


def test_parse_turn_forces_done_at_question_limit() -> None:
    """Hitting the question cap ends the interview even if the model resists."""
    turn = _parse_interview_prediction(
        _pred(message="One more?", quick_replies_json="[]", brief_json="[]", done="false"),
        asked=MAX_INTERVIEW_QUESTIONS,
    )
    assert turn["done"] is True


def test_parse_turn_tolerates_garbage_outputs() -> None:
    """Unparseable JSON fields degrade to empty lists, never raise."""
    turn = _parse_interview_prediction(
        _pred(message="hi", quick_replies_json="not json", brief_json="{broken", done="nope"),
        asked=0,
    )
    assert turn["quick_replies"] == []
    assert turn["brief"] == []
    assert turn["done"] is False


def test_parse_turn_caps_quick_replies_at_four() -> None:
    """At most four quick replies survive parsing."""
    turn = _parse_interview_prediction(
        _pred(
            message="Pick one",
            quick_replies_json='["a", "b", "c", "d", "e"]',
            brief_json="[]",
            done="false",
        ),
        asked=0,
    )
    assert turn["quick_replies"] == ["a", "b", "c", "d"]


def test_interview_inputs_shapes_and_language() -> None:
    """Inputs are valid JSON, carry the model fallback, and map the locale."""
    inputs = _interview_inputs(
        dataset_columns=["text", "label"],
        column_roles={"text": "input", "label": "output"},
        column_kinds={"text": "text"},
        sample_rows=[{"text": "hi", "label": "pos"}],
        job_model="",
        turns=[],
        locale="en",
    )
    assert json.loads(inputs["column_roles"]) == {"text": "input", "label": "output"}
    assert json.loads(inputs["sample_rows"]) == [{"text": "hi", "label": "pos"}]
    assert json.loads(inputs["transcript_json"]) == []
    assert inputs["job_model"] == "not chosen yet"
    assert inputs["reply_language"] == "English"


def test_interview_inputs_appends_question_count_note() -> None:
    """Once questions were asked, the transcript carries the limit reminder."""
    turns = [
        {"role": "assistant", "content": "Q1?"},
        {"role": "user", "content": "A1"},
    ]
    inputs = _interview_inputs(
        dataset_columns=["text"],
        column_roles={"text": "input"},
        column_kinds={},
        sample_rows=[],
        job_model="openai/gpt-4o-mini",
        turns=turns,
        locale=None,
    )
    transcript = json.loads(inputs["transcript_json"])
    assert transcript[:2] == turns
    assert transcript[-1]["role"] == "system"
    assert f"1 of at most {MAX_INTERVIEW_QUESTIONS}" in transcript[-1]["content"]
    assert inputs["job_model"] == "openai/gpt-4o-mini"
    assert inputs["reply_language"] == "Hebrew"
