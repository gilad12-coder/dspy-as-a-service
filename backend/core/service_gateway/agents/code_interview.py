"""LLM engine for the Signature & Metric authoring interview. [INTERNAL]

Mirrors the tagger's dataset interview (``core.service_gateway.tagging``):
a stateless, single-turn dspy Signature streamed over the generalist agent's
event shapes. The interviewer asks the dataset owner a handful of grounded
questions before the seed generation runs, then distills their answers into
an authoring brief — short directives the Signature/metric seed authors must
honor. The transcript is client-owned and re-sent on every turn; nothing
here touches the database.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import dspy

from ...config import settings
from .code import ReasoningStreamListener, _build_agent_lm, _reply_language
from .constants import REASONING_FIELD

logger = logging.getLogger(__name__)

MAX_INTERVIEW_QUESTIONS = 5


class CodeInterviewTurnSig(dspy.Signature):
    """Interview the dataset owner to distill a Signature & Metric brief.

    You are an authoring copilot about to write a DSPy Signature (the
    input → output contract whose docstring and field descriptions become
    the runtime prompt) and a Metric (the function that scores predictions)
    for the user's optimization job. Before writing anything, interview the
    owner. Ask ONE short, concrete question at a time — grounded in the
    sample rows — about what a high-quality output looks like, how strictly
    answers should match the reference, edge cases and dirty rows, tone /
    length / formatting / language constraints, and what should be rewarded
    or penalized when scoring. Never ask a generic question the dataset
    already answers, and never ask which LLM to use — that is chosen in a
    later step (``job_model`` tells you the current choice when one exists;
    take its strengths and quirks into account in what you ask). After at
    most five questions total (or as soon as the user asks to proceed, or
    their answers stop adding information), stop asking: set ``done`` to
    true, write a one-sentence wrap-up in ``message``, and emit the full
    brief — 4 to 10 crisp directives for the code authors. Directives state
    decisions ("Outputs must ...", "Score X lower when ..."), not process.
    Write ``message``, quick replies and the brief in ``reply_language``,
    keeping the product terms ``Signature`` and ``Metric`` in English.
    """

    dataset_columns: list[str] = dspy.InputField(desc="Every column name in the dataset.")
    column_roles: str = dspy.InputField(
        desc="JSON object mapping column name → 'input' | 'output' | 'ignore'.",
    )
    column_kinds: str = dspy.InputField(
        desc="JSON object mapping input-column name → 'text' | 'image'.",
    )
    sample_rows: str = dspy.InputField(
        desc="JSON array of up to 5 representative rows from the dataset.",
    )
    job_model: str = dspy.InputField(
        desc=(
            "The LLM the optimized program will run on, e.g. "
            "'openai/gpt-4o-mini'; 'not chosen yet' when the user has not "
            "picked one."
        ),
    )
    transcript_json: str = dspy.InputField(desc="JSON array of prior {role, content} turns.")
    reply_language: str = dspy.InputField(desc="Language every output is written in.")
    message: str = dspy.OutputField(desc="The next question, or a short wrap-up when done.")
    quick_replies_json: str = dspy.OutputField(
        desc="JSON array of 0-4 short suggested answers for a closed question; [] otherwise."
    )
    brief_json: str = dspy.OutputField(desc="JSON array of authoring-directive strings; [] until done.")
    done: str = dspy.OutputField(desc="'true' when the interview is finished, else 'false'.")


def _parse_json(text: str, fallback: Any) -> Any:
    """Parse a JSON-in-a-string model output, returning ``fallback`` on failure.

    Args:
        text: The raw output field value.
        fallback: Value returned when the text is empty or unparseable.
    """
    try:
        return json.loads(text) if text and text.strip() else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def _interview_inputs(
    dataset_columns: list[str],
    column_roles: dict[str, str],
    column_kinds: dict[str, str],
    sample_rows: list[dict[str, Any]],
    job_model: str,
    turns: list[dict[str, str]],
    locale: str | None,
) -> dict[str, Any]:
    """Assemble the ``CodeInterviewTurnSig`` inputs for one turn.

    Args:
        dataset_columns: All dataset column names.
        column_roles: Column name → role mapping.
        column_kinds: Input column → kind (text/image) mapping.
        sample_rows: Up to 5 representative rows.
        job_model: The job's target model id; empty when not chosen yet.
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    noted = list(turns)
    if asked:
        noted.append(
            {
                "role": "system",
                "content": (
                    f"Questions asked so far: {asked} of at most {MAX_INTERVIEW_QUESTIONS}."
                    " If the limit is reached you MUST finish now."
                ),
            }
        )
    transcript = json.dumps(noted, ensure_ascii=False)
    return {
        "dataset_columns": dataset_columns,
        "column_roles": json.dumps(column_roles, ensure_ascii=False),
        "column_kinds": json.dumps(column_kinds, ensure_ascii=False),
        "sample_rows": json.dumps(sample_rows, ensure_ascii=False),
        "job_model": job_model.strip() or "not chosen yet",
        "transcript_json": transcript,
        "reply_language": _reply_language(locale),
    }


def _parse_interview_prediction(pred: Any, asked: int) -> dict[str, Any]:
    """Turn a raw ``CodeInterviewTurnSig`` prediction into the client turn payload.

    Args:
        pred: The prediction (or ``None`` when the stream produced nothing).
        asked: Assistant questions asked before this turn (limit enforcement).

    Returns:
        ``{"message", "quick_replies", "brief", "done", "model"}`` — ``brief``
        is empty until ``done`` is true.
    """
    done = str(getattr(pred, "done", "")).strip().lower() in {"true", "yes", "1"}
    brief = _parse_json(getattr(pred, "brief_json", "[]"), [])
    brief = [str(b).strip() for b in brief if str(b).strip()] if isinstance(brief, list) else []
    if asked >= MAX_INTERVIEW_QUESTIONS and not done:
        done = True
    quick = _parse_json(getattr(pred, "quick_replies_json", "[]"), [])
    quick = [str(q).strip() for q in quick if str(q).strip()][:4] if isinstance(quick, list) else []
    return {
        "message": str(getattr(pred, "message", "")).strip(),
        "quick_replies": [] if done else quick,
        "brief": brief if done else [],
        "done": done,
        "model": settings.code_agent_model,
    }


async def interview_turn_stream(
    *,
    dataset_columns: list[str],
    column_roles: dict[str, str],
    column_kinds: dict[str, str],
    sample_rows: list[dict[str, Any]],
    job_model: str,
    turns: list[dict[str, str]],
    locale: str | None,
) -> Any:
    """Run one interview turn, streaming it the way the generalist agent does.

    Yields ``{"event", "data"}`` mappings ready for ``sse_from_events``:
    ``reasoning_patch`` for provider thinking tokens (same synthetic channel
    the agents use), ``message_patch`` for reply deltas, and a terminal
    ``interview_done`` carrying the parsed turn.

    Args:
        dataset_columns: All dataset column names.
        column_roles: Column name → role mapping.
        column_kinds: Input column → kind (text/image) mapping.
        sample_rows: Up to 5 representative rows.
        job_model: The job's target model id; empty when not chosen yet.
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    predict = dspy.Predict(CodeInterviewTurnSig)
    program = dspy.streamify(
        predict,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="message"),
            ReasoningStreamListener(predict=predict),
        ],
        async_streaming=True,
    )
    lm = _build_agent_lm()
    inputs = _interview_inputs(
        dataset_columns, column_roles, column_kinds, sample_rows, job_model, turns, locale
    )
    prediction: Any = None
    with dspy.context(lm=lm):
        async for chunk in program(**inputs):
            if isinstance(chunk, dspy.streaming.StreamResponse):
                if chunk.signature_field_name == REASONING_FIELD:
                    yield {"event": "reasoning_patch", "data": {"chunk": chunk.chunk}}
                elif chunk.signature_field_name == "message":
                    yield {"event": "message_patch", "data": {"chunk": chunk.chunk}}
            elif isinstance(chunk, dspy.Prediction):
                prediction = chunk
    yield {"event": "interview_done", "data": _parse_interview_prediction(prediction, asked)}
