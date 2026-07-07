"""LLM engine for AI-assisted tagging (co-tagging). [INTERNAL]

Powers the tagger's assist modes: the dataset interview that distills a
labeling rubric, silent per-row predictions during calibration, batched
review/auto-tagging, reflective rubric refinement ("deep optimize"), and the
pre-run credit estimate. Pure functions over the session payload — persistence
stays in the router; nothing here touches the database.

All structured LLM outputs are JSON-in-a-string fields (the repo-wide dspy
convention) parsed defensively, with a per-row fallback when a batch reply
cannot be parsed.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dspy

from ..billing.pricing import ModelUsage, credits_for_usage
from ..config import settings
from ..models import ColumnMapping, ModelConfig, RunRequest
from .agents.code import _reply_language
from .language_models import (
    apply_model_reasoning_config,
    build_language_model,
    usage_by_model_from_history,
)

logger = logging.getLogger(__name__)

MAX_INTERVIEW_QUESTIONS = 5
BATCH_SIZE = 10
BATCH_CONCURRENCY = 4
MAX_EXAMPLES = 40
SAMPLE_ROWS = 8
MAX_ROW_CHARS = 1200
# chars-per-token heuristic for the pre-run estimate; JSON label output per row.
CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_PER_ROW = 30
# GEPA needs enough labeled rows to split into train/val/test and still learn.
MIN_DEEP_OPTIMIZE_EXAMPLES = 10
DEEP_OPTIMIZE_MAX_EXAMPLES = 200


def assist_model_name() -> str:
    """Return the LiteLLM model id the tagging assist runs on."""
    return settings.tagger_assist_model or settings.generalist_agent_model


def _build_assist_lm() -> dspy.LM:
    """Build the assist LM from settings, mirroring the generalist agent.

    Returns:
        A cache-disabled ``dspy.LM`` on the configured tagging-assist model
        (falls back to the generalist agent's model when unset).
    """
    config = ModelConfig(
        name=assist_model_name(),
        base_url=settings.tagger_assist_base_url or settings.generalist_agent_base_url or None,
    )
    return build_language_model(apply_model_reasoning_config(config), disable_cache=True)


class InterviewTurnSig(dspy.Signature):
    """Interview the dataset owner to distill a labeling rubric.

    You are a labeling copilot preparing to tag the user's dataset for them.
    Ask ONE short, concrete question at a time — grounded in the sample rows —
    about ambiguous label boundaries, edge cases, and how to treat dirty or
    off-topic rows. Never ask generic questions the task description already
    answers. After at most five questions total (or as soon as the user asks
    to proceed, or their answers stop adding information), stop asking: set
    ``done`` to true, write a one-sentence wrap-up in ``message``, and emit
    the full rubric — 4 to 10 crisp, decision-ready rules that would let a
    stranger label exactly like the user. Rules state decisions ("X counts as
    Y when ..."), not process. Write ``message``, quick replies and the rubric
    in ``reply_language``.
    """

    task_description: str = dspy.InputField(desc="What is being labeled and the allowed labels.")
    dataset_summary: str = dspy.InputField(desc="Row count, input columns, sample values.")
    transcript_json: str = dspy.InputField(desc="JSON array of prior {role, content} turns.")
    reply_language: str = dspy.InputField(desc="Language every output is written in.")
    message: str = dspy.OutputField(desc="The next question, or a short wrap-up when done.")
    quick_replies_json: str = dspy.OutputField(
        desc="JSON array of 0-4 short suggested answers for a closed question; [] otherwise."
    )
    rubric_json: str = dspy.OutputField(desc="JSON array of rubric rule strings; [] until done.")
    done: str = dspy.OutputField(desc="'true' when the interview is finished, else 'false'.")


class TagBatchSig(dspy.Signature):
    """Label every row exactly as the instructions dictate.

    Return strictly a JSON array with one object per input row, in the same
    order: {"id": "<row id>", "label": <label>, "confidence": <0..1>,
    "reason": "<at most 12 words, in the rubric's language>"}. The label
    format is defined in the instructions (binary answer, category-name array,
    or extracted text). ``confidence`` is your honest probability that a
    careful human following the same instructions would produce your label.
    """

    task_instructions: str = dspy.InputField(desc="Task, rubric and labeled examples.")
    rows_json: str = dspy.InputField(desc="JSON array of {id, text} rows to label.")
    labels_json: str = dspy.OutputField(
        desc="JSON array, one {id, label, confidence, reason} object per row, same order."
    )


class TagOneSig(dspy.Signature):
    """Label a single row exactly as the instructions dictate.

    ``confidence`` is your honest probability that a careful human following
    the same instructions would produce your label; ``reason`` is at most 12
    words, in the rubric's language.
    """

    task_instructions: str = dspy.InputField(desc="Task, rubric and labeled examples.")
    row_text: str = dspy.InputField(desc="The row to label.")
    label_json: str = dspy.OutputField(
        desc='JSON object: {"label": <label>, "confidence": <0..1>, "reason": "..."}.'
    )


class RefineRubricSig(dspy.Signature):
    """Rewrite the labeling rubric from the human's labels and corrections.

    Study the labeled examples — especially rows where the AI's earlier guess
    was corrected by the human — and produce an improved rubric of 5 to 12
    crisp, decision-ready rules a stranger could apply to label exactly like
    the human. Keep rules that already work, sharpen the ones the corrections
    contradict, and add rules that resolve the observed disagreements. Rules
    state decisions, not process. Write the rubric in ``reply_language``.
    """

    task_description: str = dspy.InputField(desc="What is being labeled and the allowed labels.")
    current_rubric_json: str = dspy.InputField(desc="JSON array of the current rubric rules.")
    examples_json: str = dspy.InputField(
        desc="JSON array of {text, label, corrected_from?} labeled examples."
    )
    reply_language: str = dspy.InputField(desc="Language the rubric is written in.")
    rubric_json: str = dspy.OutputField(desc="JSON array of the improved rubric rule strings.")


def _parse_json(raw: str, fallback: Any) -> Any:
    """Parse a model-produced JSON string, tolerating code fences and noise.

    Args:
        raw: The raw output-field text.
        fallback: Value returned when nothing parseable is found.

    Returns:
        The parsed JSON value, or ``fallback``.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json")
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Salvage the outermost JSON array/object from surrounding prose.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return fallback


def _row_text(row: dict[str, Any]) -> str:
    """Return the flattened annotation text of a dataset row, length-capped.

    Args:
        row: A tagger ``DataRow`` payload dict.

    Returns:
        The row's ``text`` field truncated to ``MAX_ROW_CHARS``.
    """
    text = str(row.get("text") or "")
    return text[:MAX_ROW_CHARS]


def task_description(config: dict[str, Any]) -> str:
    """Describe the labeling task and its allowed labels for the LM.

    Args:
        config: The session's ``TaggerConfig`` payload.

    Returns:
        A compact English framing of the task; user-authored parts (question,
        category names, prompt) are passed through verbatim in their language.
    """
    mode = config.get("mode")
    if mode == "binary":
        question = str(config.get("question") or "Does the row match?")
        return (
            f'Binary labeling. For each row answer the question: "{question}". '
            'The label is exactly "yes" or "no".'
        )
    if mode == "multiclass":
        names = [str(c.get("label", "")).strip() for c in config.get("categories") or []]
        names = [n for n in names if n]
        return (
            "Multi-label classification. Assign each row every category that applies "
            f"from this exact list: {json.dumps(names, ensure_ascii=False)}. "
            "The label is a JSON array of the applying category names (at least one)."
        )
    prompt = str(config.get("prompt") or "Extract the relevant text.")
    return (
        f'Text extraction. For each row: "{prompt}". '
        "The label is the extracted text, taken from or grounded in the row."
    )


def summarize_dataset(
    config: dict[str, Any], columns: list[str], data: list[dict[str, Any]]
) -> str:
    """Summarize the dataset for the interview and rubric prompts.

    Args:
        config: The session's ``TaggerConfig`` payload.
        columns: All dataset column names.
        data: The full row payload; only a small sample is serialized.

    Returns:
        A compact text profile: row count, input columns, and sample rows
        spread across the dataset.
    """
    input_cols = [str(c) for c in config.get("inputColumns") or []]
    step = max(1, len(data) // SAMPLE_ROWS)
    sample = [_row_text(row) for row in data[::step][:SAMPLE_ROWS]]
    return json.dumps(
        {
            "row_count": len(data),
            "all_columns": columns,
            "input_columns": input_cols,
            "sample_rows": sample,
        },
        ensure_ascii=False,
    )


def _annotation_to_display(config: dict[str, Any], value: Any) -> Any:
    """Convert a stored annotation value to its prompt-facing display form.

    Multiclass annotations store category ids; prompts and examples use the
    human-readable category names.

    Args:
        config: The session's ``TaggerConfig`` payload.
        value: The stored annotation (str, or list of category ids).

    Returns:
        The display value (str, or list of category names).
    """
    if config.get("mode") == "multiclass" and isinstance(value, list):
        by_id = {c.get("id"): str(c.get("label", "")) for c in config.get("categories") or []}
        return [by_id.get(v, str(v)) for v in value]
    return value


def normalize_label(config: dict[str, Any], raw: Any) -> str | list[str] | None:
    """Normalize a model-produced label into the stored annotation shape.

    Args:
        config: The session's ``TaggerConfig`` payload.
        raw: The model's label (string, or list of category names).

    Returns:
        ``"yes"``/``"no"`` for binary, a non-empty list of category ids for
        multiclass, a non-empty string for freetext — or ``None`` when the
        label cannot be mapped.
    """
    mode = config.get("mode")
    if mode == "binary":
        text = str(raw).strip().lower()
        if text in {"yes", "y", "true", "1", "כן"}:
            return "yes"
        if text in {"no", "n", "false", "0", "לא"}:
            return "no"
        return None
    if mode == "multiclass":
        names = raw if isinstance(raw, list) else [raw]
        by_label = {
            str(c.get("label", "")).strip().casefold(): str(c.get("id"))
            for c in config.get("categories") or []
        }
        by_id = {str(c.get("id")) for c in config.get("categories") or []}
        ids: list[str] = []
        for name in names:
            key = str(name).strip()
            mapped = by_label.get(key.casefold()) or (key if key in by_id else None)
            if mapped and mapped not in ids:
                ids.append(mapped)
        return ids or None
    text = str(raw).strip() if raw is not None else ""
    return text or None


def select_examples(
    config: dict[str, Any],
    data: list[dict[str, Any]],
    annotations: dict[str, Any],
    assist: dict[str, Any],
    exclude_ids: set[str] | None = None,
    cap: int = MAX_EXAMPLES,
) -> list[dict[str, Any]]:
    """Pick the few-shot examples the tagging prompt is compiled from.

    Corrections (rows where the human overrode the AI's prediction) carry the
    most signal, so they are kept first when capping to ``cap``.

    Args:
        config: The session's ``TaggerConfig`` payload.
        data: The full row payload.
        annotations: The ``{row_id: value}`` final-label map.
        assist: The session's assist state (predictions + provenance).
        exclude_ids: Row ids to leave out (e.g. the rows about to be predicted).
        cap: Maximum number of examples returned.

    Returns:
        A list of ``{text, label, corrected_from?}`` example dicts.
    """
    exclude = exclude_ids or set()
    provenance = assist.get("provenance") or {}
    predictions = assist.get("predictions") or {}
    rows_by_id = {str(row.get("id")): row for row in data}
    corrections: list[dict[str, Any]] = []
    plain: list[dict[str, Any]] = []
    for row_id, value in annotations.items():
        if row_id in exclude or row_id not in rows_by_id:
            continue
        if value is None or value == "" or value == []:
            continue
        if provenance.get(row_id) == "ai_auto":
            continue
        example: dict[str, Any] = {
            "text": _row_text(rows_by_id[row_id]),
            "label": _annotation_to_display(config, value),
        }
        predicted = (predictions.get(row_id) or {}).get("value")
        if predicted is not None and predicted != value:
            example["corrected_from"] = _annotation_to_display(config, predicted)
            corrections.append(example)
        else:
            plain.append(example)
    return (corrections + plain)[:cap]


def compile_instructions(
    config: dict[str, Any], rubric: list[str], examples: list[dict[str, Any]]
) -> str:
    """Compile the tagging instructions: task + rubric + labeled examples.

    Args:
        config: The session's ``TaggerConfig`` payload.
        rubric: The labeling rubric distilled from the interview/corrections.
        examples: ``{text, label, corrected_from?}`` few-shot examples.

    Returns:
        The full instruction block given to the tagging signatures.
    """
    parts = [task_description(config)]
    if rubric:
        parts.append("Labeling rubric (binding):\n" + "\n".join(f"- {r}" for r in rubric))
    if examples:
        parts.append(
            "Labeled examples (follow them exactly; entries with 'corrected_from' are "
            "rows where an earlier AI guess was wrong and the human fixed it — treat "
            "these as the strongest signal):\n"
            + json.dumps(examples, ensure_ascii=False)
        )
    return "\n\n".join(parts)


def interview_turn(
    config: dict[str, Any],
    columns: list[str],
    data: list[dict[str, Any]],
    turns: list[dict[str, str]],
    locale: str | None,
) -> dict[str, Any]:
    """Run one interview turn and return the assistant's reply.

    Args:
        config: The session's ``TaggerConfig`` payload.
        columns: All dataset column names.
        data: The full row payload (sampled for the summary).
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.

    Returns:
        ``{"message", "quick_replies", "rubric", "done"}`` — ``rubric`` is
        empty until ``done`` is true.
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    lm = _build_assist_lm()
    with dspy.context(lm=lm):
        pred = dspy.Predict(InterviewTurnSig)(
            task_description=task_description(config)
            + (
                f"\nQuestions asked so far: {asked} of at most {MAX_INTERVIEW_QUESTIONS}."
                " If the limit is reached you MUST finish now."
                if asked
                else ""
            ),
            dataset_summary=summarize_dataset(config, columns, data),
            transcript_json=json.dumps(turns, ensure_ascii=False),
            reply_language=_reply_language(locale),
        )
    done = str(getattr(pred, "done", "")).strip().lower() in {"true", "yes", "1"}
    rubric = _parse_json(getattr(pred, "rubric_json", "[]"), [])
    rubric = [str(r).strip() for r in rubric if str(r).strip()] if isinstance(rubric, list) else []
    if asked >= MAX_INTERVIEW_QUESTIONS and not done:
        done = True
    quick = _parse_json(getattr(pred, "quick_replies_json", "[]"), [])
    quick = [str(q).strip() for q in quick if str(q).strip()][:4] if isinstance(quick, list) else []
    return {
        "message": str(getattr(pred, "message", "")).strip(),
        "quick_replies": [] if done else quick,
        "rubric": rubric if done else [],
        "done": done,
    }


def refine_rubric(
    config: dict[str, Any],
    rubric: list[str],
    examples: list[dict[str, Any]],
    locale: str | None,
) -> list[str]:
    """Reflectively rewrite the rubric from labels and corrections.

    Args:
        config: The session's ``TaggerConfig`` payload.
        rubric: The current rubric rules.
        examples: Labeled examples including ``corrected_from`` entries.
        locale: UI locale code; the rubric is written in that language.

    Returns:
        The improved rubric, or the original when the model output is unusable.
    """
    lm = _build_assist_lm()
    with dspy.context(lm=lm):
        pred = dspy.Predict(RefineRubricSig)(
            task_description=task_description(config),
            current_rubric_json=json.dumps(rubric, ensure_ascii=False),
            examples_json=json.dumps(examples, ensure_ascii=False),
            reply_language=_reply_language(locale),
        )
    improved = _parse_json(getattr(pred, "rubric_json", "[]"), [])
    if not isinstance(improved, list):
        return rubric
    cleaned = [str(r).strip() for r in improved if str(r).strip()]
    return cleaned or rubric


def _predict_batch(
    lm: dspy.LM, instructions: str, batch: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Label one batch of rows, falling back to per-row calls on parse failure.

    Args:
        lm: The assist LM (bound inside this worker thread).
        instructions: Compiled tagging instructions.
        batch: ``{id, text}`` row payloads.
        config: The session's ``TaggerConfig`` payload.

    Returns:
        ``{row_id: {value, confidence, reason}}`` for every row the model
        produced a mappable label for.
    """
    rows_json = json.dumps(
        [{"id": str(r["id"]), "text": r["text"]} for r in batch], ensure_ascii=False
    )
    parsed: Any = None
    try:
        with dspy.context(lm=lm):
            pred = dspy.Predict(TagBatchSig)(task_instructions=instructions, rows_json=rows_json)
        parsed = _parse_json(getattr(pred, "labels_json", ""), None)
    except Exception:
        logger.warning("tagging batch call failed; falling back to per-row", exc_info=True)
    results: dict[str, dict[str, Any]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row_id = str(item.get("id", "")).strip()
            value = normalize_label(config, item.get("label"))
            if not row_id or value is None:
                continue
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            results[row_id] = {
                "value": value,
                "confidence": confidence,
                "reason": str(item.get("reason", "")).strip()[:200],
            }
    missing = [r for r in batch if str(r["id"]) not in results]
    for row in missing:
        try:
            with dspy.context(lm=lm):
                one = dspy.Predict(TagOneSig)(task_instructions=instructions, row_text=row["text"])
            payload = _parse_json(getattr(one, "label_json", ""), {})
        except Exception:
            logger.warning("tagging per-row call failed for row %s", row["id"], exc_info=True)
            continue
        if not isinstance(payload, dict):
            continue
        value = normalize_label(config, payload.get("label"))
        if value is None:
            continue
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        results[str(row["id"])] = {
            "value": value,
            "confidence": confidence,
            "reason": str(payload.get("reason", "")).strip()[:200],
        }
    return results


def predict_rows(
    config: dict[str, Any],
    instructions: str,
    rows: list[dict[str, Any]],
    on_batch: Callable[[dict[str, dict[str, Any]]], None] | None = None,
    cancel: threading.Event | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Label rows in concurrent batches and report the credit cost.

    Args:
        config: The session's ``TaggerConfig`` payload.
        instructions: Compiled tagging instructions.
        rows: Row payloads (each needs ``id`` and ``text``).
        on_batch: Called with each completed batch's predictions (bulk-job
            progress persistence); called from worker threads.
        cancel: Cooperative cancellation; pending batches are skipped once set.

    Returns:
        ``(predictions, credits)`` — the merged ``{row_id: prediction}`` map
        and the credit cost of the LM calls actually made.
    """
    lm = _build_assist_lm()
    prepared = [{"id": str(r.get("id")), "text": _row_text(r)} for r in rows]
    batches = [prepared[i : i + BATCH_SIZE] for i in range(0, len(prepared), BATCH_SIZE)]
    merged: dict[str, dict[str, Any]] = {}

    def work(batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Label one batch unless the job was cancelled first."""
        if cancel is not None and cancel.is_set():
            return {}
        result = _predict_batch(lm, instructions, batch, config)
        if on_batch is not None and result:
            on_batch(result)
        return result

    with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as pool:
        for result in pool.map(work, batches):
            merged.update(result)
    usage = usage_by_model_from_history(lm)
    credits = credits_for_usage(
        ModelUsage(model=model, input_tokens=tokens[0], output_tokens=tokens[1])
        for model, tokens in usage.items()
    )
    return merged, credits


def estimate_credits_for_rows(instructions: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate the credit cost of auto-tagging the given rows.

    A chars/4 token heuristic over the compiled instructions (repeated once
    per batch) plus the row texts, with a fixed per-row output allowance.

    Args:
        instructions: Compiled tagging instructions.
        rows: The rows that would be tagged.

    Returns:
        ``{"rows", "model", "credits_low", "credits_high"}``.
    """
    model = assist_model_name()
    if not rows:
        return {"rows": 0, "model": model, "credits_low": 0, "credits_high": 0}
    batch_count = max(1, (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)
    row_chars = sum(len(_row_text(r)) for r in rows)
    input_tokens = (len(instructions) * batch_count + row_chars) // CHARS_PER_TOKEN
    output_tokens = OUTPUT_TOKENS_PER_ROW * len(rows)
    base = credits_for_usage(
        [ModelUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)]
    )
    return {
        "rows": len(rows),
        "model": model,
        "credits_low": base,
        "credits_high": max(base, int(base * 1.8)),
    }


def deep_optimize_reflection_model_name() -> str:
    """Return the LiteLLM model id GEPA reflects with during deep optimize."""
    return settings.tagger_deep_optimize_reflection_model or assist_model_name()


def _label_as_text(config: dict[str, Any], value: Any) -> str:
    """Flatten a stored annotation into the single-string label GEPA trains on.

    Args:
        config: The session's ``TaggerConfig`` payload.
        value: The stored annotation value.

    Returns:
        Binary answers and extractions verbatim; multiclass as the category
        names joined with ``"; "``.
    """
    display = _annotation_to_display(config, value)
    if isinstance(display, list):
        return "; ".join(str(d) for d in display)
    return str(display)


def build_deep_optimize_dataset(
    config: dict[str, Any],
    data: list[dict[str, Any]],
    annotations: dict[str, Any],
    assist: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the ``{text, label}`` trainset for a deep-optimize run.

    Every human-vetted label (human and ai_confirmed provenance) qualifies,
    corrections first, capped at ``DEEP_OPTIMIZE_MAX_EXAMPLES``.

    Args:
        config: The session's ``TaggerConfig`` payload.
        data: The full row payload.
        annotations: The ``{row_id: value}`` final-label map.
        assist: The session's assist state (predictions + provenance).

    Returns:
        Dataset rows shaped for the run payload's ``text``/``label`` columns.
    """
    examples = select_examples(
        config, data, annotations, assist, cap=DEEP_OPTIMIZE_MAX_EXAMPLES
    )
    rows: list[dict[str, str]] = []
    for example in examples:
        label = example["label"]
        rows.append(
            {
                "text": str(example["text"]),
                "label": "; ".join(str(v) for v in label) if isinstance(label, list) else str(label),
            }
        )
    return rows


def _deep_optimize_signature_code(config: dict[str, Any], rubric: list[str]) -> str:
    """Generate the single-signature classifier source for a deep-optimize run.

    The task description and rubric become the signature docstring — the seed
    instructions GEPA evolves. User text is sanitized so it cannot terminate
    the docstring literal.

    Args:
        config: The session's ``TaggerConfig`` payload.
        rubric: The current labeling rubric.

    Returns:
        Python source defining exactly one ``dspy.Signature`` subclass.
    """
    parts = [task_description(config)]
    if rubric:
        parts.append("Labeling rubric:\n" + "\n".join(f"- {r}" for r in rubric))
    doc = "\n\n".join(parts).replace("\\", "\\\\").replace('"""', "'''")
    mode = config.get("mode")
    label_desc = (
        "Exactly 'yes' or 'no'."
        if mode == "binary"
        else "Every applying category name, joined with '; '."
        if mode == "multiclass"
        else "The extracted text."
    )
    return (
        "class TaggerLabelSignature(dspy.Signature):\n"
        f'    """{doc}"""\n\n'
        '    text: str = dspy.InputField(desc="The row to label.")\n'
        f'    label: str = dspy.OutputField(desc="{label_desc}")\n'
    )


_BINARY_METRIC = '''\
def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Exact match on the normalized yes/no answer, with GEPA feedback."""
    def norm(value):
        text = str(value or "").strip().lower()
        if text in ("yes", "y", "true", "1"):
            return "yes"
        if text in ("no", "n", "false", "0"):
            return "no"
        return text
    gold = norm(getattr(example, "label", ""))
    guess = norm(getattr(pred, "label", ""))
    if gold == guess:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    return dspy.Prediction(
        score=0.0,
        feedback=f"Expected '{gold}' but got '{guess}'. Answer exactly 'yes' or 'no'.",
    )
'''

_MULTICLASS_METRIC = '''\
def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Set equality over the '; '-separated category names, with GEPA feedback."""
    def parts(value):
        return {p.strip().casefold() for p in str(value or "").split(";") if p.strip()}
    gold = parts(getattr(example, "label", ""))
    guess = parts(getattr(pred, "label", ""))
    if gold == guess:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    missing = "; ".join(sorted(gold - guess)) or "none"
    extra = "; ".join(sorted(guess - gold)) or "none"
    return dspy.Prediction(
        score=0.0,
        feedback=f"Wrong categories. Missing: {missing}. Extra: {extra}. "
        "Answer with every applying category name, joined by '; '.",
    )
'''

_FREETEXT_METRIC = '''\
def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Token-overlap (Dice) similarity of the extraction, with GEPA feedback."""
    def tokens(value):
        out = set()
        for raw in str(value or "").lower().split():
            word = "".join(ch for ch in raw if ch.isalnum())
            if word:
                out.add(word)
        return out
    gold = tokens(getattr(example, "label", ""))
    guess = tokens(getattr(pred, "label", ""))
    if not gold and not guess:
        return dspy.Prediction(score=1.0, feedback="Correct (both empty).")
    overlap = len(gold & guess)
    denom = len(gold) + len(guess)
    score = (2.0 * overlap / denom) if denom else 0.0
    if score >= 0.85:
        return dspy.Prediction(score=1.0, feedback="Correct.")
    return dspy.Prediction(
        score=score,
        feedback=f"Extraction differs from the expected text: "
        f"'{getattr(example, 'label', '')}'. Extract only the requested text.",
    )
'''


def _deep_optimize_metric_code(config: dict[str, Any]) -> str:
    """Return the mode-appropriate metric source for a deep-optimize run.

    Args:
        config: The session's ``TaggerConfig`` payload.
    """
    mode = config.get("mode")
    if mode == "binary":
        return _BINARY_METRIC
    if mode == "multiclass":
        return _MULTICLASS_METRIC
    return _FREETEXT_METRIC


def build_deep_optimize_request(
    config: dict[str, Any],
    rubric: list[str],
    dataset: list[dict[str, str]],
    *,
    name: str,
    username: str,
) -> RunRequest:
    """Assemble the GEPA run payload for a tagger deep-optimize.

    A single-signature ``text → label`` classifier seeded with the rubric,
    trained on the session's human-vetted labels, on the cheapest GEPA budget
    (``auto="light"``). The run is private and managed-billed like any other.

    Args:
        config: The session's ``TaggerConfig`` payload.
        rubric: The current labeling rubric (becomes the seed instructions).
        dataset: ``{text, label}`` rows from :func:`build_deep_optimize_dataset`.
        name: Display name of the run (shows in the jobs dashboard).
        username: Owner of the run.

    Returns:
        A fully-populated ``RunRequest`` ready for the submission pipeline.
    """
    base_url = settings.tagger_assist_base_url or settings.generalist_agent_base_url or None
    return RunRequest(
        name=name,
        username=username,
        description="Tagger deep-optimize: evolve the labeling guide with GEPA.",
        module_name="predict",
        signature_code=_deep_optimize_signature_code(config, rubric),
        metric_code=_deep_optimize_metric_code(config),
        optimizer_name="gepa",
        optimizer_kwargs={"auto": "light"},
        dataset=dataset,
        column_mapping=ColumnMapping(inputs={"text": "text"}, outputs={"label": "label"}),
        shuffle=True,
        seed=42,
        dataset_filename="tagger-labels.json",
        model_settings=ModelConfig(name=assist_model_name(), base_url=base_url),
        reflection_model_settings=ModelConfig(
            name=deep_optimize_reflection_model_name(), base_url=base_url
        ),
        is_private=True,
    )
