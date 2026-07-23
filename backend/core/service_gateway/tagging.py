"""LLM engine for AI-assisted tagging (co-tagging). [INTERNAL]

Powers the tagger's assist modes: the dataset interview that distills a
labeling rubric, silent per-row predictions during calibration, batched
review/auto-tagging, and the pre-run credit estimate. Pure functions over the
session payload — persistence stays in the router; nothing here touches the
database.

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
from ..models import ModelConfig
from .agents.code import ReasoningStreamListener, _reply_language
from .agents.code_interview import INTERVIEW_TURN_ATTEMPTS, normalize_options
from .agents.constants import REASONING_FIELD
from .agents.parse_salvage import salvage_prediction, strip_adapter_debris
from .language_models import (
    apply_model_reasoning_config,
    apply_reasoning_effort,
    build_language_model,
    served_model_from,
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


def assist_model_name() -> str:
    """Return the LiteLLM model id the tagging assist runs on."""
    return settings.tagger_assist_model or settings.generalist_agent_model


# Union of the per-provider effort vocabularies the composer offers ("none"
# and "xhigh" are OpenAI's floor/ceiling-adjacent tiers, "max" tops out
# Anthropic and GPT-5.6 Sol; "ultra" is a separate mode, not an effort).
_REASONING_EFFORT_LEVELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


def _sanitize_model_params(params: Any) -> dict[str, Any]:
    """Reduce stored model params to the sampling knobs tagging honors.

    ``assist`` is a free-form JSON column any API caller can write, so only
    temperature, max_tokens, top_p and the reasoning-effort extra pass
    through — coerced and clamped to the ``ModelConfig`` bounds — and
    connection fields (endpoints, keys, arbitrary LiteLLM kwargs) never
    reach the tagging LM.

    Args:
        params: The ``modelParams`` mapping stored on the assist state.

    Returns:
        Keyword arguments safe to construct a :class:`ModelConfig` with.
    """
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for key, low, high in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0)):
        value = params.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = min(high, max(low, float(value)))
    tokens = params.get("max_tokens")
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and int(tokens) >= 1:
        out["max_tokens"] = int(tokens)
    extra = params.get("extra")
    effort = extra.get("reasoning_effort") if isinstance(extra, dict) else None
    if isinstance(effort, str) and effort.lower() in _REASONING_EFFORT_LEVELS:
        out["extra"] = {"reasoning_effort": effort.lower()}
    return out


def _build_assist_lm(
    model_name: str | None = None,
    params: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    lm_extra_body: dict[str, Any] | None = None,
) -> dspy.LM:
    """Build the assist LM from settings, mirroring the generalist agent.

    Args:
        model_name: LiteLLM model id to run on; falls back to the configured
            tagging-assist model (then the generalist agent's) when empty.
        params: Sampling parameters saved alongside the chosen model
            (``assist.modelParams``); sanitized before use.
        reasoning_effort: Explicit ``reasoning_effort`` level chosen in the
            composer's model menu; ``None`` keeps the model's default.
        lm_extra_body: Extra request-body fields merged into the provider
            call (the auto router's plugin dial rides here).

    Returns:
        A cache-disabled ``dspy.LM`` on the requested model.
    """
    kwargs = _sanitize_model_params(params)
    if lm_extra_body:
        extra = dict(kwargs.get("extra") or {})
        extra["extra_body"] = {**dict(extra.get("extra_body") or {}), **lm_extra_body}
        kwargs["extra"] = extra
    config = ModelConfig(
        name=model_name or assist_model_name(),
        base_url=settings.tagger_assist_base_url or settings.generalist_agent_base_url or None,
        **kwargs,
    )
    config = apply_reasoning_effort(config, reasoning_effort)
    return build_language_model(apply_model_reasoning_config(config), disable_cache=True)


class InterviewTurnSig(dspy.Signature):
    """Interview the dataset owner to distill a labeling rubric.

    You are a labeling copilot preparing to tag the user's dataset for them.
    The user chose which columns each row's text is built from; the dataset
    summary lists the rest as excluded. Excluded columns are invisible to both
    the human tagger and the tagging model, so the task, every direction you
    offer, and every rubric rule must be decidable from the sample-row text
    alone — never propose a task that needs an excluded column.
    Ask ONE short, concrete question at a time — grounded in the sample rows —
    about ambiguous label boundaries, edge cases, and how to treat dirty or
    off-topic rows. Never ask generic questions the task description already
    answers. When the task description tells you to ask for a missing task
    definition, your first question must pin it down. After at most five
    questions total (or as
    soon as the user asks to proceed, or their answers stop adding information),
    stop asking: set
    ``done`` to true, write a one-sentence wrap-up in ``message``, and emit
    the full rubric and ``task_config_json`` — 4 to 10 crisp, decision-ready rules that would let a
    stranger label exactly like the user. Rules state decisions ("X counts as
    Y when ..."), not process. Whenever the question has a small set of likely
    answers, offer 2-4 of them in ``options_json`` — each a short pickable
    answer with a one-line description of what choosing it means — so the user
    can answer in one click. Every option must be a concrete, self-contained
    answer. The composer under the options is always the free-text path, so
    never spend an option on an escape hatch — no "other", "something else",
    "none of these", "I use my own ...", or any rewording whose real meaning
    is "I'll type it below"; when only escape hatches would fill the list,
    offer fewer options or ask an open question instead. Write ``message``,
    the options and the rubric in ``reply_language``.
    """

    task_description: str = dspy.InputField(desc="What is being labeled and the allowed labels.")
    dataset_summary: str = dspy.InputField(
        desc="Row count, input columns, excluded (invisible) columns, sample values."
    )
    transcript_json: str = dspy.InputField(desc="JSON array of prior {role, content} turns.")
    reply_language: str = dspy.InputField(desc="Language every output is written in.")
    message: str = dspy.OutputField(
        desc=(
            "The next question, or a short wrap-up when done. Plain conversational "
            "prose — never JSON, braces, or bracketed lists."
        )
    )
    # ``done`` sits right after ``message`` so it streams before the (slow)
    # options/rubric fields — the client uses it to pick the correct
    # still-generating placeholder (answer choices vs. the task contract).
    done: str = dspy.OutputField(desc="'true' when the interview is finished, else 'false'.")
    options_json: str = dspy.OutputField(
        desc=(
            "JSON array of 0-4 answer options for a closed question, each "
            '{"label": <short pickable answer, <= 6 words>, "description": '
            "<one-line note on what picking it means>}; [] for an open question."
        )
    )
    rubric_json: str = dspy.OutputField(desc="JSON array of rubric rule strings; [] until done.")
    task_config_json: str = dspy.OutputField(
        desc=(
            "JSON object containing the task definition once done; {} until done. "
            'For binary use {"question": "..."}; for multiclass use '
            '{"categories": ["..."]}; for freetext use {"prompt": "..."}. When the '
            "task description says the answer style is yours to decide, also include "
            '"mode": "binary" | "multiclass" | "freetext" next to its definition.'
        )
    )
    session_title: str = dspy.OutputField(
        desc=(
            "Once done, a short session name (2-5 words) describing the labeling "
            "task, e.g. 'Routing support tickets'; empty until done. Plain words "
            "in the reply language — no quotes, no trailing punctuation."
        )
    )


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
    label_json: str = dspy.OutputField(desc='JSON object: {"label": <label>, "confidence": <0..1>, "reason": "..."}.')


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


def effective_task_config(config: dict[str, Any], assist: dict[str, Any]) -> dict[str, Any]:
    """Merge the interview's task override over the immutable stored config.

    The ``config`` column never changes after creation; the interview's
    refinements — question, categories, prompt and, on provisional-mode
    sessions, the inferred answer style — live in ``assist.taskOverride``.
    The user's chosen tagging model (``assist.model``) and its saved
    sampling parameters (``assist.modelParams``) ride along the same way, so
    predictions, estimates and the bulk worker all see one merged view.
    Every LLM surface (router routes and the bulk worker alike) reads
    through this merge.

    Args:
        config: The stored ``TaggerConfig`` payload.
        assist: The session's assist state (carries ``taskOverride`` and the
            chosen tagging model).

    Returns:
        A copy of ``config`` with the override applied; ``modeProvisional``
        is dropped once an inferred mode wins.
    """
    merged = dict(config)
    override = (assist or {}).get("taskOverride") or {}
    mode = str(override.get("mode") or "").strip()
    if mode in {"binary", "multiclass", "freetext"}:
        merged["mode"] = mode
        merged.pop("modeProvisional", None)
    for key in ("question", "prompt"):
        value = str(override.get(key) or "").strip()
        if value:
            merged[key] = value
    categories = override.get("categories")
    if isinstance(categories, list) and categories:
        merged["categories"] = categories
    model = str((assist or {}).get("model") or "").strip()
    if model:
        merged["model"] = model
        params = (assist or {}).get("modelParams")
        if isinstance(params, dict) and params:
            merged["modelParams"] = params
    return merged


def task_description(config: dict[str, Any]) -> str:
    """Describe the labeling task and its allowed labels for the LM.

    Args:
        config: The session's ``TaggerConfig`` payload.

    Returns:
        A compact English framing of the task; user-authored parts (question,
        category names, prompt) are passed through verbatim in their language.
    """
    # A provisional-mode session (assisted setup, no interface picked) leaves
    # the answer style itself to the interview. Autopilot autonomy covers the
    # tagging phase only — the task itself is always defined with the user.
    if config.get("modeProvisional"):
        return (
            "The labeling task is not yet defined; defining it with the user is your "
            "first job. Open by briefly describing what the sample rows look like, "
            "then ask what the user wants to learn or decide about each row and what "
            "the labels will be used for (for example training an optimized prompt, "
            "filtering, or analysis). Offer 2-4 concrete task directions you infer "
            "from the data as options. You decide the answer style — binary (one "
            "yes/no question per row), multiclass (a fixed set of categories), or "
            "freetext (text extracted or written per row) — from the user's goal; "
            "ask about it only when the goal genuinely fits more than one style. "
            'Return the chosen style in the task config as "mode" together with its '
            "matching definition."
        )
    mode = config.get("mode")
    if mode == "binary":
        question = str(config.get("question") or "").strip()
        if question:
            return (
                f'Binary labeling. For each row answer the question: "{question}". The label is exactly "yes" or "no".'
            )
        if config.get("_assist_mode"):
            return (
                "Binary labeling, but the yes/no classification criterion has not been defined. "
                "Your first question must ask the user what each row should be classified for. "
                'The final label is exactly "yes" or "no".'
            )
        return (
            "Binary labeling. Apply the rubric's classification criterion to each row. "
            'The label is exactly "yes" or "no".'
        )
    if mode == "multiclass":
        names = [str(c.get("label", "")).strip() for c in config.get("categories") or []]
        names = [n for n in names if n]
        if not names and config.get("_assist_mode"):
            return (
                "Multi-label classification, but the allowed categories have not been defined. "
                "Your first question must ask the user which categories can apply to each row."
            )
        return (
            "Multi-label classification. Assign each row every category that applies "
            f"from this exact list: {json.dumps(names, ensure_ascii=False)}. "
            "The label is a JSON array of the applying category names (at least one)."
        )
    prompt = str(config.get("prompt") or "").strip()
    if not prompt:
        if config.get("_assist_mode"):
            return (
                "Open-ended text extraction, but the extraction target has not been defined. "
                "Your first question must ask the user exactly what to extract from each row."
            )
        return (
            "Open-ended text extraction: the exact text to pull from each row is "
            "the one described by the rubric rules. The label is that extracted "
            "text, grounded in the row."
        )
    return (
        f'Text extraction. For each row: "{prompt}". '
        "The label is the extracted text, taken from or grounded in the row."
    )


def summarize_dataset(config: dict[str, Any], columns: list[str], data: list[dict[str, Any]]) -> str:
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
    # Columns the user left unselected are invisible at labeling time (the row
    # text is built from the input columns only), so they are surfaced as
    # explicitly excluded rather than as available material for the task.
    excluded = [c for c in columns if c not in input_cols]
    return json.dumps(
        {
            "row_count": len(data),
            "input_columns": input_cols,
            "excluded_columns": excluded,
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
        # The Hebrew yes/no ("\u05db\u05df" / "\u05dc\u05d0") are input-normalization
        # tokens, escaped so the i18n catalog-boundary check stays clean.
        if text in {"yes", "y", "true", "1", "\u05db\u05df"}:
            return "yes"
        if text in {"no", "n", "false", "0", "\u05dc\u05d0"}:
            return "no"
        return None
    if mode == "multiclass":
        names = raw if isinstance(raw, list) else [raw]
        by_label = {
            str(c.get("label", "")).strip().casefold(): str(c.get("id")) for c in config.get("categories") or []
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


def compile_instructions(config: dict[str, Any], rubric: list[str], examples: list[dict[str, Any]]) -> str:
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
            "these as the strongest signal):\n" + json.dumps(examples, ensure_ascii=False)
        )
    return "\n\n".join(parts)


def _interview_inputs(
    config: dict[str, Any],
    columns: list[str],
    data: list[dict[str, Any]],
    turns: list[dict[str, str]],
    locale: str | None,
) -> dict[str, str]:
    """Assemble the ``InterviewTurnSig`` inputs for one turn.

    Args:
        config: The session's ``TaggerConfig`` payload.
        columns: All dataset column names.
        data: The full row payload (sampled for the summary).
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    return {
        "task_description": task_description(config)
        + (
            f"\nQuestions asked so far: {asked} of at most {MAX_INTERVIEW_QUESTIONS}."
            " If the limit is reached you MUST finish now."
            if asked
            else ""
        ),
        "dataset_summary": summarize_dataset(config, columns, data),
        "transcript_json": json.dumps(turns, ensure_ascii=False),
        "reply_language": _reply_language(locale),
    }


def _normalize_task_artifact(mode: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the mode's task artifact (question / categories / prompt).

    Args:
        mode: The annotation mode the artifact belongs to.
        raw: Parsed ``task_config_json`` output.

    Returns:
        The normalized artifact mapping, or an empty mapping when invalid.
    """
    if mode == "binary":
        question = str(raw.get("question") or "").strip()
        return {"question": question} if question else {}
    if mode == "multiclass":
        values = raw.get("categories")
        if not isinstance(values, list):
            return {}
        labels: list[str] = []
        seen: set[str] = set()
        for value in values:
            label = str(value.get("label") if isinstance(value, dict) else value).strip()
            key = label.casefold()
            if label and key not in seen:
                labels.append(label)
                seen.add(key)
        if len(labels) < 2:
            return {}
        return {"categories": [{"id": f"cat{index}", "label": label} for index, label in enumerate(labels, start=1)]}
    prompt = str(raw.get("prompt") or "").strip()
    return {"prompt": prompt} if prompt else {}


def _normalize_task_override(config: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Normalize a model-produced task definition for client and server use.

    Provisional-mode sessions carry the inferred answer style in the override
    as ``mode``; a missing ``mode`` is tolerated by inferring it from whichever
    artifact the model produced.

    Args:
        config: The session's base tagger configuration.
        raw: Parsed ``task_config_json`` output.

    Returns:
        A mode-appropriate task override, or an empty mapping when invalid.
    """
    if not isinstance(raw, dict):
        return {}
    if config.get("modeProvisional"):
        mode = str(raw.get("mode") or "").strip().lower()
        if mode not in {"binary", "multiclass", "freetext"}:
            if str(raw.get("question") or "").strip():
                mode = "binary"
            elif isinstance(raw.get("categories"), list):
                mode = "multiclass"
            elif str(raw.get("prompt") or "").strip():
                mode = "freetext"
            else:
                return {}
        artifact = _normalize_task_artifact(mode, raw)
        # Freetext works without a prompt (the rubric carries the task); the
        # other styles are unusable without their artifact.
        if mode != "freetext" and not artifact:
            return {}
        return {"mode": mode, **artifact}
    return _normalize_task_artifact(str(config.get("mode") or ""), raw)


def _parse_interview_prediction(pred: Any, asked: int, config: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw ``InterviewTurnSig`` prediction into the client turn payload.

    Args:
        pred: The prediction (or ``None`` when the stream produced nothing).
        asked: Assistant questions asked before this turn (limit enforcement).
        config: The session's tagger configuration.

    Returns:
        The client turn payload. ``options`` is a list of pickable answers;
        ``rubric`` and ``task_override`` stay empty until ``done`` is true.
    """
    done = str(getattr(pred, "done", "")).strip().lower() in {"true", "yes", "1"}
    rubric = _parse_json(getattr(pred, "rubric_json", "[]"), [])
    rubric = [str(r).strip() for r in rubric if str(r).strip()] if isinstance(rubric, list) else []
    if asked >= MAX_INTERVIEW_QUESTIONS and not done:
        done = True
    options = normalize_options(_parse_json(getattr(pred, "options_json", "[]"), []))
    task_override = _normalize_task_override(
        config,
        _parse_json(getattr(pred, "task_config_json", "{}"), {}),
    )
    # session_title is the signature's last output field, so a malformed
    # terminal adapter marker leaks into its tail — strip it before the name
    # reaches the session card. The DB name column caps at 200; 80 keeps
    # session cards to one line.
    title = strip_adapter_debris(str(getattr(pred, "session_title", ""))).strip().strip("\"'")[:80]
    return {
        "message": str(getattr(pred, "message", "")).strip(),
        "options": [] if done else options,
        "rubric": rubric if done else [],
        "task_override": task_override if done else {},
        "title": title if done else "",
        "done": done,
        "model": assist_model_name(),
    }


# A reply that opens like JSON, a code fence or an adapter marker is leaked
# structure, not prose; these markers are the transition points where a stream
# that began as prose drifts into the payload's remaining fields.
_LEAK_PREFIXES = ("{", "[", "`")
_LEAK_MARKERS = ("[[ ##", '"options_json"', '"rubric_json"', '"task_config_json"', '"session_title"')


class _MessageLeakGuard:
    """Keep raw structured output out of the streamed reply channel.

    dspy's field listener leaks the whole payload as ``message`` deltas when
    the model answers in the other adapter's format (the same minimax-class
    drift ``salvage_prediction`` covers). The parsed turn always arrives via
    ``interview_done``, so leaked deltas are dropped rather than repaired: a
    stream that opens like structured output is muted entirely, and one that
    drifts into a marker mid-way is reset client-side and muted from there on.
    """

    def __init__(self) -> None:
        self._seen = ""
        self._sent = False
        self._muted = False

    def feed(self, chunk: str) -> tuple[str, bool]:
        """Classify one reply delta.

        Args:
            chunk: The raw ``message`` delta from the stream listener.

        Returns:
            ``(text, reset)`` — ``text`` is what may be forwarded (empty while
            muted or still all-whitespace); ``reset`` asks the client to drop
            partial reply text it already rendered.
        """
        if self._muted:
            return "", False
        self._seen += chunk
        head = self._seen.lstrip()
        if not head:
            return "", False
        if head.startswith(_LEAK_PREFIXES) or any(marker in self._seen for marker in _LEAK_MARKERS):
            self._muted = True
            reset, self._sent = self._sent, False
            return "", reset
        # First forward flushes whatever whitespace was buffered ahead of it.
        text = chunk if self._sent else self._seen
        self._sent = True
        return text, False


def interview_turn(
    config: dict[str, Any],
    columns: list[str],
    data: list[dict[str, Any]],
    turns: list[dict[str, str]],
    locale: str | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    lm_extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one interview turn and return the assistant's reply (non-streaming).

    Args:
        config: The session's ``TaggerConfig`` payload.
        columns: All dataset column names.
        data: The full row payload (sampled for the summary).
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.
        model: LiteLLM id conducting the interview; ``None`` runs the default.
        reasoning_effort: Explicit effort level for ``model``; ``None`` keeps
            the model's default.
        lm_extra_body: Extra request-body fields for the LM call (the auto
            router's plugin dial when the composer picked an Auto tier).

    Returns:
        ``{"message", "options", "rubric", "done"}`` — ``rubric`` is empty
        until ``done`` is true.
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    lm = _build_assist_lm(model, reasoning_effort=reasoning_effort, lm_extra_body=lm_extra_body)
    with dspy.context(lm=lm):
        pred = dspy.Predict(InterviewTurnSig)(**_interview_inputs(config, columns, data, turns, locale))
    turn = _parse_interview_prediction(pred, asked, config)
    if model:
        turn["model"] = model
    return turn


async def interview_turn_stream(
    config: dict[str, Any],
    columns: list[str],
    data: list[dict[str, Any]],
    turns: list[dict[str, str]],
    locale: str | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    lm_extra_body: dict[str, Any] | None = None,
) -> Any:
    """Run one interview turn, streaming it the way the generalist agent does.

    Yields ``{"event", "data"}`` mappings ready for ``sse_from_events``:
    ``reasoning_patch`` for provider thinking tokens (same synthetic channel
    the agents use), ``message_patch`` for reply deltas, ``message_end`` once
    the reply is fully streamed (the remaining structured fields — options,
    rubric, task — are still generating), ``turn_hint`` with ``{"final"}`` as
    soon as the streamed ``done`` field settles (the client picks the matching
    still-generating placeholder), ``message_reset`` when a failed attempt is
    being retried or leaked structure was dropped (the client must drop
    partial text and the hint), and a terminal ``interview_done`` carrying the
    parsed turn. Reply deltas pass through :class:`_MessageLeakGuard` so raw
    payload text never reaches the visible reply.

    A turn whose terminal parse fails is first salvaged from the raw response
    (minimax-class models answer in chat-adapter format even under the JSON
    fallback — see ``agents.parse_salvage``) and only then retried from
    scratch, mirroring the code interview. A finished turn that carries no
    rubric rules is retried the same way — a "done" turn without a rubric
    would hand the user an empty contract card.

    Args:
        config: The session's ``TaggerConfig`` payload.
        columns: All dataset column names.
        data: The full row payload (sampled for the summary).
        turns: Prior ``{role, content}`` turns, oldest first.
        locale: UI locale code; replies are written in that language.
        model: LiteLLM id conducting the interview; ``None`` runs the default.
        reasoning_effort: Explicit effort level for ``model``; ``None`` keeps
            the model's default.
        lm_extra_body: Extra request-body fields for the LM call (the auto
            router's plugin dial when the composer picked an Auto tier).
    """
    asked = sum(1 for t in turns if t.get("role") == "assistant")
    predict = dspy.Predict(InterviewTurnSig)
    lm = _build_assist_lm(model, reasoning_effort=reasoning_effort, lm_extra_body=lm_extra_body)
    inputs = _interview_inputs(config, columns, data, turns, locale)
    turn: dict[str, Any] = {}
    for attempt in range(INTERVIEW_TURN_ATTEMPTS):
        # Stream listeners are single-use; rebuild the program per attempt.
        program = dspy.streamify(
            predict,
            stream_listeners=[
                dspy.streaming.StreamListener(signature_field_name="message"),
                dspy.streaming.StreamListener(signature_field_name="done"),
                ReasoningStreamListener(predict=predict),
            ],
            async_streaming=True,
        )
        if attempt:
            yield {"event": "message_reset", "data": {}}
        guard = _MessageLeakGuard()
        prediction: Any = None
        done_text = ""
        try:
            with dspy.context(lm=lm):
                async for chunk in program(**inputs):
                    if isinstance(chunk, dspy.streaming.StreamResponse):
                        if chunk.signature_field_name == REASONING_FIELD:
                            yield {"event": "reasoning_patch", "data": {"chunk": chunk.chunk}}
                        elif chunk.signature_field_name == "message":
                            text, reset = guard.feed(chunk.chunk)
                            if reset:
                                yield {"event": "message_reset", "data": {}}
                            if text:
                                yield {"event": "message_patch", "data": {"chunk": text}}
                            if chunk.is_last_chunk:
                                yield {"event": "message_end", "data": {}}
                        elif chunk.signature_field_name == "done":
                            done_text += chunk.chunk
                            if chunk.is_last_chunk:
                                yield {
                                    "event": "turn_hint",
                                    "data": {"final": "true" in done_text.lower()},
                                }
                    elif isinstance(chunk, dspy.Prediction):
                        prediction = chunk
        except Exception as err:
            prediction = salvage_prediction(err)
            if prediction is None:
                if attempt + 1 >= INTERVIEW_TURN_ATTEMPTS:
                    raise
                logger.warning("tagger interview turn failed; retrying", exc_info=True)
                continue
        turn = _parse_interview_prediction(prediction, asked, config)
        if turn["done"] and not turn["rubric"] and attempt + 1 < INTERVIEW_TURN_ATTEMPTS:
            logger.warning("tagger interview finished without a rubric; retrying")
            continue
        break
    if model and turn:
        turn["model"] = model
    turn["served_model"] = served_model_from(lm)
    yield {"event": "interview_done", "data": turn}


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
    rows_json = json.dumps([{"id": str(r["id"]), "text": r["text"]} for r in batch], ensure_ascii=False)
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
        config: The session's effective config; when it carries the user's
            chosen tagging model (``model``) and its saved sampling
            parameters (``modelParams``), predictions run on them.
        instructions: Compiled tagging instructions.
        rows: Row payloads (each needs ``id`` and ``text``).
        on_batch: Called with each completed batch's predictions (bulk-job
            progress persistence); called from worker threads.
        cancel: Cooperative cancellation; pending batches are skipped once set.

    Returns:
        ``(predictions, credits)`` — the merged ``{row_id: prediction}`` map
        and the credit cost of the LM calls actually made.
    """
    lm = _build_assist_lm(str(config.get("model") or "").strip() or None, config.get("modelParams"))
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
        ModelUsage(model=model, input_tokens=tokens[0], output_tokens=tokens[1]) for model, tokens in usage.items()
    )
    return merged, credits


def estimate_credits_for_rows(
    instructions: str, rows: list[dict[str, Any]], model: str | None = None
) -> dict[str, Any]:
    """Estimate the credit cost of auto-tagging the given rows.

    A chars/4 token heuristic over the compiled instructions (repeated once
    per batch) plus the row texts, with a fixed per-row output allowance.

    Args:
        instructions: Compiled tagging instructions.
        rows: The rows that would be tagged.
        model: LiteLLM id of the session's chosen tagging model; falls back
            to the configured default when empty.

    Returns:
        ``{"rows", "model", "credits_low", "credits_high"}``.
    """
    model = (model or "").strip() or assist_model_name()
    if not rows:
        return {"rows": 0, "model": model, "credits_low": 0, "credits_high": 0}
    batch_count = max(1, (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)
    row_chars = sum(len(_row_text(r)) for r in rows)
    input_tokens = (len(instructions) * batch_count + row_chars) // CHARS_PER_TOKEN
    output_tokens = OUTPUT_TOKENS_PER_ROW * len(rows)
    base = credits_for_usage([ModelUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)])
    return {
        "rows": len(rows),
        "model": model,
        "credits_low": base,
        "credits_high": max(base, int(base * 1.8)),
    }
