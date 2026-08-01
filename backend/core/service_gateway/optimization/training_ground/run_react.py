"""Live react optimization for the generic ``POST /run`` path.

ReAct is a generic module like ``predict``/``cot`` that additionally carries a
tool roster. This module builds a live :class:`RetryingReActV2` over a resolved
roster and optimizes its instructions *and* tool descriptions with GEPA's
tool-aware DSPy adapter
(``gepa.adapters.dspy_adapter.DspyAdapter`` with ``enable_tool_optimization``),
scoring every candidate with the same standard
``(gold, pred, trace, pred_name, pred_trace)`` metric the predict/cot path uses.

There is no replay: rollouts execute the live MCP tools, so the optimized
program behaves at eval time exactly as it will when served. Two building
blocks live here:

- :func:`resolve_react_tools` materialises the tool roster (live MCP listing or
  a persisted snapshot) plus its schema-hash map.
- :func:`run_react_optimization` wires a seed ``RetryingReActV2`` through
  ``gepa.optimize`` and returns the servable program state, the
  baseline/optimized scalars, and the tool-description overlay for the bundle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import dspy
import gepa
from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from core.config import Settings
from core.constants import (
    DETAIL_BASELINE,
    DETAIL_OPTIMIZED,
    PROGRESS_BASELINE,
    PROGRESS_OPTIMIZED,
)

from ..logged_scores import reset_logged_metrics
from ..optimizers import build_target_score_stopper
from ..retrying_react import RetryingReActV2
from ..timing import (
    STAGE_BASELINE,
    STAGE_EVALUATION,
    STAGE_TRAINING,
    track_stage,
)
from ..tool_overlay import hash_tool_schema
from .gepa_adapter import (
    _candidate_tool_arg_descriptions,
    _candidate_tool_descriptions,
    _candidate_tool_names,
    seed_candidate_from_program,
)

logger = logging.getLogger(__name__)

TERMINAL_TOOL_NAMES = frozenset({"submit", "finish"})
"""The synthetic loop-exit tool the program appends to every roster — ``submit``
on ReActV2, ``finish`` on classic ReAct. Excluded from the optimizable tool
surface (snapshot rosters, overlay) because it carries no user-tunable
description — it is DSPy's fixed loop-exit action."""

_AUTO_BUDGETS: dict[str, int] = {
    "light": 500,
    "medium": 2000,
    "heavy": 8000,
}
"""Translation table for an ``--auto`` tier into ``max_metric_calls``.

GEPA's public API does not accept an ``auto`` enum; we resolve it here so the
budget stays ergonomic. Numbers are operator-tunable via an explicit
``max_metric_calls`` for any campaign that doesn't fit the table."""

_TOOL_SEVERITY_ATTR = "_skynet_severity"


def _severity_from_annotations(annotations: Any) -> str | None:
    """Map MCP tool annotations to an approval severity, or ``None`` if unstated.

    Mirrors the frontend ``ApprovalSeverity`` tiers from the MCP annotation
    hints: a read-only tool is ``info``, an explicitly destructive one is
    ``destructive``, and a declared-but-mutating tool is ``warning``. When the
    server states no relevant hint we return ``None`` so the surface never
    fabricates a severity it was not told.

    Args:
        annotations: The MCP ``ToolAnnotations`` object (or ``None``).

    Returns:
        ``"info"``/``"warning"``/``"destructive"``, or ``None`` when no hint
        applies.
    """
    if annotations is None:
        return None
    if getattr(annotations, "readOnlyHint", None) is True:
        return "info"
    if getattr(annotations, "destructiveHint", None) is True:
        return "destructive"
    if getattr(annotations, "readOnlyHint", None) is False:
        return "warning"
    return None


def set_tool_severity(tool: dspy.Tool, severity: str | None) -> None:
    """Stash a derived approval severity on a wrapped tool (no-op when ``None``).

    Args:
        tool: The ``dspy.Tool`` to annotate.
        severity: The severity tier, or ``None`` to leave the tool unmarked.
    """
    if severity is not None:
        setattr(tool, _TOOL_SEVERITY_ATTR, severity)


def tool_severity(tool: dspy.Tool) -> str | None:
    """Read a tool's stashed approval severity, or ``None`` when unmarked.

    Args:
        tool: The ``dspy.Tool`` to read.

    Returns:
        The severity tier set by :func:`set_tool_severity`, or ``None``.
    """
    return getattr(tool, _TOOL_SEVERITY_ATTR, None)


async def _list_live_tools(mcp_url: str, auth_header: str | None) -> list[dspy.Tool]:
    """Open one MCP session and return the live ``dspy.Tool`` roster.

    Each tool is wrapped via ``dspy.Tool.from_mcp_tool`` so the react rollouts
    can invoke it, and carries its derived approval severity read from
    ``.annotations`` (see :func:`set_tool_severity`).

    Args:
        mcp_url: MCP server URL.
        auth_header: Optional ``Authorization`` header to forward.

    Returns:
        List of dspy.Tool objects, one per MCP-exposed tool, each carrying its
        derived approval severity.
    """
    headers = {"Authorization": auth_header} if auth_header else None
    async with (
        streamablehttp_client(mcp_url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listing = await session.list_tools()
        tools: list[dspy.Tool] = []
        for tool in listing.tools:
            wrapped = dspy.Tool.from_mcp_tool(session, tool)
            set_tool_severity(
                wrapped, _severity_from_annotations(getattr(tool, "annotations", None))
            )
            tools.append(wrapped)
        return tools


def extract_program_state(program: dspy.Module) -> dict[str, Any]:
    """Return the JSON-shaped state dict from ``program.save``.

    Mirrors the path used by
    ``core/service_gateway/optimization/artifacts.persist_program``
    so the persisted ``program_state`` round-trips with
    ``fresh_program.load_state(state)`` at runtime.

    Args:
        program: The DSPy module whose state is serialised.

    Returns:
        The JSON-decoded ``program_state`` dict written by ``program.save``.
    """
    with tempfile.TemporaryDirectory(prefix="tg_bundle_") as tmpdir:
        state_path = Path(tmpdir) / "program.json"
        program.save(str(state_path), save_program=False)
        return json.loads(state_path.read_text())


class _JobLogGepaLogger:
    """Forward GEPA's free-text progress lines into the job log sink.

    GEPA reports per-iteration status (e.g. ``"Iteration N: Valset score for new
    program: X"``) through a ``gepa.logging.LoggerProtocol`` object. Its default
    ``StdOutLogger`` prints to stdout, which the worker's ``JobLogHandler`` never
    captures, so the score chart (``extractScoresFromLogs``) stays empty for
    react runs. Forwarding to a module logger lets those records propagate up to
    the ``core.service_gateway.optimization`` handler instead.
    """

    def __init__(self, sink: logging.Logger):
        """Store the logger the GEPA lines are forwarded to.

        Args:
            sink: Logger whose records reach the job's ``JobLogHandler``.
        """
        self._sink = sink

    def log(self, message: str) -> None:
        """Forward one GEPA progress line to the job log sink.

        Args:
            message: The free-text progress line emitted by GEPA's engine.
        """
        self._sink.info(message)


DEFAULT_MAX_ITERS = 8
"""ReActV2 loop budget for the react /run seed program. Mirrors the generalist
CLI default so a /run-produced bundle behaves like a CLI-produced one."""


def _as_sequence(value: Any) -> list[Any]:
    """Coerce a cell value into a list, tolerating JSON-string cells.

    Args:
        value: Raw cell value (sequence, JSON string, or other).

    Returns:
        A list — empty when the value is missing or not array-shaped.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _dataset_snapshot_tools(
    dataset: Sequence[Mapping[str, Any]] | None,
) -> list[dspy.Tool]:
    """Rebuild a ``dspy.Tool`` roster from a dataset-carried snapshot sidecar.

    A persisted snapshot tool source carries the roster on a reserved
    ``__tool_snapshot__`` key (the first row that has it wins), a list of specs::

        {"name": str, "description": str, "args": {<arg>: {<schema>}},
         "severity": str | None}

    Each spec becomes a ``dspy.Tool`` whose ``name``/``desc``/``args``/severity
    are read for the seed instructions, schema-hash snapshot, and overlay. The
    serve path uses this to reconstruct the surface a run was optimized against.

    Args:
        dataset: Raw dataset rows; scanned for the snapshot sidecar.

    Returns:
        One ``dspy.Tool`` per snapshot spec (``submit`` excluded), in spec
        order. Empty when no sidecar is present.
    """
    specs = _find_snapshot_specs(dataset)
    tools: list[dspy.Tool] = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        name = str(spec.get("name") or "").strip()
        if not name or name in TERMINAL_TOOL_NAMES:
            continue
        desc = str(spec.get("description") or spec.get("desc") or "")
        args = spec.get("args") if isinstance(spec.get("args"), Mapping) else {}
        severity = spec.get("severity")
        tools.append(
            _snapshot_tool(
                name=name,
                desc=desc,
                args=dict(args),
                severity=str(severity) if severity else None,
            )
        )
    return tools


def _find_snapshot_specs(
    dataset: Sequence[Mapping[str, Any]] | None,
) -> list[Any]:
    """Return the tool-snapshot spec list from the dataset sidecar.

    Args:
        dataset: Raw dataset rows.

    Returns:
        The first row's ``__tool_snapshot__`` list, or an empty list when no
        row carries one.
    """
    if not dataset:
        return []
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        sidecar = row.get("__tool_snapshot__")
        if isinstance(sidecar, str):
            sidecar = _as_sequence(sidecar)
        if isinstance(sidecar, (list, tuple)) and sidecar:
            return list(sidecar)
    return []


def _snapshot_tool(
    *, name: str, desc: str, args: dict[str, Any], severity: str | None = None
) -> dspy.Tool:
    """Construct a ``dspy.Tool`` carrying a snapshot spec's schema.

    The serve path reconstructs the roster from these specs; the callable body
    is a placeholder because serve re-binds live MCP callables when it executes.

    Args:
        name: Tool name.
        desc: Tool description.
        args: Per-arg JSON schema map.
        severity: Optional approval severity to carry through (so a
            snapshot-sourced run keeps the live roster's per-tool severity).

    Returns:
        A ``dspy.Tool`` whose ``args`` mirror the snapshot spec.
    """

    def _noop(**_kwargs: Any) -> None:
        """Placeholder body — serve re-binds live callables before executing."""

    _noop.__name__ = name
    _noop.__doc__ = desc
    tool = dspy.Tool(_noop, name=name, desc=desc, args=args or None)
    set_tool_severity(tool, severity)
    return tool


def resolve_react_tools(
    tool_source: Any,
    signature_cls: type,
    settings: Settings,
    *,
    dataset: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dspy.Tool], dict[str, str]]:
    """Materialise the react tool roster + its schema-hash snapshot.

    Two sources, mirroring ``ToolSource.kind``:

    - ``live_mcp`` lists the live MCP roster via :func:`_list_live_tools`,
      optionally filtering and reordering by ``tool_source.tool_filter``.
    - ``dataset_snapshot`` rebuilds the roster from the dataset's reserved
      ``__tool_snapshot__`` sidecar (run path) or the source's own persisted
      ``tool_snapshot`` specs (serve path) via :func:`_dataset_snapshot_tools`.

    The schema-hash map is computed with :func:`hash_tool_schema` over whatever
    roster was resolved, so it stays 1:1 with the tools the seed program sees.

    Args:
        tool_source: A ``ToolSource``-shaped object or a persisted mapping
            (``kind`` plus optional ``mcp_url``/``mcp_auth_header``/
            ``tool_filter``/``tool_snapshot``); read field-tolerantly via
            :func:`_ts` so the run and serve paths share one resolver.
        signature_cls: The resolved signature (unused today; accepted so the
            call site stays stable as the snapshot path grows).
        settings: Runtime settings — supplies the default MCP URL.
        dataset: Raw dataset rows, required for the ``dataset_snapshot`` kind.

    Returns:
        ``(tools, schema_hashes)`` — the resolved roster and its
        ``{tool_name: sha256_hex}`` snapshot.

    Raises:
        ValueError: When ``tool_source.kind`` is unrecognised, or when a
            snapshot source carries no tools.
    """
    _ = signature_cls
    kind = _ts(tool_source, "kind")
    if kind == "live_mcp":
        mcp_url = _ts(tool_source, "mcp_url") or settings.generalist_agent_mcp_url
        tools = asyncio.run(
            _list_live_tools(mcp_url, _ts(tool_source, "mcp_auth_header"))
        )
        tools = _apply_tool_filter(tools, _ts(tool_source, "tool_filter"))
    elif kind == "dataset_snapshot":
        tools = _dataset_snapshot_tools(_snapshot_source(tool_source, dataset))
        tools = _apply_tool_filter(tools, _ts(tool_source, "tool_filter"))
        if not tools:
            raise ValueError(
                "dataset_snapshot tool source carried no tools — expected a "
                "'__tool_snapshot__' sidecar with at least one spec."
            )
    else:
        raise ValueError(f"Unknown tool_source.kind {kind!r}.")
    schema_hashes = {tool.name: hash_tool_schema(tool) for tool in tools}
    return tools, schema_hashes


def _ts(tool_source: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a tool source that may be an object or a mapping.

    The run path passes a ``ToolSource`` model (attribute access); the serve
    path re-sources from a persisted dict (mapping access). Reads through
    both so the same resolver serves either shape.

    Args:
        tool_source: A ``ToolSource``-shaped object or a plain mapping.
        key: Field name to read.
        default: Returned when the key is absent on either shape.

    Returns:
        The stored value, or ``default`` when absent.
    """
    if isinstance(tool_source, Mapping):
        return tool_source.get(key, default)
    return getattr(tool_source, key, default)


def _snapshot_source(
    tool_source: Any, dataset: Sequence[Mapping[str, Any]] | None
) -> Sequence[Mapping[str, Any]] | None:
    """Pick the snapshot-spec carrier for a ``dataset_snapshot`` source.

    Serve persists the snapshot specs on the source itself under
    ``tool_snapshot``; the run path carries them on the dataset's
    ``__tool_snapshot__`` sidecar. When the source carries them, wrap the
    specs in a synthetic single-row sidecar so :func:`_dataset_snapshot_tools`
    can read them without the original dataset.

    Args:
        tool_source: The ``dataset_snapshot`` tool source (object or mapping).
        dataset: Raw dataset rows for the run path.

    Returns:
        A row sequence carrying the snapshot sidecar, or ``dataset`` when the
        source has no persisted ``tool_snapshot``.
    """
    snapshot = _ts(tool_source, "tool_snapshot")
    if snapshot:
        return [{"__tool_snapshot__": snapshot}]
    return dataset


def _apply_tool_filter(
    tools: list[dspy.Tool], tool_filter: list[str] | None
) -> list[dspy.Tool]:
    """Filter and order a tool roster by ``tool_filter``.

    When ``tool_filter`` is set the result keeps only the named tools, in the
    filter's order (names absent from the roster are skipped). When ``None`` the
    roster is returned unchanged.

    Args:
        tools: The resolved tool roster.
        tool_filter: Optional ordered allow-list of tool names.

    Returns:
        The filtered, reordered roster (or the original when no filter).
    """
    if not tool_filter:
        return tools
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in tool_filter if name in by_name]


def _build_feedback_map(
    seed_program: dspy.Module, metric: Callable[..., Any]
) -> dict[str, Callable[..., Any]]:
    """Build GEPA's per-predictor feedback map from a standard DSPy metric.

    Mirrors ``dspy.GEPA``'s ``feedback_fn_creator``: each predictor gets a
    closure that calls the user metric with GEPA's 5-arg reflection signature
    ``(gold, pred, trace, pred_name, pred_trace)`` and normalises the result
    into the ``{score, feedback}`` shape the gepa-package adapter expects. The
    only react-specific predictor (``react``) and the inner ``extract`` both get
    an entry, but only the component GEPA actually proposes is ever read.

    Args:
        seed_program: The seed ReAct program whose predictors are mapped.
        metric: The standard ``(gold, pred, trace, pred_name, pred_trace)``
            metric, returning a float or a ``dspy.Prediction``-like object with
            ``score``/``feedback``.

    Returns:
        ``{predictor_name: feedback_fn}`` for every named predictor.
    """

    def _make(pred_name: str, predictor: Any) -> Callable[..., Any]:
        """Bind one predictor's feedback closure over the shared metric."""

        def feedback_fn(
            predictor_output: dict[str, Any],
            predictor_inputs: dict[str, Any],
            module_inputs: Any,
            module_outputs: Any,
            captured_trace: Any,
        ) -> Any:
            """Score one predictor invocation and wrap it as score+feedback."""
            trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
            # Fresh log_metrics slot per call — same residue guard as
            # MinibatchRecorder; react runs don't aggregate logged scores yet.
            reset_logged_metrics()
            outcome = metric(
                module_inputs,
                module_outputs,
                captured_trace,
                pred_name,
                trace_for_pred,
            )
            if hasattr(outcome, "feedback"):
                if outcome["feedback"] is None:
                    outcome["feedback"] = f"This trajectory got a score of {outcome['score']}."
                return outcome
            return {
                "score": outcome,
                "feedback": f"This trajectory got a score of {outcome}.",
            }

        return feedback_fn

    return {name: _make(name, pred) for name, pred in seed_program.named_predictors()}


def _best_candidate(result: Any) -> dict[str, str]:
    """Pull the best candidate dict out of GEPA's result wrapper.

    Args:
        result: The ``GEPAResult`` returned by ``gepa.optimize``.

    Returns:
        The best candidate as a ``{component: text}`` dict.

    Raises:
        ValueError: When the result exposes no dict-shaped best candidate.
    """
    best = getattr(result, "best_candidate", None) or getattr(result, "candidate", None)
    if isinstance(best, dict):
        return dict(best)
    raise ValueError(
        f"Unable to extract best candidate from GEPA result of type {type(result)!r}."
    )


def _scores_and_outputs(
    adapter: DspyAdapter, candidate: dict[str, str], examples: list[dspy.Example]
) -> tuple[list[float], list[Any]]:
    """Score one candidate over ``examples`` and return scalars + predictions.

    Runs the candidate through the adapter's own ``evaluate`` (the same
    ``dspy.Evaluate`` path GEPA uses internally) so baseline and optimized
    numbers are apples-to-apples with the optimization loop.

    Args:
        adapter: The configured gepa-package adapter.
        candidate: The candidate component dict to evaluate.
        examples: Held-out examples to score.

    Returns:
        ``(per_example_scalars, per_example_predictions)`` in ``examples``
        order; predictions are raw ``dspy.Prediction`` objects (or ``None`` for
        a failed rollout) so the caller can surface per-field answers.
    """
    if not examples:
        return [], []
    batch = adapter.evaluate(examples, candidate, capture_traces=False)
    return list(batch.scores), list(batch.outputs)


def run_react_optimization(
    *,
    signature_cls: type,
    tools: list[dspy.Tool],
    schema_hashes: dict[str, str],
    metric: Callable[..., Any],
    train: list[dspy.Example],
    val: list[dspy.Example],
    test: list[dspy.Example],
    student_lm: dspy.LM,
    reflection_lm: dspy.LM,
    max_metric_calls: int = _AUTO_BUDGETS["medium"],
    target_score: float | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed: int = 0,
    num_threads: int | None = None,
    run_dir: str | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    timing_callbacks: Sequence[Any] = (),
) -> dict[str, Any]:
    """Optimise a live react program and report baseline-vs-optimized scalars.

    Builds a seed :class:`RetryingReActV2` over ``signature_cls`` + ``tools``,
    then runs ``gepa.optimize`` with the tool-aware gepa-package adapter
    (``enable_tool_optimization=True``) so GEPA jointly evolves the inner
    instruction prompt and the tool descriptions. The seed and best candidate
    are both scored on ``test`` through the adapter's own evaluator (live tool
    execution) so the reported delta matches what serving will reproduce.

    Args:
        signature_cls: Resolved DSPy signature the program is built around.
        tools: Tool roster from :func:`resolve_react_tools`.
        schema_hashes: ``{tool_name: hash}`` snapshot carried into the overlay.
        metric: Standard ``(gold, pred, trace, pred_name, pred_trace)`` metric.
        train: Train split (plain ``dspy.Example`` rows).
        val: Validation split passed to ``gepa.optimize`` as the valset.
        test: Held-out split scored for the reported baseline/optimized metrics.
        student_lm: Candidate-rollout model (bound by the caller's
            ``dspy.context``).
        reflection_lm: Reflective-proposer model used by GEPA's adapter.
        max_metric_calls: GEPA metric-call budget (default medium = 2000).
        target_score: Optional validation score target in the API's 0–100
            percentage scale. GEPA stops when its best validation candidate
            reaches the target, while ``max_metric_calls`` remains a ceiling.
        max_iters: ReActV2 loop budget for the seed program.
        seed: RNG seed shared by GEPA and the adapter.
        num_threads: Eval/rollout thread count for the adapter; ``None`` keeps
            DSPy's default.
        run_dir: Directory where GEPA persists its state (``gepa_state.bin``) so
            the trajectory watcher can stream the candidate tree; ``None`` keeps
            the run in-memory with no persisted state.
        progress_callback: Job-level progress sink; receives the baseline and
            optimized scalar events.
        timing_callbacks: Stage-timing callbacks (typically the generation-LM
            :class:`GenLMTimingCallback`) to attribute student-rollout latency
            to the baseline/training/evaluation stages. Empty by default.

    Returns:
        A dict with the servable ``program_state``, baseline/optimized scalar
        means and per-example scalars/outputs, and a ``tool_overlay``
        (descriptions, arg descriptions, names, schema hashes, max_iters) for
        the bundle.
    """
    seed_program = RetryingReActV2(signature_cls, tools, max_iters=max_iters)
    seed_candidate = seed_candidate_from_program(seed_program)
    adapter = DspyAdapter(
        student_module=seed_program,
        metric_fn=metric,
        feedback_map=_build_feedback_map(seed_program, metric),
        num_threads=num_threads,
        rng=random.Random(seed),
        reflection_lm=reflection_lm,
        enable_tool_optimization=True,
    )

    # Score the seed on the held-out test set up front and surface it before the
    # GEPA loop starts, mirroring the scalar run path: the early baseline event
    # lets the live score card render as soon as the baseline is known instead
    # of only after the whole loop finishes.
    with track_stage(STAGE_BASELINE, *timing_callbacks):
        baseline_scalars, baseline_outputs = _scores_and_outputs(
            adapter, seed_candidate, test
        )
    if progress_callback:
        progress_callback(PROGRESS_BASELINE, {DETAIL_BASELINE: _mean(baseline_scalars)})

    target_stopper = build_target_score_stopper(target_score)
    stop_callbacks = [target_stopper] if target_stopper is not None else None
    with track_stage(STAGE_TRAINING, *timing_callbacks):
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=(lambda x: adapter.stripped_lm_call(x)[0]),
            max_metric_calls=max_metric_calls,
            stop_callbacks=stop_callbacks,
            seed=seed,
            run_dir=run_dir,
            # GEPA defaults both off, which left react runs without a score chart
            # or a progress bar: logger=None prints iteration lines to stdout
            # (never reaching job_logs), and display_progress_bar=False means no
            # rollouts bar for capture_tqdm to relay as optimizer_progress.
            logger=_JobLogGepaLogger(logger),
            display_progress_bar=True,
        )
    best_candidate = _best_candidate(result)
    best_program = adapter.build_program(best_candidate)
    program_state = extract_program_state(best_program)

    with track_stage(STAGE_EVALUATION, *timing_callbacks):
        optimized_scalars, optimized_outputs = _scores_and_outputs(
            adapter, best_candidate, test
        )
    if progress_callback:
        progress_callback(
            PROGRESS_OPTIMIZED, {DETAIL_OPTIMIZED: _mean(optimized_scalars)}
        )

    return {
        "program_state": program_state,
        "baseline_scalar": _mean(baseline_scalars),
        "optimized_scalar": _mean(optimized_scalars),
        "baseline_scalars_per_example": baseline_scalars,
        "optimized_scalars_per_example": optimized_scalars,
        "baseline_outputs_per_example": baseline_outputs,
        "optimized_outputs_per_example": optimized_outputs,
        "target_score_reached": (
            bool(target_stopper.reached) if target_stopper is not None else None
        ),
        "tool_overlay": {
            "tool_descriptions": _candidate_tool_descriptions(best_candidate),
            "tool_arg_descriptions": _candidate_tool_arg_descriptions(best_candidate),
            "tool_names": _candidate_tool_names(best_candidate),
            "tool_schema_hashes": schema_hashes,
            "max_iters": max_iters,
        },
    }


def _mean(values: list[float]) -> float:
    """Arithmetic mean, ``0.0`` for an empty list."""
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "DEFAULT_MAX_ITERS",
    "resolve_react_tools",
    "run_react_optimization",
]
