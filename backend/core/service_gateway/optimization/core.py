"""High-level :class:`DspyService` facade for running optimization jobs.

Glues together signature/metric loading, dataset splitting, optimizer
instantiation, baseline-vs-optimized evaluation, artifact persistence,
and progress callbacks for both single-run and grid-search payloads.
The companion modules in this package handle the individual concerns;
this file orchestrates them.
"""

import functools
import json
import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from ...config import settings as app_settings
from ...constants import (
    DETAIL_BASELINE,
    DETAIL_OPTIMIZED,
    DETAIL_TEST,
    DETAIL_TRAIN,
    DETAIL_VAL,
    META_COMPILE_KWARGS,
    META_MODEL_IDENTIFIER,
    META_MODULE_KWARGS,
    META_OPTIMIZER,
    META_OPTIMIZER_KWARGS,
    OPTIMIZER_NAME_GEPA,
    PROGRESS_BASELINE,
    PROGRESS_CANDIDATE,
    PROGRESS_GRID_PAIR_COMPLETED,
    PROGRESS_GRID_PAIR_FAILED,
    PROGRESS_GRID_PAIR_STARTED,
    PROGRESS_MINIBATCH,
    PROGRESS_OPTIMIZED,
    PROGRESS_REJECTED,
    PROGRESS_SPLITS_READY,
)
from ...exceptions import ServiceError
from ...models import (
    GridSearchRequest,
    GridSearchResponse,
    LMActivity,
    LMStageStats,
    PairResult,
    RunRequest,
    RunResponse,
    SplitCounts,
)
from ...models.artifacts import ProgramArtifact, ReactOverlay
from ...registry import (
    ResolverError,
    ServiceRegistry,
    UnknownRegistrationError,
    resolve_module_factory,
    resolve_optimizer_factory,
)
from ...worker.log_handler import set_current_pair_index
from ..language_models import apply_model_reasoning_config, build_language_model
from ..react_compat import REACT_CLASS
from ..safe_exec import validate_metric_code, validate_signature_code
from .artifacts import persist_program
from .data import (
    extract_signature_fields,
    image_input_field_names,
    load_metric_from_code,
    load_signature_from_code,
    rows_to_examples,
    split_examples,
)
from .llm_error import enrich_error_message
from .optimizers import (
    compile_program,
    evaluate_on_test,
    instantiate_optimizer,
    preflight_metric_check,
    validate_optimizer_kwargs,
    validate_optimizer_signature,
)
from .progress import capture_tqdm
from .timing import (
    STAGE_BASELINE,
    STAGE_EVALUATION,
    STAGE_TRAINING,
    GenLMTimingCallback,
    ReflectionLMTimingCallback,
    track_stage,
)
from .training_ground import run_react
from .training_ground.run_react import _AUTO_BUDGETS, tool_severity
from .trajectory import (
    GRID_PAIR_RESULT_FILENAME,
    capture_proposal_prompts,
    emit_valset_event,
    gepa_log_dir,
    maybe_wrap_minibatch_recorder,
    trajectory_watch,
)
from .validators import (
    require_mapping_columns_in_dataset,
    require_mapping_matches_signature,
)

logger = logging.getLogger(__name__)


def _resolve_max_metric_calls(optimizer_kwargs: dict[str, Any]) -> int:
    """Resolve the GEPA metric-call budget from a react submission's kwargs.

    Mirrors the precedence the generic GEPA path gets for free from
    ``instantiate_optimizer``: an explicit ``max_metric_calls`` wins, otherwise
    the ``auto`` preset ("light"/"medium"/"heavy") is translated through
    ``_AUTO_BUDGETS``. Only when neither is supplied do we fall back to the
    medium budget. Previously this site read ``max_metric_calls`` alone and
    hardcoded medium, so the ``auto`` preset chosen in the wizard was silently
    ignored for react jobs (a "light" run got the 2000-call medium budget).

    Args:
        optimizer_kwargs: The raw optimizer kwargs from the run payload.

    Returns:
        The metric-call ceiling to hand GEPA.
    """
    if "max_metric_calls" in optimizer_kwargs:
        return int(optimizer_kwargs["max_metric_calls"])
    auto = optimizer_kwargs.get("auto")
    if auto is not None:
        return _AUTO_BUDGETS[auto]
    return _AUTO_BUDGETS["medium"]


def _require_metric_compatible_with_optimizer(optimizer_name: str, param_names: list[str]) -> None:
    """Reject metrics whose arity is incompatible with the chosen optimizer.

    GEPA hands the metric 5 positional args during reflection —
    ``(gold, pred, trace, pred_name, pred_trace)``. Any shorter signature
    raises ``TypeError`` on every iteration, leaving GEPA unable to propose
    a new candidate and the optimized score pinned to the baseline. The
    API ``/validate-code`` route already enforces this, but submissions
    bypassing the wizard (the generalist agent, direct ``POST /run``)
    used to slip through to a "successful" no-op run.

    Args:
        optimizer_name: The optimizer key from the run/grid-search payload.
        param_names: Metric parameter names returned by ``validate_metric_code``.

    Raises:
        ServiceError: When the metric signature is incompatible with the optimizer.
    """
    if optimizer_name == OPTIMIZER_NAME_GEPA and len(param_names) < 5:
        raise ServiceError(
            "GEPA metric must accept 5 arguments: "
            "(gold, pred, trace, pred_name, pred_trace). "
            f"Found {len(param_names)}: ({', '.join(param_names)}). "
            "See https://dspy.ai/api/optimizers/GEPA for details."
        )


def _tool_to_snapshot_spec(tool: Any) -> dict[str, Any]:
    """Serialize a ``dspy.Tool`` into a dataset-snapshot spec.

    The spec round-trips through ``run_react._snapshot_tool`` at serve time, so
    the rebuilt roster keeps the same ``name``/``desc``/``args`` — and therefore
    the same schema hash — as the roster the run was optimized against.

    Args:
        tool: The resolved ``dspy.Tool`` from the run's roster.

    Returns:
        A ``{"name", "description", "args"}`` spec mirroring the tool's schema,
        plus ``"severity"`` when the live roster carried an approval hint.
    """
    spec: dict[str, Any] = {
        "name": tool.name,
        "description": tool.desc or "",
        "args": dict(tool.args) if isinstance(tool.args, dict) else {},
    }
    severity = tool_severity(tool)
    if severity is not None:
        spec["severity"] = severity
    return spec


def _react_prediction_outputs(
    prediction: Any, output_fields: Iterable[str]
) -> dict[str, Any]:
    """Pull a react rollout's per-field answer off its ``Prediction``.

    Mirrors the on-demand ``evaluate-examples`` endpoint's extraction
    (``getattr(prediction, sig_field)`` keyed by signature output field) so the
    stored test-result ``outputs`` match the shape the DataTab already reads.
    Non-scalar field values are stringified so the result stays JSON-serializable
    for persistence.

    Args:
        prediction: The rollout ``Prediction``, or ``None`` for a failed rollout.
        output_fields: Signature output field names (``column_mapping.outputs``
            keys) to surface.

    Returns:
        ``{output_field: value}``; empty when ``prediction`` is ``None``.
    """
    if prediction is None:
        return {}
    outputs: dict[str, Any] = {}
    for field_name in output_fields:
        value = getattr(prediction, field_name, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            outputs[field_name] = value
        else:
            outputs[field_name] = str(value)
    return outputs


def _build_lm_activity(
    gen_timing: GenLMTimingCallback,
    refl_timing: ReflectionLMTimingCallback | None,
) -> LMActivity:
    """Compose an ``LMActivity`` payload from per-stage callback summaries.

    Only stages that recorded at least one call get a ``LMStageStats``
    entry; the frontend renders zero-call stages from :data:`STAGE_ORDER`
    on its own, so we keep the wire payload sparse.

    Args:
        gen_timing: Generation-LM timing callback, always present.
        refl_timing: Reflection-LM timing callback, or ``None`` when the
            run did not use a reflection LM (non-GEPA optimizer).

    Returns:
        An ``LMActivity`` whose ``generation`` and ``reflection`` dicts
        contain one entry per stage with calls > 0.
    """
    generation: dict[str, LMStageStats] = {}
    for stage, (calls, avg_ms) in gen_timing.stage_summary().items():
        generation[stage] = LMStageStats(calls=calls, avg_response_time_ms=avg_ms)
    reflection: dict[str, LMStageStats] = {}
    if refl_timing is not None:
        for stage, (calls, avg_ms) in refl_timing.stage_summary().items():
            reflection[stage] = LMStageStats(calls=calls, avg_response_time_ms=avg_ms)
    return LMActivity(generation=generation, reflection=reflection)


@dataclass
class _GridPairContext:
    """Shared state threaded through grid-search pair workers.

    Groups every value a pair worker previously pulled from the
    ``run_grid_search`` closure so the worker can live at module scope.
    Counters are mutated through ``results_lock`` (never directly); pair
    workers read the snapshot inside the lock so progress callbacks see a
    consistent view.
    """

    total_pairs: int
    module_factory: Any
    module_kwargs: dict[str, Any]
    payload: GridSearchRequest
    optimizer_factory: Any
    metric: Any
    splits: Any
    artifact_id: str | None
    progress_callback: Callable[[str, dict[str, Any]], None] | None
    # Worker-owned base dir for resumable grids; each pair writes its GEPA state
    # and (on success) ``result.json`` under ``<base>/pair_<i>``. None = ephemeral.
    gepa_log_dir_base: str | None = None
    results_lock: threading.Lock = field(default_factory=threading.Lock)
    completed: int = 0
    failed: int = 0


def _write_pair_result_json(pair_dir: str, result: PairResult) -> None:
    """Atomically write a completed pair's result for the worker to persist.

    The worker reads ``<pair_dir>/result.json`` to durably record finished pairs,
    so a resumed grid skips them. Written via a temp file + ``os.replace`` so the
    worker never sees a partial write.

    Args:
        pair_dir: The pair's worker-owned directory.
        result: The completed :class:`PairResult` to serialize.
    """
    path = Path(pair_dir) / GRID_PAIR_RESULT_FILENAME
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(result.model_dump(mode="json")), encoding="utf-8")
    tmp.replace(path)


def _tag_candidate_event(
    callback: Callable[[str, dict[str, Any]], None],
    pair_index: int,
    event: str,
    metrics: dict[str, Any],
) -> None:
    """Forward a progress event, stamping ``pair_index`` onto trajectory metrics.

    Candidate and rejected-proposal events come out of the per-pair GEPA state
    file unaware of which grid pair produced them; the frontend uses
    ``pair_index`` to scope the trajectory tree to a single pair, so we
    inject it here before forwarding. Other events pass through untouched.

    Args:
        callback: The downstream progress callback to forward into.
        pair_index: Zero-based grid pair index to stamp onto trajectory metrics.
        event: Event name as produced by ``TrajectoryWatcher``.
        metrics: Event metrics dict from the watcher.
    """
    if event in (PROGRESS_CANDIDATE, PROGRESS_REJECTED, PROGRESS_MINIBATCH):
        callback(event, {**metrics, "pair_index": pair_index})
    else:
        callback(event, metrics)


def _run_grid_pair(
    ctx: _GridPairContext,
    i: int,
    gen_cfg: Any,
    ref_cfg: Any,
) -> PairResult:
    """Execute one (generation_model, reflection_model) pair and return its ``PairResult``.

    Mirrors the per-pair flow of ``run_grid_search``: compiles the module,
    evaluates baseline and optimized programs on the test split, persists
    the better program, and emits progress events through
    ``ctx.progress_callback``. On failure returns a ``PairResult`` carrying
    the stringified exception rather than propagating — the caller treats
    unsuccessful pairs as soft failures.

    Args:
        ctx: Shared grid-pair context with payload, factories, and counters.
        i: Zero-based pair index for this worker.
        gen_cfg: Generation-model config for this pair.
        ref_cfg: Reflection-model config for this pair.

    Returns:
        The :class:`PairResult` for this pair (carrying an ``error`` string
        when execution failed).
    """
    set_current_pair_index(i)
    pair_label = f"{gen_cfg.name} + {ref_cfg.name}"
    # On a resumable grid each pair gets its own dir under the worker-owned base,
    # so the worker can restore its checkpoint (resume) and persist its result.
    pair_dir = str(Path(ctx.gepa_log_dir_base) / f"pair_{i}") if ctx.gepa_log_dir_base else None
    logger.info("Grid pair %d/%d: %s", i + 1, ctx.total_pairs, pair_label)
    if ctx.progress_callback:
        ctx.progress_callback(
            PROGRESS_GRID_PAIR_STARTED,
            {
                "pair_index": i,
                "total_pairs": ctx.total_pairs,
                "generation_model": gen_cfg.name,
                "reflection_model": ref_cfg.name,
            },
        )

    pair_start = datetime.now(UTC)
    try:
        program = ctx.module_factory(**dict(ctx.module_kwargs))
        # Caching stays ON here: forcing cache off across the GEPA
        # training/eval region suppresses GEPA's recognized valset
        # ``Evaluate`` / rollouts tqdm bar, which is the only bar the
        # progress proxy emits from (regression from #23/#24). Per-stage
        # activity tracking does not need cache-off — it works through the
        # timing callbacks' stage state, independent of the LM cache.
        language_model = build_language_model(gen_cfg)
        reflection_lm = build_language_model(ref_cfg) if ref_cfg is not None else None
        gen_timing = GenLMTimingCallback(language_model)
        refl_timing = ReflectionLMTimingCallback(reflection_lm) if reflection_lm is not None else None
        callbacks: list[Any] = [gen_timing]
        if refl_timing is not None:
            callbacks.append(refl_timing)

        with dspy.context(lm=language_model, callbacks=callbacks), gepa_log_dir(
            ctx.payload.optimizer_name, pair_dir
        ) as trajectory_log_dir:
            trajectory_callback: Callable[[str, dict[str, Any]], None] | None = (
                functools.partial(_tag_candidate_event, ctx.progress_callback, i)
                if ctx.progress_callback is not None
                else None
            )
            training_metric = maybe_wrap_minibatch_recorder(
                ctx.metric,
                ctx.splits.val,
                ctx.payload.optimizer_name,
                trajectory_callback,
                ctx.payload.module_name,
            )
            optimizer = instantiate_optimizer(
                ctx.optimizer_factory,
                ctx.payload.optimizer_name,
                ctx.payload.optimizer_kwargs,
                training_metric,
                ref_cfg,
                reflection_lm=reflection_lm,
                log_dir=trajectory_log_dir,
            )
            baseline = None
            baseline_test_results: list[dict] = []
            if ctx.splits.test:
                with track_stage(STAGE_BASELINE, *callbacks):
                    baseline, baseline_test_results = evaluate_on_test(
                        program,
                        ctx.splits.test,
                        ctx.metric,
                        collect_per_example=True,
                    )
                if ctx.progress_callback and baseline is not None:
                    ctx.progress_callback(
                        PROGRESS_BASELINE,
                        {
                            DETAIL_BASELINE: baseline,
                            "pair_index": i,
                        },
                    )

            with (
                capture_tqdm(ctx.progress_callback),
                track_stage(STAGE_TRAINING, *callbacks),
                capture_proposal_prompts(ctx.payload.optimizer_name),
                trajectory_watch(trajectory_log_dir, trajectory_callback),
            ):
                compiled = compile_program(
                    optimizer=optimizer,
                    program=program,
                    splits=ctx.splits,
                    metric=ctx.metric,
                    compile_kwargs=ctx.payload.compile_kwargs,
                )

            optimized = None
            optimized_test_results: list[dict] = []
            if ctx.splits.test:
                with track_stage(STAGE_EVALUATION, *callbacks):
                    optimized, optimized_test_results = evaluate_on_test(
                        compiled,
                        ctx.splits.test,
                        ctx.metric,
                        collect_per_example=True,
                    )
                if ctx.progress_callback and optimized is not None:
                    ctx.progress_callback(
                        PROGRESS_OPTIMIZED,
                        {
                            DETAIL_OPTIMIZED: optimized,
                            "pair_index": i,
                        },
                    )

        best = compiled
        if baseline is not None and optimized is not None and optimized < baseline:
            logger.warning(
                "Grid pair %d: optimized (%.4f) worse than baseline (%.4f) — keeping baseline",
                i,
                optimized,
                baseline,
            )
            best = program
            optimized = baseline

        art_id = f"{ctx.artifact_id}_pair_{i}" if ctx.artifact_id else None
        program_artifact = persist_program(best, art_id)

        improvement = None
        if baseline is not None and optimized is not None:
            improvement = optimized - baseline

        pair_runtime = (datetime.now(UTC) - pair_start).total_seconds()
        pair_lm_calls = len(language_model.history) if hasattr(language_model, "history") else None
        _, pair_avg_ms = gen_timing.summary()
        pair_lm_activity = _build_lm_activity(gen_timing, refl_timing)
        result = PairResult(
            pair_index=i,
            generation_model=gen_cfg.name,
            reflection_model=ref_cfg.name,
            generation_reasoning_effort=gen_cfg.extra.get("reasoning_effort"),
            reflection_reasoning_effort=ref_cfg.extra.get("reasoning_effort"),
            baseline_test_metric=baseline,
            optimized_test_metric=optimized,
            metric_improvement=improvement,
            runtime_seconds=round(pair_runtime, 2),
            num_lm_calls=pair_lm_calls,
            avg_response_time_ms=pair_avg_ms,
            lm_activity=pair_lm_activity,
            program_artifact=program_artifact,
            baseline_test_results=baseline_test_results,
            optimized_test_results=optimized_test_results,
        )
        with ctx.results_lock:
            ctx.completed += 1
        logger.info(
            "Grid pair %d/%d completed: baseline=%.4f optimized=%.4f (%.1fs)",
            i + 1,
            ctx.total_pairs,
            baseline if baseline is not None else float("nan"),
            optimized if optimized is not None else float("nan"),
            pair_runtime,
        )
        if ctx.progress_callback:
            with ctx.results_lock:
                c, f = ctx.completed, ctx.failed
            ctx.progress_callback(
                PROGRESS_GRID_PAIR_COMPLETED,
                {
                    "pair_index": i,
                    "total_pairs": ctx.total_pairs,
                    "generation_model": gen_cfg.name,
                    "reflection_model": ref_cfg.name,
                    "baseline_test_metric": baseline,
                    "optimized_test_metric": optimized,
                    "metric_improvement": improvement,
                    "runtime_seconds": round(pair_runtime, 2),
                    "completed_so_far": c,
                    "failed_so_far": f,
                },
            )
        # Record the finished pair so a resumed grid keeps it instead of re-running.
        if pair_dir is not None:
            _write_pair_result_json(pair_dir, result)
        set_current_pair_index(None)
        return result

    except Exception as exc:
        pair_runtime = (datetime.now(UTC) - pair_start).total_seconds()
        # DSPy reports a generic "Execution cancelled" when an LM call fails;
        # recover the real cause (billing, auth, rate limit) it only logged.
        error_msg = enrich_error_message(str(exc))
        result = PairResult(
            pair_index=i,
            generation_model=gen_cfg.name,
            reflection_model=ref_cfg.name,
            generation_reasoning_effort=gen_cfg.extra.get("reasoning_effort"),
            reflection_reasoning_effort=ref_cfg.extra.get("reasoning_effort"),
            error=error_msg,
            runtime_seconds=round(pair_runtime, 2),
        )
        with ctx.results_lock:
            ctx.failed += 1
        logger.warning(
            "Grid pair %d/%d failed (%s): %s",
            i + 1,
            ctx.total_pairs,
            pair_label,
            error_msg,
        )
        if ctx.progress_callback:
            with ctx.results_lock:
                c, f = ctx.completed, ctx.failed
            ctx.progress_callback(
                PROGRESS_GRID_PAIR_FAILED,
                {
                    "pair_index": i,
                    "total_pairs": ctx.total_pairs,
                    "generation_model": gen_cfg.name,
                    "reflection_model": ref_cfg.name,
                    "error": error_msg,
                    "completed_so_far": c,
                    "failed_so_far": f,
                },
            )
        set_current_pair_index(None)
        return result


class DspyService:
    """High-level coordinator between FastAPI payloads and DSPy runtimes."""

    def __init__(
        self,
        registry: ServiceRegistry,
        default_seed: int | None = None,
    ):
        """Initialize DspyService with a module/optimizer registry and optional default seed.

        Args:
            registry: Module/optimizer registry used to resolve named factories.
            default_seed: Optional default split seed when a payload omits one.
        """
        self.registry = registry
        self.default_seed = default_seed

    def run(
        self,
        payload: RunRequest,
        *,
        artifact_id: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        gepa_log_dir_path: str | None = None,
    ) -> RunResponse:
        """Execute a single DSPy optimization run and return a structured response.

        Loads the signature, module, metric, and optimizer from the payload,
        splits the dataset, runs baseline evaluation, compiles the program, then
        evaluates the optimized result.  If the optimized program scores lower than
        the baseline on the test set, the baseline program is returned instead and
        the reported optimized metric is swapped to the baseline value.

        Args:
            payload: The validated run request to execute.
            artifact_id: Optional identifier carried into artifact storage.
            progress_callback: Optional callback receiving ``(event, detail)`` updates.
            gepa_log_dir_path: Worker-owned directory for GEPA's ``gepa_state.bin``.
                When the directory already holds a prior state file the GEPA
                engine resumes from it; when ``None`` an ephemeral tempdir is
                used and the run starts fresh.

        Returns:
            A populated :class:`RunResponse` summarising the run.
        """
        run_start = datetime.now(UTC)
        logger.info(
            "Starting DSPy run: module=%s optimizer=%s dataset_rows=%d",
            payload.module_name,
            payload.optimizer_name,
            len(payload.dataset),
        )

        signature_cls = load_signature_from_code(payload.signature_code)
        signature_inputs, signature_outputs = extract_signature_fields(signature_cls)
        logger.info(
            "Loaded signature %s with inputs=%s outputs=%s",
            signature_cls.__name__,
            signature_inputs,
            signature_outputs,
        )
        require_mapping_matches_signature(payload.column_mapping, signature_inputs, signature_outputs)
        if payload.module_name.lower() == "react":
            return self._run_react(
                payload,
                signature_cls=signature_cls,
                run_start=run_start,
                artifact_id=artifact_id,
                progress_callback=progress_callback,
                gepa_log_dir_path=gepa_log_dir_path,
            )
        module_factory, auto_signature = self._get_module_factory(payload.module_name)
        module_kwargs = dict(payload.module_kwargs)
        if auto_signature or "signature" not in module_kwargs:
            module_kwargs["signature"] = signature_cls
        program = module_factory(**module_kwargs)
        # Breadcrumb for the dspy.* escape hatch: a non-aliased path gets
        # auto_signature=False and trusts a user-supplied signature, so an
        # opaque constructor error is otherwise indistinguishable from an
        # aliased-module failure in job_logs.
        logger.info(
            "Instantiated module %s (auto_signature=%s signature_injected=%s)",
            payload.module_name,
            auto_signature,
            "signature" in module_kwargs,
        )

        metric = load_metric_from_code(payload.metric_code)
        metric_identifier = getattr(metric, "__name__", "inline_metric")

        # Caching stays ON here: forcing cache off across the GEPA
        # training/eval region suppresses GEPA's recognized valset
        # ``Evaluate`` / rollouts tqdm bar, which is the only bar the
        # progress proxy emits from (regression from #23/#24). Per-stage
        # activity tracking does not need cache-off — it works through the
        # timing callbacks' stage state, independent of the LM cache.
        language_model = build_language_model(payload.model_settings)
        # Build the reflection LM up front (when supplied) so the timing
        # callback can be bound to its identity. ``instantiate_optimizer``
        # would otherwise build it internally, leaving us no handle.
        reflection_lm = (
            build_language_model(payload.reflection_model_settings)
            if payload.reflection_model_settings is not None
            else None
        )
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)

        examples = rows_to_examples(
            payload.dataset,
            payload.column_mapping,
            image_input_fields=image_input_field_names(signature_cls),
        )
        logger.info("Converted dataset to %d DSPy examples", len(examples))

        splits = split_examples(
            examples,
            payload.split_fractions,
            shuffle=payload.shuffle,
            seed=payload.seed or self.default_seed,
        )
        logger.info(
            "Split dataset -> train=%d val=%d test=%d",
            len(splits.train),
            len(splits.val),
            len(splits.test),
        )
        if progress_callback:
            progress_callback(
                PROGRESS_SPLITS_READY,
                {
                    DETAIL_TRAIN: len(splits.train),
                    DETAIL_VAL: len(splits.val),
                    DETAIL_TEST: len(splits.test),
                },
            )
            emit_valset_event(splits.val, progress_callback)

        gen_timing = GenLMTimingCallback(language_model)
        refl_timing = ReflectionLMTimingCallback(reflection_lm) if reflection_lm is not None else None
        callbacks: list[Any] = [gen_timing]
        if refl_timing is not None:
            callbacks.append(refl_timing)
        with gepa_log_dir(payload.optimizer_name, gepa_log_dir_path) as trajectory_log_dir:
            training_metric = maybe_wrap_minibatch_recorder(
                metric,
                splits.val,
                payload.optimizer_name,
                progress_callback,
                payload.module_name,
            )
            optimizer = instantiate_optimizer(
                optimizer_factory,
                payload.optimizer_name,
                payload.optimizer_kwargs,
                training_metric,
                payload.reflection_model_settings,
                reflection_lm=reflection_lm,
                log_dir=trajectory_log_dir,
            )
            with dspy.context(lm=language_model, callbacks=callbacks):
                baseline_test_metric = None
                baseline_test_results: list[dict] = []
                if splits.test:
                    with track_stage(STAGE_BASELINE, *callbacks):
                        baseline_test_metric, baseline_test_results = evaluate_on_test(
                            program,
                            splits.test,
                            metric,
                            collect_per_example=True,
                        )
                    logger.info(
                        "%s baseline test metric: %s", payload.module_name, baseline_test_metric
                    )
                    if progress_callback and baseline_test_metric is not None:
                        progress_callback(
                            PROGRESS_BASELINE,
                            {DETAIL_BASELINE: baseline_test_metric},
                        )

                # Fail fast on a structurally broken metric (e.g. wrong field
                # names or isinstance(gold, dict) gating) before spending the
                # optimizer budget grinding at 0%. A correct metric always
                # scores a perfect prediction > 0, so legitimately hard tasks
                # are never blocked.
                preflight_metric_check(metric, splits.train, signature_outputs)

                logger.info(
                    "Compiling %s program via optimizer=%s",
                    payload.module_name,
                    payload.optimizer_name,
                )
                with (
                    capture_tqdm(progress_callback),
                    track_stage(STAGE_TRAINING, *callbacks),
                    capture_proposal_prompts(payload.optimizer_name),
                    trajectory_watch(trajectory_log_dir, progress_callback),
                ):
                    compiled_program = compile_program(
                        optimizer=optimizer,
                        program=program,
                        splits=splits,
                        metric=metric,
                        compile_kwargs=payload.compile_kwargs,
                    )
                logger.info("Optimizer compile completed successfully")

                optimized_test_metric = None
                optimized_test_results: list[dict] = []
                if splits.test:
                    with track_stage(STAGE_EVALUATION, *callbacks):
                        optimized_test_metric, optimized_test_results = evaluate_on_test(
                            compiled_program,
                            splits.test,
                            metric,
                            collect_per_example=True,
                        )
                    logger.info("Optimized test metric: %s", optimized_test_metric)
                    if progress_callback and optimized_test_metric is not None:
                        progress_callback(
                            PROGRESS_OPTIMIZED,
                            {DETAIL_OPTIMIZED: optimized_test_metric},
                        )

                best_program = compiled_program
                if (
                    baseline_test_metric is not None
                    and optimized_test_metric is not None
                    and optimized_test_metric < baseline_test_metric
                ):
                    logger.warning(
                        "Optimized program (%.4f) is worse than baseline (%.4f) — returning baseline program",
                        optimized_test_metric,
                        baseline_test_metric,
                    )
                    best_program = program
                    # Swap so the "optimized" metric reflects what we're actually returning
                    optimized_test_metric = baseline_test_metric

        program_artifact = persist_program(best_program, artifact_id)
        if program_artifact:
            logger.info("Program artifact created: %s", program_artifact.path)

        split_counts = SplitCounts(train=len(splits.train), val=len(splits.val), test=len(splits.test))

        details: dict[str, Any] = {
            DETAIL_TRAIN: split_counts.train,
            DETAIL_VAL: split_counts.val,
            DETAIL_TEST: split_counts.test,
            DETAIL_BASELINE: baseline_test_metric,
            DETAIL_OPTIMIZED: optimized_test_metric,
        }

        optimization_metadata = {
            META_OPTIMIZER: payload.optimizer_name,
            META_OPTIMIZER_KWARGS: payload.optimizer_kwargs,
            META_COMPILE_KWARGS: payload.compile_kwargs,
            META_MODULE_KWARGS: payload.module_kwargs,
            META_MODEL_IDENTIFIER: payload.model_settings.normalized_identifier(),
        }

        metric_improvement = None
        if baseline_test_metric is not None and optimized_test_metric is not None:
            metric_improvement = optimized_test_metric - baseline_test_metric

        runtime_seconds = (datetime.now(UTC) - run_start).total_seconds()
        # num_lm_calls comes from LM history (preserves counting for mocked LMs);
        # avg is wall-clock time spent inside the generation LM only, via the callback.
        num_lm_calls = len(language_model.history) if hasattr(language_model, "history") else None
        _, avg_response_time_ms = gen_timing.summary()
        lm_activity = _build_lm_activity(gen_timing, refl_timing)
        response = RunResponse(
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
            metric_name=metric_identifier,
            split_counts=split_counts,
            baseline_test_metric=baseline_test_metric,
            optimized_test_metric=optimized_test_metric,
            metric_improvement=metric_improvement,
            optimization_metadata=optimization_metadata,
            details=details,
            program_artifact_path=program_artifact.path if program_artifact else None,
            program_artifact=program_artifact,
            runtime_seconds=runtime_seconds,
            num_lm_calls=num_lm_calls,
            avg_response_time_ms=avg_response_time_ms,
            lm_activity=lm_activity,
            baseline_test_results=baseline_test_results,
            optimized_test_results=optimized_test_results,
        )

        logger.info(
            "DSPy run finished: module=%s optimizer=%s status=success",
            payload.module_name,
            payload.optimizer_name,
        )
        return response

    def _run_react(
        self,
        payload: RunRequest,
        *,
        signature_cls: type,
        run_start: datetime,
        artifact_id: str | None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
        gepa_log_dir_path: str | None = None,
    ) -> RunResponse:
        """Optimize a generic react program over a live tool roster and assemble the response.

        React is a generic module like predict/cot that additionally carries a
        tool roster, so this mirrors :meth:`run` end-to-end: it resolves the
        roster from ``payload.tool_source``, converts the dataset into plain
        ``dspy.Example`` rows, splits them, then delegates ``gepa.optimize`` to
        :func:`run_react.run_react_optimization`, which scores the seed and the
        optimized candidate against the *live* MCP tools with the same standard
        ``(gold, pred, trace, pred_name, pred_trace)`` metric the scalar path
        uses. There is no replay: rollouts execute the live tools, so the
        optimized program behaves at eval time exactly as it will when served.
        The servable program state is persisted into a :class:`ProgramArtifact`
        carrying the react tool overlay.

        The scalar (non-react) ``run`` path never reaches here — the branch in
        :meth:`run` returns early — so this method owns the react contract
        end-to-end without disturbing the existing flow.

        Args:
            payload: The validated react run request.
            signature_cls: The signature resolved from ``payload.signature_code``.
            run_start: Wall-clock start stamped by :meth:`run`, reused for the
                reported runtime.
            artifact_id: Optional identifier carried into artifact storage.
            progress_callback: Optional ``(event, detail)`` progress sink.
            gepa_log_dir_path: Worker-owned directory for GEPA's state file; a
                prior state file in it resumes the run, ``None`` starts fresh.

        Returns:
            A populated :class:`RunResponse` whose scalar metrics mirror the
            baseline/optimized means and whose ``program_artifact`` carries the
            react tool overlay.
        """
        tools, schema_hashes = run_react.resolve_react_tools(
            payload.tool_source,
            signature_cls,
            app_settings,
            dataset=payload.dataset,
        )

        metric = load_metric_from_code(payload.metric_code)
        metric_identifier = getattr(metric, "__name__", "inline_metric")

        examples = rows_to_examples(
            payload.dataset,
            payload.column_mapping,
            image_input_fields=image_input_field_names(signature_cls),
        )
        splits = split_examples(
            examples,
            payload.split_fractions,
            shuffle=payload.shuffle,
            seed=payload.seed or self.default_seed,
        )
        if progress_callback:
            progress_callback(
                PROGRESS_SPLITS_READY,
                {
                    DETAIL_TRAIN: len(splits.train),
                    DETAIL_VAL: len(splits.val),
                    DETAIL_TEST: len(splits.test),
                },
            )
            # Pareto cells in the candidate tree are indexed by valset position;
            # GEPA scores against the replay valset built from splits.val in the
            # same order, so emitting splits.val here lines the cells up 1:1.
            emit_valset_event(splits.val, progress_callback)

        # Normalize through the reasoning-config helper so a minimax/reasoning
        # student model gets a safe max_tokens floor + extras automatically;
        # without it a bare minimax ModelConfig truncates into malformed
        # tool_calls (a dspy ToolCalls ValidationError) during the react loop.
        student_lm = build_language_model(
            apply_model_reasoning_config(payload.model_settings)
        )
        reflection_lm = (
            build_language_model(
                apply_model_reasoning_config(payload.reflection_model_settings)
            )
            if payload.reflection_model_settings is not None
            else student_lm
        )
        max_metric_calls = _resolve_max_metric_calls(payload.optimizer_kwargs)
        seed = payload.seed if payload.seed is not None else (self.default_seed or 0)

        gen_timing = GenLMTimingCallback(student_lm)
        # Mirror the scalar run's trajectory wiring so react gets the same
        # candidate tree: GEPA persists state into trajectory_log_dir,
        # trajectory_watch streams candidate/rejected events, capture_proposal_prompts
        # records rejected proposal prompts, and capture_tqdm drives the live bar.
        with (
            gepa_log_dir(payload.optimizer_name, gepa_log_dir_path) as trajectory_log_dir,
            dspy.context(lm=student_lm, callbacks=[gen_timing]),
            capture_tqdm(progress_callback),
            capture_proposal_prompts(payload.optimizer_name),
            trajectory_watch(trajectory_log_dir, progress_callback),
        ):
            # Same minibatch recorder the scalar path wraps its metric in, so the
            # per-example reflection feedback and the
            # valset candidate outputs populate the trajectory drawer for react too.
            training_metric = maybe_wrap_minibatch_recorder(
                metric,
                splits.val,
                payload.optimizer_name,
                progress_callback,
                payload.module_name,
            )
            result = run_react.run_react_optimization(
                signature_cls=signature_cls,
                tools=tools,
                schema_hashes=schema_hashes,
                metric=training_metric,
                train=splits.train,
                val=splits.val,
                test=splits.test,
                student_lm=student_lm,
                reflection_lm=reflection_lm,
                max_metric_calls=max_metric_calls,
                seed=seed,
                run_dir=trajectory_log_dir,
                progress_callback=progress_callback,
                timing_callbacks=(gen_timing,),
            )

        overlay = result["tool_overlay"]
        program_artifact = self._persist_react_program(
            signature_cls=signature_cls,
            tools=tools,
            program_state=result["program_state"],
            overlay=overlay,
            tool_source=payload.tool_source,
            artifact_id=artifact_id,
        )

        baseline_scalar = result["baseline_scalar"]
        optimized_scalar = result["optimized_scalar"]
        metric_improvement = optimized_scalar - baseline_scalar

        if progress_callback:
            # baseline_evaluated is emitted early inside run_react_optimization
            # (before the GEPA loop) so the live score card renders mid-run;
            # here we only surface the optimized score after the best candidate
            # is scored on the held-out test set.
            progress_callback(PROGRESS_OPTIMIZED, {DETAIL_OPTIMIZED: optimized_scalar})

        split_counts = SplitCounts(
            train=len(splits.train), val=len(splits.val), test=len(splits.test)
        )
        details: dict[str, Any] = {
            DETAIL_TRAIN: split_counts.train,
            DETAIL_VAL: split_counts.val,
            DETAIL_TEST: split_counts.test,
            DETAIL_BASELINE: baseline_scalar,
            DETAIL_OPTIMIZED: optimized_scalar,
        }
        optimization_metadata = {
            META_OPTIMIZER: payload.optimizer_name,
            META_OPTIMIZER_KWARGS: payload.optimizer_kwargs,
            META_COMPILE_KWARGS: payload.compile_kwargs,
            META_MODULE_KWARGS: payload.module_kwargs,
            META_MODEL_IDENTIFIER: payload.model_settings.normalized_identifier(),
            "max_metric_calls": max_metric_calls,
        }

        runtime_seconds = (datetime.now(UTC) - run_start).total_seconds()
        num_lm_calls = len(student_lm.history) if hasattr(student_lm, "history") else None
        _, avg_response_time_ms = gen_timing.summary()

        # Map per-example scalars into the standard EvalExampleResult shape so the
        # DataTab score overlay + Compare per-row diff render for react.
        # ``outputs`` carries the rollout's per-field answer (keyed by signature
        # output field, matching the evaluate-examples endpoint) so the data view
        # can show *what* the agent produced per example, not just the score;
        # ``pass`` mirrors the scalar-run convention (score > 0).
        output_fields = list(payload.column_mapping.outputs.keys())
        baseline_outputs = result.get("baseline_outputs_per_example") or []
        optimized_outputs = result.get("optimized_outputs_per_example") or []
        baseline_test_results = [
            {
                "index": i,
                "outputs": _react_prediction_outputs(
                    baseline_outputs[i] if i < len(baseline_outputs) else None,
                    output_fields,
                ),
                "score": float(s),
                "pass": float(s) > 0,
            }
            for i, s in enumerate(result.get("baseline_scalars_per_example") or [])
        ]
        optimized_test_results = [
            {
                "index": i,
                "outputs": _react_prediction_outputs(
                    optimized_outputs[i] if i < len(optimized_outputs) else None,
                    output_fields,
                ),
                "score": float(s),
                "pass": float(s) > 0,
            }
            for i, s in enumerate(result.get("optimized_scalars_per_example") or [])
        ]

        logger.info(
            "React run finished: baseline=%.4f optimized=%.4f delta=%.4f",
            baseline_scalar,
            optimized_scalar,
            metric_improvement,
        )
        return RunResponse(
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
            metric_name=metric_identifier,
            split_counts=split_counts,
            baseline_test_metric=baseline_scalar,
            optimized_test_metric=optimized_scalar,
            metric_improvement=metric_improvement,
            optimization_metadata=optimization_metadata,
            details=details,
            program_artifact_path=program_artifact.path if program_artifact else None,
            program_artifact=program_artifact,
            runtime_seconds=runtime_seconds,
            num_lm_calls=num_lm_calls,
            avg_response_time_ms=avg_response_time_ms,
            # Generation-stage activity: student rollouts are bucketed into
            # baseline/training/evaluation via the timing_callbacks passed into
            # run_react_optimization. The reflection LM runs inside gepa.optimize
            # off the dspy callback path, so it stays untracked (None) and its
            # column is hidden by the activity panel.
            lm_activity=_build_lm_activity(gen_timing, None),
            baseline_test_results=baseline_test_results,
            optimized_test_results=optimized_test_results,
        )

    @staticmethod
    def _persist_react_program(
        *,
        signature_cls: type,
        tools: list[Any],
        program_state: dict[str, Any],
        overlay: dict[str, Any],
        tool_source: Any,
        artifact_id: str | None,
    ) -> ProgramArtifact | None:
        """Rebuild the optimized react program from state and persist it with its overlay.

        The optimized candidate is realized by loading ``program_state`` onto a
        fresh ``ReActV2`` shell (same signature + tool roster the seed used), so
        ``persist_program`` writes the canonical state JSON and extracts the
        prompt exactly as the scalar path does. The react tool overlay is then
        attached so a served bundle can reconstruct the tool surface: a
        ``live_mcp`` source persists ``{kind, mcp_url, tool_filter}`` (never the
        secret ``mcp_auth_header``), while a ``dataset_snapshot`` source persists
        ``{kind, tool_filter, tool_snapshot}`` so serve rebuilds the roster
        without the original dataset.

        Args:
            signature_cls: Signature the program was built around.
            tools: Tool roster resolved for the run.
            program_state: State dict returned by ``run_react_optimization``.
            overlay: ``tool_overlay`` dict from ``run_react_optimization``.
            tool_source: The originating ``ToolSource`` (or ``None``).
            artifact_id: Optional identifier carried into artifact storage.

        Returns:
            The persisted :class:`ProgramArtifact` with ``react_overlay`` set,
            or ``None`` when persistence produced no artifact.
        """
        program = REACT_CLASS(
            signature_cls, tools=tools, max_iters=overlay["max_iters"]
        )
        program.load_state(program_state)
        artifact = persist_program(program, artifact_id)
        if artifact is None:
            return None
        source_meta: dict[str, Any] = {}
        if tool_source is not None:
            source_meta["kind"] = tool_source.kind
            if tool_source.tool_filter is not None:
                source_meta["tool_filter"] = list(tool_source.tool_filter)
            if tool_source.kind == "live_mcp":
                if tool_source.mcp_url is not None:
                    source_meta["mcp_url"] = tool_source.mcp_url
                # mcp_auth_header is intentionally never persisted — it is a
                # secret; serve re-sources the live MCP with auth None.
            elif tool_source.kind == "dataset_snapshot":
                # Persist the resolved roster as snapshot specs so serve can
                # rebuild the tool surface without the original dataset.
                source_meta["tool_snapshot"] = [
                    _tool_to_snapshot_spec(tool) for tool in tools
                ]
        tool_names = overlay.get("tool_names")
        tool_severities = {
            tool.name: severity
            for tool in tools
            if (severity := tool_severity(tool)) is not None
        }
        artifact.react_overlay = ReactOverlay(
            tool_descriptions=dict(overlay["tool_descriptions"]),
            tool_arg_descriptions={
                name: dict(args) for name, args in overlay["tool_arg_descriptions"].items()
            },
            tool_schema_hashes=dict(overlay["tool_schema_hashes"]),
            max_iters=int(overlay["max_iters"]),
            tool_source=source_meta,
            tool_names=dict(tool_names) if tool_names else None,
            tool_severities=tool_severities,
        )
        return artifact

    def run_grid_search(
        self,
        payload: GridSearchRequest,
        *,
        artifact_id: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        gepa_log_dir_path: str | None = None,
        completed_pairs: dict[int, dict[str, Any]] | None = None,
    ) -> GridSearchResponse:
        """Run one optimization per (generation_model, reflection_model) pair and return all results.

        Pairs are executed concurrently (up to 4 threads).  Each pair follows the
        same baseline-vs-optimized fallback logic as ``run()``.  Failed pairs are
        recorded with their error message rather than aborting the whole search.

        On resume, ``completed_pairs`` (``pair_index`` → stored ``PairResult``) are
        kept as-is and skipped, and ``gepa_log_dir_path`` is the worker-owned base
        dir whose ``pair_<i>`` subdirs seed each in-flight pair's GEPA checkpoint.

        Args:
            payload: The validated grid-search request to execute.
            artifact_id: Optional identifier carried into per-pair artifacts.
            progress_callback: Optional callback receiving ``(event, detail)`` updates.

        Returns:
            A populated :class:`GridSearchResponse` with per-pair results
            and the best overall pair.
        """
        grid_start = datetime.now(UTC)
        pairs = [(gen_cfg, ref_cfg) for gen_cfg in payload.generation_models for ref_cfg in payload.reflection_models]
        total_pairs = len(pairs)
        logger.info(
            "Starting grid search: %d pairs, module=%s optimizer=%s",
            total_pairs,
            payload.module_name,
            payload.optimizer_name,
        )

        signature_cls = load_signature_from_code(payload.signature_code)
        signature_inputs, signature_outputs = extract_signature_fields(signature_cls)
        require_mapping_matches_signature(
            payload.column_mapping,
            signature_inputs,
            signature_outputs,
        )
        module_factory, auto_signature = self._get_module_factory(payload.module_name)
        module_kwargs = dict(payload.module_kwargs)
        if auto_signature or "signature" not in module_kwargs:
            module_kwargs["signature"] = signature_cls

        metric = load_metric_from_code(payload.metric_code)
        metric_identifier = getattr(metric, "__name__", "inline_metric")
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)

        examples = rows_to_examples(
            payload.dataset,
            payload.column_mapping,
            image_input_fields=image_input_field_names(signature_cls),
        )
        splits = split_examples(
            examples,
            payload.split_fractions,
            shuffle=payload.shuffle,
            seed=payload.seed or self.default_seed,
        )
        split_counts = SplitCounts(
            train=len(splits.train),
            val=len(splits.val),
            test=len(splits.test),
        )
        if progress_callback:
            progress_callback(
                PROGRESS_SPLITS_READY,
                {
                    DETAIL_TRAIN: split_counts.train,
                    DETAIL_VAL: split_counts.val,
                    DETAIL_TEST: split_counts.test,
                    "total_pairs": total_pairs,
                },
            )
            emit_valset_event(splits.val, progress_callback)

        pair_results: list[PairResult] = [None] * total_pairs  # type: ignore[list-item]
        completed = completed_pairs or {}
        # Resume: keep already-finished pairs verbatim so they are neither re-run
        # nor lost, and run only the rest.
        for idx, stored in completed.items():
            if 0 <= idx < total_pairs:
                pair_results[idx] = PairResult.model_validate(stored)
        pending = [(i, gen_cfg, ref_cfg) for i, (gen_cfg, ref_cfg) in enumerate(pairs) if i not in completed]

        grid_ctx = _GridPairContext(
            total_pairs=total_pairs,
            module_factory=module_factory,
            module_kwargs=module_kwargs,
            payload=payload,
            optimizer_factory=optimizer_factory,
            metric=metric,
            splits=splits,
            artifact_id=artifact_id,
            progress_callback=progress_callback,
            gepa_log_dir_base=gepa_log_dir_path,
            completed=len(completed),
        )
        if completed:
            logger.info(
                "Grid resume: %d/%d pairs already complete, running %d",
                len(completed),
                total_pairs,
                len(pending),
            )

        if pending:
            max_workers = min(len(pending), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_grid_pair, grid_ctx, i, gen_cfg, ref_cfg): i
                    for i, gen_cfg, ref_cfg in pending
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    pair_results[idx] = future.result()

        successful = [p for p in pair_results if p.error is None and p.optimized_test_metric is not None]
        best_pair = (
            max(successful, key=lambda p: p.optimized_test_metric if p.optimized_test_metric is not None else float("-inf"))
            if successful
            else None
        )

        grid_runtime = (datetime.now(UTC) - grid_start).total_seconds()
        completed_count = len([p for p in pair_results if p.error is None])
        failed_count = len([p for p in pair_results if p.error is not None])

        logger.info(
            "Grid search finished: %d/%d completed, %d failed, best=%s (%.1fs total)",
            completed_count,
            total_pairs,
            failed_count,
            f"{best_pair.generation_model}+{best_pair.reflection_model}" if best_pair else "none",
            grid_runtime,
        )

        return GridSearchResponse(
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
            metric_name=metric_identifier,
            split_counts=split_counts,
            total_pairs=total_pairs,
            completed_pairs=completed_count,
            failed_pairs=failed_count,
            pair_results=pair_results,
            best_pair=best_pair,
            runtime_seconds=round(grid_runtime, 2),
        )

    def validate_grid_search_payload(self, payload: GridSearchRequest) -> None:
        """Validate a GridSearchRequest without executing any optimization.

        User-authored signature and metric code is exec'd inside isolated
        subprocesses via ``safe_exec`` — this runs in the API/worker-engine
        process, so we cannot trust the exec to live here.

        Args:
            payload: The grid-search request to validate.

        Raises:
            ServiceError: When any field fails validation (mapping mismatch,
                unknown module/optimizer, malformed kwargs, etc.).
        """
        intro = validate_signature_code(payload.signature_code)
        require_mapping_matches_signature(payload.column_mapping, intro.input_fields, intro.output_fields)
        require_mapping_columns_in_dataset(payload.column_mapping, payload.dataset)
        metric_info = validate_metric_code(payload.metric_code)
        _require_metric_compatible_with_optimizer(payload.optimizer_name, metric_info.param_names)
        self._get_module_factory(payload.module_name)
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)
        validate_optimizer_signature(optimizer_factory, payload.optimizer_name)
        validate_optimizer_kwargs(
            optimizer_factory,
            payload.optimizer_kwargs,
            payload.optimizer_name,
        )

    def validate_payload(self, payload: RunRequest) -> None:
        """Validate a RunRequest without executing any optimization.

        User-authored signature and metric code is exec'd inside isolated
        subprocesses via ``safe_exec`` — this runs in the API/worker-engine
        process, so we cannot trust the exec to live here.

        Args:
            payload: The run request to validate.

        Raises:
            ServiceError: When any field fails validation (mapping mismatch,
                unknown module/optimizer, malformed kwargs, etc.).
        """
        logger.info(
            "Validating payload for module=%s optimizer=%s dataset_rows=%d",
            payload.module_name,
            payload.optimizer_name,
            len(payload.dataset),
        )

        intro = validate_signature_code(payload.signature_code)
        require_mapping_matches_signature(payload.column_mapping, intro.input_fields, intro.output_fields)
        require_mapping_columns_in_dataset(payload.column_mapping, payload.dataset)
        if payload.module_name.lower() == "react":
            self._validate_react_payload(payload)
        else:
            metric_info = validate_metric_code(payload.metric_code)
            _require_metric_compatible_with_optimizer(payload.optimizer_name, metric_info.param_names)
        self._get_module_factory(payload.module_name)
        optimizer_factory = self._get_optimizer_factory(payload.optimizer_name)
        validate_optimizer_signature(optimizer_factory, payload.optimizer_name)
        validate_optimizer_kwargs(optimizer_factory, payload.optimizer_kwargs, payload.optimizer_name)

    def _validate_react_payload(self, payload: RunRequest) -> None:
        """Validate the react-specific portion of a run payload.

        React is a generic module that scores rollouts with the same standard
        ``(gold, pred, trace, pred_name, pred_trace)`` metric the predict/cot
        path uses, so the 5-arg arity gate applies identically. The only
        react-specific requirement is a ``tool_source`` naming the live MCP
        roster (or the persisted snapshot) the rollouts execute against.

        Args:
            payload: The react run request to validate.

        Raises:
            ServiceError: When ``tool_source`` or ``metric_code`` is missing, or
                when the metric is arity-incompatible with the optimizer.
        """
        if payload.tool_source is None:
            raise ServiceError("react runs require tool_source.")
        if payload.metric_code is None:
            raise ServiceError("react runs require metric_code.")
        metric_info = validate_metric_code(payload.metric_code)
        _require_metric_compatible_with_optimizer(
            payload.optimizer_name, metric_info.param_names
        )

    def _get_module_factory(self, name: str) -> tuple[Callable[..., Any], bool]:
        """Resolve a module factory by name from registry or built-in resolver.

        Args:
            name: The module factory name to resolve.

        Returns:
            A tuple ``(factory, auto_signature)`` where ``auto_signature``
            is False for registry entries and True for resolver-provided ones.

        Raises:
            ServiceError: When ``name`` cannot be resolved.
        """
        try:
            factory = self.registry.get_module(name)
        except UnknownRegistrationError:
            try:
                resolved = resolve_module_factory(name)
            except ResolverError as exc:
                raise ServiceError(str(exc)) from exc
            logger.info(
                "Module '%s' not in ServiceRegistry; resolved via built-in resolver",
                name,
            )
            return resolved
        logger.info("Resolved module '%s' from ServiceRegistry (user-registered)", name)
        return factory, False

    def _get_optimizer_factory(self, name: str) -> Callable[..., Any]:
        """Resolve an optimizer factory by name from registry or built-in resolver.

        Args:
            name: The optimizer factory name to resolve.

        Returns:
            The resolved optimizer factory callable.

        Raises:
            ServiceError: When ``name`` cannot be resolved.
        """
        try:
            return self.registry.get_optimizer(name)
        except UnknownRegistrationError:
            try:
                return resolve_optimizer_factory(name)
            except ResolverError as exc:
                raise ServiceError(str(exc)) from exc
