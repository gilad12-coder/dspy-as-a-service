"""Unit tests for the pure pieces of the react /run core.

Covers the ``dataset_snapshot`` branch of ``resolve_react_tools`` (roster
rebuild from the reserved sidecar / persisted specs, including per-tool
severity) and the ``gepa.optimize`` call contract of ``run_react_optimization``
verified with the live model + adapter stubbed out. The full model-driven path
(live MCP tools + real ``gepa.optimize``) is exercised by the gateway
integration suite.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from core.config import settings
from core.models.submissions import ToolSource
from core.service_gateway.optimization.core import _tool_to_snapshot_spec
from core.service_gateway.optimization.timing import (
    STAGE_BASELINE,
    STAGE_EVALUATION,
    STAGE_TRAINING,
    GenLMTimingCallback,
)
from core.service_gateway.optimization.training_ground import run_react as run_react_mod
from core.service_gateway.optimization.training_ground.registry import hash_tool_schema
from core.service_gateway.optimization.training_ground.run_react import (
    _JobLogGepaLogger,
    _severity_from_annotations,
    resolve_react_tools,
    run_react_optimization,
    set_tool_severity,
    tool_severity,
)


def test_resolve_react_tools_dataset_snapshot_rebuilds_roster() -> None:
    """A dataset snapshot sidecar rebuilds a dspy.Tool roster + schema hashes."""
    snapshot = [
        {
            "name": "search",
            "description": "Search the web.",
            "args": {"q": {"type": "string", "description": "query"}},
        },
        {"name": "lookup", "description": "Look up an id.", "args": {}},
        {"name": "submit", "description": "reserved", "args": {}},
    ]
    dataset = [
        {"question": "x", "__tool_snapshot__": snapshot},
        {"question": "y"},
    ]
    tool_source = ToolSource(kind="dataset_snapshot")

    tools, hashes = resolve_react_tools(
        tool_source, signature_cls=object, settings=settings, dataset=dataset
    )

    names = [tool.name for tool in tools]
    assert names == ["search", "lookup"]
    assert "submit" not in names
    search_tool = next(tool for tool in tools if tool.name == "search")
    assert search_tool.desc == "Search the web."
    assert set(hashes) == {"search", "lookup"}
    assert hashes["search"] == hash_tool_schema(search_tool)


def test_severity_from_annotations_maps_mcp_hints() -> None:
    """MCP annotation hints map to the three approval tiers, else ``None``."""
    read_only = SimpleNamespace(readOnlyHint=True, destructiveHint=None)
    destructive = SimpleNamespace(readOnlyHint=False, destructiveHint=True)
    mutating = SimpleNamespace(readOnlyHint=False, destructiveHint=False)
    open_world = SimpleNamespace(readOnlyHint=None, destructiveHint=None)

    assert _severity_from_annotations(read_only) == "info"
    assert _severity_from_annotations(destructive) == "destructive"
    assert _severity_from_annotations(mutating) == "warning"
    assert _severity_from_annotations(open_world) is None
    assert _severity_from_annotations(None) is None


def test_tool_to_snapshot_spec_serializes_severity() -> None:
    """A tool's stashed severity is serialized into its spec; omitted when unset."""

    def _noop(**_kwargs: object) -> None:
        """Placeholder body for a snapshot tool."""

    _noop.__name__ = "wipe"
    marked = dspy.Tool(_noop, name="wipe", desc="Delete it.", args=None)
    set_tool_severity(marked, "destructive")
    unmarked = dspy.Tool(_noop, name="peek", desc="Read it.", args=None)

    assert _tool_to_snapshot_spec(marked)["severity"] == "destructive"
    assert "severity" not in _tool_to_snapshot_spec(unmarked)


def test_resolve_react_tools_dataset_snapshot_carries_severity() -> None:
    """A snapshot spec's ``severity`` round-trips onto the rebuilt tool."""
    snapshot = [
        {"name": "wipe", "description": "Delete it.", "args": {}, "severity": "destructive"},
        {"name": "peek", "description": "Read it.", "args": {}},
    ]
    dataset = [{"__tool_snapshot__": snapshot}]
    tool_source = ToolSource(kind="dataset_snapshot")

    tools, _ = resolve_react_tools(
        tool_source, signature_cls=object, settings=settings, dataset=dataset
    )

    by_name = {tool.name: tool for tool in tools}
    assert tool_severity(by_name["wipe"]) == "destructive"
    assert tool_severity(by_name["peek"]) is None


def test_resolve_react_tools_dataset_snapshot_honours_tool_filter() -> None:
    """``tool_filter`` keeps and reorders the snapshot roster."""
    snapshot = [
        {"name": "alpha", "description": "a", "args": {}},
        {"name": "beta", "description": "b", "args": {}},
        {"name": "gamma", "description": "c", "args": {}},
    ]
    dataset = [{"__tool_snapshot__": snapshot}]
    tool_source = ToolSource(kind="dataset_snapshot", tool_filter=["gamma", "alpha"])

    tools, _ = resolve_react_tools(
        tool_source, signature_cls=object, settings=settings, dataset=dataset
    )

    assert [tool.name for tool in tools] == ["gamma", "alpha"]


def test_resolve_react_tools_dataset_snapshot_without_sidecar_raises() -> None:
    """A dataset_snapshot source with no sidecar is a hard error."""
    tool_source = ToolSource(kind="dataset_snapshot")
    with pytest.raises(ValueError, match="carried no tools"):
        resolve_react_tools(
            tool_source, signature_cls=object, settings=settings, dataset=[{"q": "x"}]
        )


def test_resolve_react_tools_unknown_kind_raises() -> None:
    """An unrecognised tool_source kind fails fast."""
    bad_source = SimpleNamespace(kind="carrier_pigeon", tool_filter=None)
    with pytest.raises(ValueError, match=r"Unknown tool_source\.kind"):
        resolve_react_tools(bad_source, signature_cls=object, settings=settings)


def test_resolve_react_tools_accepts_persisted_dict_snapshot() -> None:
    """A persisted dict tool_source (serve path) rebuilds the roster without a dataset."""
    persisted = {
        "kind": "dataset_snapshot",
        "tool_filter": ["search"],
        "tool_snapshot": [
            {
                "name": "search",
                "description": "Search the web.",
                "args": {"q": {"type": "string"}},
            },
            {"name": "lookup", "description": "Look up an id.", "args": {}},
        ],
    }

    tools, hashes = resolve_react_tools(
        persisted, signature_cls=object, settings=settings, dataset=None
    )

    assert [tool.name for tool in tools] == ["search"]
    assert set(hashes) == {"search"}
    assert hashes["search"] == hash_tool_schema(tools[0])


class _StopAfterOptimizeError(Exception):
    """Sentinel raised by the gepa.optimize spy to halt before the scoring tail."""


def _stub_optimize_prelude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the seed/adapter construction shared by the optimize-contract tests.

    Replaces ``RetryingReActV2`` (seed program), ``seed_candidate_from_program``,
    ``_build_feedback_map``, and the gepa-package ``DspyAdapter`` so
    ``run_react_optimization`` reaches ``gepa.optimize`` without a live model or
    a real tool-aware adapter.

    Args:
        monkeypatch: Pytest fixture used to install the stubs on ``run_react``.
    """
    monkeypatch.setattr(run_react_mod, "RetryingReActV2", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        run_react_mod, "seed_candidate_from_program", lambda program: {"seed": True}
    )
    monkeypatch.setattr(run_react_mod, "_build_feedback_map", lambda program, metric: {})
    monkeypatch.setattr(run_react_mod, "DspyAdapter", lambda **k: MagicMock())


def test_job_log_gepa_logger_routes_to_logging(caplog: pytest.LogCaptureFixture) -> None:
    """``_JobLogGepaLogger.log`` forwards a GEPA line to its sink at INFO level."""
    sink = logging.getLogger("test.gepa.sink")
    line = "Iteration 3: Valset score for new program: 0.42"
    with caplog.at_level(logging.INFO, logger="test.gepa.sink"):
        _JobLogGepaLogger(sink).log(line)
    assert line in caplog.text


def test_run_react_optimization_passes_progress_bar_and_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_react_optimization`` wires ``display_progress_bar`` + a logging bridge into gepa.optimize.

    Guards the regression where react runs emitted no ``optimizer_progress`` (so
    the tqdm bar stayed empty) and no GEPA ``Iteration N`` lines into job_logs
    (so the score chart stayed empty) because both ``gepa.optimize`` arguments
    were omitted. The optimize call is stubbed, so no live model runs.
    """
    captured: dict[str, object] = {}

    def _spy_optimize(**kwargs: object) -> None:
        """Record the optimize kwargs, then abort before the scoring tail."""
        captured.update(kwargs)
        raise _StopAfterOptimizeError

    _stub_optimize_prelude(monkeypatch)
    monkeypatch.setattr(run_react_mod, "_scores_and_outputs", lambda *a, **k: ([1.0], [{}]))
    monkeypatch.setattr(run_react_mod.gepa, "optimize", _spy_optimize)

    with pytest.raises(_StopAfterOptimizeError):
        run_react_optimization(
            signature_cls=object,
            tools=[],
            schema_hashes={},
            metric=MagicMock(),
            train=[],
            val=[],
            test=[],
            student_lm=MagicMock(),
            reflection_lm=MagicMock(),
        )

    assert captured["display_progress_bar"] is True
    wired_logger = captured["logger"]
    assert isinstance(wired_logger, _JobLogGepaLogger)
    # The sink must sit under core.service_gateway.optimization so the worker's
    # JobLogHandler (attached to that parent) captures GEPA's iteration lines.
    assert wired_logger._sink.name.startswith("core.service_gateway.optimization")


def test_run_react_optimization_passes_target_score_stopper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """React GEPA runs receive the optional validation target callback."""
    captured: dict[str, object] = {}

    def _spy_optimize(**kwargs: object) -> None:
        """Record the optimize kwargs, then abort before the scoring tail."""
        captured.update(kwargs)
        raise _StopAfterOptimizeError

    _stub_optimize_prelude(monkeypatch)
    monkeypatch.setattr(run_react_mod, "_scores_and_outputs", lambda *a, **k: ([1.0], [{}]))
    monkeypatch.setattr(run_react_mod.gepa, "optimize", _spy_optimize)

    with pytest.raises(_StopAfterOptimizeError):
        run_react_optimization(
            signature_cls=object,
            tools=[],
            schema_hashes={},
            metric=MagicMock(),
            train=[],
            val=[],
            test=[],
            student_lm=MagicMock(),
            reflection_lm=MagicMock(),
            target_score=85,
        )

    stopper = captured["stop_callbacks"][0]
    assert stopper.threshold == pytest.approx(0.85)


def test_run_react_optimization_buckets_lm_activity_per_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Student rollouts are attributed to the baseline/training/evaluation stages.

    Guards the regression where react runs left the LM-activity panel empty:
    ``run_react_optimization`` ran its three model-call regions without a
    ``track_stage`` wrapper, so the per-stage buckets the panel reads stayed
    empty and the tab rendered "no activity". A real
    :class:`GenLMTimingCallback` is threaded in here and each stubbed region
    simulates one matched student call; the callback must then report exactly
    one call bucketed under each of the three stages.
    """
    student_lm = MagicMock()
    gen_timing = GenLMTimingCallback(student_lm)
    call_ids = iter(range(100))

    def _simulate_student_call() -> None:
        """Record one matched student-LM call against whichever stage is active."""
        call_id = f"call-{next(call_ids)}"
        gen_timing.on_lm_start(call_id, student_lm, {})
        gen_timing.on_lm_end(call_id, {})

    def _fake_scores_and_outputs(*_args: object, **_kwargs: object) -> tuple[list, list]:
        """Stand in for ``_scores_and_outputs`` with one simulated student call."""
        _simulate_student_call()
        return [1.0], [{}]

    def _fake_optimize(**_kwargs: object) -> MagicMock:
        """Stand in for ``gepa.optimize`` with one simulated training call."""
        _simulate_student_call()
        return MagicMock()

    _stub_optimize_prelude(monkeypatch)
    monkeypatch.setattr(run_react_mod, "_scores_and_outputs", _fake_scores_and_outputs)
    monkeypatch.setattr(run_react_mod.gepa, "optimize", _fake_optimize)
    monkeypatch.setattr(run_react_mod, "_best_candidate", lambda result: {"opt": True})
    monkeypatch.setattr(run_react_mod, "extract_program_state", lambda program: {})
    monkeypatch.setattr(run_react_mod, "_candidate_tool_descriptions", lambda c: {})
    monkeypatch.setattr(run_react_mod, "_candidate_tool_arg_descriptions", lambda c: {})
    monkeypatch.setattr(run_react_mod, "_candidate_tool_names", lambda c: {})

    run_react_optimization(
        signature_cls=object,
        tools=[],
        schema_hashes={},
        metric=MagicMock(),
        train=[],
        val=[],
        test=[],
        student_lm=student_lm,
        reflection_lm=MagicMock(),
        timing_callbacks=(gen_timing,),
    )

    stages = gen_timing.stage_summary()
    assert set(stages) == {STAGE_BASELINE, STAGE_TRAINING, STAGE_EVALUATION}
    assert all(calls == 1 for calls, _avg in stages.values())
