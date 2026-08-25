"""Tests for the generalist-agent tool gating and approval registry."""

from __future__ import annotations

import asyncio
from typing import cast

import dspy
import litellm
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from core.service_gateway.agents import generalist as generalist_module
from core.service_gateway.agents.code import _agent_error_payload, _SubmitArgExtractor
from core.service_gateway.agents.generalist import (
    ApprovalRegistry,
    GeneralistSig,
    WizardState,
    _needs_approval,
    _TurnAuthoringFlag,
    _wrap_tool_with_approval,
    tools_for,
    validate_wizard_patch_order,
)
from core.storage.models import Base


def test_empty_state_hides_dataset_and_submit_tools() -> None:
    """An empty wizard state hides dataset/code/submit tools, exposing only discovery."""
    allowed = tools_for(WizardState())
    assert "request_code_authoring" not in allowed
    assert "validate_code_validate_code_post" not in allowed
    assert "submit_job_run_post" not in allowed
    assert "submit_grid_search_grid_search_post" not in allowed
    assert "list_models_for_agent" in allowed


def test_comparison_tool_is_not_exposed() -> None:
    """Keep cross-optimization comparison out of every agent state."""
    assert "compare_jobs_optimizations_compare_post" not in tools_for(WizardState())


def test_dataset_ready_unlocks_diagnostics_but_not_code_without_name() -> None:
    """``dataset_ready`` exposes the diagnostic tools, but code authoring stays
    hidden until the run is named — mirroring the wizard's Basics → Code order."""
    allowed = tools_for(WizardState(dataset_ready=True, columns_configured=True))
    assert "validate_code_validate_code_post" in allowed
    assert "profile_datasets_profile_post" in allowed
    assert "request_code_authoring" not in allowed
    assert "submit_job_run_post" not in allowed


def test_named_dataset_ready_unlocks_code_authoring() -> None:
    """Naming the run (with the dataset ready) opens the Signature/Metric step."""
    allowed = tools_for(
        WizardState(job_name="Sentiment run", dataset_ready=True, columns_configured=True)
    )
    assert "request_code_authoring" in allowed
    assert "submit_job_run_post" not in allowed


def test_full_readiness_unlocks_submit() -> None:
    """Full readiness exposes the submit tool."""
    allowed = tools_for(
        WizardState(
            job_name="My run",
            dataset_ready=True,
            columns_configured=True,
            signature_code="class S(dspy.Signature): ...",
            metric_code="def metric(): return 1.0",
            model_configured=True,
            # GEPA (the default optimizer) reflects on a second model, so the
            # gate requires a reflection model alongside the generation one.
            reflection_model_config={"name": "openai/gpt-4o-mini"},
        )
    )
    assert "submit_job_run_post" in allowed
    # Grid search is intentionally NOT exposed to the agent; users reach
    # it through the wizard UI directly. See generalist._READY_TO_SUBMIT_TOOLS.
    assert "submit_grid_search_grid_search_post" not in allowed


def test_workflow_ready_unlocks_submit() -> None:
    """A workflow run readies on an authored graph + metric, not a Signature."""
    base = {
        "job_name": "Graph run",
        "dataset_ready": True,
        "columns_configured": True,
        "module_name": "workflow",
        "metric_code": "def metric(gold, pred, trace, pred_name, pred_trace): return 1.0",
        "model_configured": True,
        "reflection_model_config": {"name": "openai/gpt-4o-mini"},
    }
    # No graph (or an empty one) is not an authored program — submit stays locked
    # even though there is no Signature to satisfy the single-module gate.
    assert "submit_job_run_post" not in tools_for(cast(WizardState, base))
    assert "submit_job_run_post" not in tools_for(
        cast(WizardState, {**base, "workflow": {"nodes": [], "edges": []}})
    )
    ready = {**base, "workflow": {"nodes": [{"id": "input"}, {"id": "out"}], "edges": []}}
    assert "submit_job_run_post" in tools_for(cast(WizardState, ready))


def test_grid_job_type_swaps_submit_tool() -> None:
    """A grid run exposes the grid submit tool and hides the single-run one."""
    base = {
        "job_name": "Sweep",
        "dataset_ready": True,
        "columns_configured": True,
        "signature_code": "class S(dspy.Signature): ...",
        "metric_code": "def metric(gold, pred, trace, pred_name, pred_trace): return 1.0",
        "job_type": "grid_search",
        "use_all_generation_models": True,
        "use_all_reflection_models": True,
    }
    allowed = tools_for(cast(WizardState, base))
    # Exactly one submit surface, matching job_type — never both (the state
    # that made the model oscillate between two submit tools).
    assert "submit_grid_search_grid_search_post" in allowed
    assert "submit_job_run_post" not in allowed


def test_grid_requires_generation_and_reflection_model_lists() -> None:
    """Grid submit stays locked until both model lists (or use_all flags) are set."""
    base = {
        "job_name": "Sweep",
        "dataset_ready": True,
        "columns_configured": True,
        "signature_code": "class S(dspy.Signature): ...",
        "metric_code": "def metric(gold, pred, trace, pred_name, pred_trace): return 1.0",
        "job_type": "grid_search",
    }
    assert "submit_grid_search_grid_search_post" not in tools_for(cast(WizardState, base))
    # A generation list alone is not enough for GEPA (it reflects on a second).
    gen_only = {**base, "generation_models": [{"name": "openai/gpt-4o-mini"}]}
    assert "submit_grid_search_grid_search_post" not in tools_for(cast(WizardState, gen_only))
    both = {
        **base,
        "generation_models": [{"name": "openai/gpt-4o-mini"}],
        "reflection_models": [{"name": "openai/gpt-4o"}],
    }
    assert "submit_grid_search_grid_search_post" in tools_for(cast(WizardState, both))


def test_gepa_without_reflection_model_keeps_submit_locked() -> None:
    """GEPA with a generation model but no reflection model must NOT unlock submit.

    Mirrors the manual wizard's ``reflection_model_required`` gate: submitting
    GEPA without a ``reflection_model_config`` is a known 422, so the agent's
    submit tool stays hidden until the reflection model is set.
    """
    base = WizardState(
        job_name="My run",
        dataset_ready=True,
        columns_configured=True,
        signature_code=(
            'class Sentiment(dspy.Signature):\n'
            '    review: str = dspy.InputField()\n'
            '    label: str = dspy.OutputField()\n'
        ),
        metric_code="def metric(gold, pred, trace=None): return 1.0",
        model_config={"name": "openai/gpt-4o-mini"},
    )
    assert "submit_job_run_post" not in tools_for(base)
    with_reflection = {**base, "reflection_model_config": {"name": "openai/gpt-4o-mini"}}
    assert "submit_job_run_post" in tools_for(cast(WizardState, with_reflection))


# Un-edited templates the frontend seeds into the wizard from the first turn
# (frontend/src/features/submit/lib/build-signature.ts and build-metric.ts,
# fallback branch with no columns mapped). Submitting these triggers the
# server's "Missing inputs: ['input_field']" 400, so the gate must reject them.
_PLACEHOLDER_SIGNATURE = (
    'class MySignature(dspy.Signature):\n'
    '    """Describe the task here."""\n\n'
    '    # inputs\n'
    '    input_field: str = dspy.InputField(desc="")\n\n'
    '    # outputs\n'
    '    output_field: str = dspy.OutputField(desc="")\n'
)
_PLACEHOLDER_METRIC = (
    'def metric(gold: dspy.Example, pred: dspy.Prediction, trace: bool = None,'
    ' pred_name: str = None, pred_trace: list = None) -> dspy.Prediction:\n'
    '    fields = ["output_field"]\n'
    '    total = len(fields)\n'
    '    correct = 0\n'
    '    return dspy.Prediction(score=correct / total if total else 0.0)\n'
)


def test_placeholder_code_keeps_submit_locked() -> None:
    """The seeded placeholder Signature/Metric must NOT unlock submit.

    Everything else is ready (name + dataset + model), but the wizard still
    holds the frontend's un-edited templates rather than authored code.
    """
    allowed = tools_for(
        WizardState(
            job_name="My run",
            dataset_ready=True,
            columns_configured=True,
            signature_code=_PLACEHOLDER_SIGNATURE,
            metric_code=_PLACEHOLDER_METRIC,
            model_configured=True,
        )
    )
    assert "submit_job_run_post" not in allowed
    assert "request_code_authoring" in allowed


def test_authored_code_unlocks_submit() -> None:
    """Replacing the placeholders with real authored code unlocks submit."""
    allowed = tools_for(
        WizardState(
            job_name="My run",
            dataset_ready=True,
            columns_configured=True,
            signature_code=(
                'class Sentiment(dspy.Signature):\n'
                '    """Classify the sentiment of a review."""\n\n'
                '    review: str = dspy.InputField(desc="the review text")\n'
                '    label: str = dspy.OutputField(desc="positive or negative")\n'
            ),
            metric_code=(
                'def metric(gold, pred, trace=None):\n'
                '    return float(gold.label.strip().lower() == str(pred.label).strip().lower())\n'
            ),
            model_configured=True,
            reflection_model_config={"name": "openai/gpt-4o-mini"},
        )
    )
    assert "submit_job_run_post" in allowed


def test_column_mapped_template_unlocks_submit() -> None:
    """A column-mapped template with the default docstring is NOT a placeholder.

    Once the user maps columns, build-signature.ts emits real field names
    (e.g. question/answer) but keeps the default ``"Describe the task here."``
    docstring. That is a valid, submittable Signature — its fields match the
    column mapping — so the gate must not key on the docstring. Regression for
    the agent looping back into ``request_code_authoring`` instead of advancing
    to the Model step after the code card produced a column-mapped signature.
    """
    allowed = tools_for(
        WizardState(
            job_name="My run",
            dataset_ready=True,
            columns_configured=True,
            signature_code=(
                'class MySignature(dspy.Signature):\n'
                '    """Describe the task here."""\n\n'
                '    # inputs\n'
                '    question: str = dspy.InputField(desc="")\n\n'
                '    # outputs\n'
                '    answer: str = dspy.OutputField(desc="")\n'
            ),
            metric_code=(
                'def metric(gold, pred, trace=None):\n'
                '    fields = ["answer"]\n'
                '    return float(getattr(pred, "answer", None) == gold.answer)\n'
            ),
            model_configured=True,
            reflection_model_config={"name": "openai/gpt-4o-mini"},
        )
    )
    assert "submit_job_run_post" in allowed


def test_missing_any_submit_precondition_keeps_submit_hidden() -> None:
    """Submit tools stay hidden when any single readiness criterion is missing.

    Dataset readiness is satisfied by ``columns_configured`` OR ``dataset_ready``
    (the wizard flips ``columns_configured`` once roles are assigned; a freshly
    sample-staged dataset only sets ``dataset_ready``). Flipping both together
    is what hides the submit tools, so this test groups them into one
    "dataset_ready" criterion.
    """
    base: dict[str, object] = {
        "job_name": "My run",
        "dataset_ready": True,
        "columns_configured": True,
        "signature_code": "x",
        "metric_code": "y",
        "model_configured": True,
        "reflection_model_config": {"name": "openai/gpt-4o-mini"},
    }
    # Sanity-check the base is genuinely submittable, so each removal below is
    # the sole reason submit disappears (not a second missing precondition).
    assert "submit_job_run_post" in tools_for(cast(WizardState, base))
    criteria: dict[str, dict[str, object]] = {
        "job_name": {"job_name": ""},
        "dataset_ready": {"dataset_ready": False, "columns_configured": False},
        "signature_code": {"signature_code": ""},
        "metric_code": {"metric_code": ""},
        "model_configured": {"model_configured": False},
        "reflection_model_config": {"reflection_model_config": {}},
    }
    for label, overrides in criteria.items():
        state = {**base, **overrides}
        assert "submit_job_run_post" not in tools_for(cast(WizardState, state)), (
            f"submit_job leaked with {label} missing"
        )


def test_order_allows_name_first() -> None:
    """Setting ``job_name`` on an empty wizard is in order (Basics is first)."""
    assert validate_wizard_patch_order({"job_name": "My run"}, WizardState()) is None


def test_order_rejects_dataset_roles_before_name() -> None:
    """Column roles (Data) require the run to be named first (Basics)."""
    err = validate_wizard_patch_order(
        {"column_roles": {"q": "input", "a": "output"}}, WizardState()
    )
    assert err is not None
    assert "Basics" in err


def test_order_allows_name_and_dataset_in_one_patch() -> None:
    """A single patch may set the name AND column roles — each satisfies its own step."""
    patch = {"job_name": "My run", "column_roles": {"q": "input", "a": "output"}}
    assert validate_wizard_patch_order(patch, WizardState()) is None


def test_order_rejects_params_before_dataset() -> None:
    """Params can't be set before the dataset is ready, even with a name set."""
    err = validate_wizard_patch_order(
        {"optimizer_name": "gepa"}, WizardState(job_name="My run")
    )
    assert err is not None
    assert "Data" in err


def test_order_rejects_model_before_code() -> None:
    """The Model step is gated on the Code step being authored first."""
    state = WizardState(job_name="My run", dataset_ready=True, columns_configured=True)
    err = validate_wizard_patch_order({"model_config": {"name": "openai/gpt-4o"}}, state)
    assert err is not None
    assert "Code" in err


def test_order_allows_model_after_code() -> None:
    """With name + dataset + code present, setting the model is in order."""
    state = WizardState(
        job_name="My run",
        dataset_ready=True,
        columns_configured=True,
        signature_code="class S(dspy.Signature): ...",
        metric_code="def m(): return 1.0",
    )
    assert (
        validate_wizard_patch_order({"model_config": {"name": "openai/gpt-4o"}}, state)
        is None
    )


def test_order_allows_model_after_column_mapped_template() -> None:
    """Setting the model is in order once a column-mapped template is present.

    Regression for the agent rejecting ``model_config`` with "Do these first:
    Code" — and looping back into authoring — when the Code step already held a
    valid column-mapped signature that merely kept the default docstring.
    """
    state = WizardState(
        job_name="My run",
        dataset_ready=True,
        columns_configured=True,
        signature_code=(
            'class MySignature(dspy.Signature):\n'
            '    """Describe the task here."""\n\n'
            '    question: str = dspy.InputField(desc="")\n'
            '    answer: str = dspy.OutputField(desc="")\n'
        ),
        metric_code='def metric(gold, pred, trace=None):\n    fields = ["answer"]\n    return 1.0\n',
    )
    assert (
        validate_wizard_patch_order({"model_config": {"name": "openai/gpt-5.4-nano"}}, state)
        is None
    )


def test_order_ignores_unmapped_fields() -> None:
    """A patch with no step-mapped fields passes (code fields are rejected elsewhere)."""
    assert validate_wizard_patch_order({"signature_code": "x"}, WizardState()) is None


def test_always_tools_include_discovery_and_post_submit() -> None:
    """The always-on toolset includes discovery and post-submit lifecycle tools."""
    allowed = tools_for(WizardState())
    assert "list_models_for_agent" in allowed
    assert "get_registry_snapshot_registry_get" in allowed
    assert "list_jobs_optimizations_get" in allowed
    assert "cancel_job_optimizations" in allowed
    assert "rename_job_optimizations" in allowed


def test_read_only_reach_tools_always_available() -> None:
    """Wallet and tagging-session reads are always-on and never gate."""
    allowed = tools_for(WizardState())
    assert "get_wallet_for_agent" in allowed
    assert "list_tagging_sessions_for_agent" in allowed
    for name in ("get_wallet_for_agent", "list_tagging_sessions_for_agent"):
        assert _needs_approval(name, "ask") is False


def test_dataset_library_tools_always_available() -> None:
    """The saved-dataset library read + by-reference picker are always exposed."""
    allowed = tools_for(WizardState())
    assert "list_datasets_for_agent" in allowed
    assert "request_user_dataset_from_library" in allowed


def test_pair_inference_trigger_always_available() -> None:
    """The grid-search pair inference trigger is always exposed and never gates."""
    assert "request_user_pair_inference" in tools_for(WizardState())
    assert _needs_approval("request_user_pair_inference", "ask") is False


def test_yolo_never_gates() -> None:
    """Yolo trust-mode never gates any tool."""
    for name in ("delete_job_optimizations", "submit_job_run_post", "rename_job_optimizations"):
        assert _needs_approval(name, "yolo") is False


def test_ask_gates_every_mutation() -> None:
    """Ask trust-mode gates every mutating tool."""
    assert _needs_approval("delete_job_optimizations", "ask") is True
    assert _needs_approval("rename_job_optimizations", "ask") is True
    assert _needs_approval("submit_job_run_post", "ask") is True


def test_auto_safe_gates_only_destructive() -> None:
    """Auto-safe gates only destructive operations."""
    assert _needs_approval("rename_job_optimizations", "auto_safe") is False
    assert _needs_approval("toggle_pin_job_optimizations", "auto_safe") is False
    assert _needs_approval("delete_job_optimizations", "auto_safe") is True
    assert _needs_approval("submit_job_run_post", "auto_safe") is True


def _make_fake_tool(name: str, return_value: str = "ok") -> dspy.Tool:
    """Build a ``dspy.Tool`` whose async ``func`` returns the given value."""
    async def func(**kwargs):
        return return_value

    return dspy.Tool(func=func, name=name, desc="test tool", args={}, arg_types={}, arg_desc={})


@pytest.mark.asyncio
async def test_wrap_bypasses_when_no_approval_needed() -> None:
    """Wrapped tool runs straight through when no approval is needed."""
    events: list[dict] = []
    registry = ApprovalRegistry()
    tool = _wrap_tool_with_approval(
        _make_fake_tool("rename_job_optimizations", return_value="renamed"),
        trust_mode="auto_safe",
        registry=registry,
        emit=events.append,
        outer_loop=asyncio.get_running_loop(),
    )
    # Drive the async body directly: the sync ``__call__`` schedules onto
    # ``outer_loop`` via ``run_coroutine_threadsafe`` for DSPy's worker-thread
    # dispatch, but inside the test loop we exercise the same logic by awaiting
    # ``_async_body`` so we don't deadlock blocking on the running loop.
    result = await tool.func._async_body()
    assert result == "renamed"
    event_names = [e["event"] for e in events]
    assert "pending_approval" not in event_names
    assert event_names == ["tool_start", "tool_end"]


@pytest.mark.asyncio
async def test_wrap_emits_pending_and_runs_on_approve() -> None:
    """Wrapped tool emits ``pending_approval`` and runs once approved."""
    events: list[dict] = []
    registry = ApprovalRegistry()
    tool = _wrap_tool_with_approval(
        _make_fake_tool("delete_job_optimizations", return_value="deleted"),
        trust_mode="ask",
        registry=registry,
        emit=events.append,
        outer_loop=asyncio.get_running_loop(),
    )
    call_task = asyncio.create_task(tool.func._async_body())
    for _ in range(20):
        await asyncio.sleep(0)
        if any(e["event"] == "pending_approval" for e in events):
            break
    pending = next((e for e in events if e["event"] == "pending_approval"), None)
    assert pending is not None
    call_id = pending["data"]["id"]
    assert registry.resolve(call_id, True) is True
    result = await call_task
    assert result == "deleted"
    resolved = next((e for e in events if e["event"] == "approval_resolved"), None)
    assert resolved is not None
    assert resolved["data"]["approved"] is True


@pytest.mark.asyncio
async def test_denial_returns_observation_not_exception() -> None:
    """A denied approval surfaces a string observation instead of raising."""
    events: list[dict] = []
    registry = ApprovalRegistry()
    tool = _wrap_tool_with_approval(
        _make_fake_tool("submit_job_run_post", return_value="should not run"),
        trust_mode="ask",
        registry=registry,
        emit=events.append,
        outer_loop=asyncio.get_running_loop(),
    )
    call_task = asyncio.create_task(tool.func._async_body())
    for _ in range(20):
        await asyncio.sleep(0)
        if events:
            break
    call_id = events[0]["data"]["id"]
    registry.resolve(call_id, False)
    result = await call_task
    assert result == "User declined"


def _make_recording_tool(name: str) -> tuple[dspy.Tool, dict]:
    """Build a ``dspy.Tool`` whose async ``func`` records the kwargs it receives.

    Args:
        name: Registered tool name.

    Returns:
        The tool and the dict its ``func`` populates with received kwargs.
    """
    seen: dict = {}

    async def func(**kwargs):
        seen.update(kwargs)
        return "ok"

    tool = dspy.Tool(func=func, name=name, desc="test tool", args={}, arg_types={}, arg_desc={})
    return tool, seen


@pytest.mark.asyncio
async def test_submit_injects_validated_code_over_agent_supplied() -> None:
    """Submit sources Signature/Metric from the snapshot, discarding agent code."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    wizard_state = cast(
        WizardState,
        {
            "signature_code": "class Good(dspy.Signature): ...",
            "metric_code": "def good(gold, pred, trace, pred_name, pred_trace): return 1.0",
            "staged_dataset_id": "ds_123",
        },
    )
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        staged_dataset_id="ds_123",
        wizard_state=wizard_state,
    )
    await tool.func._async_body(
        signature_code="BROKEN {",
        metric_code="def m(example, prediction, trace): return 1.0",
    )
    assert seen["signature_code"] == "class Good(dspy.Signature): ..."
    assert seen["metric_code"] == (
        "def good(gold, pred, trace, pred_name, pred_trace): return 1.0"
    )
    assert seen["staged_dataset_id"] == "ds_123"


@pytest.mark.asyncio
async def test_submit_without_snapshot_code_leaves_agent_args() -> None:
    """With no authored code in the snapshot, submit args pass through unchanged."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {}),
    )
    await tool.func._async_body(signature_code="agent_sig", metric_code="agent_metric")
    assert seen["signature_code"] == "agent_sig"
    assert seen["metric_code"] == "agent_metric"


@pytest.mark.asyncio
async def test_submit_injects_wizard_description() -> None:
    """Submit carries the wizard's job_description — stripped and clipped to 280."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {"job_description": "  " + "x" * 300 + "  "}),
    )
    await tool.func._async_body()
    assert seen["description"] == "x" * 280


@pytest.mark.asyncio
async def test_submit_keeps_agent_supplied_description() -> None:
    """An agent-supplied description wins over the wizard snapshot's."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {"job_description": "from the wizard"}),
    )
    await tool.func._async_body(description="from the agent")
    assert seen["description"] == "from the agent"


@pytest.mark.asyncio
async def test_submit_injects_source_dataset_id() -> None:
    """A library-dataset run injects source_dataset_id when no dataset is supplied."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        source_dataset_id="lib_42",
        wizard_state=cast(WizardState, {"source_dataset_id": "lib_42"}),
    )
    await tool.func._async_body()
    assert seen["source_dataset_id"] == "lib_42"


@pytest.mark.asyncio
async def test_submit_staged_dataset_wins_over_source() -> None:
    """Staged and library ids are mutually exclusive: staged injects, source does not."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        staged_dataset_id="ds_1",
        source_dataset_id="lib_1",
        wizard_state=cast(WizardState, {}),
    )
    await tool.func._async_body()
    assert seen["staged_dataset_id"] == "ds_1"
    assert "source_dataset_id" not in seen


@pytest.mark.asyncio
async def test_submit_injects_workflow_graph_over_signature() -> None:
    """A workflow submit ships the authored graph and drops signature_code."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    graph = {"nodes": [{"id": "input"}, {"id": "out"}], "edges": []}
    wizard_state = cast(
        WizardState,
        {
            "module_name": "workflow",
            "workflow": graph,
            "metric_code": "def good(gold, pred, trace, pred_name, pred_trace): return 1.0",
            "staged_dataset_id": "ds_9",
        },
    )
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        staged_dataset_id="ds_9",
        wizard_state=wizard_state,
    )
    # The agent leaves a stale Signature in its args; the workflow snapshot wins.
    await tool.func._async_body(signature_code="class Leftover(dspy.Signature): ...")
    assert seen["module_name"] == "workflow"
    assert seen["workflow"] == graph
    assert "signature_code" not in seen
    assert seen["metric_code"] == "def good(gold, pred, trace, pred_name, pred_trace): return 1.0"
    assert seen["staged_dataset_id"] == "ds_9"


@pytest.mark.asyncio
async def test_submit_defaults_to_private() -> None:
    """Submit forces ``is_private`` to private when the snapshot never set it."""
    tool, seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {}),
    )
    await tool.func._async_body()
    assert seen["is_private"] is True


@pytest.mark.asyncio
async def test_submit_respects_explicit_public() -> None:
    """A snapshot ``is_private=False`` (user asked for public) reaches submit."""
    tool, seen = _make_recording_tool("submit_grid_search_grid_search_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {"is_private": False}),
    )
    await tool.func._async_body()
    assert seen["is_private"] is False


@pytest.mark.asyncio
async def test_non_submit_tool_does_not_inject_code() -> None:
    """Code injection is scoped to submit tools; other tools are untouched."""
    tool, seen = _make_recording_tool("update_wizard_state")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {"signature_code": "snap", "metric_code": "snap"}),
    )
    await tool.func._async_body(job_name="x")
    assert "signature_code" not in seen
    assert "metric_code" not in seen


@pytest.mark.asyncio
async def test_profile_injects_staged_dataset_id() -> None:
    """Profiling a staged dataset gets the opaque id injected so the backend rehydrates."""
    tool, seen = _make_recording_tool("profile_datasets_profile_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        staged_dataset_id="ds_123",
        wizard_state=cast(WizardState, {}),
    )
    await tool.func._async_body(column_mapping={"inputs": {}, "outputs": {}})
    assert seen["staged_dataset_id"] == "ds_123"


@pytest.mark.asyncio
async def test_profile_inline_dataset_not_overridden_by_staged_id() -> None:
    """An inline dataset on the profile call suppresses staged-id injection."""
    tool, seen = _make_recording_tool("profile_datasets_profile_post")
    _wrap_tool_with_approval(
        tool,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        staged_dataset_id="ds_123",
        wizard_state=cast(WizardState, {}),
    )
    await tool.func._async_body(
        dataset=[{"q": "x"}], column_mapping={"inputs": {}, "outputs": {}}
    )
    assert "staged_dataset_id" not in seen


def test_registry_resolve_unknown_returns_false() -> None:
    """Resolving an unknown call id returns ``False``."""
    registry = ApprovalRegistry()
    assert registry.resolve("does-not-exist", True) is False


def test_submit_arg_extractor_streams_value_incrementally() -> None:
    """Incremental chunks of a submit tool_calls payload yield growing deltas."""
    ext = _SubmitArgExtractor("assistant_message")
    deltas: list[str] = []
    for chunk in (
        '{"tool_calls": [',
        '{"name": "submit", "args": ',
        '{"assistant_message": "',
        "Shal",
        "om, ",
        "Gilad",
        '!"}}]}',
    ):
        delta = ext.feed(chunk)
        if delta:
            deltas.append(delta)
    assert "".join(deltas) == "Shalom, Gilad!"


def test_submit_arg_extractor_ignores_non_submit_calls() -> None:
    """A non-submit tool call produces no delta, and reset clears the buffer."""
    ext = _SubmitArgExtractor("assistant_message")
    assert ext.feed('{"tool_calls": [{"name": "list_models", "args": {}}]}') is None
    ext.reset()
    delta = ext.feed(
        '{"tool_calls": [{"name": "submit", "args": {"assistant_message": "Done!"}}]}'
    )
    assert delta == "Done!"


def test_submit_arg_extractor_handles_malformed_json() -> None:
    """A partial / malformed chunk returns ``None`` without raising."""
    ext = _SubmitArgExtractor("reply")
    assert ext.feed('{"tool_calls": [{"name": "submit') is None


def test_submit_arg_extractor_picks_submit_among_parallel_calls() -> None:
    """Parallel tool calls including submit still resolve to the submit arg."""
    ext = _SubmitArgExtractor("reply")
    parallel = (
        '{"tool_calls": [{"name": "foo", "args": {}}, '
        '{"name": "submit", "args": {"reply": "yo"}}]}'
    )
    assert ext.feed(parallel) == "yo"


def test_submit_arg_extractor_is_idempotent_on_repeat_feed() -> None:
    """Re-feeding the same buffer or an empty chunk yields no new delta."""
    ext = _SubmitArgExtractor("reply")
    full = '{"tool_calls": [{"name": "submit", "args": {"reply": "hi"}}]}'
    assert ext.feed(full) == "hi"
    assert ext.feed("") is None


@pytest.mark.asyncio
async def test_submit_blocked_when_authoring_requested_same_turn() -> None:
    """A submit that follows request_code_authoring in one turn is denied.

    ``request_code_authoring`` writes its authored code back to the wizard
    asynchronously, so the new Signature/Metric is not in this turn's snapshot.
    The shared turn flag must cause the same-turn submit to short-circuit with
    a denial observation rather than ship stale code into a doomed run.
    """
    flag = _TurnAuthoringFlag()
    authoring, _ = _make_recording_tool("request_code_authoring")
    submit, submit_seen = _make_recording_tool("submit_job_run_post")
    common = {
        "trust_mode": "yolo",
        "registry": ApprovalRegistry(),
        "emit": lambda _e: None,
        "outer_loop": asyncio.get_running_loop(),
        "wizard_state": cast(WizardState, {}),
        "authoring_flag": flag,
    }
    _wrap_tool_with_approval(authoring, **common)
    _wrap_tool_with_approval(submit, **common)

    await authoring.func._async_body(goal="fix the signature field names")
    result = await submit.func._async_body(name="My run")

    assert "Submit blocked" in result
    assert submit_seen == {}


@pytest.mark.asyncio
async def test_submit_runs_when_no_authoring_this_turn() -> None:
    """Submit runs normally when request_code_authoring did NOT fire this turn.

    The happy path — submit on a later turn, once the authored code is in the
    snapshot — must be unaffected by the same-turn backstop.
    """
    flag = _TurnAuthoringFlag()
    submit, submit_seen = _make_recording_tool("submit_job_run_post")
    _wrap_tool_with_approval(
        submit,
        trust_mode="yolo",
        registry=ApprovalRegistry(),
        emit=lambda _e: None,
        outer_loop=asyncio.get_running_loop(),
        wizard_state=cast(WizardState, {}),
        authoring_flag=flag,
    )
    result = await submit.func._async_body(name="My run")
    assert result == "ok"
    assert submit_seen["name"] == "My run"


def test_system_prompt_forbids_submit_in_authoring_turn() -> None:
    """The system prompt must keep the never-submit-in-an-authoring-turn rule.

    Guards against a future prompt edit silently dropping the ordering rule that
    is the primary defense for this bug.
    """
    prompt = GeneralistSig.__doc__ or ""
    assert "NEVER call ``submit_job_run_post`` in the SAME turn as" in prompt
    assert "request_code_authoring" in prompt


def _approval_engine():
    """In-memory shared-across-threads SQLite engine with the ORM schema."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return engine


def test_resolve_or_persist_without_engine_rejects_unknown_id() -> None:
    """Store-less registries keep the old contract: unknown id is unresolved."""
    registry = ApprovalRegistry()
    assert registry.resolve_or_persist("nope", True) is False


@pytest.mark.asyncio
async def test_confirm_on_another_replica_reaches_waiting_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decision persisted by one replica resolves another replica's wait loop."""
    monkeypatch.setattr(generalist_module, "_DURABLE_POLL_SECONDS", 0.01)
    engine = _approval_engine()
    streaming = ApprovalRegistry()
    streaming.bind_engine(engine)
    confirming = ApprovalRegistry()
    confirming.bind_engine(engine)

    fut = streaming.register("call-cross")
    # The confirming replica holds no future for this id — it must persist.
    assert confirming.resolve_or_persist("call-cross", True) is True
    assert await streaming.wait_for_decision("call-cross", fut) is True
    # The decision row is consumed on delivery.
    assert streaming._take_durable("call-cross") is None


@pytest.mark.asyncio
async def test_wait_for_decision_expires_as_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirm that never arrives expires the wait as a decline, not a hang."""
    monkeypatch.setattr(generalist_module, "APPROVAL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(generalist_module, "_DURABLE_POLL_SECONDS", 0.01)
    registry = ApprovalRegistry()
    fut = registry.register("call-lost")
    assert await registry.wait_for_decision("call-lost", fut) is False


@pytest.mark.asyncio
async def test_local_resolve_still_wins_instantly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-process fast path resolves without touching the durable store."""
    registry = ApprovalRegistry()
    fut = registry.register("call-local")
    task = asyncio.create_task(registry.wait_for_decision("call-local", fut))
    await asyncio.sleep(0)
    assert registry.resolve_or_persist("call-local", False) is True
    assert await task is False


def test_agent_error_payload_flags_litellm_context_overflow() -> None:
    """litellm's typed context-window error carries the machine code."""
    exc = litellm.ContextWindowExceededError(
        "The prompt exceeds the model limit", model="gpt-4o", llm_provider="openai"
    )
    payload = _agent_error_payload(exc)
    assert payload["code"] == "context_too_long"
    assert payload["error"]


def test_agent_error_payload_flags_provider_overflow_message() -> None:
    """An untyped provider 400 mentioning context length is classified too."""
    exc = RuntimeError(
        "BadRequestError: This model's maximum context length is 128000 tokens, "
        "however you requested 191694 tokens"
    )
    assert _agent_error_payload(exc)["code"] == "context_too_long"


def test_agent_error_payload_walks_groups_and_causes() -> None:
    """The classifier sees through exception groups and __cause__ chains."""
    inner = ValueError("prompt is too long: 250000 tokens > 200000 maximum")
    wrapper = RuntimeError("agent turn failed")
    wrapper.__cause__ = inner
    group = BaseExceptionGroup("unhandled errors in a TaskGroup", [wrapper])
    assert _agent_error_payload(group)["code"] == "context_too_long"


def test_agent_error_payload_plain_error_has_no_code() -> None:
    """Unclassified failures keep the text-only payload."""
    payload = _agent_error_payload(RuntimeError("boom"))
    assert payload == {"error": "RuntimeError: boom"}
