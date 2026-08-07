"""Tests for workflow program compilation, execution, and deep validation."""

from __future__ import annotations

import dspy
import pytest
from dspy.teleprompt.gepa.gepa_flex_utils import enumerate_flex_submodules
from pydantic import ValidationError

from core.exceptions import ServiceError
from core.models import WorkflowSpec, workflow_tool_users
from core.registry.resolvers import ResolverError
from core.service_gateway.optimization import workflow as workflow_module
from core.service_gateway.optimization.workflow import (
    WORKFLOW_NODE_ATTR_PREFIX,
    WorkflowNodeExecutionError,
    WorkflowProgram,
    build_workflow_program,
    capture_node_traces,
    validate_workflow,
)

_SIG = (
    "import dspy\n"
    "class Summarize(dspy.Signature):\n"
    "    text: str = dspy.InputField()\n"
    "    summary: str = dspy.OutputField()\n"
)

_TRANSFORM = "def transform(text):\n    return {'shout': text.upper()}\n"


class _FakeModule(dspy.Module):
    """A stand-in signature module returning canned prediction fields."""

    def __init__(self, fn):
        """Store the fabrication function.

        Args:
            fn: Callable mapping input kwargs to an output-field dict.
        """
        super().__init__()
        self._fn = fn

    def forward(self, **kwargs):
        """Fabricate a prediction from the stored function."""
        return dspy.Prediction(**self._fn(**kwargs))


def _spec(nodes: list[dict], edges: list[dict]) -> WorkflowSpec:
    """Parse a spec from node/edge dicts.

    Args:
        nodes: Node dicts.
        edges: Edge dicts.

    Returns:
        The validated ``WorkflowSpec``.
    """
    return WorkflowSpec.model_validate({"nodes": nodes, "edges": edges})


def _branch_merge_program() -> WorkflowProgram:
    """Build a branch+merge program from fake signature modules."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "left", "kind": "signature", "signature_code": _SIG},
            {"id": "right", "kind": "signature", "signature_code": _SIG},
            {"id": "out", "kind": "output", "fields": [{"name": "a"}, {"name": "b"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "left", "target_port": "text"},
            {"source": "inp", "source_port": "text", "target": "right", "target_port": "text"},
            {"source": "left", "source_port": "summary", "target": "out", "target_port": "a"},
            {"source": "right", "source_port": "summary", "target": "out", "target_port": "b"},
        ],
    )
    return WorkflowProgram(
        spec,
        signature_modules={
            "left": _FakeModule(lambda text: {"summary": f"L:{text}"}),
            "right": _FakeModule(lambda text: {"summary": f"R:{text}"}),
        },
        signature_outputs={"left": ["summary"], "right": ["summary"]},
        transforms={},
        tools={},
    )


def test_branch_merge_execution_and_traces():
    """Branches both see the input; the output anchor gathers both; traces replay it."""
    program = _branch_merge_program()
    with capture_node_traces() as traces:
        prediction = program(text="hi")
    assert prediction.a == "L:hi"
    assert prediction.b == "R:hi"
    assert [t["node_id"] for t in traces] == ["inp", "left", "right", "out"]
    assert traces[1]["inputs"] == {"text": "hi"}
    assert traces[3]["outputs"] == {"a": "L:hi", "b": "R:hi"}
    assert all(t["error"] is None for t in traces)


def test_no_trace_recording_without_sink():
    """Execution outside capture_node_traces records nothing (optimization path)."""
    program = _branch_merge_program()
    prediction = program(text="hi")
    assert prediction.a == "L:hi"


def test_missing_workflow_input_raises_with_node_id():
    """Omitting a workflow input names the input anchor."""
    program = _branch_merge_program()
    with pytest.raises(WorkflowNodeExecutionError, match="missing workflow inputs"):
        program()


def test_failing_node_is_named_and_traced():
    """A raising node surfaces its id and leaves an error trace entry."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "boom", "kind": "signature", "signature_code": _SIG},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "boom", "target_port": "text"},
            {"source": "boom", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )

    def _explode(**_kwargs):
        raise ValueError("kaput")

    program = WorkflowProgram(
        spec,
        signature_modules={"boom": _FakeModule(_explode)},
        signature_outputs={"boom": ["summary"]},
        transforms={},
        tools={},
    )
    with capture_node_traces() as traces, pytest.raises(WorkflowNodeExecutionError, match="'boom' failed"):
        program(text="x")
    assert traces[-1]["node_id"] == "boom"
    assert "kaput" in traces[-1]["error"]


def test_transform_workflow_builds_and_executes():
    """build_workflow_program loads real transform code and executes it end to end."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {
                "id": "shout",
                "kind": "transform",
                "transform_code": _TRANSFORM,
                "input_fields": [{"name": "text"}],
                "output_fields": [{"name": "shout"}],
            },
            {"id": "out", "kind": "output", "fields": [{"name": "shout"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "shout", "target_port": "text"},
            {"source": "shout", "source_port": "shout", "target": "out", "target_port": "shout"},
        ],
    )
    program, schema_hashes = build_workflow_program(spec)
    assert schema_hashes == {}
    prediction = program(text="quiet")
    assert prediction.shout == "QUIET"


def test_transform_missing_output_field_raises():
    """A transform omitting a declared output field names the node."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {
                "id": "bad",
                "kind": "transform",
                "transform_code": "def transform(text):\n    return {'wrong': text}\n",
                "input_fields": [{"name": "text"}],
                "output_fields": [{"name": "expected"}],
            },
            {"id": "out", "kind": "output", "fields": [{"name": "expected"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "bad", "target_port": "text"},
            {"source": "bad", "source_port": "expected", "target": "out", "target_port": "expected"},
        ],
    )
    program, _ = build_workflow_program(spec)
    with pytest.raises(WorkflowNodeExecutionError, match="missing output fields"):
        program(text="x")


def test_mcp_node_calls_tool():
    """An mcp node feeds its inputs to the tool and stores the result on its port."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "q"}]},
            {
                "id": "lookup",
                "kind": "mcp",
                "tool_name": "search",
                "input_fields": [{"name": "q"}],
                "output_field": {"name": "result"},
            },
            {"id": "out", "kind": "output", "fields": [{"name": "result"}]},
        ],
        [
            {"source": "inp", "source_port": "q", "target": "lookup", "target_port": "q"},
            {"source": "lookup", "source_port": "result", "target": "out", "target_port": "result"},
        ],
    )
    tool = dspy.Tool(lambda q: f"found:{q}", name="search", desc="find things")
    program = WorkflowProgram(
        spec,
        signature_modules={},
        signature_outputs={},
        transforms={},
        tools={"lookup": tool},
    )
    prediction = program(q="cats")
    assert prediction.result == "found:cats"


def test_build_workflow_program_registers_predictors():
    """Signature nodes become named sub-modules so GEPA can discover them."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "summarize", "kind": "signature", "signature_code": _SIG},
            {"id": "polish", "kind": "signature", "signature_code": _SIG, "module_name": "cot"},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "summarize", "target_port": "text"},
            {"source": "summarize", "source_port": "summary", "target": "polish", "target_port": "text"},
            {"source": "polish", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )
    program, _ = build_workflow_program(spec)
    predictor_names = [name for name, _pred in program.named_predictors()]
    assert any(name.startswith(f"{WORKFLOW_NODE_ATTR_PREFIX}summarize") for name in predictor_names)
    assert any(name.startswith(f"{WORKFLOW_NODE_ATTR_PREFIX}polish") for name in predictor_names)
    assert len(predictor_names) == 2


def _flex_node_spec(tool_filter: list[str] | None = None) -> WorkflowSpec:
    """Build a two-node graph whose second node is a flex (code-optimized) step.

    Args:
        tool_filter: Tools the flex node opts into, or ``None`` for no tools.

    Returns:
        The workflow spec.
    """
    return _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "draft", "kind": "signature", "signature_code": _SIG},
            {
                "id": "refine",
                "kind": "signature",
                "signature_code": _SIG,
                "module_name": "flex",
                "tool_filter": tool_filter,
            },
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "draft", "target_port": "text"},
            {"source": "draft", "source_port": "summary", "target": "refine", "target_port": "text"},
            {"source": "refine", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )


def test_flex_node_is_discovered_as_a_code_component():
    """A flex node becomes a dspy.Flex GEPA optimizes as code, not as instructions."""
    program, _ = build_workflow_program(_flex_node_spec())
    flex_path = f"{WORKFLOW_NODE_ATTR_PREFIX}refine"
    assert isinstance(getattr(program, flex_path), dspy.Flex)
    assert set(enumerate_flex_submodules(program)) == {flex_path}
    # A Flex's update unit is its module_src, so it contributes no instruction
    # predictor: the draft node's is the only prompt GEPA tunes as text.
    predictor_names = [name for name, _pred in program.named_predictors()]
    assert predictor_names == [f"{WORKFLOW_NODE_ATTR_PREFIX}draft"]


def test_flex_node_state_roundtrip_carries_rewritten_source():
    """The GEPA-rewritten source saves and reloads under the flex node's path."""
    spec = _flex_node_spec()
    flex_path = f"{WORKFLOW_NODE_ATTR_PREFIX}refine"
    rewritten = (
        "class SummarizeModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.predict = dspy.Predict('text -> summary')\n"
        "\n"
        "    def forward(self, **inputs):\n"
        "        return dspy.Prediction(summary=self.predict(**inputs).summary)\n"
    )
    first, _ = build_workflow_program(spec)
    state = first.dump_state()
    state[flex_path]["module_src"] = rewritten

    second, _ = build_workflow_program(spec)
    second.load_state(state)
    assert getattr(second, flex_path).module_src == rewritten
    assert second.dump_state()[flex_path]["module_src"] == rewritten


def _stub_roster(monkeypatch, *names: str) -> None:
    """Replace run-level tool resolution with a fixed roster of no-op tools.

    Args:
        monkeypatch: Fixture used to patch the resolver.
        names: Tool names the fake tool_source exposes, in roster order.
    """
    roster = [dspy.Tool(lambda **kwargs: "", name=name, desc=f"{name} tool") for name in names]
    monkeypatch.setattr(workflow_module, "resolve_react_tools", lambda *args, **kwargs: (roster, {}))


def test_flex_node_without_tools_is_not_a_tool_user():
    """A flex node naming no tools builds without a run-level tool_source."""
    spec = _flex_node_spec()
    assert workflow_tool_users(spec) == []
    program, _ = build_workflow_program(spec)
    assert "dspy.Predict(" in getattr(program, f"{WORKFLOW_NODE_ATTR_PREFIX}refine").module_src


def test_flex_node_tool_filter_opts_into_the_roster(monkeypatch):
    """Naming tools hands them to the Flex, whose baseline becomes a dspy.RLM.

    Args:
        monkeypatch: Fixture used to stub run-level tool resolution.
    """
    _stub_roster(monkeypatch, "search", "fetch")
    spec = _flex_node_spec(tool_filter=["search"])
    assert workflow_tool_users(spec) == ["refine"]

    program, _ = build_workflow_program(spec, tool_source={"kind": "live_mcp"})
    baseline = getattr(program, f"{WORKFLOW_NODE_ATTR_PREFIX}refine").module_src
    assert "dspy.RLM(" in baseline
    assert "tools=[search]" in baseline


def test_flex_node_tool_filter_rejects_unknown_tools(monkeypatch):
    """A flex node naming a tool the roster lacks fails the build.

    Args:
        monkeypatch: Fixture used to stub run-level tool resolution.
    """
    _stub_roster(monkeypatch, "search")
    with pytest.raises(ServiceError, match="references unknown tools"):
        build_workflow_program(_flex_node_spec(tool_filter=["nope"]), tool_source={"kind": "live_mcp"})


def test_flex_node_rejects_empty_tool_filter():
    """An empty flex tool_filter is refused: null already says "no tools"."""
    with pytest.raises(ValidationError, match="must name at least one tool"):
        _flex_node_spec(tool_filter=[])


def test_unresolvable_node_module_names_the_node(monkeypatch):
    """A DSPy build without dspy.Flex fails as a node-anchored ServiceError.

    Args:
        monkeypatch: Fixture used to make the module resolver fail.
    """

    real_resolver = workflow_module.resolve_module_factory

    def _flex_is_missing(name: str):
        if name == "flex":
            raise ResolverError("Module 'dspy' has no attribute 'Flex'.")
        return real_resolver(name)

    monkeypatch.setattr(workflow_module, "resolve_module_factory", _flex_is_missing)
    with pytest.raises(ServiceError, match="Workflow node 'refine' requests module 'flex'"):
        build_workflow_program(_flex_node_spec())


def test_state_roundtrip_reconstructs_identically():
    """dump_state on one build loads cleanly onto a fresh build (serve reconstruction)."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "summarize", "kind": "signature", "signature_code": _SIG},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "summarize", "target_port": "text"},
            {"source": "summarize", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )
    first, _ = build_workflow_program(spec)
    state = first.dump_state()
    second, _ = build_workflow_program(spec)
    second.load_state(state)
    assert second.dump_state() == state


def test_build_rejects_tool_nodes_without_tool_source():
    """A tool-using graph cannot be built without a run-level tool_source."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "q"}]},
            {
                "id": "lookup",
                "kind": "mcp",
                "tool_name": "search",
                "input_fields": [{"name": "q"}],
                "output_field": {"name": "result"},
            },
            {"id": "out", "kind": "output", "fields": [{"name": "result"}]},
        ],
        [
            {"source": "inp", "source_port": "q", "target": "lookup", "target_port": "q"},
            {"source": "lookup", "source_port": "result", "target": "out", "target_port": "result"},
        ],
    )
    with pytest.raises(ServiceError, match="tool_source is required"):
        build_workflow_program(spec)


def test_validate_workflow_checks_signature_ports():
    """Deep validation introspects signature code and rejects a bad edge port."""
    good = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "summarize", "kind": "signature", "signature_code": _SIG},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "summarize", "target_port": "text"},
            {"source": "summarize", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )
    introspection = validate_workflow(good)
    assert introspection.signature_fields["summarize"] == (["text"], ["summary"])
    assert introspection.input_fields == ["text"]
    assert introspection.output_fields == ["summary"]

    bad = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "summarize", "kind": "signature", "signature_code": _SIG},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "summarize", "target_port": "wrong_port"},
            {"source": "summarize", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )
    with pytest.raises(ServiceError, match="no input field 'wrong_port'"):
        validate_workflow(bad)


def test_validate_workflow_rejects_unconnected_signature_input():
    """Deep validation catches a signature input no edge feeds."""
    two_input_sig = (
        "import dspy\n"
        "class Compose(dspy.Signature):\n"
        "    text: str = dspy.InputField()\n"
        "    tone: str = dspy.InputField()\n"
        "    summary: str = dspy.OutputField()\n"
    )
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {"id": "compose", "kind": "signature", "signature_code": two_input_sig},
            {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "compose", "target_port": "text"},
            {"source": "compose", "source_port": "summary", "target": "out", "target_port": "summary"},
        ],
    )
    with pytest.raises(ServiceError, match="unconnected input fields: \\['tone'\\]"):
        validate_workflow(spec)


def test_validate_workflow_checks_transform_params():
    """Deep validation rejects a transform whose params disagree with declared inputs."""
    spec = _spec(
        [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {
                "id": "shout",
                "kind": "transform",
                "transform_code": "def transform(other_name):\n    return {'shout': other_name}\n",
                "input_fields": [{"name": "text"}],
                "output_fields": [{"name": "shout"}],
            },
            {"id": "out", "kind": "output", "fields": [{"name": "shout"}]},
        ],
        [
            {"source": "inp", "source_port": "text", "target": "shout", "target_port": "text"},
            {"source": "shout", "source_port": "shout", "target": "out", "target_port": "shout"},
        ],
    )
    with pytest.raises(ServiceError, match="do not match the declared input fields"):
        validate_workflow(spec)
