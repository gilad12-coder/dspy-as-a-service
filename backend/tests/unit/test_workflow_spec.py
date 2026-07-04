"""Unit tests for the workflow graph spec models and submission wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import WorkflowSpec, workflow_topological_order
from core.models.submissions import GridSearchRequest, RunRequest

_SIG = (
    "import dspy\n"
    "class Summarize(dspy.Signature):\n"
    "    text: str = dspy.InputField()\n"
    "    summary: str = dspy.OutputField()\n"
)

_MODEL_CFG = {"name": "openai/gpt-4o-mini"}


def _linear_nodes() -> list[dict]:
    """Return nodes for a minimal valid input → signature → output chain."""
    return [
        {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
        {"id": "summarize", "kind": "signature", "signature_code": _SIG},
        {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
    ]


def _linear_edges() -> list[dict]:
    """Return edges wiring the minimal chain end to end."""
    return [
        {"source": "inp", "source_port": "text", "target": "summarize", "target_port": "text"},
        {"source": "summarize", "source_port": "summary", "target": "out", "target_port": "summary"},
    ]


def _spec(nodes: list[dict] | None = None, edges: list[dict] | None = None) -> WorkflowSpec:
    """Validate and return a spec built from the given (or default) graph.

    Args:
        nodes: Node dicts; defaults to the minimal chain.
        edges: Edge dicts; defaults to the minimal chain wiring.

    Returns:
        The parsed ``WorkflowSpec``.
    """
    return WorkflowSpec.model_validate(
        {"nodes": nodes if nodes is not None else _linear_nodes(), "edges": edges if edges is not None else _linear_edges()}
    )


def test_valid_linear_spec_parses():
    """The minimal chain parses and exposes its end-to-end fields."""
    spec = _spec()
    assert spec.input_field_names() == ["text"]
    assert spec.output_field_names() == ["summary"]
    assert workflow_topological_order(spec) == ["inp", "summarize", "out"]


def test_branch_merge_spec_parses_with_stable_topo_order():
    """A branch+merge DAG parses; topological order is deterministic by spec order."""
    nodes = [
        {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
        {"id": "left", "kind": "signature", "signature_code": _SIG},
        {"id": "right", "kind": "signature", "signature_code": _SIG},
        {
            "id": "merge",
            "kind": "transform",
            "transform_code": "def transform(a, b):\n    return {'joined': a + b}\n",
            "input_fields": [{"name": "a"}, {"name": "b"}],
            "output_fields": [{"name": "joined"}],
        },
        {"id": "out", "kind": "output", "fields": [{"name": "joined"}]},
    ]
    edges = [
        {"source": "inp", "source_port": "text", "target": "left", "target_port": "text"},
        {"source": "inp", "source_port": "text", "target": "right", "target_port": "text"},
        {"source": "left", "source_port": "summary", "target": "merge", "target_port": "a"},
        {"source": "right", "source_port": "summary", "target": "merge", "target_port": "b"},
        {"source": "merge", "source_port": "joined", "target": "out", "target_port": "joined"},
    ]
    spec = _spec(nodes, edges)
    assert workflow_topological_order(spec) == ["inp", "left", "right", "merge", "out"]


def test_duplicate_node_ids_rejected():
    """Two nodes sharing an id fail structural validation."""
    nodes = _linear_nodes() + [{"id": "summarize", "kind": "signature", "signature_code": _SIG}]
    with pytest.raises(ValidationError, match="Duplicate node id"):
        _spec(nodes)


def test_exactly_one_input_and_output_required():
    """Zero or multiple anchors are rejected."""
    extra_input = _linear_nodes() + [{"id": "inp2", "kind": "input", "fields": [{"name": "text"}]}]
    with pytest.raises(ValidationError, match="exactly one input node"):
        _spec(extra_input)
    no_output = [node for node in _linear_nodes() if node["id"] != "out"]
    with pytest.raises(ValidationError, match="exactly one output node"):
        _spec(no_output, edges=[_linear_edges()[0]])


def test_cycle_rejected():
    """A directed cycle between signature nodes fails validation."""
    nodes = [
        {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
        {"id": "a", "kind": "signature", "signature_code": _SIG},
        {"id": "b", "kind": "signature", "signature_code": _SIG},
        {"id": "out", "kind": "output", "fields": [{"name": "summary"}]},
    ]
    edges = [
        {"source": "inp", "source_port": "text", "target": "a", "target_port": "text"},
        {"source": "a", "source_port": "summary", "target": "b", "target_port": "text"},
        # Signature-node port names are deep-validated, so the structural
        # pass sees this only as a back-edge closing the a→b→a cycle.
        {"source": "b", "source_port": "summary", "target": "a", "target_port": "extra"},
        {"source": "b", "source_port": "summary", "target": "out", "target_port": "summary"},
    ]
    with pytest.raises(ValidationError, match="cycle"):
        _spec(nodes, edges)


def test_unknown_edge_endpoint_rejected():
    """An edge naming a nonexistent node fails validation."""
    edges = _linear_edges() + [
        {"source": "ghost", "source_port": "x", "target": "out", "target_port": "summary"}
    ]
    with pytest.raises(ValidationError, match="unknown source node"):
        _spec(edges=edges)


def test_multiple_producers_for_one_port_rejected():
    """Two edges feeding the same input port fail validation."""
    nodes = _linear_nodes()
    nodes.insert(2, {"id": "second", "kind": "signature", "signature_code": _SIG})
    edges = _linear_edges() + [
        {"source": "inp", "source_port": "text", "target": "second", "target_port": "text"},
        {"source": "second", "source_port": "summary", "target": "out", "target_port": "summary"},
    ]
    with pytest.raises(ValidationError, match="fed by more than one edge"):
        _spec(nodes, edges)


def test_unfed_output_field_rejected():
    """An output anchor field with no incoming edge fails validation."""
    nodes = _linear_nodes()
    nodes[-1]["fields"].append({"name": "orphan"})
    with pytest.raises(ValidationError, match="'orphan' is not connected"):
        _spec(nodes)


def test_unreachable_node_rejected():
    """A node disconnected from the input anchor fails validation."""
    nodes = _linear_nodes() + [{"id": "island", "kind": "signature", "signature_code": _SIG}]
    with pytest.raises(ValidationError, match="not reachable from the input node"):
        _spec(nodes)


def test_declared_port_existence_checked():
    """An edge referencing a port the input anchor does not declare fails."""
    edges = _linear_edges()
    edges[0]["source_port"] = "missing_port"
    with pytest.raises(ValidationError, match="no output port 'missing_port'"):
        _spec(edges=edges)


def test_anchor_direction_enforced():
    """Edges into the input anchor or out of the output anchor fail."""
    edges = _linear_edges() + [
        {"source": "summarize", "source_port": "summary", "target": "inp", "target_port": "text"}
    ]
    with pytest.raises(ValidationError, match="input node cannot be an edge target"):
        _spec(edges=edges)


def test_tool_filter_only_valid_on_react_nodes():
    """A predict node carrying tool_filter is rejected."""
    nodes = _linear_nodes()
    nodes[1]["tool_filter"] = ["search"]
    with pytest.raises(ValidationError, match="tool_filter is only valid"):
        _spec(nodes)


def _run_request_payload(**overrides) -> dict:
    """Build a valid workflow RunRequest payload dict, applying overrides.

    Args:
        **overrides: Top-level keys to replace in the payload.

    Returns:
        The payload dict ready for ``RunRequest.model_validate``.
    """
    payload = {
        "module_name": "workflow",
        "workflow": {"nodes": _linear_nodes(), "edges": _linear_edges()},
        "metric_code": "def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):\n    return 1.0\n",
        "optimizer_name": "gepa",
        "dataset": [{"t": "hello", "s": "hi"}],
        "column_mapping": {"inputs": {"text": "t"}, "outputs": {"summary": "s"}},
        "model_config": _MODEL_CFG,
    }
    payload.update(overrides)
    return payload


def test_run_request_accepts_workflow_module():
    """A workflow run parses without a top-level signature_code."""
    request = RunRequest.model_validate(_run_request_payload())
    assert request.workflow is not None
    assert request.signature_code is None


def test_run_request_requires_workflow_for_workflow_module():
    """module_name='workflow' without a workflow spec is rejected."""
    with pytest.raises(ValidationError, match="workflow is required"):
        RunRequest.model_validate(_run_request_payload(workflow=None))


def test_run_request_rejects_workflow_on_other_modules():
    """A workflow spec on a predict run is rejected."""
    with pytest.raises(ValidationError, match="only valid when module_name"):
        RunRequest.model_validate(_run_request_payload(module_name="predict", signature_code=_SIG))


def test_run_request_still_requires_signature_code_for_scalar_modules():
    """A predict run without signature_code is rejected."""
    with pytest.raises(ValidationError, match="signature_code is required"):
        RunRequest.model_validate(_run_request_payload(module_name="predict", workflow=None))


def test_run_request_requires_tool_source_for_tool_nodes():
    """A workflow containing an mcp node requires a run-level tool_source."""
    nodes = _linear_nodes()
    nodes.insert(
        2,
        {
            "id": "lookup",
            "kind": "mcp",
            "tool_name": "search",
            "input_fields": [{"name": "query"}],
            "output_field": {"name": "result"},
        },
    )
    edges = _linear_edges() + [
        {"source": "inp", "source_port": "text", "target": "lookup", "target_port": "query"},
    ]
    # Keep the mcp node on a path to the output.
    nodes[-1]["fields"].append({"name": "extra"})
    edges.append({"source": "lookup", "source_port": "result", "target": "out", "target_port": "extra"})
    payload = _run_request_payload(workflow={"nodes": nodes, "edges": edges})
    with pytest.raises(ValidationError, match="tool_source is required"):
        RunRequest.model_validate(payload)
    payload["tool_source"] = {"kind": "live_mcp", "mcp_url": "http://localhost:9"}
    assert RunRequest.model_validate(payload).tool_source is not None


def test_grid_search_rejects_workflow_module():
    """Grid search does not support workflow modules in v1."""
    payload = _run_request_payload()
    payload.pop("model_config")
    payload["generation_models"] = [_MODEL_CFG]
    payload["reflection_models"] = [_MODEL_CFG]
    with pytest.raises(ValidationError, match="Grid search does not support workflow"):
        GridSearchRequest.model_validate(payload)
