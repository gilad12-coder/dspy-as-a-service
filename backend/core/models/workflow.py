"""Workflow graph spec models for composite multi-node optimization runs.

A workflow is a DAG of nodes submitted as ``module_name="workflow"`` on a
run request. Exactly one ``input`` anchor (its fields are the workflow's
input ports, fed from mapped dataset columns) and one ``output`` anchor
(its fields are the final outputs the metric scores and serve returns)
frame any number of ``signature`` / ``transform`` / ``mcp`` nodes wired by
port-level edges.

Validation is two-phase, mirroring the rest of the submission pipeline:
this module performs the structural pass (pure graph checks, no exec of
user code) at the Pydantic boundary; the deep pass (signature/transform
introspection via subprocess exec, port checks against introspected
fields) lives in ``service_gateway/optimization/workflow.py``.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

WORKFLOW_MODULE_NAME = "workflow"

# Node ids double as sub-module attribute names on the compiled program, so
# GEPA predictor paths (``n_<id>.predict``) stay stable and identifier-safe.
_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"

MAX_WORKFLOW_NODES = 50
MAX_WORKFLOW_EDGES = 200


class WorkflowNodePosition(BaseModel):
    """Canvas coordinates persisted so the builder re-opens exactly as left."""

    x: float
    y: float


class WorkflowField(BaseModel):
    """A declared port on an anchor, transform, or mcp node."""

    name: str = Field(pattern=_IDENTIFIER_PATTERN, max_length=64)
    # Python type expression as authored ("str", "list[str]", ...). Opaque to
    # the server in v1; the canvas uses it for port coloring and client-side
    # type-compatibility checks.
    annotation: str = "str"
    description: str | None = None


class _WorkflowNodeBase(BaseModel):
    """Fields shared by every workflow node kind."""

    id: str = Field(pattern=_IDENTIFIER_PATTERN, max_length=64)
    name: str | None = Field(default=None, max_length=120)
    position: WorkflowNodePosition | None = None


def _require_unique_field_names(fields: list[WorkflowField], node_id: str, label: str) -> None:
    """Reject duplicate port names within one declared-field list.

    Args:
        fields: Declared ports to check.
        node_id: Owning node id, used in the error message.
        label: Which field list is being checked (e.g. ``fields``).

    Raises:
        ValueError: When two declared ports share a name.
    """
    seen: set[str] = set()
    for field in fields:
        if field.name in seen:
            raise ValueError(f"Node '{node_id}' declares duplicate {label} name '{field.name}'.")
        seen.add(field.name)


class WorkflowInputNode(_WorkflowNodeBase):
    """The single entry anchor; its fields are the workflow's input ports."""

    kind: Literal["input"]
    fields: list[WorkflowField] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_fields(self) -> WorkflowInputNode:
        """Reject duplicate port names.

        Returns:
            The validated node.
        """
        _require_unique_field_names(self.fields, self.id, "field")
        return self


class WorkflowOutputNode(_WorkflowNodeBase):
    """The single exit anchor; its fields are the workflow's final outputs."""

    kind: Literal["output"]
    fields: list[WorkflowField] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_fields(self) -> WorkflowOutputNode:
        """Reject duplicate port names.

        Returns:
            The validated node.
        """
        _require_unique_field_names(self.fields, self.id, "field")
        return self


class WorkflowSignatureNode(_WorkflowNodeBase):
    """An LLM step: a dspy.Signature run by a per-node module choice."""

    kind: Literal["signature"]
    module_name: Literal["predict", "cot", "react"] = "predict"
    signature_code: str = Field(min_length=1)
    tool_filter: list[str] | None = Field(
        default=None,
        description=(
            "React nodes only: restrict the node's tool roster to these tool names "
            "out of the run-level tool_source. Null means the full roster."
        ),
    )

    @model_validator(mode="after")
    def _check_tool_filter(self) -> WorkflowSignatureNode:
        """Reject a tool filter on a non-react node.

        Returns:
            The validated node.

        Raises:
            ValueError: When ``tool_filter`` is set but ``module_name`` is not react.
        """
        if self.tool_filter is not None and self.module_name != "react":
            raise ValueError(f"Node '{self.id}': tool_filter is only valid when module_name is 'react'.")
        return self


class WorkflowTransformNode(_WorkflowNodeBase):
    """A non-LLM Python step reshaping data between nodes."""

    kind: Literal["transform"]
    transform_code: str = Field(min_length=1)
    input_fields: list[WorkflowField] = Field(min_length=1)
    output_fields: list[WorkflowField] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_fields(self) -> WorkflowTransformNode:
        """Reject duplicate port names on either side.

        Returns:
            The validated node.
        """
        _require_unique_field_names(self.input_fields, self.id, "input_field")
        _require_unique_field_names(self.output_fields, self.id, "output_field")
        return self


class WorkflowMcpNode(_WorkflowNodeBase):
    """An explicit MCP tool call as a graph step."""

    kind: Literal["mcp"]
    tool_name: str = Field(min_length=1, max_length=128)
    input_fields: list[WorkflowField] = Field(default_factory=list)
    output_field: WorkflowField = Field(default_factory=lambda: WorkflowField(name="result"))

    @model_validator(mode="after")
    def _check_fields(self) -> WorkflowMcpNode:
        """Reject duplicate input port names.

        Returns:
            The validated node.
        """
        _require_unique_field_names(self.input_fields, self.id, "input_field")
        return self


WorkflowNode = Annotated[
    Union[
        WorkflowInputNode,
        WorkflowOutputNode,
        WorkflowSignatureNode,
        WorkflowTransformNode,
        WorkflowMcpNode,
    ],
    Field(discriminator="kind"),
]


class WorkflowEdge(BaseModel):
    """A port-level data-flow connection between two nodes."""

    source: str
    source_port: str = Field(pattern=_IDENTIFIER_PATTERN, max_length=64)
    target: str
    target_port: str = Field(pattern=_IDENTIFIER_PATTERN, max_length=64)


class WorkflowSpec(BaseModel):
    """The full workflow graph carried on a run request."""

    nodes: list[WorkflowNode] = Field(min_length=2, max_length=MAX_WORKFLOW_NODES)
    edges: list[WorkflowEdge] = Field(default_factory=list, max_length=MAX_WORKFLOW_EDGES)

    @property
    def input_node(self) -> WorkflowInputNode:
        """Return the single input anchor (validation guarantees presence)."""
        return next(node for node in self.nodes if node.kind == "input")

    @property
    def output_node(self) -> WorkflowOutputNode:
        """Return the single output anchor (validation guarantees presence)."""
        return next(node for node in self.nodes if node.kind == "output")

    def node_by_id(self, node_id: str) -> WorkflowNode:
        """Look up a node by id.

        Args:
            node_id: The node id to find.

        Returns:
            The matching node.

        Raises:
            KeyError: When no node carries ``node_id``.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def input_field_names(self) -> list[str]:
        """Return the workflow's input port names (mapped from dataset columns)."""
        return [field.name for field in self.input_node.fields]

    def output_field_names(self) -> list[str]:
        """Return the workflow's final output port names (scored by the metric)."""
        return [field.name for field in self.output_node.fields]

    def edges_into(self, node_id: str) -> list[WorkflowEdge]:
        """Return every edge targeting ``node_id``.

        Args:
            node_id: The consuming node id.

        Returns:
            Edges whose target is ``node_id``, in spec order.
        """
        return [edge for edge in self.edges if edge.target == node_id]

    @model_validator(mode="after")
    def _validate_graph(self) -> WorkflowSpec:
        """Run the structural (exec-free) graph validation pass.

        Checks anchors, id uniqueness, edge endpoint/port sanity for nodes
        whose ports are declared (anchors, transform, mcp), single-producer
        inputs, acyclicity, and input→output connectivity. Signature-node
        port names require exec-based introspection and are checked in the
        deep pass instead.

        Returns:
            The validated spec.

        Raises:
            ValueError: On any structural violation, with the offending node
                or edge named so the canvas can anchor the error.
        """
        nodes_by_id: dict[str, WorkflowNode] = {}
        for node in self.nodes:
            if node.id in nodes_by_id:
                raise ValueError(f"Duplicate node id '{node.id}'.")
            nodes_by_id[node.id] = node

        input_nodes = [node for node in self.nodes if node.kind == "input"]
        output_nodes = [node for node in self.nodes if node.kind == "output"]
        if len(input_nodes) != 1:
            raise ValueError(f"Workflow must contain exactly one input node (found {len(input_nodes)}).")
        if len(output_nodes) != 1:
            raise ValueError(f"Workflow must contain exactly one output node (found {len(output_nodes)}).")
        input_id = input_nodes[0].id
        output_id = output_nodes[0].id

        consumed_ports: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in nodes_by_id:
                raise ValueError(f"Edge references unknown source node '{edge.source}'.")
            if edge.target not in nodes_by_id:
                raise ValueError(f"Edge references unknown target node '{edge.target}'.")
            if edge.source == edge.target:
                raise ValueError(f"Edge on node '{edge.source}' connects the node to itself.")
            if edge.source == output_id:
                raise ValueError("The output node cannot be an edge source.")
            if edge.target == input_id:
                raise ValueError("The input node cannot be an edge target.")
            port_key = (edge.target, edge.target_port)
            if port_key in consumed_ports:
                raise ValueError(
                    f"Input port '{edge.target_port}' on node '{edge.target}' is fed by more than one edge."
                )
            consumed_ports.add(port_key)
            _check_declared_source_port(nodes_by_id[edge.source], edge.source_port)
            _check_declared_target_port(nodes_by_id[edge.target], edge.target_port)

        for field in output_nodes[0].fields:
            if (output_id, field.name) not in consumed_ports:
                raise ValueError(f"Output field '{field.name}' is not connected to any node.")

        _require_acyclic(self.nodes, self.edges)
        _require_connected(self.nodes, self.edges, input_id=input_id, output_id=output_id)
        return self


def _declared_output_ports(node: WorkflowNode) -> list[str] | None:
    """Return a node's declared output port names, or None when introspection is required.

    Args:
        node: The node to inspect.

    Returns:
        Port names for anchor/transform/mcp nodes; ``None`` for signature
        nodes (their ports come from exec-based introspection).
    """
    if node.kind == "input":
        return [field.name for field in node.fields]
    if node.kind == "transform":
        return [field.name for field in node.output_fields]
    if node.kind == "mcp":
        return [node.output_field.name]
    return None


def _declared_input_ports(node: WorkflowNode) -> list[str] | None:
    """Return a node's declared input port names, or None when introspection is required.

    Args:
        node: The node to inspect.

    Returns:
        Port names for anchor/transform/mcp nodes; ``None`` for signature nodes.
    """
    if node.kind == "output":
        return [field.name for field in node.fields]
    if node.kind in ("transform", "mcp"):
        return [field.name for field in node.input_fields]
    return None


def _check_declared_source_port(node: WorkflowNode, port: str) -> None:
    """Reject an edge whose source port is absent from a declared-port node.

    Args:
        node: The edge's source node.
        port: The source port name.

    Raises:
        ValueError: When the node declares its ports and ``port`` is not among them.
    """
    declared = _declared_output_ports(node)
    if declared is not None and port not in declared:
        raise ValueError(f"Node '{node.id}' has no output port '{port}'.")


def _check_declared_target_port(node: WorkflowNode, port: str) -> None:
    """Reject an edge whose target port is absent from a declared-port node.

    Args:
        node: The edge's target node.
        port: The target port name.

    Raises:
        ValueError: When the node declares its ports and ``port`` is not among them.
    """
    declared = _declared_input_ports(node)
    if declared is not None and port not in declared:
        raise ValueError(f"Node '{node.id}' has no input port '{port}'.")


def workflow_topological_order(spec: WorkflowSpec) -> list[str]:
    """Return node ids in a deterministic topological order.

    Kahn's algorithm with ties broken by spec node order, so the execution
    order — and therefore GEPA predictor discovery order — is stable for a
    given spec JSON across processes.

    Args:
        spec: A structurally valid workflow spec.

    Returns:
        Every node id, dependency-ordered.

    Raises:
        ValueError: When the graph contains a cycle (defensive; the spec
            validator already rejects cycles).
    """
    order_index = {node.id: index for index, node in enumerate(spec.nodes)}
    indegree = {node.id: 0 for node in spec.nodes}
    outgoing: dict[str, set[str]] = {node.id: set() for node in spec.nodes}
    for edge in spec.edges:
        # Parallel edges between the same pair (different ports) count once.
        if edge.target not in outgoing[edge.source]:
            outgoing[edge.source].add(edge.target)
            indegree[edge.target] += 1

    ready = [(order_index[nid], nid) for nid, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        _, node_id = heapq.heappop(ready)
        ordered.append(node_id)
        for child in sorted(outgoing[node_id], key=order_index.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (order_index[child], child))
    if len(ordered) != len(spec.nodes):
        cyclic = sorted(nid for nid, degree in indegree.items() if degree > 0)
        raise ValueError(f"Workflow contains a cycle involving: {cyclic}.")
    return ordered


def _require_acyclic(nodes: list[WorkflowNode], edges: list[WorkflowEdge]) -> None:
    """Reject graphs containing a directed cycle.

    Args:
        nodes: All workflow nodes.
        edges: All workflow edges.

    Raises:
        ValueError: When a cycle exists, naming the nodes involved.
    """
    indegree = {node.id: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        indegree[edge.target] += 1
        outgoing[edge.source].append(edge.target)
    ready = deque(nid for nid, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for child in outgoing[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        cyclic = sorted(nid for nid, degree in indegree.items() if degree > 0)
        raise ValueError(f"Workflow contains a cycle involving: {cyclic}.")


def _require_connected(
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
    *,
    input_id: str,
    output_id: str,
) -> None:
    """Reject nodes that do not lie on a path from the input anchor to the output anchor.

    Args:
        nodes: All workflow nodes.
        edges: All workflow edges.
        input_id: Id of the input anchor.
        output_id: Id of the output anchor.

    Raises:
        ValueError: When any node is unreachable from the input or cannot
            reach the output.
    """
    forward: dict[str, list[str]] = {node.id: [] for node in nodes}
    backward: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        forward[edge.source].append(edge.target)
        backward[edge.target].append(edge.source)

    reachable = _flood(forward, input_id)
    co_reachable = _flood(backward, output_id)
    for node in nodes:
        if node.id != input_id and node.id not in reachable:
            raise ValueError(f"Node '{node.id}' is not reachable from the input node.")
        if node.id != output_id and node.id not in co_reachable:
            raise ValueError(f"Node '{node.id}' has no path to the output node.")


def _flood(adjacency: dict[str, list[str]], start: str) -> set[str]:
    """Return every node reachable from ``start`` over ``adjacency``.

    Args:
        adjacency: Node id to neighbor ids.
        start: Starting node id (excluded from the result unless revisited).

    Returns:
        The set of reached node ids.
    """
    seen: set[str] = set()
    frontier = deque(adjacency.get(start, ()))
    while frontier:
        node_id = frontier.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        frontier.extend(adjacency.get(node_id, ()))
    return seen
