"""Compile a workflow graph spec into a runnable composed DSPy module.

``build_workflow_program`` turns a structurally-valid ``WorkflowSpec`` into a
``WorkflowProgram`` — a ``dspy.Module`` whose signature nodes are named
sub-modules (so GEPA discovers and jointly optimizes every node, tuning
instructions for predict/cot/react nodes and rewriting code for flex
ones), whose transform nodes execute user Python, and whose mcp nodes
call tools from the run-level roster. The same builder reconstructs
the module shell at serve time before ``load_state`` overlays the optimized
instructions, so construction must stay deterministic for a given spec.

``validate_workflow`` is the deep validation pass (exec-based, subprocess
isolated) complementing the structural pass in ``models/workflow.py``: it
introspects per-node signature/transform code and checks every edge port
and required input against the introspected fields.

Per-node execution traces are captured through a ``ContextVar`` sink so
serve and dry-run can replay the graph without slowing optimization rollouts
(no sink set → zero recording).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import dspy

from ...config import settings as app_settings
from ...exceptions import ServiceError
from ...models import WorkflowSpec, workflow_tool_users, workflow_topological_order
from ...models.workflow import WorkflowNode
from ...registry.resolvers import ResolverError, resolve_module_factory
from ..react_compat import REACT_CLASS
from ..safe_exec import validate_signature_code, validate_transform_code
from .data import extract_signature_fields, load_signature_from_code, load_transform_from_code
from .training_ground.run_react import resolve_react_tools

# Node ids become sub-module attribute names under this prefix so a node id
# like ``forward`` can never shadow a dspy.Module method, and GEPA predictor
# paths stay recognizable (``n_<node_id>...``).
WORKFLOW_NODE_ATTR_PREFIX = "n_"

_trace_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar("workflow_trace_sink", default=None)


class WorkflowNodeExecutionError(RuntimeError):
    """A workflow node failed at execution time.

    Carries the failing ``node_id`` so serve/dry-run responses can anchor
    the error to a canvas node instead of a generic failure banner.
    """

    def __init__(self, node_id: str, message: str) -> None:
        """Store the failing node and build the display message.

        Args:
            node_id: Id of the node that raised.
            message: Human-readable failure description.
        """
        super().__init__(f"Workflow node '{node_id}' failed: {message}")
        self.node_id = node_id


@contextmanager
def capture_node_traces() -> Iterator[list[dict[str, Any]]]:
    """Collect per-node execution traces for calls made inside the context.

    Yields:
        A list that fills with one trace dict per executed node:
        ``{node_id, kind, name, inputs, outputs, elapsed_ms, error}``.
        Recording is skipped entirely when no sink is active, so
        optimization rollouts pay nothing.
    """
    sink: list[dict[str, Any]] = []
    token = _trace_sink.set(sink)
    try:
        yield sink
    finally:
        _trace_sink.reset(token)


def _record_trace(
    node: WorkflowNode,
    inputs: dict[str, Any],
    outputs: dict[str, Any] | None,
    elapsed_ms: float,
    error: str | None = None,
) -> None:
    """Append a node execution record to the active trace sink, if any.

    Args:
        node: The node that just executed.
        inputs: Port values the node consumed.
        outputs: Port values the node produced (``None`` on failure).
        elapsed_ms: Wall-clock execution time in milliseconds.
        error: Failure description when the node raised.
    """
    sink = _trace_sink.get()
    if sink is None:
        return
    sink.append(
        {
            "node_id": node.id,
            "kind": node.kind,
            "name": node.name or node.id,
            "inputs": dict(inputs),
            "outputs": dict(outputs) if outputs is not None else None,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": error,
        }
    )


class WorkflowProgram(dspy.Module):
    """A composed DSPy program executing a workflow DAG in topological order.

    Signature nodes are plain attributes (``n_<node_id>``) holding their
    DSPy modules, which is exactly what GEPA's predictor discovery walks —
    every node is optimized jointly against the single end-to-end metric.
    A flex node is a ``dspy.Flex``, which GEPA picks up by type and whose
    rewritten code becomes its component instead of an instruction string;
    every other node contributes its predictors' instructions. Transforms
    and tools are held in dicts so they stay invisible to that discovery.
    """

    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        signature_modules: dict[str, Any],
        signature_outputs: dict[str, list[str]],
        transforms: dict[str, Callable[..., Any]],
        tools: dict[str, Any],
    ) -> None:
        """Assemble the program from pre-built node runners.

        Args:
            spec: The validated workflow spec.
            signature_modules: Node id to instantiated DSPy module.
            signature_outputs: Node id to the signature's output field names,
                used to project prediction objects onto the node's ports.
            transforms: Node id to loaded transform callable.
            tools: Node id to resolved ``dspy.Tool`` for mcp nodes.
        """
        super().__init__()
        self.spec = spec
        self.execution_order = workflow_topological_order(spec)
        self.signature_outputs = dict(signature_outputs)
        self._transforms = dict(transforms)
        self._tools = dict(tools)
        for node_id, module in signature_modules.items():
            setattr(self, WORKFLOW_NODE_ATTR_PREFIX + node_id, module)

    def node_module(self, node_id: str) -> Any:
        """Return the DSPy module backing a signature node.

        Args:
            node_id: The signature node's id.

        Returns:
            The sub-module registered for the node.
        """
        return getattr(self, WORKFLOW_NODE_ATTR_PREFIX + node_id)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Execute the DAG on one example and return the final outputs.

        Args:
            **kwargs: Workflow input port values (the input anchor's fields).

        Returns:
            A ``dspy.Prediction`` carrying the output anchor's fields.

        Raises:
            WorkflowNodeExecutionError: When any node fails, naming the node.
        """
        input_node = self.spec.input_node
        missing = [field.name for field in input_node.fields if field.name not in kwargs]
        if missing:
            raise WorkflowNodeExecutionError(input_node.id, f"missing workflow inputs: {missing}")

        values: dict[str, dict[str, Any]] = {}
        final_outputs: dict[str, Any] = {}
        for node_id in self.execution_order:
            node = self.spec.node_by_id(node_id)
            if node.kind == "input":
                outputs = {field.name: kwargs[field.name] for field in node.fields}
                values[node_id] = outputs
                _record_trace(node, {}, outputs, 0.0)
                continue

            inputs = {
                edge.target_port: values[edge.source][edge.source_port] for edge in self.spec.edges_into(node_id)
            }
            started = time.perf_counter()
            try:
                outputs = self._execute_node(node, inputs)
            except WorkflowNodeExecutionError as exc:
                _record_trace(node, inputs, None, (time.perf_counter() - started) * 1000.0, error=str(exc))
                raise
            except Exception as exc:
                _record_trace(node, inputs, None, (time.perf_counter() - started) * 1000.0, error=str(exc))
                raise WorkflowNodeExecutionError(node_id, str(exc)) from exc
            values[node_id] = outputs
            _record_trace(node, inputs, outputs, (time.perf_counter() - started) * 1000.0)
            if node.kind == "output":
                final_outputs = outputs

        return dspy.Prediction(**final_outputs)

    def _execute_node(self, node: WorkflowNode, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one non-input node and return its port values.

        Args:
            node: The node to execute.
            inputs: Port values gathered from the node's in-edges.

        Returns:
            The node's output port values.

        Raises:
            WorkflowNodeExecutionError: When a transform returns a malformed
                result or omits a declared output field.
        """
        if node.kind == "signature":
            prediction = self.node_module(node.id)(**inputs)
            # Project onto the signature's declared outputs only: react
            # predictions additionally carry trajectory bookkeeping that must
            # not leak onto downstream ports.
            return {name: getattr(prediction, name) for name in self.signature_outputs[node.id]}
        if node.kind == "transform":
            result = self._transforms[node.id](**inputs)
            if not isinstance(result, dict):
                raise WorkflowNodeExecutionError(
                    node.id, f"transform must return a dict, got {type(result).__name__}."
                )
            missing = [field.name for field in node.output_fields if field.name not in result]
            if missing:
                raise WorkflowNodeExecutionError(node.id, f"transform result is missing output fields: {missing}.")
            return {field.name: result[field.name] for field in node.output_fields}
        if node.kind == "mcp":
            with dspy.context(allow_tool_async_sync_conversion=True):
                result = self._tools[node.id](**inputs)
            return {node.output_field.name: result}
        return dict(inputs)


def build_workflow_program(
    spec: WorkflowSpec,
    *,
    tool_source: Any = None,
    dataset: list[dict[str, Any]] | None = None,
) -> tuple[WorkflowProgram, dict[str, str]]:
    """Instantiate the composed program for a workflow spec.

    Deterministic for a given spec: node modules are created in spec order,
    so predictor discovery order — and therefore saved-state keys — are
    stable between the optimizing worker and the serving API.

    Args:
        spec: The structurally valid workflow spec.
        tool_source: The run-level ``ToolSource`` (or persisted mapping);
            required when the graph contains react or mcp nodes.
        dataset: Raw dataset rows, needed only for ``dataset_snapshot``
            tool sources.

    Returns:
        ``(program, tool_schema_hashes)`` — the runnable program and the
        roster's schema-hash snapshot (empty when no tools are used).

    Raises:
        ServiceError: When a tool-using graph lacks a ``tool_source``, a
            node references an unknown tool, or node code fails to load.
    """
    roster: list[Any] = []
    schema_hashes: dict[str, str] = {}
    if workflow_tool_users(spec):
        if tool_source is None:
            raise ServiceError("workflow contains tool-using nodes; tool_source is required.")
        roster, schema_hashes = resolve_react_tools(tool_source, None, app_settings, dataset=dataset)
    roster_by_name = {tool.name: tool for tool in roster}

    signature_modules: dict[str, Any] = {}
    signature_outputs: dict[str, list[str]] = {}
    transforms: dict[str, Callable[..., Any]] = {}
    tools: dict[str, Any] = {}
    for node in spec.nodes:
        if node.kind == "signature":
            signature_cls = load_signature_from_code(node.signature_code)
            _, output_fields = extract_signature_fields(signature_cls)
            signature_outputs[node.id] = output_fields
            if node.module_name == "react":
                signature_modules[node.id] = REACT_CLASS(
                    signature_cls, tools=_filter_roster(roster, node.tool_filter, node.id)
                )
            else:
                try:
                    factory, _auto = resolve_module_factory(node.module_name)
                except ResolverError as exc:
                    # Reachable for flex on a dspy build without ``dspy.Flex``;
                    # name the node so the canvas can anchor the error.
                    raise ServiceError(
                        f"Workflow node '{node.id}' requests module '{node.module_name}', "
                        f"which this DSPy build cannot provide: {exc}"
                    ) from exc
                module_kwargs: dict[str, Any] = {"signature": signature_cls}
                if node.tool_filter is not None:
                    # Flex is the only other tool-capable module; giving it tools
                    # swaps its baseline from dspy.Predict to dspy.RLM.
                    module_kwargs["tools"] = _filter_roster(roster, node.tool_filter, node.id)
                signature_modules[node.id] = factory(**module_kwargs)
        elif node.kind == "transform":
            transforms[node.id] = load_transform_from_code(node.transform_code)
        elif node.kind == "mcp":
            tool = roster_by_name.get(node.tool_name)
            if tool is None:
                raise ServiceError(
                    f"Workflow node '{node.id}' references tool '{node.tool_name}', which is not in the "
                    f"resolved roster: {sorted(roster_by_name)}."
                )
            tools[node.id] = tool

    program = WorkflowProgram(
        spec,
        signature_modules=signature_modules,
        signature_outputs=signature_outputs,
        transforms=transforms,
        tools=tools,
    )
    return program, schema_hashes


def _filter_roster(roster: list[Any], tool_filter: list[str] | None, node_id: str) -> list[Any]:
    """Narrow the run-level roster to a node's tool filter.

    Args:
        roster: The resolved run-level tool roster.
        tool_filter: Tool names the node is limited to, or ``None`` for all.
        node_id: The node's id, for error messages.

    Returns:
        The filtered roster, preserving roster order.

    Raises:
        ServiceError: When the filter names a tool absent from the roster.
    """
    if tool_filter is None:
        return list(roster)
    available = {tool.name for tool in roster}
    missing = [name for name in tool_filter if name not in available]
    if missing:
        raise ServiceError(f"Workflow node '{node_id}' tool_filter references unknown tools: {missing}.")
    wanted = set(tool_filter)
    return [tool for tool in roster if tool.name in wanted]


class WorkflowIntrospection:
    """Deep-validation result: per-node ports plus the workflow's end-to-end fields."""

    def __init__(
        self,
        *,
        signature_fields: dict[str, tuple[list[str], list[str]]],
        input_fields: list[str],
        output_fields: list[str],
    ) -> None:
        """Store the introspected shape.

        Args:
            signature_fields: Node id to ``(input_fields, output_fields)``
                introspected from the node's signature code.
            input_fields: The workflow's input port names.
            output_fields: The workflow's final output port names.
        """
        self.signature_fields = signature_fields
        self.input_fields = input_fields
        self.output_fields = output_fields


def validate_workflow(spec: WorkflowSpec) -> WorkflowIntrospection:
    """Run the deep (exec-based) validation pass over a workflow spec.

    Complements the structural pass in ``models/workflow.py``: signature and
    transform code is introspected in isolated subprocesses, then every edge
    port touching a signature node and every node's required inputs are
    checked against the real fields.

    Args:
        spec: The structurally valid workflow spec.

    Returns:
        The introspected workflow shape.

    Raises:
        ServiceError: When node code fails to load, a port references a
            missing field, or a required input port is unconnected — always
            naming the offending node so the canvas can anchor the error.
    """
    signature_fields: dict[str, tuple[list[str], list[str]]] = {}
    for node in spec.nodes:
        if node.kind == "signature":
            try:
                intro = validate_signature_code(node.signature_code)
            except ServiceError as exc:
                raise ServiceError(f"Workflow node '{node.id}': {exc}") from exc
            if intro.image_input_fields:
                raise ServiceError(
                    f"Workflow node '{node.id}' declares image input fields "
                    f"({intro.image_input_fields}); image fields are not supported inside workflows yet."
                )
            signature_fields[node.id] = (list(intro.input_fields), list(intro.output_fields))
        elif node.kind == "transform":
            try:
                transform_intro = validate_transform_code(node.transform_code)
            except ServiceError as exc:
                raise ServiceError(f"Workflow node '{node.id}': {exc}") from exc
            declared = sorted(field.name for field in node.input_fields)
            actual = sorted(transform_intro.param_names)
            if declared != actual:
                raise ServiceError(
                    f"Workflow node '{node.id}': transform parameters {actual} do not match the "
                    f"declared input fields {declared}."
                )

    for edge in spec.edges:
        source = spec.node_by_id(edge.source)
        target = spec.node_by_id(edge.target)
        if source.kind == "signature" and edge.source_port not in signature_fields[source.id][1]:
            raise ServiceError(
                f"Workflow node '{source.id}' has no output field '{edge.source_port}' "
                f"(available: {signature_fields[source.id][1]})."
            )
        if target.kind == "signature" and edge.target_port not in signature_fields[target.id][0]:
            raise ServiceError(
                f"Workflow node '{target.id}' has no input field '{edge.target_port}' "
                f"(available: {signature_fields[target.id][0]})."
            )

    fed_ports = {(edge.target, edge.target_port) for edge in spec.edges}
    for node in spec.nodes:
        if node.kind == "signature":
            required = signature_fields[node.id][0]
        elif node.kind in ("transform", "mcp"):
            required = [field.name for field in node.input_fields]
        else:
            continue
        unconnected = [name for name in required if (node.id, name) not in fed_ports]
        if unconnected:
            raise ServiceError(f"Workflow node '{node.id}' has unconnected input fields: {unconnected}.")

    return WorkflowIntrospection(
        signature_fields=signature_fields,
        input_fields=spec.input_field_names(),
        output_fields=spec.output_field_names(),
    )
