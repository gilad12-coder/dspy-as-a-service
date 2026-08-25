"""Assemble a self-contained, runnable export of a compiled DSPy program.

The platform serves optimized programs through its own inference API, which
ties a caller to a Skynet-hosted endpoint. This module instead packages the
exact artifact the gateway persists — DSPy's state-only JSON plus the signature
source and module recipe — into a zip the user can run anywhere with plain
``dspy`` and their own LM key. The bundle mirrors how the gateway itself rebuilds
a program in ``_helpers._materialize_program`` (signature_code -> module factory
-> ``load_state``), so an exported program reconstructs to the one that was
optimized.
"""

from __future__ import annotations

import io
import json
import zipfile
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ....constants import (
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODULE_KWARGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_SIGNATURE_CODE,
    PAYLOAD_OVERVIEW_WORKFLOW,
)
from ....models import ProgramArtifact

EXPORT_FORMAT_VERSION = 1

# Standalone loader shipped inside the bundle. Plain string (NOT an f-string):
# the ``{...}`` below are the loader's own runtime f-strings and must survive
# verbatim into the generated file. Depends on ``dspy`` only — no platform code.
_LOADER_PY = '''"""Standalone loader for an exported Skynet / DSPy program.

Rebuilds the optimized program from the files in this folder using plain
``dspy`` — no Skynet account, platform API, or network call back to the service.
Bring your own LM provider key (OpenAI, Anthropic, ... via LiteLLM).

Quick start
-----------
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...            # or your provider's key
    python load_program.py                  # smoke-loads and prints the program

Use it in your own code
-----------------------
    from load_program import load_program, configure_lm

    configure_lm("openai/gpt-4o-mini")       # any LiteLLM model string
    program = load_program()
    result = program(question="...")          # use YOUR signature's input fields
    print(result)

Workflow exports rebuild the whole graph the same way — see workflow.json and
the "Workflow graphs" section of README.md.
"""

from __future__ import annotations

import copy
import heapq
import importlib
import json
import pathlib

import dspy

_HERE = pathlib.Path(__file__).resolve().parent
_META = json.loads((_HERE / "metadata.json").read_text(encoding="utf-8"))

# Short aliases the platform uses; a stored module_name may instead be a
# fully-qualified ``dspy.*`` path, which the resolver below also accepts.
_MODULE_ALIASES = {
    "predict": ("dspy.Predict",),
    "cot": ("dspy.modules.ChainOfThought", "dspy.ChainOfThought"),
    "react": ("dspy.ReActV2", "dspy.ReAct"),
    "flex": ("dspy.Flex",),
}

# Workflow node ids become sub-module attributes under this prefix, matching
# the platform builder so the saved per-node state keys line up on load.
_NODE_PREFIX = "n_"


def _exec_user_code(code, origin):
    """Execute one snippet of exported user code and return its namespace."""
    namespace = {"dspy": dspy}
    # dont_inherit=True keeps this file's ``from __future__ import annotations``
    # from stringizing the user code's annotations.
    exec(compile(code, origin, "exec", dont_inherit=True), namespace)
    return namespace


def _signature_from_code(code, origin):
    """Return the single dspy.Signature subclass defined by a code snippet."""
    found = [
        obj
        for obj in _exec_user_code(code, origin).values()
        if isinstance(obj, type)
        and issubclass(obj, dspy.Signature)
        and obj is not dspy.Signature
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"{origin} must define exactly one dspy.Signature subclass, found {len(found)}"
        )
    return found[0]


def _load_signature():
    """Return the task signature defined by signature.py."""
    return _signature_from_code(
        (_HERE / "signature.py").read_text(encoding="utf-8"), "signature.py"
    )


def _load_transform(code, node_id):
    """Return the callable a workflow transform node's code defines."""
    namespace = _exec_user_code(code, f"{node_id}_transform.py")
    transform = namespace.get("transform")
    if not callable(transform):
        candidates = [
            obj
            for name, obj in namespace.items()
            if callable(obj) and not isinstance(obj, type) and not name.startswith("__") and name != "dspy"
        ]
        if len(candidates) == 1:
            transform = candidates[0]
    if not callable(transform):
        raise RuntimeError(f"workflow node {node_id!r}: transform code defines no callable named 'transform'")
    return transform


def _resolve_module(name):
    """Resolve a module alias or ``dspy.*`` path to a dspy module class."""
    paths = _MODULE_ALIASES.get(name.lower(), (name,))
    for path in paths:
        if not path.startswith("dspy."):
            raise RuntimeError(f"unsupported module {name!r}; expected a dspy.* class")
        try:
            module_path, attribute = path.rsplit(".", 1)
            obj = getattr(importlib.import_module(module_path), attribute)
        except (AttributeError, ModuleNotFoundError):
            continue
        if callable(obj):
            return obj
    raise RuntimeError(f"module {name!r} is unavailable in this dspy installation")


def _prepare_tools(tools, overlay=None):
    """Clone supplied tools and apply the optimized ReAct tool overlay.

    Args:
        tools: Callables or ``dspy.Tool`` objects supplied by the user.
        overlay: Saved descriptions, argument descriptions, and tool names.

    Returns:
        An isolated list of ``dspy.Tool`` objects matching the optimized roster.
    """
    normalized = [tool if isinstance(tool, dspy.Tool) else dspy.Tool(tool) for tool in (tools or [])]
    isolated = [
        tool.model_copy(deep=True) if hasattr(tool, "model_copy") else copy.deepcopy(tool)
        for tool in normalized
    ]
    if not overlay:
        return isolated

    by_name = {tool.name: tool for tool in isolated}
    for tool_name, description in (overlay.get("tool_descriptions") or {}).items():
        tool = by_name.get(tool_name)
        if tool is not None and description:
            tool.desc = description
    for tool_name, arg_descriptions in (overlay.get("tool_arg_descriptions") or {}).items():
        tool = by_name.get(tool_name)
        if tool is None or not isinstance(tool.args, dict):
            continue
        for arg_name, description in arg_descriptions.items():
            schema = tool.args.get(arg_name)
            if isinstance(schema, dict) and description:
                schema["description"] = description

    proposed_names = overlay.get("tool_names") or {}
    original_names = set(by_name)
    desired = {
        tool.name: proposed_names.get(tool.name) or tool.name
        for tool in isolated
    }
    claimants = {}
    for canonical, desired_name in desired.items():
        claimants.setdefault(desired_name, []).append(canonical)
    for tool in isolated:
        canonical = tool.name
        desired_name = desired[canonical]
        if (
            desired_name != canonical
            and len(claimants[desired_name]) == 1
            and desired_name not in original_names
        ):
            tool.name = desired_name
    return isolated


def _topological_order(spec):
    """Order workflow node ids by dependency, ties broken by position in the spec."""
    index = {node["id"]: position for position, node in enumerate(spec["nodes"])}
    indegree = {node_id: 0 for node_id in index}
    outgoing = {node_id: set() for node_id in index}
    for edge in spec.get("edges", []):
        # Parallel edges between the same pair (different ports) count once.
        if edge["target"] not in outgoing[edge["source"]]:
            outgoing[edge["source"]].add(edge["target"])
            indegree[edge["target"]] += 1
    ready = [(index[node_id], node_id) for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, node_id = heapq.heappop(ready)
        ordered.append(node_id)
        for child in sorted(outgoing[node_id], key=index.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (index[child], child))
    if len(ordered) != len(index):
        raise RuntimeError("workflow.json contains a cycle")
    return ordered


class WorkflowProgram(dspy.Module):
    """Runs an exported workflow graph node by node, in dependency order."""

    def __init__(self, spec, modules, output_fields, transforms, tools):
        """Assemble the graph from its pre-built node runners."""
        super().__init__()
        self.spec = spec
        self.nodes = {node["id"]: node for node in spec["nodes"]}
        self.execution_order = _topological_order(spec)
        self._output_fields = output_fields
        self._transforms = transforms
        self._tools = tools
        for node_id, module in modules.items():
            setattr(self, _NODE_PREFIX + node_id, module)

    def forward(self, **kwargs):
        """Execute the graph on one example and return the output anchor's fields."""
        values = {}
        final = {}
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            kind = node["kind"]
            if kind == "input":
                missing = [f["name"] for f in node["fields"] if f["name"] not in kwargs]
                if missing:
                    raise RuntimeError(f"missing workflow inputs: {missing}")
                values[node_id] = {f["name"]: kwargs[f["name"]] for f in node["fields"]}
                continue

            inputs = {
                edge["target_port"]: values[edge["source"]][edge["source_port"]]
                for edge in self.spec.get("edges", [])
                if edge["target"] == node_id
            }
            if kind == "signature":
                prediction = getattr(self, _NODE_PREFIX + node_id)(**inputs)
                # Project onto declared outputs only: a react prediction also
                # carries trajectory bookkeeping that must not reach downstream ports.
                outputs = {name: getattr(prediction, name) for name in self._output_fields[node_id]}
            elif kind == "transform":
                result = self._transforms[node_id](**inputs)
                if not isinstance(result, dict):
                    raise RuntimeError(f"workflow node {node_id!r}: transform must return a dict")
                outputs = {f["name"]: result[f["name"]] for f in node["output_fields"]}
            elif kind == "mcp":
                with dspy.context(allow_tool_async_sync_conversion=True):
                    outputs = {node["output_field"]["name"]: self._tools[node_id](**inputs)}
            else:
                outputs = dict(inputs)
            values[node_id] = outputs
            if kind == "output":
                final = outputs
        return dspy.Prediction(**final)


def _node_tools(node, roster):
    """Pick the tools one signature node runs with, or None when it takes none.

    ReAct is defined by tool use, so it gets the whole roster unless it names a
    filter; every other module takes tools only by naming them.
    """
    tool_filter = node.get("tool_filter")
    if tool_filter is None:
        return list(roster.values()) if node.get("module_name") == "react" else None
    missing = [name for name in tool_filter if name not in roster]
    if missing:
        raise RuntimeError(f"workflow node {node['id']!r} needs tools that were not supplied: {missing}")
    wanted = set(tool_filter)
    return [tool for name, tool in roster.items() if name in wanted]


def _build_workflow(tools):
    """Rebuild the workflow graph shell described by workflow.json."""
    spec = json.loads((_HERE / "workflow.json").read_text(encoding="utf-8"))
    tool_users = [
        node["id"]
        for node in spec["nodes"]
        if node["kind"] == "mcp"
        or (node["kind"] == "signature" and (node.get("module_name") == "react" or node.get("tool_filter")))
    ]
    if tool_users and not tools:
        raise RuntimeError(
            f"This workflow has tool-using nodes {tool_users}. Re-supply the tools "
            "you optimized against: load_program(tools=[my_tool, ...]). Tools are "
            "matched to nodes by name."
        )

    roster = {tool.name: tool for tool in _prepare_tools(tools)}
    modules, output_fields, transforms, node_tools = {}, {}, {}, {}
    for node in spec["nodes"]:
        node_id = node["id"]
        if node["kind"] == "signature":
            signature = _signature_from_code(node["signature_code"], f"{node_id}_signature.py")
            output_fields[node_id] = list(signature.output_fields)
            factory = _resolve_module(node.get("module_name") or "predict")
            selected = _node_tools(node, roster)
            modules[node_id] = (
                factory(signature=signature)
                if selected is None
                else factory(signature=signature, tools=selected)
            )
        elif node["kind"] == "transform":
            transforms[node_id] = _load_transform(node["transform_code"], node_id)
        elif node["kind"] == "mcp":
            if node["tool_name"] not in roster:
                raise RuntimeError(
                    f"workflow node {node_id!r} calls tool {node['tool_name']!r}, which was not supplied"
                )
            node_tools[node_id] = roster[node["tool_name"]]
    return WorkflowProgram(spec, modules, output_fields, transforms, node_tools)


def load_program(tools=None):
    """Rebuild the optimized program and load its trained state.

    Pass ``tools=[...]`` for ReAct programs and for workflows containing
    tool-using nodes (the same tool callables you optimized against); other
    module types ignore it.
    """
    if _META.get("is_workflow"):
        program = _build_workflow(tools)
    else:
        signature = _load_signature()
        factory = _resolve_module(_META["module_name"])
        kwargs = dict(_META.get("module_kwargs") or {})
        kwargs.pop("signature", None)
        if _META.get("is_react"):
            if not tools:
                raise RuntimeError(
                    "This is a ReAct program. Re-supply the tools you optimized "
                    "against: load_program(tools=[my_tool, ...]). The optimized tool "
                    "descriptions are in react_overlay.json."
                )
            overlay = json.loads((_HERE / "react_overlay.json").read_text(encoding="utf-8"))
            kwargs.pop("tools", None)
            kwargs["max_iters"] = overlay.get("max_iters", kwargs.get("max_iters", 20))
            program = factory(
                signature=signature,
                tools=_prepare_tools(tools, overlay),
                **kwargs,
            )
        else:
            kwargs["signature"] = signature
            program = factory(**kwargs)
    state = json.loads((_HERE / "program.json").read_text(encoding="utf-8"))
    program.load_state(state)
    return program


def configure_lm(model=None):
    """Point dspy at an LM. Defaults to the model the program was optimized on."""
    dspy.configure(lm=dspy.LM(model or _META.get("model") or "openrouter/openai/gpt-4o-mini"))


if __name__ == "__main__":
    configure_lm()
    loaded = load_program()
    print(f"Loaded {type(loaded).__name__} (optimization {_META.get('optimization_id')})")
    print(loaded)
'''


_MODULE_HEADER = (
    "# Optimized module source (dspy.Flex).\n"
    "#\n"
    "# GEPA rewrote this program's code alongside its prompt. This file is a\n"
    "# human-readable copy for reference; the loader reconstructs the program from\n"
    "# program.json (which embeds the same source), so you do not import this file.\n"
    "\n"
)


def _optimized_module_file(module_src: str) -> str:
    """Render the readable ``optimized_module.py`` body for a Flex export.

    Args:
        module_src: The GEPA-rewritten module source from the artifact.

    Returns:
        The module source prefixed with a ``#`` comment header, kept as comments
        so a leading ``from __future__`` import in the source stays first-statement
        valid.
    """
    return _MODULE_HEADER + module_src.rstrip("\n") + "\n"


def _installed_dspy_version() -> str | None:
    """Return the installed ``dspy`` version, or ``None`` when unavailable.

    Returns:
        The version string of the ``dspy`` distribution, or ``None`` when the
        package metadata cannot be located.
    """
    try:
        return version("dspy")
    except PackageNotFoundError:
        return None


def _build_metadata(optimization_id: str, artifact: ProgramArtifact, overview: dict[str, Any]) -> dict[str, Any]:
    """Build the ``metadata.json`` payload the loader reads at runtime.

    Args:
        optimization_id: Optimization id the export belongs to.
        artifact: The compiled program artifact being exported.
        overview: Parsed payload-overview dict supplying the module recipe.

    Returns:
        A JSON-serializable dict carrying the module recipe, default model, and
        provenance the standalone loader and README need.
    """
    module_name = overview.get(PAYLOAD_OVERVIEW_MODULE_NAME) or "predict"
    module_leaf = str(module_name).rsplit(".", 1)[-1].lower()
    return {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "optimization_id": optimization_id,
        "module_name": module_name,
        "module_kwargs": dict(overview.get(PAYLOAD_OVERVIEW_MODULE_KWARGS, {})),
        "model": overview.get(PAYLOAD_OVERVIEW_MODEL_NAME),
        "optimizer": overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
        "dspy_version": _installed_dspy_version(),
        "is_react": artifact.react_overlay is not None or module_leaf in {"react", "reactv2"},
        "is_flex": (
            artifact.optimized_module_src is not None
            or bool(artifact.optimized_component_srcs)
            or module_leaf == "flex"
        ),
        "flex_components": sorted(artifact.optimized_component_srcs),
        "is_workflow": bool(overview.get(PAYLOAD_OVERVIEW_WORKFLOW)),
    }


def _build_readme(metadata: dict[str, Any]) -> str:
    """Render the bundle README from the export metadata.

    Args:
        metadata: The metadata dict produced by :func:`_build_metadata`.

    Returns:
        Markdown documenting how to run the exported program standalone.
    """
    dspy_version = metadata.get("dspy_version") or "unknown"
    model = metadata.get("model") or "your provider's model (e.g. openai/gpt-4o-mini)"
    react_note = (
        "\n## ReAct tools\n\n"
        "This program is a ReAct agent, so its tool roster is **not** baked into "
        "the saved state. Re-supply the same tools you optimized against:\n\n"
        "```python\n"
        "program = load_program(tools=[my_tool, my_other_tool])\n"
        "```\n\n"
        "The optimized tool/argument descriptions are in `react_overlay.json` for reference.\n"
        if metadata.get("is_react")
        else ""
    )
    components = metadata.get("flex_components") or []
    flex_where = (
        "sources are in `optimized_modules/` — one file per Flex submodule"
        if components
        else "source is in `optimized_module.py`"
    )
    flex_contents: list[str] = []
    if components:
        flex_contents = [
            f"- `optimized_modules/` — the GEPA-rewritten source of each Flex submodule: {', '.join(components)}."
        ]
    elif metadata.get("is_flex"):
        flex_contents = ["- `optimized_module.py` — the GEPA-rewritten program source, human-readable."]
    flex_note = (
        "\n## Optimized code (Flex)\n\n"
        "GEPA rewrote this program's **code** as well as its prompt. The rewritten "
        f"{flex_where} for you to read; `load_program()` rebuilds it from "
        "`program.json` (which embeds the same source), so you do not import those "
        "files directly.\n"
        if metadata.get("is_flex")
        else ""
    )
    is_workflow = metadata.get("is_workflow")
    program_entry = (
        "- `workflow.json` — the graph: every node's module and code, plus the edges wiring their ports."
        if is_workflow
        else "- `signature.py` — the task signature the program was built on."
    )
    workflow_note = (
        "\n## Workflow graphs\n\n"
        "This program is a graph, not a single module. `load_program()` rebuilds one "
        "DSPy module per signature node — attached as `n_<node_id>`, which is how the "
        "saved state is keyed — runs transform nodes as plain Python, and executes "
        "everything in dependency order. Call it with the input anchor's field names.\n\n"
        "Nodes that use tools (ReAct nodes, MCP nodes, and Flex nodes with a "
        "`tool_filter`) need the roster re-supplied, matched to nodes **by name**:\n\n"
        "```python\n"
        "program = load_program(tools=[my_tool, my_other_tool])\n"
        "```\n"
        if is_workflow
        else ""
    )
    lines = [
        "# Exported DSPy program",
        "",
        f"Optimization `{metadata.get('optimization_id')}`, exported from Skynet.",
        "",
        "This is the actual compiled program — not a hosted endpoint. You run it",
        "yourself with plain `dspy` and your own LM key. Nothing here calls back",
        "to the platform.",
        "",
        "## Contents",
        "",
        "- `program.json` — the optimized DSPy state (`module.save(..., save_program=False)`).",
        program_entry,
        "- `load_program.py` — rebuilds the module and loads the state. Run or import it.",
        "- `metadata.json` — the module recipe (module name, kwargs, default model).",
        "- `prompt.json` — the optimized instructions and few-shot demos, human-readable.",
        *flex_contents,
        "- `requirements.txt` — the `dspy` version this was trained on.",
        "",
        "## Run it",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "export OPENAI_API_KEY=sk-...   # or your provider's key",
        "python load_program.py         # smoke-loads and prints the program",
        "```",
        "",
        "```python",
        "from load_program import load_program, configure_lm",
        "",
        f'configure_lm("{model}")',
        "program = load_program()",
        'result = program(field_name="...")   # use your signature\'s input fields',
        "print(result)",
        "```",
        react_note,
        workflow_note,
        flex_note,
        "## Reproducibility",
        "",
        f"Trained against `dspy=={dspy_version}`. Pin the same version for an exact",
        "match — DSPy's state format can drift across majors.",
        "",
    ]
    return "\n".join(lines)


def build_program_export_zip(
    *,
    optimization_id: str,
    artifact: ProgramArtifact,
    overview: dict[str, Any],
) -> bytes:
    """Package a runnable, self-contained DSPy program export as zip bytes.

    Callers must validate first that ``artifact.program_state_json`` is present
    along with the program definition the module shape calls for — the
    overview's ``workflow`` for a workflow run, its ``signature_code``
    otherwise; this builder assumes them.

    Args:
        optimization_id: Optimization id the export belongs to.
        artifact: The compiled program artifact (state JSON, optional react
            overlay, optimized prompt).
        overview: Parsed payload-overview dict supplying the program definition
            and the module recipe.

    Returns:
        The bytes of a zip archive containing the program state, the signature
        source or workflow graph, a standalone loader, metadata, and a README.
    """
    metadata = _build_metadata(optimization_id, artifact, overview)
    workflow_spec = overview.get(PAYLOAD_OVERVIEW_WORKFLOW)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("program.json", json.dumps(artifact.program_state_json, indent=2, ensure_ascii=False))
        if workflow_spec:
            archive.writestr("workflow.json", json.dumps(workflow_spec, indent=2, ensure_ascii=False))
        else:
            archive.writestr("signature.py", overview.get(PAYLOAD_OVERVIEW_SIGNATURE_CODE) or "")
        archive.writestr("load_program.py", _LOADER_PY)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        if artifact.optimized_prompt is not None:
            archive.writestr(
                "prompt.json",
                artifact.optimized_prompt.model_dump_json(indent=2),
            )
        if artifact.optimized_module_src is not None:
            archive.writestr(
                "optimized_module.py",
                _optimized_module_file(artifact.optimized_module_src),
            )
        for path, module_src in sorted(artifact.optimized_component_srcs.items()):
            archive.writestr(
                f"optimized_modules/{path}.py",
                _optimized_module_file(module_src),
            )
        if metadata["is_react"]:
            react_overlay = (
                artifact.react_overlay.model_dump()
                if artifact.react_overlay is not None
                else {
                    "tool_descriptions": {},
                    "tool_arg_descriptions": {},
                    "tool_schema_hashes": {},
                    "max_iters": metadata["module_kwargs"].get("max_iters", 20),
                    "tool_names": None,
                }
            )
            archive.writestr(
                "react_overlay.json",
                json.dumps(react_overlay, indent=2, ensure_ascii=False),
            )
        dspy_version = metadata.get("dspy_version")
        requirement = f"dspy=={dspy_version}\n" if dspy_version else "dspy\n"
        archive.writestr("requirements.txt", requirement)
        archive.writestr("README.md", _build_readme(metadata))

    return buffer.getvalue()
