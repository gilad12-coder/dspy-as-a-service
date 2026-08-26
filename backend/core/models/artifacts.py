"""Compiled DSPy program artifact models (prompts, demos, serialized pickle)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# Workflow nodes register as ``n_<node_id>`` attributes (see workflow.py's
# WORKFLOW_NODE_ATTR_PREFIX), so their saved predictor-state keys carry that
# prefix. Hardcoded rather than imported to keep the models layer free of an
# optimization-layer dependency.
_WORKFLOW_NODE_KEY_PREFIX = "n_"

# DSPy stamps these bookkeeping keys onto a serialized demo alongside the real
# field values; they are not prompt content, so the state back-fill drops them.
_DEMO_METADATA_KEYS = frozenset({"augmented", "dspy_uuid"})


class OptimizedDemo(BaseModel):
    """A single few-shot demonstration example from an optimized predictor."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)


class OptimizedPredictor(BaseModel):
    """Extracted prompt and demos from a single predictor in the compiled program."""

    predictor_name: str
    signature_name: str | None = None
    instructions: str
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    demos: list[OptimizedDemo] = Field(default_factory=list)
    formatted_prompt: str = Field(
        default="",
        description="Complete prompt as a single formatted string including instructions and demos.",
    )


# Tool-level overlay carried by a react ``POST /run`` artifact: the optimized
# per-tool descriptions and arg descriptions, the schema-hash snapshot the seed
# program was built against, the ReActV2 loop budget, and the originating tool
# source. Phase B persists this alongside the program state so a served react
# bundle can reconstruct its tool surface.
class ReactOverlay(BaseModel):
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    tool_arg_descriptions: dict[str, dict[str, str]] = Field(default_factory=dict)
    tool_schema_hashes: dict[str, str] = Field(default_factory=dict)
    max_iters: int
    tool_source: dict[str, Any] | None = None
    # GEPA-proposed agent-facing display names, ``{canonical: proposed}``. Serve
    # renames the re-sourced canonical tools to these AFTER drift-check + desc/arg
    # overlays. None (the default) preserves pre-rename behavior exactly.
    tool_names: dict[str, str] | None = Field(default=None)
    # Per-tool approval severity (``info``/``warning``/``destructive``) derived
    # from the source MCP's tool annotations, ``{tool_name: severity}``. Only
    # tools whose server stated a hint appear; omitted tools carry no severity so
    # the UI never fabricates one. Empty by default for pre-severity artifacts.
    tool_severities: dict[str, str] = Field(default_factory=dict)


class NodeArtifact(BaseModel):
    """The optimized surface of a single workflow node.

    Bundles whichever of a node's optimized outputs exist: a signature node
    carries its ``optimized_prompt`` (and, for react nodes, a ``react_overlay``);
    a flex node carries its rewritten ``optimized_src``. Keyed under the node's
    component path (``n_<node_id>``) in :attr:`ProgramArtifact.optimized_nodes`.
    """

    optimized_prompt: OptimizedPredictor | None = Field(
        default=None,
        description="Extracted prompt and demos for a signature node's predictor.",
    )
    react_overlay: ReactOverlay | None = Field(
        default=None,
        description=(
            "Tool overlay for a react node. Unset until per-node overlays are "
            "captured; back-fill from state cannot recover it."
        ),
    )
    optimized_src: str | None = Field(
        default=None,
        description="GEPA-rewritten source for a flex node. Unset for non-flex nodes.",
    )


def _field_label(field: dict[str, Any]) -> str | None:
    """Render one saved signature field as a prompt label line.

    Args:
        field: A ``{"prefix": ..., "description": ...}`` entry from a saved
            ``signature.fields`` list.

    Returns:
        A ``"Prefix: description"`` line — the description dropped when it is a
        ``${...}`` adapter placeholder or empty — or ``None`` when the field
        carries no usable prefix.
    """
    prefix = field.get("prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        return None
    desc = field.get("description")
    if isinstance(desc, str):
        desc = desc.strip()
        if desc and not (desc.startswith("${") and desc.endswith("}")):
            return f"{prefix} {desc}"
    return prefix


def _demo_fields(demo: dict[str, Any]) -> dict[str, Any]:
    """Strip DSPy bookkeeping keys from a serialized demo dict.

    Args:
        demo: One serialized demonstration from a predictor's ``demos`` state.

    Returns:
        The demo's field values, without DSPy metadata or private keys.
    """
    return {
        key: value
        for key, value in demo.items()
        if isinstance(key, str) and not key.startswith("_") and key not in _DEMO_METADATA_KEYS
    }


def _predictor_from_node_state(predictor_name: str, state: dict[str, Any]) -> OptimizedPredictor | None:
    """Rebuild a workflow node's prompt view from its saved predictor state.

    Recovers instructions, few-shot demos, and a readable prompt string from
    the state JSON alone — the surface a node needs when its run was persisted
    before per-node prompts were extracted at write time. State carries field
    *prefixes* (what the LM actually sees) but not field names or input/output
    roles, so fields render by prefix and demos carry no role split; the
    live-program extractor recovers those for new runs.

    Args:
        predictor_name: The predictor's saved-state key (a ``named_predictors``
            path such as ``n_polish.predict``).
        state: The predictor's saved state (``{signature, demos, ...}``).

    Returns:
        The reconstructed :class:`OptimizedPredictor`, or ``None`` when the
        state carries no signature (e.g. a flex node's code-only state).
    """
    signature = state.get("signature")
    if not isinstance(signature, dict):
        return None

    raw_instructions = signature.get("instructions")
    instructions = raw_instructions if isinstance(raw_instructions, str) else ""

    field_lines: list[str] = []
    fields = signature.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict):
                line = _field_label(field)
                if line is not None:
                    field_lines.append(line)

    raw_demos = state.get("demos")
    demos: list[OptimizedDemo] = (
        [OptimizedDemo(inputs=_demo_fields(demo)) for demo in raw_demos if isinstance(demo, dict)]
        if isinstance(raw_demos, list)
        else []
    )

    parts: list[str] = []
    if instructions:
        parts.append(instructions)
        parts.append("")
    if field_lines:
        parts.append("Fields:")
        parts.extend(field_lines)
        parts.append("")
    if demos:
        parts.append("---")
        parts.append("Examples:")
        parts.append("")
        for index, demo in enumerate(demos, 1):
            parts.append(f"Example {index}:")
            for name, value in demo.inputs.items():
                parts.append(f"  {name}: {value}")
            parts.append("")

    return OptimizedPredictor(
        predictor_name=predictor_name,
        instructions=instructions,
        demos=demos,
        formatted_prompt="\n".join(parts).strip(),
    )


class ProgramArtifact(BaseModel):
    """Serializable payload that carries the optimized DSPy program files."""

    path: str | None = Field(
        default=None,
        description="Absolute path on the server where the artifact lives.",
    )
    program_state_json: dict[str, Any] | None = Field(
        default=None,
        description=(
            "State-only JSON dump from ``module.save(path.json)``. Loaded by "
            "reconstructing the module from signature_code + module_name and "
            "calling ``program.load(json_path)``."
        ),
    )
    program_pickle_base64: str | None = Field(
        default=None,
        description=(
            "Deprecated. Base64-encoded ``program.pkl`` retained only so jobs "
            "saved before the JSON migration can still be served. New jobs "
            "leave this field unset."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="metadata.json contents already parsed into a dict.",
    )
    optimized_prompt: OptimizedPredictor | None = Field(
        default=None,
        description="Extracted prompt and demos from the compiled program predictor.",
    )
    react_overlay: ReactOverlay | None = Field(
        default=None,
        description=(
            "Tool-level overlay for a react run: optimized tool/arg descriptions, "
            "the schema-hash snapshot, and the ReActV2 loop budget. Unset for "
            "non-react artifacts."
        ),
    )
    optimized_module_src: str | None = Field(
        default=None,
        description=(
            "GEPA-rewritten module source for a dspy.Flex program: the optimized "
            "Python that runs in the serve sandbox. Unset for non-Flex artifacts, "
            "whose optimization lands in the prompt rather than the code."
        ),
    )
    optimized_component_srcs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "GEPA-rewritten module source per Flex submodule, keyed by its component "
            "path (a workflow's flex node is 'n_<node_id>'). Empty unless the program "
            "nests Flex modules rather than being one itself."
        ),
    )
    optimized_nodes: dict[str, NodeArtifact] = Field(
        default_factory=dict,
        description=(
            "Per-node optimized surface for a workflow program, keyed by component "
            "path ('n_<node_id>'): each node's prompt, react overlay, or rewritten "
            "code. Empty for scalar (single-module) programs."
        ),
    )

    @model_validator(mode="after")
    def _backfill_module_src(self) -> ProgramArtifact:
        """Derive the Flex sources from persisted program state when absent.

        A top-level Flex saves ``module_src`` at the root of ``program.save(json)``;
        nested ones (a workflow's flex nodes) save it under their component path, so
        state is scanned one level deep too. Back-filling on validation surfaces the
        code for old and new artifacts alike without a data migration; non-Flex state
        has no ``module_src`` key, so this is a no-op there.

        Returns:
            The validated artifact, with the Flex sources populated wherever they
            are recoverable.
        """
        if not isinstance(self.program_state_json, dict):
            return self
        if self.optimized_module_src is None:
            src = self.program_state_json.get("module_src")
            if isinstance(src, str) and src.strip():
                self.optimized_module_src = src
        if not self.optimized_component_srcs:
            self.optimized_component_srcs = {
                path: state["module_src"]
                for path, state in self.program_state_json.items()
                if isinstance(state, dict)
                and isinstance(state.get("module_src"), str)
                and state["module_src"].strip()
            }
        return self

    @model_validator(mode="after")
    def _fold_flex_src_into_nodes(self) -> ProgramArtifact:
        """Mirror each flex node's rewritten source into ``optimized_nodes``.

        ``optimized_nodes`` is the unified per-node view the workflow-as-artifact
        UI reads; flex sources are still carried in ``optimized_component_srcs``
        (and back-filled there for old artifacts) so this folds them in under the
        same ``n_<node_id>`` key without a data migration. Runs after
        ``_backfill_module_src`` so the sources are already populated. Prompts are
        set at persist time; this only supplies the code half, so a flex-only node
        still gets an entry.

        Returns:
            The validated artifact, with each flex source attached to its node.
        """
        for path, src in self.optimized_component_srcs.items():
            node = self.optimized_nodes.get(path)
            if node is None:
                self.optimized_nodes[path] = NodeArtifact(optimized_src=src)
            elif node.optimized_src is None:
                node.optimized_src = src
        return self

    @model_validator(mode="after")
    def _backfill_prompts_into_nodes(self) -> ProgramArtifact:
        """Rebuild per-node prompts from saved state for pre-extraction runs.

        Workflow runs persisted before prompts were extracted at write time
        carry each node's tuned prompt only inside ``program_state_json``, keyed
        by predictor path. This surfaces them under ``optimized_nodes`` — keyed
        by node path (``n_<node_id>``, folding a cot node's ``.predict``
        predictor back to its node) — so old and new workflow artifacts render
        identically with no data migration. Nodes already carrying a prompt
        (new runs, extracted at persist time) are left untouched, and the first
        predictor per node wins, mirroring the write-time extractor. Scalar
        runs, whose predictors carry no ``n_`` prefix, are unaffected.

        Returns:
            The validated artifact, with each recoverable node prompt attached.
        """
        if not isinstance(self.program_state_json, dict):
            return self
        for key, node_state in self.program_state_json.items():
            if not (isinstance(key, str) and key.startswith(_WORKFLOW_NODE_KEY_PREFIX)):
                continue
            if not isinstance(node_state, dict):
                continue
            node_path = key.split(".", 1)[0]
            existing = self.optimized_nodes.get(node_path)
            if existing is not None and existing.optimized_prompt is not None:
                continue
            prompt = _predictor_from_node_state(key, node_state)
            if prompt is None:
                continue
            if existing is None:
                self.optimized_nodes[node_path] = NodeArtifact(optimized_prompt=prompt)
            else:
                existing.optimized_prompt = prompt
        return self
