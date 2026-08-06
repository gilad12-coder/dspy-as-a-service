"""Compiled DSPy program artifact models (prompts, demos, serialized pickle)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _backfill_module_src(self) -> ProgramArtifact:
        """Derive ``optimized_module_src`` from persisted Flex state when absent.

        Flex saves ``module_src`` at the top level of ``program.save(json)``, so
        artifacts persisted before this field existed still carry the code inside
        ``program_state_json``. Back-filling on validation surfaces it for old and
        new Flex artifacts alike without a data migration; non-Flex state has no
        ``module_src`` key, so this is a no-op there.

        Returns:
            The validated artifact, with ``optimized_module_src`` populated when a
            Flex source is recoverable.
        """
        if self.optimized_module_src is None and isinstance(self.program_state_json, dict):
            src = self.program_state_json.get("module_src")
            if isinstance(src, str) and src.strip():
                self.optimized_module_src = src
        return self
