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
"""

from __future__ import annotations

import json
import pathlib

import dspy

_HERE = pathlib.Path(__file__).resolve().parent
_META = json.loads((_HERE / "metadata.json").read_text(encoding="utf-8"))

# Short aliases the platform uses; a stored module_name may instead be a
# fully-qualified ``dspy.*`` path, which the resolver below also accepts.
_MODULE_ALIASES = {
    "predict": "dspy.Predict",
    "cot": "dspy.ChainOfThought",
    "react": "dspy.ReAct",
}


def _load_signature():
    """Execute signature.py and return the single dspy.Signature subclass it defines."""
    namespace = {"dspy": dspy}
    code = (_HERE / "signature.py").read_text(encoding="utf-8")
    # dont_inherit=True keeps this file's ``from __future__ import annotations``
    # from stringizing the signature's field annotations.
    exec(compile(code, "signature.py", "exec", dont_inherit=True), namespace)
    found = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type)
        and issubclass(obj, dspy.Signature)
        and obj is not dspy.Signature
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"signature.py must define exactly one dspy.Signature subclass, found {len(found)}"
        )
    return found[0]


def _resolve_module(name):
    """Resolve a module alias or ``dspy.*`` path to a dspy module class."""
    path = _MODULE_ALIASES.get(name.lower(), name)
    if not path.startswith("dspy."):
        raise RuntimeError(f"unsupported module {name!r}; expected a dspy.* class")
    obj = dspy
    for attr in path.split(".")[1:]:
        obj = getattr(obj, attr)
    return obj


def load_program(tools=None):
    """Rebuild the optimized program and load its trained state.

    Pass ``tools=[...]`` only for ReAct programs (the same tool callables you
    optimized against); other module types ignore it.
    """
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
        program = factory(signature, tools=tools, **kwargs)
    else:
        program = factory(signature, **kwargs)
    state = json.loads((_HERE / "program.json").read_text(encoding="utf-8"))
    program.load_state(state)
    return program


def configure_lm(model=None):
    """Point dspy at an LM. Defaults to the model the program was optimized on."""
    dspy.configure(lm=dspy.LM(model or _META.get("model") or "openai/gpt-4o-mini"))


if __name__ == "__main__":
    configure_lm()
    loaded = load_program()
    print(f"Loaded {type(loaded).__name__} (optimization {_META.get('optimization_id')})")
    print(loaded)
'''


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
    return {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "optimization_id": optimization_id,
        "module_name": overview.get(PAYLOAD_OVERVIEW_MODULE_NAME) or "predict",
        "module_kwargs": dict(overview.get(PAYLOAD_OVERVIEW_MODULE_KWARGS, {})),
        "model": overview.get(PAYLOAD_OVERVIEW_MODEL_NAME),
        "optimizer": overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
        "dspy_version": _installed_dspy_version(),
        "is_react": artifact.react_overlay is not None,
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
        "- `signature.py` — the task signature the program was built on.",
        "- `load_program.py` — rebuilds the module and loads the state. Run or import it.",
        "- `metadata.json` — the module recipe (module name, kwargs, default model).",
        "- `prompt.json` — the optimized instructions and few-shot demos, human-readable.",
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

    Callers must validate first that ``artifact.program_state_json`` and the
    overview's ``signature_code`` are both present; this builder assumes them.

    Args:
        optimization_id: Optimization id the export belongs to.
        artifact: The compiled program artifact (state JSON, optional react
            overlay, optimized prompt).
        overview: Parsed payload-overview dict supplying ``signature_code`` and
            the module recipe.

    Returns:
        The bytes of a zip archive containing the program state, signature
        source, a standalone loader, metadata, and a README.
    """
    metadata = _build_metadata(optimization_id, artifact, overview)
    signature_code = overview.get(PAYLOAD_OVERVIEW_SIGNATURE_CODE) or ""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("program.json", json.dumps(artifact.program_state_json, indent=2, ensure_ascii=False))
        archive.writestr("signature.py", signature_code)
        archive.writestr("load_program.py", _LOADER_PY)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        if artifact.optimized_prompt is not None:
            archive.writestr(
                "prompt.json",
                artifact.optimized_prompt.model_dump_json(indent=2),
            )
        if artifact.react_overlay is not None:
            archive.writestr(
                "react_overlay.json",
                artifact.react_overlay.model_dump_json(indent=2),
            )
        dspy_version = metadata.get("dspy_version")
        requirement = f"dspy=={dspy_version}\n" if dspy_version else "dspy\n"
        archive.writestr("requirements.txt", requirement)
        archive.writestr("README.md", _build_readme(metadata))

    return buffer.getvalue()
