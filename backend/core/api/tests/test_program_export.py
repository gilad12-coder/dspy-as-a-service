"""Tests for the runnable DSPy program export bundle.

Verifies the zip a user downloads from ``/optimizations/{id}/program-export``
actually reconstructs the optimized program with plain ``dspy`` — no platform
code on the path — so the export is independent of the hosted serving endpoint.
"""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import dspy
import pytest

from ...models import OptimizedPredictor, ProgramArtifact
from ...models.artifacts import ReactOverlay
from ..routers.optimizations._program_export import build_program_export_zip

_SIGNATURE_CODE = '''import dspy


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
'''

_OPTIMIZED_INSTRUCTIONS = "OPTIMIZED: reason step by step, then answer concisely."


def _persisted_artifact(module_alias: str = "predict") -> tuple[ProgramArtifact, dict]:
    """Build a ProgramArtifact + overview the way the gateway persists them.

    Args:
        module_alias: Module the program is built from — ``"predict"`` or
            ``"cot"`` (ChainOfThought). Other tests rely on the ``predict``
            default, so changing it only affects callers that pass it.

    Returns:
        A ``(ProgramArtifact, overview)`` pair carrying a real compiled DSPy
        program's state-only JSON plus the reconstruction recipe.
    """
    namespace: dict = {"dspy": dspy}
    exec(compile(_SIGNATURE_CODE, "<sig>", "exec", dont_inherit=True), namespace)
    factory = {"predict": dspy.Predict, "cot": dspy.ChainOfThought}[module_alias]
    program = factory(namespace["QA"])
    demo = dspy.Example(question="2+2?", answer="four").with_inputs("question")
    # Set on every predictor so the optimized state survives for both the
    # flat Predict and ChainOfThought (whose instructions live on an inner one).
    for predictor in program.predictors():
        predictor.signature = predictor.signature.with_instructions(_OPTIMIZED_INSTRUCTIONS)
        predictor.demos = [demo]
    state = program.dump_state()

    artifact = ProgramArtifact(
        program_state_json=state,
        optimized_prompt=OptimizedPredictor(
            predictor_name="self",
            instructions=_OPTIMIZED_INSTRUCTIONS,
            input_fields=["question"],
            output_fields=["answer"],
        ),
    )
    overview = {
        "signature_code": _SIGNATURE_CODE,
        "module_name": module_alias,
        "module_kwargs": {},
        "model_name": "openai/gpt-4o-mini",
        "optimizer_name": "gepa",
    }
    return artifact, overview


def _load_program_from_zip(zip_bytes: bytes, dest: Path):
    """Extract the export and import its standalone loader module.

    Args:
        zip_bytes: The bundle produced by :func:`build_program_export_zip`.
        dest: Directory to extract the bundle into.

    Returns:
        The imported ``load_program`` module object, loaded from ``dest`` so its
        ``__file__``-relative reads resolve against the extracted files.
    """
    zipfile.ZipFile(io.BytesIO(zip_bytes)).extractall(dest)
    spec = importlib.util.spec_from_file_location("exported_loader", dest / "load_program.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_contains_expected_entries() -> None:
    """The export zip ships state, signature, loader, metadata, prompt, and docs."""
    artifact, overview = _persisted_artifact()

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-export", artifact=artifact, overview=overview
    )

    names = set(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())
    assert {
        "program.json",
        "signature.py",
        "load_program.py",
        "metadata.json",
        "prompt.json",
        "requirements.txt",
        "README.md",
    } <= names


def test_metadata_records_module_recipe() -> None:
    """metadata.json carries the module recipe the loader rebuilds from."""
    artifact, overview = _persisted_artifact()

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-export", artifact=artifact, overview=overview
    )

    meta = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("metadata.json"))
    assert meta["module_name"] == "predict"
    assert meta["model"] == "openai/gpt-4o-mini"
    assert meta["is_react"] is False
    assert meta["optimization_id"] == "abcd1234-export"


def test_loader_uses_only_dspy_and_stdlib() -> None:
    """The shipped loader must not import platform code, or it isn't standalone."""
    artifact, overview = _persisted_artifact()

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-export", artifact=artifact, overview=overview
    )

    loader_src = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("load_program.py").decode("utf-8")
    assert "import core" not in loader_src
    assert "from core" not in loader_src
    assert "import dspy" in loader_src


@pytest.mark.parametrize(
    ("module_alias", "expected_type"),
    [("predict", "Predict"), ("cot", "ChainOfThought")],
)
def test_export_reconstructs_optimized_program(tmp_path, module_alias, expected_type) -> None:
    """The standalone loader rebuilds the program with its optimized state intact.

    Asserts module-agnostically via ``predictors()`` so it covers both a flat
    ``Predict`` and ``ChainOfThought`` (whose optimized instructions and demos
    live on an inner predictor).
    """
    artifact, overview = _persisted_artifact(module_alias)
    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-export", artifact=artifact, overview=overview
    )

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    program = loader.load_program()

    assert type(program).__name__ == expected_type
    predictors = list(program.predictors())
    assert any(p.signature.instructions == _OPTIMIZED_INSTRUCTIONS for p in predictors)
    assert any(len(p.demos) == 1 and p.demos[0]["answer"] == "four" for p in predictors)


def test_react_export_requires_tools(tmp_path) -> None:
    """A react export ships its overlay and refuses to load without a tool roster."""
    artifact, overview = _persisted_artifact()
    artifact = artifact.model_copy(update={"react_overlay": ReactOverlay(max_iters=5)})
    overview["module_name"] = "react"

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-react", artifact=artifact, overview=overview
    )

    names = set(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())
    assert "react_overlay.json" in names
    meta = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("metadata.json"))
    assert meta["is_react"] is True

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    with pytest.raises(RuntimeError, match="ReAct"):
        loader.load_program()
