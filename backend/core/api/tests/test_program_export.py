"""Tests for the runnable DSPy program export bundle.

Verifies the zip a user downloads from ``/optimizations/{id}/program-export``
actually reconstructs the optimized program with plain ``dspy`` — no platform
code on the path — so the export is independent of the hosted serving endpoint.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from ...models import OptimizedPredictor, ProgramArtifact, WorkflowSpec
from ...models.artifacts import ReactOverlay
from ...service_gateway.optimization.workflow import build_workflow_program
from ..routers.optimizations._program_export import build_program_export_zip

_SIGNATURE_CODE = '''import dspy


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
'''

_OPTIMIZED_INSTRUCTIONS = "OPTIMIZED: reason step by step, then answer concisely."

_WORKFLOW_SPEC = {
    "nodes": [
        {"id": "start", "kind": "input", "fields": [{"name": "question"}]},
        {"id": "draft", "kind": "signature", "module_name": "predict", "signature_code": _SIGNATURE_CODE},
        {
            "id": "shout",
            "kind": "transform",
            "transform_code": 'def transform(answer):\n    return {"final": answer.upper()}\n',
            "input_fields": [{"name": "answer"}],
            "output_fields": [{"name": "final"}],
        },
        {"id": "end", "kind": "output", "fields": [{"name": "final"}]},
    ],
    "edges": [
        {"source": "start", "source_port": "question", "target": "draft", "target_port": "question"},
        {"source": "draft", "source_port": "answer", "target": "shout", "target_port": "answer"},
        {"source": "shout", "source_port": "final", "target": "end", "target_port": "final"},
    ],
}


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


def _persisted_workflow_artifact(spec: dict | None = None) -> tuple[ProgramArtifact, dict]:
    """Build the artifact + overview a workflow run persists.

    State comes from the platform's own builder, so the saved per-node keys are
    exactly what the standalone loader has to line up with.

    Args:
        spec: Graph to compile, defaulting to the module-level two-step graph.

    Returns:
        A ``(ProgramArtifact, overview)`` pair whose signature node carries
        optimized instructions.
    """
    workflow = spec or _WORKFLOW_SPEC
    program, _hashes = build_workflow_program(WorkflowSpec.model_validate(workflow))
    for predictor in program.predictors():
        predictor.signature = predictor.signature.with_instructions(_OPTIMIZED_INSTRUCTIONS)

    overview = {
        "module_name": "workflow",
        "workflow": workflow,
        "model_name": "openai/gpt-4o-mini",
        "optimizer_name": "gepa",
    }
    return ProgramArtifact(program_state_json=program.dump_state()), overview


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


def test_flex_export_ships_readable_module_source() -> None:
    """A Flex export adds ``optimized_module.py`` with the rewritten source and flags is_flex."""
    artifact, overview = _persisted_artifact()
    module_src = "import dspy\n\n\nclass QAModule(dspy.Module):\n    pass\n"
    artifact = artifact.model_copy(update={"optimized_module_src": module_src})
    overview["module_name"] = "flex"

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-flex", artifact=artifact, overview=overview
    )

    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert "optimized_module.py" in set(archive.namelist())
    module_file = archive.read("optimized_module.py").decode("utf-8")
    assert "class QAModule(dspy.Module):" in module_file
    meta = json.loads(archive.read("metadata.json"))
    assert meta["is_flex"] is True
    assert "optimized_module.py" in archive.read("README.md").decode("utf-8")


def test_nested_flex_export_ships_one_file_per_component() -> None:
    """A workflow's flex nodes each get a readable file under ``optimized_modules/``."""
    artifact, overview = _persisted_artifact()
    module_src = "import dspy\n\n\nclass RefineModule(dspy.Module):\n    pass\n"
    artifact = artifact.model_copy(update={"optimized_component_srcs": {"n_refine": module_src}})
    overview["module_name"] = "workflow"

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow", artifact=artifact, overview=overview
    )

    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = set(archive.namelist())
    assert "optimized_modules/n_refine.py" in names
    assert "optimized_module.py" not in names
    assert "class RefineModule(dspy.Module):" in archive.read("optimized_modules/n_refine.py").decode("utf-8")
    meta = json.loads(archive.read("metadata.json"))
    assert meta["is_flex"] is True
    assert meta["flex_components"] == ["n_refine"]
    assert "optimized_modules/" in archive.read("README.md").decode("utf-8")


def test_non_flex_export_omits_module_source() -> None:
    """A non-Flex export leaves out ``optimized_module.py`` and marks is_flex False."""
    artifact, overview = _persisted_artifact()

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-export", artifact=artifact, overview=overview
    )

    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert "optimized_module.py" not in set(archive.namelist())
    meta = json.loads(archive.read("metadata.json"))
    assert meta["is_flex"] is False


def test_workflow_bundle_ships_the_graph_instead_of_a_signature() -> None:
    """A workflow has no top-level signature: the graph is the program definition."""
    artifact, overview = _persisted_workflow_artifact()

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow", artifact=artifact, overview=overview
    )

    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = set(archive.namelist())
    assert "workflow.json" in names
    assert "signature.py" not in names
    assert json.loads(archive.read("metadata.json"))["is_workflow"] is True
    assert "workflow.json" in archive.read("README.md").decode("utf-8")


def test_workflow_export_rebuilds_and_runs_the_graph(tmp_path) -> None:
    """The standalone loader reconstructs every node, restores state, and executes the DAG."""
    artifact, overview = _persisted_workflow_artifact()
    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow", artifact=artifact, overview=overview
    )

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    program = loader.load_program()

    assert program.n_draft.signature.instructions == _OPTIMIZED_INSTRUCTIONS
    with dspy.context(lm=DummyLM([{"answer": "four"}])):
        prediction = program(question="2+2?")
    assert prediction.final == "FOUR"


def test_workflow_export_restores_a_flex_node_rewritten_source(tmp_path) -> None:
    """A flex node's GEPA-rewritten code survives the round-trip through the bundle."""
    spec = copy.deepcopy(_WORKFLOW_SPEC)
    spec["nodes"][1]["module_name"] = "flex"
    artifact, overview = _persisted_workflow_artifact(spec)
    rewritten = "class QAModule(dspy.Module):\n    def forward(self, **inputs):\n        return None\n"
    state = dict(artifact.program_state_json)
    state["n_draft"] = {**state["n_draft"], "module_src": rewritten}
    artifact = artifact.model_copy(update={"program_state_json": state})

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow-flex", artifact=artifact, overview=overview
    )

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    program = loader.load_program()

    assert isinstance(program.n_draft, dspy.Flex)
    assert program.n_draft.module_src == rewritten


def test_workflow_export_runs_an_mcp_node_against_the_supplied_roster(tmp_path) -> None:
    """An MCP node picks its tool out of the roster by name and calls it."""
    spec = {
        "nodes": [
            {"id": "start", "kind": "input", "fields": [{"name": "question"}]},
            {
                "id": "lookup",
                "kind": "mcp",
                "tool_name": "search",
                "input_fields": [{"name": "query"}],
                "output_field": {"name": "result"},
            },
            {"id": "end", "kind": "output", "fields": [{"name": "result"}]},
        ],
        "edges": [
            {"source": "start", "source_port": "question", "target": "lookup", "target_port": "query"},
            {"source": "lookup", "source_port": "result", "target": "end", "target_port": "result"},
        ],
    }
    # model_dump() is the exact shape the overview persists; validating here keeps
    # the fixture to graphs the canvas would actually have accepted.
    overview = {"module_name": "workflow", "workflow": WorkflowSpec.model_validate(spec).model_dump()}
    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow-mcp",
        artifact=ProgramArtifact(program_state_json={}),
        overview=overview,
    )

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    tool = dspy.Tool(lambda query: f"hit:{query}", name="search", desc="Look something up")
    program = loader.load_program(tools=[tool])

    assert program(question="dspy").result == "hit:dspy"


def test_workflow_export_refuses_to_load_without_the_tools_a_node_needs(tmp_path) -> None:
    """A tool-using node names itself rather than silently rebuilding a toolless graph."""
    artifact, overview = _persisted_workflow_artifact()
    spec = copy.deepcopy(_WORKFLOW_SPEC)
    spec["nodes"][1]["module_name"] = "react"
    overview["workflow"] = spec

    zip_bytes = build_program_export_zip(
        optimization_id="abcd1234-workflow-react", artifact=artifact, overview=overview
    )

    loader = _load_program_from_zip(zip_bytes, tmp_path)
    with pytest.raises(RuntimeError, match="draft"):
        loader.load_program()


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
