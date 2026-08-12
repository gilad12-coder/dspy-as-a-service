"""Tests for ``core.service_gateway.optimization.artifacts``.

Covers prompt-string formatting, optimized-predictor extraction, and the
``persist_program`` save/cleanup contract.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import ServiceError
from core.models import OptimizedDemo, OptimizedPredictor, ProgramArtifact
from core.service_gateway.optimization.artifacts import (
    _format_prompt_string,
    extract_optimized_nodes,
    extract_optimized_prompt,
    persist_program,
)


def test_format_prompt_string_instructions_only() -> None:
    """Instructions alone round-trip unchanged when no fields or demos exist."""
    result = _format_prompt_string("Do the thing.", [], [], [], None)

    assert result == "Do the thing."


def test_format_prompt_string_includes_input_and_output_field_labels() -> None:
    """Field labels are rendered as ``[Input]``/``[Output]`` markers."""
    result = _format_prompt_string("Instr", ["question"], ["answer"], [], _mock_sig())

    assert "[Input] question" in result
    assert "[Output] answer" in result


def test_format_prompt_string_demo_block_present() -> None:
    """A single demo's inputs and outputs appear under an ``Examples:`` header."""
    demo = OptimizedDemo(inputs={"q": "What?"}, outputs={"a": "42"})

    result = _format_prompt_string("Instr", ["q"], ["a"], [demo], _mock_sig())

    assert "Examples:" in result
    assert "What?" in result
    assert "42" in result


def test_format_prompt_string_empty_everything_returns_empty_string() -> None:
    """All-empty inputs collapse to an empty string."""
    result = _format_prompt_string("", [], [], [], None)

    assert result == ""


def test_format_prompt_string_multiple_demos_numbered() -> None:
    """Multiple demos are numbered ``Example 1:``, ``Example 2:``..."""
    demo1 = OptimizedDemo(inputs={"q": "q1"}, outputs={"a": "a1"})
    demo2 = OptimizedDemo(inputs={"q": "q2"}, outputs={"a": "a2"})

    result = _format_prompt_string("X", ["q"], ["a"], [demo1, demo2], _mock_sig())

    assert "Example 1:" in result
    assert "Example 2:" in result


def test_extract_optimized_prompt_no_predictors_returns_none() -> None:
    """A program with zero predictors yields ``None``."""
    program = MagicMock()
    program.named_predictors.return_value = []

    result = extract_optimized_prompt(program)

    assert result is None


def test_extract_optimized_prompt_named_predictors_raises_returns_none() -> None:
    """An exception from ``named_predictors()`` must NOT propagate — extraction is best-effort."""
    program = MagicMock()
    program.named_predictors.side_effect = RuntimeError("boom")

    result = extract_optimized_prompt(program)

    assert result is None


def test_extract_optimized_prompt_no_signature_returns_none() -> None:
    """A predictor lacking a ``signature`` attribute yields ``None``."""
    predictor = MagicMock(spec=[])  # no .signature attribute
    program = MagicMock()
    program.named_predictors.return_value = [("predict", predictor)]

    result = extract_optimized_prompt(program)

    assert result is None


def test_extract_optimized_prompt_happy_path_returns_optimized_predictor() -> None:
    """A well-formed predictor returns an ``OptimizedPredictor`` carrying its name."""
    sig = _mock_sig()
    predictor = MagicMock()
    predictor.signature = sig
    predictor.demos = []
    program = MagicMock()
    program.named_predictors.return_value = [("predict", predictor)]

    result = extract_optimized_prompt(program)

    assert isinstance(result, OptimizedPredictor)
    assert result.predictor_name == "predict"


def test_extract_optimized_prompt_demos_extracted() -> None:
    """Demo input/output cells are projected into the returned ``OptimizedPredictor``."""
    sig = _mock_sig(input_fields=["q"], output_fields=["a"])
    demo = MagicMock()
    demo.q = "What?"
    demo.a = "42"
    predictor = MagicMock()
    predictor.signature = sig
    predictor.demos = [demo]
    program = MagicMock()
    program.named_predictors.return_value = [("predict", predictor)]

    result = extract_optimized_prompt(program)

    assert result is not None
    assert len(result.demos) == 1
    assert result.demos[0].inputs == {"q": "What?"}
    assert result.demos[0].outputs == {"a": "42"}


def test_extract_optimized_nodes_scalar_program_returns_empty() -> None:
    """A scalar program's predictors carry no node prefix, so no per-node map is built."""
    predictor = MagicMock()
    predictor.signature = _mock_sig()
    predictor.demos = []
    program = MagicMock()
    program.named_predictors.return_value = [("predict", predictor)]

    assert extract_optimized_nodes(program) == {}


def test_extract_optimized_nodes_named_predictors_raises_returns_empty() -> None:
    """Extraction is best-effort: an enumeration error yields an empty map, not a raise."""
    program = MagicMock()
    program.named_predictors.side_effect = RuntimeError("boom")

    assert extract_optimized_nodes(program) == {}


def test_extract_optimized_nodes_groups_predictors_by_node() -> None:
    """Workflow predictors group under their ``n_<node_id>`` prefix, one entry per node."""
    summarize = MagicMock()
    summarize.signature = _mock_sig(instructions="Summarize.")
    summarize.demos = []
    polish = MagicMock()
    polish.signature = _mock_sig(instructions="Polish.")
    polish.demos = []
    program = MagicMock()
    # A cot node nests its predictor under ``.predict``; grouping must fold it back
    # to the node key.
    program.named_predictors.return_value = [("n_summarize", summarize), ("n_polish.predict", polish)]

    nodes = extract_optimized_nodes(program)

    assert set(nodes) == {"n_summarize", "n_polish"}
    assert nodes["n_summarize"].optimized_prompt is not None
    assert nodes["n_summarize"].optimized_prompt.instructions == "Summarize."
    assert nodes["n_polish"].optimized_prompt.instructions == "Polish."


def test_extract_optimized_nodes_keeps_first_predictor_per_node() -> None:
    """A node with several predictors keeps the first, mirroring the scalar path."""
    first = MagicMock()
    first.signature = _mock_sig(instructions="First.")
    first.demos = []
    second = MagicMock()
    second.signature = _mock_sig(instructions="Second.")
    second.demos = []
    program = MagicMock()
    program.named_predictors.return_value = [("n_agent.react", first), ("n_agent.extract", second)]

    nodes = extract_optimized_nodes(program)

    assert set(nodes) == {"n_agent"}
    assert nodes["n_agent"].optimized_prompt.instructions == "First."


def test_persist_program_populates_optimized_nodes_for_workflow() -> None:
    """persist_program extracts a per-node map from a workflow-shaped program."""
    node_pred = MagicMock()
    node_pred.signature = _mock_sig(instructions="Node instr.")
    node_pred.demos = []
    program = _fake_save_program({"demos": [], "metadata": {"dspy_version": "0"}})
    program.named_predictors.return_value = [("n_draft", node_pred)]

    result = persist_program(program, artifact_id="wf")

    assert result is not None
    assert set(result.optimized_nodes) == {"n_draft"}
    assert result.optimized_nodes["n_draft"].optimized_prompt.instructions == "Node instr."


def test_persist_program_scalar_leaves_optimized_nodes_empty() -> None:
    """A scalar program persists with an empty per-node map."""
    program = _fake_save_program({"demos": [], "metadata": {"dspy_version": "0"}})

    result = persist_program(program, artifact_id="scalar")

    assert result is not None
    assert result.optimized_nodes == {}


def test_persist_program_save_failure_raises_service_error() -> None:
    """A ``save`` failure surfaces as ``ServiceError`` with the save-failure prefix."""
    program = MagicMock()
    program.save.side_effect = RuntimeError("disk full")

    with pytest.raises(ServiceError, match="Failed to save program artifact"):
        persist_program(program, artifact_id="test-id")


def test_persist_program_missing_files_raises_service_error(tmp_path) -> None:
    """Missing artifact files raise the load-failure ``ServiceError``."""
    program = MagicMock()
    program.save.return_value = None  # creates nothing

    with pytest.raises(ServiceError, match="Failed to load program artifact"):
        persist_program(program, artifact_id="test-id")


def test_persist_program_writes_state_json_into_artifact() -> None:
    """The saved JSON state is parsed into ``program_state_json`` verbatim."""
    state = {"demos": [{"q": "hi", "a": "bye"}], "metadata": {"dspy_version": "0"}}
    program = _fake_save_program(state)

    result = persist_program(program, artifact_id=None)

    assert result is not None
    assert result.program_state_json == state
    assert result.program_pickle_base64 is None


def test_persist_program_no_temp_files_left_on_disk() -> None:
    """No ``dspy_artifact_*`` temp directories survive a successful persist call."""
    program = _fake_save_program({"demos": [], "metadata": {"dspy_version": "0"}})
    tmp_root = Path(tempfile.gettempdir())
    tmp_dirs_before = {p.name for p in tmp_root.iterdir()}

    persist_program(program, artifact_id="cleanup-test")

    tmp_dirs_after = {p.name for p in tmp_root.iterdir()}
    dspy_dirs = [d for d in tmp_dirs_after - tmp_dirs_before if d.startswith("dspy_artifact_")]
    assert dspy_dirs == [], f"Temp dirs not cleaned up: {dspy_dirs}"


def test_persist_program_returns_program_artifact_model() -> None:
    """``persist_program`` returns a ``ProgramArtifact`` with no on-disk path."""
    program = _fake_save_program({"demos": [], "metadata": {"dspy_version": "0"}})

    result = persist_program(program, artifact_id="abc")

    assert isinstance(result, ProgramArtifact)
    assert result.path is None


def test_persist_program_surfaces_metadata_from_state() -> None:
    """``ProgramArtifact.metadata`` mirrors the ``metadata`` block inside the JSON state."""
    meta = {"dspy_version": "2.5.0", "extras": {"k": "v"}}
    program = _fake_save_program({"demos": [], "metadata": meta})

    result = persist_program(program, artifact_id="meta")

    assert result is not None
    assert result.metadata == meta


def test_persist_program_raises_service_error_when_save_fails() -> None:
    """An ``OSError`` from ``save`` is wrapped as the save-failure ``ServiceError``."""
    program = MagicMock()
    program.save.side_effect = OSError("no space left on device")

    with pytest.raises(ServiceError, match="Failed to save program artifact"):
        persist_program(program, artifact_id="save-fail")


def test_persist_program_raises_service_error_when_json_missing() -> None:
    """A missing ``program.json`` produces the load-failure ``ServiceError``."""
    def _save_nothing(path: str, save_program: bool = False) -> None:
        """Stand in for ``program.save`` while writing no file."""
        # Deliberately write no file so read_text() raises FileNotFoundError.
        return

    program = MagicMock()
    program.save.side_effect = _save_nothing
    program.named_predictors.return_value = []

    with pytest.raises(ServiceError, match="Failed to load program artifact"):
        persist_program(program, artifact_id="json-fail")


def test_persist_program_cleans_up_tempdir_on_error() -> None:
    """The temp dir is removed even when ``ServiceError`` propagates."""
    known_path = tempfile.mkdtemp(prefix="dspy_sentinel_")
    try:
        program = MagicMock()
        program.save.side_effect = RuntimeError("boom")

        with (
            patch("core.service_gateway.optimization.artifacts.tempfile.mkdtemp", return_value=known_path),
            patch("core.service_gateway.optimization.artifacts.shutil.rmtree") as mock_rmtree,
        ):
            with pytest.raises(ServiceError):
                persist_program(program, artifact_id="cleanup-err")

            mock_rmtree.assert_called_once_with(known_path, ignore_errors=True)
    finally:
        if Path(known_path).exists():
            shutil.rmtree(known_path, ignore_errors=True)


def test_persist_program_cleans_up_tempdir_on_success() -> None:
    """Symmetric to the error-cleanup test: cleanup must run on the happy path too."""
    known_path = tempfile.mkdtemp(prefix="dspy_sentinel_success_")
    try:
        state = {"demos": [], "metadata": {"dspy_version": "0"}}

        def _save_state(path: str, save_program: bool = False) -> None:
            """Stand in for ``program.save`` by writing ``state`` to ``path``."""
            Path(path).write_text(json.dumps(state))

        program = MagicMock()
        program.save.side_effect = _save_state
        program.named_predictors.return_value = []

        with (
            patch("core.service_gateway.optimization.artifacts.tempfile.mkdtemp", return_value=known_path),
            patch("core.service_gateway.optimization.artifacts.shutil.rmtree") as mock_rmtree,
        ):
            result = persist_program(program, artifact_id="cleanup-ok")

            assert result is not None
            mock_rmtree.assert_called_once_with(known_path, ignore_errors=True)
    finally:
        if Path(known_path).exists():
            shutil.rmtree(known_path, ignore_errors=True)


def _mock_sig(
    *,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
    instructions: str = "test instructions",
) -> MagicMock:
    """Build a ``MagicMock`` shaped like a DSPy ``Signature`` class."""
    sig = MagicMock()
    sig.instructions = instructions
    sig.__class__.__name__ = "MockSignature"
    sig.input_fields = {k: MagicMock(json_schema_extra={}) for k in (input_fields or ["question"])}
    sig.output_fields = {k: MagicMock(json_schema_extra={}) for k in (output_fields or ["answer"])}
    return sig


def _fake_save_program(state: dict) -> MagicMock:
    """Return a fake program whose ``save`` writes ``state`` to the requested JSON path."""

    def _save(path: str, save_program: bool = False) -> None:
        """Write ``state`` as JSON to ``path`` in place of a real save."""
        Path(path).write_text(json.dumps(state))

    program = MagicMock()
    program.save.side_effect = _save
    program.named_predictors.return_value = []
    return program
