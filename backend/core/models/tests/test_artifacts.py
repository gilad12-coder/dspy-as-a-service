"""Tests for ProgramArtifact, OptimizedPredictor, and OptimizedDemo models."""

from __future__ import annotations

import pytest

from core.models.artifacts import OptimizedDemo, OptimizedPredictor, ProgramArtifact, ReactOverlay


def test_optimized_demo_defaults_empty_dicts() -> None:
    """Verify OptimizedDemo defaults inputs and outputs to empty dicts."""
    d = OptimizedDemo()

    assert d.inputs == {}
    assert d.outputs == {}


def test_optimized_demo_stores_values() -> None:
    """Verify OptimizedDemo round-trips inputs and outputs through construction."""
    d = OptimizedDemo(inputs={"q": "What is 2+2?"}, outputs={"a": "4"})

    assert d.inputs == {"q": "What is 2+2?"}
    assert d.outputs == {"a": "4"}


def test_optimized_predictor_required_fields() -> None:
    """Verify OptimizedPredictor requires predictor_name and instructions."""
    p = OptimizedPredictor(predictor_name="pred0", instructions="Do X.")

    assert p.predictor_name == "pred0"
    assert p.instructions == "Do X."


def test_optimized_predictor_defaults() -> None:
    """Verify OptimizedPredictor optional fields default to expected empties."""
    p = OptimizedPredictor(predictor_name="pred0", instructions="Do X.")

    assert p.signature_name is None
    assert p.input_fields == []
    assert p.output_fields == []
    assert p.demos == []
    assert p.formatted_prompt == ""


def test_optimized_predictor_accepts_demos() -> None:
    """Verify OptimizedPredictor stores nested OptimizedDemo entries."""
    p = OptimizedPredictor(
        predictor_name="pred0",
        instructions="Do X.",
        demos=[OptimizedDemo(inputs={"q": "hi"}, outputs={"a": "bye"})],
    )

    assert len(p.demos) == 1
    assert p.demos[0].inputs == {"q": "hi"}


def test_optimized_predictor_with_full_fields() -> None:
    """Verify OptimizedPredictor populates every field when provided."""
    p = OptimizedPredictor(
        predictor_name="pred0",
        instructions="Do X.",
        signature_name="MySignature",
        input_fields=["question"],
        output_fields=["answer"],
        formatted_prompt="question: ...\nanswer: ...",
    )

    assert p.signature_name == "MySignature"
    assert p.input_fields == ["question"]
    assert p.output_fields == ["answer"]
    assert "question" in p.formatted_prompt


def test_program_artifact_all_defaults_none() -> None:
    """Verify ProgramArtifact defaults every field to None."""
    art = ProgramArtifact()

    assert art.path is None
    assert art.program_pickle_base64 is None
    assert art.metadata is None
    assert art.optimized_prompt is None


def test_program_artifact_with_nested_predictor() -> None:
    """Verify ProgramArtifact stores a nested OptimizedPredictor with demos."""
    art = ProgramArtifact(
        optimized_prompt=OptimizedPredictor(
            predictor_name="pred0",
            instructions="Do X.",
            demos=[OptimizedDemo(inputs={"q": "a"}, outputs={"a": "b"})],
        )
    )

    assert art.optimized_prompt is not None
    assert len(art.optimized_prompt.demos) == 1


def test_program_artifact_with_metadata() -> None:
    """Verify ProgramArtifact persists path and arbitrary metadata fields."""
    art = ProgramArtifact(
        path="/opt/artifacts/job123",
        metadata={"score": 0.95, "num_demos": 3},
    )

    assert art.path == "/opt/artifacts/job123"
    assert art.metadata is not None
    assert art.metadata["score"] == pytest.approx(0.95)


def test_program_artifact_with_pickle() -> None:
    """Verify ProgramArtifact stores a base64-encoded program pickle."""
    art = ProgramArtifact(program_pickle_base64="abc123==")

    assert art.program_pickle_base64 == "abc123=="


def test_program_artifact_backfills_module_src_from_flex_state() -> None:
    """A Flex artifact derives optimized_module_src from top-level program_state_json."""
    src = "import dspy\n\n\nclass M(dspy.Module):\n    pass\n"
    art = ProgramArtifact(program_state_json={"module_src": src, "lm": None})

    assert art.optimized_module_src == src


def test_program_artifact_backfill_ignores_non_flex_state() -> None:
    """Predictor-keyed (non-Flex) state has no module_src, so the field stays None."""
    art = ProgramArtifact(program_state_json={"self": {"demos": []}})

    assert art.optimized_module_src is None


def test_program_artifact_explicit_module_src_survives_validation() -> None:
    """An explicitly set optimized_module_src is not overwritten by the back-fill."""
    art = ProgramArtifact(
        program_state_json={"module_src": "old"},
        optimized_module_src="explicit",
    )

    assert art.optimized_module_src == "explicit"


def test_program_artifact_react_overlay_defaults_none() -> None:
    """A ProgramArtifact leaves ``react_overlay`` unset for non-react artifacts."""
    art = ProgramArtifact(path="/opt/artifacts/scalar")

    assert art.react_overlay is None


def test_program_artifact_react_overlay_round_trip() -> None:
    """A react overlay survives a ``model_dump`` / ``model_validate`` round-trip."""
    overlay = ReactOverlay(
        tool_descriptions={"search": "optimized search"},
        tool_arg_descriptions={"search": {"query": "the search query"}},
        tool_schema_hashes={"search": "deadbeef"},
        max_iters=6,
        tool_source={"kind": "live_mcp", "mcp_url": "http://localhost:9000/mcp"},
    )
    art = ProgramArtifact(path="/opt/artifacts/react", react_overlay=overlay)

    restored = ProgramArtifact.model_validate(art.model_dump())

    assert restored.react_overlay is not None
    assert restored.react_overlay.tool_descriptions == {"search": "optimized search"}
    assert restored.react_overlay.tool_arg_descriptions == {"search": {"query": "the search query"}}
    assert restored.react_overlay.tool_schema_hashes == {"search": "deadbeef"}
    assert restored.react_overlay.max_iters == 6
    assert restored.react_overlay.tool_source == {
        "kind": "live_mcp",
        "mcp_url": "http://localhost:9000/mcp",
    }
