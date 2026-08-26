"""Tests for ProgramArtifact, OptimizedPredictor, and OptimizedDemo models."""

from __future__ import annotations

import pytest

from core.models.artifacts import (
    NodeArtifact,
    OptimizedDemo,
    OptimizedPredictor,
    ProgramArtifact,
    ReactOverlay,
)


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
    assert art.optimized_component_srcs == {}


def test_program_artifact_backfills_component_srcs_from_nested_flex_state() -> None:
    """A workflow's flex nodes surface keyed by their component path."""
    src = "import dspy\n\n\nclass M(dspy.Module):\n    pass\n"
    art = ProgramArtifact(
        program_state_json={
            "n_draft": {"demos": [], "signature": {}},
            "n_refine": {"module_src": src, "lm": None},
        }
    )

    assert art.optimized_component_srcs == {"n_refine": src}
    # The program is a workflow, not a Flex, so the scalar stays empty.
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


def test_node_artifact_defaults_all_none() -> None:
    """A bare NodeArtifact leaves every optimized surface unset."""
    node = NodeArtifact()

    assert node.optimized_prompt is None
    assert node.react_overlay is None
    assert node.optimized_src is None


def test_program_artifact_optimized_nodes_defaults_empty() -> None:
    """A scalar artifact carries no per-node map."""
    art = ProgramArtifact(path="/opt/artifacts/scalar")

    assert art.optimized_nodes == {}


def test_optimized_nodes_folds_in_flex_src_for_flex_only_node() -> None:
    """A workflow's flex node surfaces its rewritten code under optimized_nodes."""
    src = "import dspy\n\n\nclass M(dspy.Module):\n    pass\n"
    art = ProgramArtifact(
        program_state_json={
            "n_draft": {"demos": [], "signature": {}},
            "n_refine": {"module_src": src, "lm": None},
        }
    )

    assert art.optimized_nodes["n_refine"].optimized_src == src
    assert art.optimized_nodes["n_refine"].optimized_prompt is None


def test_optimized_nodes_fold_preserves_existing_prompt_entry() -> None:
    """Folding flex src attaches code to a node that already carries a prompt."""
    prompt = OptimizedPredictor(predictor_name="n_hybrid", instructions="Do X.")
    src = "import dspy\n\n\nclass M(dspy.Module):\n    pass\n"
    art = ProgramArtifact(
        program_state_json={"n_hybrid": {"module_src": src}},
        optimized_nodes={"n_hybrid": NodeArtifact(optimized_prompt=prompt)},
    )

    node = art.optimized_nodes["n_hybrid"]
    assert node.optimized_prompt is prompt
    assert node.optimized_src == src


def test_optimized_nodes_round_trip() -> None:
    """optimized_nodes survives a model_dump / model_validate round-trip."""
    prompt = OptimizedPredictor(predictor_name="n_summarize", instructions="Summarize.")
    art = ProgramArtifact(optimized_nodes={"n_summarize": NodeArtifact(optimized_prompt=prompt)})

    restored = ProgramArtifact.model_validate(art.model_dump())

    assert restored.optimized_nodes["n_summarize"].optimized_prompt is not None
    assert restored.optimized_nodes["n_summarize"].optimized_prompt.instructions == "Summarize."


def test_optimized_nodes_backfills_prompt_from_state() -> None:
    """An old workflow run surfaces a node's prompt, demos, and fields from state."""
    art = ProgramArtifact(
        program_state_json={
            "n_step": {
                "demos": [{"q": "2+2", "a": "4", "augmented": True, "_meta": 1}],
                "signature": {
                    "instructions": "Answer.",
                    "fields": [
                        {"prefix": "Q:", "description": "the question"},
                        {"prefix": "A:", "description": "${a}"},
                    ],
                },
                "lm": None,
            }
        }
    )

    prompt = art.optimized_nodes["n_step"].optimized_prompt
    assert prompt is not None
    assert prompt.predictor_name == "n_step"
    assert prompt.instructions == "Answer."
    # DSPy bookkeeping keys are dropped; the real field values survive.
    assert prompt.demos == [OptimizedDemo(inputs={"q": "2+2", "a": "4"})]
    assert "Q: the question" in prompt.formatted_prompt
    # The ``${a}`` adapter placeholder is elided, leaving the bare prefix.
    assert "A:" in prompt.formatted_prompt
    assert "${a}" not in prompt.formatted_prompt
    assert "q: 2+2" in prompt.formatted_prompt


def test_optimized_nodes_backfill_groups_cot_predict_under_node() -> None:
    """A cot node's ``.predict`` predictor state folds back to its node key."""
    art = ProgramArtifact(
        program_state_json={
            "n_polish.predict": {"demos": [], "signature": {"instructions": "Polish.", "fields": []}}
        }
    )

    assert set(art.optimized_nodes) == {"n_polish"}
    prompt = art.optimized_nodes["n_polish"].optimized_prompt
    assert prompt is not None
    assert prompt.predictor_name == "n_polish.predict"
    assert prompt.instructions == "Polish."


def test_optimized_nodes_backfill_skips_flex_only_state() -> None:
    """A flex node's code-only state yields a src entry but no prompt."""
    src = "import dspy\n\n\nclass M(dspy.Module):\n    pass\n"
    art = ProgramArtifact(program_state_json={"n_flex": {"module_src": src, "lm": None}})

    node = art.optimized_nodes["n_flex"]
    assert node.optimized_src == src
    assert node.optimized_prompt is None


def test_optimized_nodes_backfill_preserves_persist_time_prompt() -> None:
    """A prompt extracted at write time is not overwritten by the state back-fill."""
    persisted = OptimizedPredictor(predictor_name="n_x", instructions="Persisted.")
    art = ProgramArtifact(
        program_state_json={"n_x": {"signature": {"instructions": "FromState.", "fields": []}}},
        optimized_nodes={"n_x": NodeArtifact(optimized_prompt=persisted)},
    )

    assert art.optimized_nodes["n_x"].optimized_prompt is persisted


def test_optimized_nodes_backfill_first_predictor_per_node_wins() -> None:
    """A react node's two predictors collapse to one prompt from the first in state."""
    art = ProgramArtifact(
        program_state_json={
            "n_r.react": {"signature": {"instructions": "React step.", "fields": []}},
            "n_r.extract": {"signature": {"instructions": "Extract step.", "fields": []}},
        }
    )

    assert set(art.optimized_nodes) == {"n_r"}
    assert art.optimized_nodes["n_r"].optimized_prompt.instructions == "React step."


def test_optimized_nodes_backfill_ignores_scalar_state() -> None:
    """Scalar predictors carry no ``n_`` prefix, so no per-node map is built."""
    art = ProgramArtifact(
        program_state_json={
            "self": {"demos": [], "signature": {"instructions": "x", "fields": []}},
            "metadata": {"dependency_versions": {}},
        }
    )

    assert art.optimized_nodes == {}
