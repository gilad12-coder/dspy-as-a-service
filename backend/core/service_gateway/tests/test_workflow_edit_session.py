"""Unit tests for the code agent's workflow graph tools and JSON helpers.

Covers the no-LLM surface of the workflow-aware code agent: op-level
guardrails on :class:`_WorkflowEditSession` (anchors immutable, unique
ids, single-producer ports, cycle prevention), the SSE events each op
emits, and the parse/validate helpers the seed's repair loop drives.
"""

from __future__ import annotations

import json

import pytest

from core.service_gateway.agents.code import (
    _parse_workflow_json,
    _reply_language,
    _validate_workflow_dict,
    _WorkflowEditSession,
)

_SIG = (
    "class Step(dspy.Signature):\n"
    '    """Do the step."""\n\n'
    "    q: str = dspy.InputField()\n"
    "    a: str = dspy.OutputField()\n"
)


def _linear_spec() -> dict:
    """Build a minimal valid input → signature → output graph dict."""
    return {
        "nodes": [
            {"id": "input", "kind": "input", "fields": [{"name": "q"}]},
            {"id": "step_1", "kind": "signature", "module_name": "predict", "signature_code": _SIG},
            {"id": "output", "kind": "output", "fields": [{"name": "a"}]},
        ],
        "edges": [
            {"source": "input", "source_port": "q", "target": "step_1", "target_port": "q"},
            {"source": "step_1", "source_port": "a", "target": "output", "target_port": "a"},
        ],
    }


@pytest.fixture()
def events() -> list[dict]:
    """Collect the SSE events a session emits during a test."""
    return []


@pytest.fixture()
def session(events: list[dict]) -> _WorkflowEditSession:
    """Build a session over the linear starter graph with a capturing emitter."""
    return _WorkflowEditSession(
        workflow=_linear_spec(),
        metric_code="def metric(gold, pred, trace=None): pass",
        emit=events.append,
    )


def _events_named(events: list[dict], name: str) -> list[dict]:
    """Filter captured events by event name."""
    return [e for e in events if e["event"] == name]


def test_connect_and_disconnect_round_trip(session: _WorkflowEditSession, events: list[dict]) -> None:
    """A disconnect of an existing edge succeeds and emits a workflow snapshot."""
    obs = session.disconnect("ניתוק", "step_1", "a", "output", "a")
    assert "Disconnected" in obs
    assert "remaining issues" in obs
    obs = session.connect("חיבור", "step_1", "a", "output", "a")
    assert "graph is valid" in obs
    snapshots = _events_named(events, "workflow_replace")
    assert len(snapshots) == 2
    assert snapshots[-1]["data"]["changed_node_id"] == "output"


def test_connect_rejects_fed_port_and_cycle(session: _WorkflowEditSession) -> None:
    """Double-feeding a port or closing a cycle is rejected before mutation."""
    fed = session.connect("כפול", "input", "q", "step_1", "q")
    assert "already fed" in fed
    cycle = session.connect("מעגל", "step_1", "a", "step_1", "q")
    assert "cannot feed itself" in cycle
    edges_before = list(session.workflow["edges"])
    back = session.connect("אחורה", "output", "a", "step_1", "extra")
    assert "cycle" in back
    assert session.workflow["edges"] == edges_before


def test_add_node_validates_kind_id_and_signature(session: _WorkflowEditSession, events: list[dict]) -> None:
    """add_node rejects anchors, duplicate ids, and broken signature code."""
    anchor = session.add_node("עוגן", '{"id": "input2", "kind": "input", "fields": []}')
    assert "kind 'signature' or 'transform'" in anchor
    dup = session.add_node(
        "כפול",
        json.dumps({"id": "step_1", "kind": "signature", "module_name": "predict", "signature_code": _SIG}),
    )
    assert "already taken" in dup
    broken = session.add_node(
        "שבור", '{"id": "step_2", "kind": "signature", "module_name": "predict", "signature_code": "class X("}'
    )
    assert "invalid" in broken
    ok = session.add_node(
        "הוספה",
        '{"id": "step_2", "kind": "transform", "transform_code": "def transform(a):\\n    return {\\"b\\": a}", '
        '"input_fields": [{"name": "a"}], "output_fields": [{"name": "b"}]}',
    )
    assert "added" in ok
    assert _events_named(events, "workflow_replace")[-1]["data"]["changed_node_id"] == "step_2"


def test_update_node_keeps_id_and_kind(session: _WorkflowEditSession) -> None:
    """update_node rejects id/kind changes and applies same-shape rewrites."""
    wrong_kind = session.update_node(
        "סוג", "step_1", '{"id": "step_1", "kind": "transform", "transform_code": "def transform(q):\\n    return {\\"a\\": q}", "input_fields": [{"name": "q"}], "output_fields": [{"name": "a"}]}'
    )
    assert "same id and kind" in wrong_kind
    new_sig = _SIG.replace("Do the step.", "Do it better.")
    ok = session.update_node(
        "עדכון",
        "step_1",
        json.dumps({"id": "step_1", "kind": "signature", "module_name": "cot", "signature_code": new_sig}),
    )
    assert "updated" in ok
    node = next(n for n in session.workflow["nodes"] if n["id"] == "step_1")
    assert node["module_name"] == "cot"


def test_remove_node_protects_anchors_and_drops_edges(session: _WorkflowEditSession) -> None:
    """remove_node refuses anchors and removes a node together with its edges."""
    assert "cannot be removed" in session.remove_node("עוגן", "output")
    obs = session.remove_node("מחיקה", "step_1")
    assert "removed" in obs
    assert all(n["id"] != "step_1" for n in session.workflow["nodes"])
    assert session.workflow["edges"] == []


def test_edit_metric_validates_and_caps(session: _WorkflowEditSession, events: list[dict]) -> None:
    """edit_metric validates the new code and rejects a second edit per turn."""
    bad = session.edit_metric("שבור", "def metric(:")
    assert "invalid" in bad
    good = session.edit_metric(
        "עדכון", "def metric(gold, pred, trace=None):\n    return dspy.Prediction(score=1.0, feedback='ok')\n"
    )
    assert "replaced" in good
    assert _events_named(events, "metric_replace")
    again = session.edit_metric(
        "שוב", "def metric(gold, pred, trace=None):\n    return dspy.Prediction(score=0.0, feedback='x')\n"
    )
    assert "already replaced" in again


def test_edit_budget_caps_graph_ops(events: list[dict]) -> None:
    """The per-turn edit budget rejects ops after MAX_EDITS successes."""
    session = _WorkflowEditSession(workflow=_linear_spec(), metric_code="", emit=events.append)
    for i in range(_WorkflowEditSession.MAX_EDITS):
        obs = (
            session.disconnect("ניתוק", "step_1", "a", "output", "a")
            if i % 2 == 0
            else session.connect("חיבור", "step_1", "a", "output", "a")
        )
        assert "rejected" not in obs
    capped = session.connect("עוד", "input", "q", "output", "a")
    assert "too many graph edits" in capped


def test_parse_workflow_json_strips_fences() -> None:
    """Markdown-fenced JSON parses; garbage returns a parse error."""
    spec, err = _parse_workflow_json('```json\n{"nodes": [], "edges": []}\n```')
    assert err == ""
    assert spec == {"nodes": [], "edges": []}
    spec, err = _parse_workflow_json("not json")
    assert spec is None
    assert "does not parse" in err
    spec, err = _parse_workflow_json("[1, 2]")
    assert spec is None
    assert "single object" in err


def test_validate_workflow_dict_names_broken_node() -> None:
    """Structural errors surface; a broken signature node is named."""
    assert _validate_workflow_dict(_linear_spec()) == ""
    missing_edge = _linear_spec()
    missing_edge["edges"].pop()
    assert "not connected" in _validate_workflow_dict(missing_edge)
    broken_sig = _linear_spec()
    broken_sig["nodes"][1]["signature_code"] = "class X("
    assert "Node 'step_1'" in _validate_workflow_dict(broken_sig)


def test_reply_language_maps_locales_with_hebrew_fallback() -> None:
    """UI locale codes resolve to language names; unknowns fall back to Hebrew."""
    assert _reply_language("he") == "Hebrew"
    assert _reply_language("en") == "English"
    assert _reply_language("en-GB") == "English"
    assert _reply_language("fr-CA") == "French"
    assert _reply_language("zh-Hans") == "Chinese"
    assert _reply_language("HE") == "Hebrew"
    assert _reply_language(None) == "Hebrew"
    assert _reply_language("") == "Hebrew"
    assert _reply_language("xx") == "Hebrew"
