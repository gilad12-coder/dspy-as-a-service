"""Tests for the ``/workflows/dry-run`` endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..routers.workflows import create_workflows_router
from .conftest import bypass_auth

_MODEL_CFG = {"name": "openai/gpt-4o-mini"}


@pytest.fixture
def workflows_client() -> TestClient:
    """Build a ``TestClient`` exposing only the workflows router.

    Returns:
        A ``TestClient`` over a minimal FastAPI app with auth bypassed.
    """
    app = FastAPI()
    app.include_router(create_workflows_router())
    bypass_auth(app)
    return TestClient(app, raise_server_exceptions=False)


def _transform_workflow(transform_code: str) -> dict:
    """Build a transform-only workflow payload around the given code.

    Args:
        transform_code: Source for the single transform node.

    Returns:
        The workflow spec dict.
    """
    return {
        "nodes": [
            {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
            {
                "id": "shout",
                "kind": "transform",
                "transform_code": transform_code,
                "input_fields": [{"name": "text"}],
                "output_fields": [{"name": "shout"}],
            },
            {"id": "out", "kind": "output", "fields": [{"name": "shout"}]},
        ],
        "edges": [
            {"source": "inp", "source_port": "text", "target": "shout", "target_port": "text"},
            {"source": "shout", "source_port": "shout", "target": "out", "target_port": "shout"},
        ],
    }


def test_dry_run_executes_and_returns_traces(workflows_client: TestClient) -> None:
    """A valid transform-only workflow runs and returns outputs plus per-node traces."""
    resp = workflows_client.post(
        "/workflows/dry-run",
        json={
            "workflow": _transform_workflow("def transform(text):\n    return {'shout': text.upper()}\n"),
            "inputs": {"text": "quiet"},
            "model_config": _MODEL_CFG,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["outputs"] == {"shout": "QUIET"}
    assert body["error"] is None
    assert [t["node_id"] for t in body["node_traces"]] == ["inp", "shout", "out"]
    assert body["node_traces"][1]["outputs"] == {"shout": "QUIET"}


def test_dry_run_node_failure_returns_200_with_failing_node(workflows_client: TestClient) -> None:
    """A raising node is an expected outcome: 200 with error, node id, and traces."""
    resp = workflows_client.post(
        "/workflows/dry-run",
        json={
            "workflow": _transform_workflow(
                "def transform(text):\n    raise ValueError('kaput: ' + text)\n"
            ),
            "inputs": {"text": "boom"},
            "model_config": _MODEL_CFG,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["outputs"] is None
    assert body["failed_node_id"] == "shout"
    assert "kaput" in body["error"]
    assert body["node_traces"][-1]["node_id"] == "shout"
    assert "kaput" in body["node_traces"][-1]["error"]


def test_dry_run_rejects_invalid_graph_with_detail(workflows_client: TestClient) -> None:
    """Deep-validation failures return 400 with the node-anchored detail."""
    resp = workflows_client.post(
        "/workflows/dry-run",
        json={
            "workflow": _transform_workflow(
                "def transform(wrong_param):\n    return {'shout': wrong_param}\n"
            ),
            "inputs": {"text": "x"},
            "model_config": _MODEL_CFG,
        },
    )

    assert resp.status_code == 400
    assert "shout" in str(resp.json())


def test_dry_run_rejects_missing_inputs(workflows_client: TestClient) -> None:
    """Missing workflow inputs are rejected before any execution."""
    resp = workflows_client.post(
        "/workflows/dry-run",
        json={
            "workflow": _transform_workflow("def transform(text):\n    return {'shout': text}\n"),
            "inputs": {},
            "model_config": _MODEL_CFG,
        },
    )

    assert resp.status_code == 400
    assert "text" in str(resp.json())


def test_dry_run_rejects_structurally_invalid_graph(workflows_client: TestClient) -> None:
    """A structurally broken graph fails Pydantic validation with 422."""
    workflow = _transform_workflow("def transform(text):\n    return {'shout': text}\n")
    workflow["edges"] = []
    resp = workflows_client.post(
        "/workflows/dry-run",
        json={
            "workflow": workflow,
            "inputs": {"text": "x"},
            "model_config": _MODEL_CFG,
        },
    )

    assert resp.status_code == 422
