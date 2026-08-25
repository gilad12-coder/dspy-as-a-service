"""Tests for the ``/workflows/dry-run`` endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ...billing.byok_vault import ProviderKeyVault
from ...config import settings
from ...storage.models import Base
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
            "workflow": _transform_workflow("def transform(text):\n    raise ValueError('kaput: ' + text)\n"),
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
            "workflow": _transform_workflow("def transform(wrong_param):\n    return {'shout': wrong_param}\n"),
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


def test_dry_run_resolves_stored_custom_byok_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow dry runs use a verified custom provider connection from the vault."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(
        settings,
        "byok_vault_key",
        SecretStr(Fernet.generate_key().decode("utf-8")),
    )
    response = SimpleNamespace(status_code=200, is_success=True)
    with patch("core.billing.byok_vault.httpx.get", return_value=response):
        ProviderKeyVault(engine=engine).save_key(
            "alice",
            "custom",
            "private-secret",
            api_base="https://inference.example/v1",
        )
    app = FastAPI()
    app.include_router(create_workflows_router(job_store=SimpleNamespace(engine=engine)))
    bypass_auth(app)
    client = TestClient(app, raise_server_exceptions=False)
    builder = MagicMock(return_value=MagicMock())

    with patch("core.api.routers.workflows.build_language_model", builder):
        resp = client.post(
            "/workflows/dry-run",
            json={
                "workflow": _transform_workflow("def transform(text):\n    return {'shout': text.upper()}\n"),
                "inputs": {"text": "quiet"},
                "model_config": {
                    "name": "openai/private-chat",
                    "token_source": "byok",
                    "byok_provider": "custom",
                },
            },
        )

    assert resp.status_code == 200
    resolved = builder.call_args.args[0]
    assert resolved.base_url == "https://inference.example/v1"
    assert resolved.extra["api_key"] == "private-secret"
