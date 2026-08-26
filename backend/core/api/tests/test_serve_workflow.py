"""Tests for serving workflow runs: materialization, anchor fields, node traces."""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...models import ProgramArtifact, WorkflowSpec
from ...service_gateway.optimization.workflow import build_workflow_program

# noinspection PyProtectedMember
from ..routers import _helpers
from ..routers.serve import create_serve_router
from .conftest import bypass_auth
from .mocks import _BaseFakeJobStore, make_run_result

_TRANSFORM_SPEC = {
    "nodes": [
        {"id": "inp", "kind": "input", "fields": [{"name": "text"}]},
        {
            "id": "shout",
            "kind": "transform",
            "transform_code": "def transform(text):\n    return {'shout': text.upper()}\n",
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

_MCP_SPEC = {
    "nodes": [
        {"id": "inp", "kind": "input", "fields": [{"name": "q"}]},
        {
            "id": "lookup",
            "kind": "mcp",
            "tool_name": "search",
            "input_fields": [{"name": "q"}],
            "output_field": {"name": "result"},
        },
        {"id": "out", "kind": "output", "fields": [{"name": "result"}]},
    ],
    "edges": [
        {"source": "inp", "source_port": "q", "target": "lookup", "target_port": "q"},
        {"source": "lookup", "source_port": "result", "target": "out", "target_port": "result"},
    ],
}


# noinspection PyProtectedMember
@pytest.fixture(autouse=True)
def _clear_program_cache() -> Generator[None, None, None]:
    """Reset the in-process program cache around every test in this file.

    Yields:
        ``None`` once the cache is cleared; cleared again on teardown.
    """
    _helpers.clear_program_cache()
    yield
    _helpers.clear_program_cache()


@pytest.fixture
def serve_store() -> _BaseFakeJobStore:
    """Provide a fresh fake job store.

    Returns:
        A new ``_BaseFakeJobStore`` instance.
    """
    return _BaseFakeJobStore()


@pytest.fixture
def serve_client(serve_store: _BaseFakeJobStore) -> TestClient:
    """Build a ``TestClient`` exposing only the serve router.

    Args:
        serve_store: Fake job store wired into the router factory.

    Returns:
        A ``TestClient`` over a minimal FastAPI app.
    """
    app = FastAPI()
    app.include_router(create_serve_router(job_store=serve_store))
    bypass_auth(app)
    return TestClient(app, raise_server_exceptions=False)


def _saved_state(spec_dict: dict, tmp_path) -> dict:
    """Persist a freshly built workflow program exactly like the run path does.

    Args:
        spec_dict: Workflow spec dict to build.
        tmp_path: pytest tmp directory for the intermediate state file.

    Returns:
        The state JSON dict ``persist_program`` would have stored.
    """
    program, _ = build_workflow_program(WorkflowSpec.model_validate(spec_dict))
    state_path = tmp_path / "program.json"
    program.save(str(state_path), save_program=False)
    return json.loads(state_path.read_text())


def _seed_workflow_job(
    store: _BaseFakeJobStore,
    opt_id: str,
    spec_dict: dict,
    state: dict | None,
    overview_extra: dict | None = None,
) -> None:
    """Seed a success workflow job whose program must be materialized from state.

    Args:
        store: Fake job store to seed into.
        opt_id: Job id to seed under.
        spec_dict: Workflow spec dict persisted on the overview.
        state: Program state JSON for the artifact (``None`` allowed).
        overview_extra: Extra fields merged into the payload overview.
    """
    artifact = ProgramArtifact(path=None, metadata=None, program_state_json=state, optimized_prompt=None)
    result = make_run_result(artifact)
    overview = {
        "model_name": "openai/gpt-4o-mini",
        "module_name": "workflow",
        "workflow": spec_dict,
        **(overview_extra or {}),
    }
    store.seed_job(opt_id, status="success", payload_overview=overview, result=result.model_dump())


def test_workflow_serve_materializes_and_traces(
    serve_client: TestClient, serve_store: _BaseFakeJobStore, tmp_path
) -> None:
    """A workflow job is rebuilt from spec+state and serves with node traces."""
    _seed_workflow_job(serve_store, "wf", _TRANSFORM_SPEC, _saved_state(_TRANSFORM_SPEC, tmp_path))

    resp = serve_client.post("/serve/wf", json={"inputs": {"text": "quiet"}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["outputs"] == {"shout": "QUIET"}
    assert body["input_fields"] == ["text"]
    assert body["output_fields"] == ["shout"]
    assert [t["node_id"] for t in body["node_traces"]] == ["inp", "shout", "out"]


def test_workflow_serve_info_reports_anchor_fields(
    serve_client: TestClient, serve_store: _BaseFakeJobStore, tmp_path
) -> None:
    """Serve info for a workflow reflects the anchor fields, not the first predictor."""
    _seed_workflow_job(serve_store, "wfi", _TRANSFORM_SPEC, _saved_state(_TRANSFORM_SPEC, tmp_path))

    resp = serve_client.get("/serve/wfi/info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["module_name"] == "workflow"
    assert body["input_fields"] == ["text"]
    assert body["output_fields"] == ["shout"]


def test_workflow_serve_rejects_tool_graph_without_tool_source(
    serve_client: TestClient, serve_store: _BaseFakeJobStore
) -> None:
    """A tool-using workflow without a persisted tool_source cannot be re-materialized."""
    _seed_workflow_job(serve_store, "wft", _MCP_SPEC, {"metadata": {}})

    resp = serve_client.post("/serve/wft", json={"inputs": {"q": "cats"}})

    assert resp.status_code == 409


def test_workflow_serve_rejects_snapshot_tool_source(
    serve_client: TestClient, serve_store: _BaseFakeJobStore
) -> None:
    """dataset_snapshot tool sources are not yet servable for workflows."""
    _seed_workflow_job(
        serve_store,
        "wfs",
        _MCP_SPEC,
        {"metadata": {}},
        overview_extra={"tool_source": {"kind": "dataset_snapshot"}},
    )

    resp = serve_client.post("/serve/wfs", json={"inputs": {"q": "cats"}})

    assert resp.status_code == 409
