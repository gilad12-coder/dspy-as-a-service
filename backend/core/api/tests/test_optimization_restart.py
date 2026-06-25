"""Tests for restart: POST /optimizations/{id}/restart.

Restart re-runs a terminal failed/cancelled optimization from scratch *in place*:
the same id is flipped back to ``pending`` after its prior attempt's result,
checkpoint and child artefacts are wiped — unlike retry (which mints a new id)
and unlike resume (which continues from a checkpoint). These tests pin the
endpoint's preconditions and the in-place reset.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..routers.optimizations import create_optimizations_router
from .conftest import bypass_auth
from .mocks import _BaseFakeJobStore


@pytest.fixture
def store() -> _BaseFakeJobStore:
    """Provide a fresh fake job store per test."""
    return _BaseFakeJobStore()


@pytest.fixture
def client(store: _BaseFakeJobStore) -> TestClient:
    """Build a TestClient exposing only the optimizations router over the fake store.

    Args:
        store: Fake job store wired into the router factory.

    Returns:
        A ``TestClient`` over a minimal FastAPI app.
    """
    app = FastAPI()
    app.include_router(create_optimizations_router(job_store=store, get_worker_ref=lambda: None))
    bypass_auth(app)
    return TestClient(app, raise_server_exceptions=False)


def test_restart_requeues_failed_run_in_place(client: TestClient, store: _BaseFakeJobStore) -> None:
    """A failed run restarts in place: 202, same id, pending, attempts reset to 0."""
    store.seed_job("r1", status="failed", attempts=2, username="alice")
    resp = client.post("/optimizations/r1/restart")
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["optimization_id"] == "r1"
    assert body["status"] == "pending"
    job = store.get_job("r1")
    assert job["status"] == "pending"
    assert job["attempts"] == 0


def test_restart_clears_prior_attempt(client: TestClient, store: _BaseFakeJobStore) -> None:
    """The prior attempt's result and checkpoint are discarded by a restart."""
    store.seed_job("r2", status="failed", result={"score": 0.9}, username="alice")
    store.save_gepa_checkpoint("r2", b"GEPA-STATE", iteration=7)
    resp = client.post("/optimizations/r2/restart")
    assert resp.status_code == 202, resp.text
    job = store.get_job("r2")
    assert job["result"] is None
    assert store.has_gepa_checkpoint("r2") is False


def test_restart_allows_cancelled_run(client: TestClient, store: _BaseFakeJobStore) -> None:
    """A cancelled run can also be restarted from scratch."""
    store.seed_job("c1", status="cancelled", username="alice")
    resp = client.post("/optimizations/c1/restart")
    assert resp.status_code == 202, resp.text


@pytest.mark.parametrize("status", ["running", "pending", "validating", "success"])
def test_restart_rejects_non_terminal_failure(client: TestClient, store: _BaseFakeJobStore, status: str) -> None:
    """Only failed/cancelled runs may be restarted — others 409 and stay untouched."""
    oid = f"s_{status}"
    store.seed_job(oid, status=status, username="alice")
    resp = client.post(f"/optimizations/{oid}/restart")
    assert resp.status_code == 409
    assert store.get_job(oid)["status"] == status


def test_restart_unknown_id_is_404(client: TestClient) -> None:
    """Restarting a missing optimization is a 404."""
    resp = client.post("/optimizations/nope/restart")
    assert resp.status_code == 404
