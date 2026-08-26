"""Tests for the ``/datasets/profile`` route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from core.exceptions import AppError
from core.i18n_en import t_en
from core.i18n_keys import I18nKey

from ..auth import AuthenticatedUser, get_authenticated_user
from ..routers.datasets import create_datasets_router
from .mocks import FakeJobStore


@pytest.fixture
def datasets_client() -> TestClient:
    """Build a ``TestClient`` exposing the datasets router with error handlers.

    Returns:
        A ``TestClient`` over a minimal FastAPI app with ``AppError`` and
        validation handlers wired in to mirror production behaviour.
    """
    app = FastAPI()
    app.include_router(create_datasets_router(job_store=object()))
    app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
        username="alice", role="user", groups=()
    )

    # Mirror the app-level AppError handler so ValidationError (400) surfaces
    # as a proper HTTP response instead of bubbling up as a 500.
    @app.exception_handler(AppError)
    async def _app_error_handler(_request, exc: AppError) -> JSONResponse:
        """Map ``AppError`` to a JSON envelope mirroring production behaviour."""
        content = {"error": exc.error_code.lower(), "detail": exc.message}
        if exc.code:
            content["code"] = exc.code
            content["params"] = exc.params
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_request, exc: RequestValidationError) -> JSONResponse:
        """Map Pydantic validation errors to a 422 JSON response (test mirror)."""
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": exc.errors()})

    return TestClient(app, raise_server_exceptions=False)


def test_profile_returns_plan_and_profile(datasets_client: TestClient) -> None:
    """Profiling a sized dataset returns a plan and shape profile."""
    payload = {
        "dataset": [
            {"q": "q1", "a": "yes"},
            {"q": "q2", "a": "no"},
            {"q": "q3", "a": "yes"},
            {"q": "q4", "a": "no"},
        ]
        * 100,
        "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        "seed": 42,
    }

    resp = datasets_client.post("/datasets/profile", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["row_count"] == 400
    assert body["profile"]["target"]["name"] == "a"
    assert body["plan"]["seed"] == 42
    counts = body["plan"]["counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 400


def test_profile_empty_dataset_returns_400(datasets_client: TestClient) -> None:
    """An empty dataset is rejected with a 400 carrying English detail and i18n code."""
    payload = {
        "dataset": [],
        "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
    }

    resp = datasets_client.post("/datasets/profile", json=payload)

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == t_en(I18nKey.DATASET_PROFILE_EMPTY)
    assert body["code"] == I18nKey.DATASET_PROFILE_EMPTY.value


def test_profile_missing_column_mapping_returns_422(datasets_client: TestClient) -> None:
    """``column_mapping`` is required; its absence is a 422."""
    resp = datasets_client.post("/datasets/profile", json={"dataset": [{"q": "x"}]})

    assert resp.status_code == 422


def test_profile_omitted_seed_is_still_populated(datasets_client: TestClient) -> None:
    """A missing ``seed`` is still echoed back as a populated integer."""
    payload = {
        "dataset": [{"q": f"q{i}", "a": "yes"} for i in range(50)],
        "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
    }

    resp = datasets_client.post("/datasets/profile", json=payload)

    assert resp.status_code == 200
    assert isinstance(resp.json()["plan"]["seed"], int)


@pytest.fixture
def staged_client() -> tuple[TestClient, FakeJobStore]:
    """Build a datasets ``TestClient`` with a fake store and a fixed auth user.

    Returns:
        The bound client and the backing ``FakeJobStore`` so tests can stage
        rows directly and assert the GET rehydration path.
    """
    store = FakeJobStore()
    app = FastAPI()
    app.include_router(create_datasets_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
        username="alice", role="user", groups=()
    )
    return TestClient(app, raise_server_exceptions=False), store


def test_get_staged_dataset_round_trips_rows_and_columns(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """A staged dataset is fetchable by id with rows + first-row column order."""
    client, store = staged_client
    rows = [{"q": "q1", "a": "yes"}, {"q": "q2", "a": "no"}]
    staged_id = store.stage_dataset(username="alice", dataset_filename="d.json", rows=rows)

    resp = client.get(f"/datasets/staged/{staged_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["staged_dataset_id"] == staged_id
    assert body["columns"] == ["q", "a"]
    assert body["rows"] == rows
    assert body["row_count"] == 2


def test_get_staged_dataset_unknown_id_returns_404(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """An unknown staged id is a 404, not an empty 200."""
    client, _ = staged_client

    resp = client.get("/datasets/staged/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "dataset.staged.not_found"


def test_get_staged_dataset_is_scoped_to_owner(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """A dataset staged by another user is invisible (404), not readable."""
    client, store = staged_client
    staged_id = store.stage_dataset(
        username="mallory", dataset_filename="d.json", rows=[{"q": "x", "a": "y"}]
    )

    resp = client.get(f"/datasets/staged/{staged_id}")

    assert resp.status_code == 404


def test_profile_rehydrates_staged_dataset_by_id(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """Profiling with only a staged id rehydrates the rows and returns a profile."""
    client, store = staged_client
    rows = [{"q": f"q{i}", "a": "yes" if i % 2 else "no"} for i in range(60)]
    staged_id = store.stage_dataset(username="alice", dataset_filename="d.json", rows=rows)

    resp = client.post(
        "/datasets/profile",
        json={
            "staged_dataset_id": staged_id,
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["profile"]["row_count"] == 60


def test_profile_unknown_staged_id_returns_404(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """An unknown staged id surfaces a 404 rather than an empty-dataset 400."""
    client, _ = staged_client

    resp = client.post(
        "/datasets/profile",
        json={
            "staged_dataset_id": "does-not-exist",
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        },
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "dataset.staged.not_found"


def test_profile_inline_dataset_ignores_staged_id(
    staged_client: tuple[TestClient, FakeJobStore],
) -> None:
    """An inline dataset wins; a staged id alongside it is not rehydrated."""
    client, store = staged_client
    staged_id = store.stage_dataset(
        username="alice", dataset_filename="d.json", rows=[{"q": "x", "a": "y"}]
    )

    resp = client.post(
        "/datasets/profile",
        json={
            "dataset": [{"q": f"q{i}", "a": "yes"} for i in range(40)],
            "staged_dataset_id": staged_id,
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["profile"]["row_count"] == 40
