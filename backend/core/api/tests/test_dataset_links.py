"""Tests for the optimization⇄dataset links and submit-by-reference path.

Exercises the producer/consumer wiring task that connects the personal dataset
library to optimization submission against an in-memory SQLite store (the
sibling routers' pattern: a ``RemoteDBJobStore`` subclass that skips the
pgvector bootstrap so ``Base.metadata.create_all`` stands up every table). The
submissions router is mounted alongside the library router with a worker signal
stub, so a by-reference submit, the dataset→optimizations reverse link, and
saving a run's dataset all round-trip through real rows.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...constants import PAYLOAD_OVERVIEW_SOURCE_DATASET_ID
from ...storage.models import Base, BillingCustomerModel
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..converters import overview_to_base_fields, parse_overview
from ..errors import DomainError
from ..routers import submissions as _sub_mod
from ..routers.dataset_library import create_dataset_library_router
from ..routers.submissions import create_submissions_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())

_ROWS = [{"question": "2+2", "answer": "4"}, {"question": "3+3", "answer": "6"}]
_SCHEMA = {
    "column_order": ["question", "answer"],
    "column_roles": {"question": "input", "answer": "output"},
    "column_kinds": {"question": "text", "answer": "text"},
}
_MAPPING = {"inputs": {"q": "question"}, "outputs": {"a": "answer"}}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for the link tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


class _QueueSignalWorker:
    """Worker stub that accepts a local queue hint after DB persistence."""

    def enqueue_job(self, optimization_id: str) -> None:
        """Accept a queue hint for an already-persisted job.

        Args:
            optimization_id: Id of the pending job.
        """


class _FakeService:
    """Service stub whose validate methods always pass."""

    def validate_payload(self, payload: Any) -> None:
        """Accept any run payload (no-op)."""

    def validate_grid_search_payload(self, payload: Any) -> None:
        """Accept any grid-search payload (no-op)."""


def _app_for(store: _MemStore, user: AuthenticatedUser, *, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Mount the submissions + library routers on a store, authed as ``user``.

    Args:
        store: Backing store both routers share.
        user: Identity the auth dependency resolves to for every request.
        monkeypatch: Fixture used to stub the worker and notifier.

    Returns:
        A FastAPI app whose ``DomainError``s render the production envelope.
    """
    worker = _QueueSignalWorker()
    monkeypatch.setattr(_sub_mod, "get_worker", lambda *a, **kw: worker)
    monkeypatch.setattr(_sub_mod, "notify_job_started", lambda **_: None)

    # No free allowance exists, so the authed user is funded explicitly to pass
    # the 402 credit gate on run submissions.
    with Session(store.engine) as session:
        if session.get(BillingCustomerModel, user.username) is None:
            session.add(
                BillingCustomerModel(
                    username=user.username,
                    stripe_customer_id=f"cus_{user.username}",
                    credit_balance=10_000,
                    grant_remaining=0,
                )
            )
            session.commit()

    app = FastAPI()
    app.include_router(create_submissions_router(service=_FakeService(), job_store=store))
    app.include_router(create_dataset_library_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: user

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return app


def _client(store: _MemStore, user: AuthenticatedUser, *, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a test client over the link routers authed as ``user``.

    Args:
        store: Backing store the client shares.
        user: Identity the auth dependency resolves to.
        monkeypatch: Fixture used to stub the worker and notifier.

    Returns:
        A ``TestClient`` driving the mounted app.
    """
    return TestClient(_app_for(store, user, monkeypatch=monkeypatch), raise_server_exceptions=False)


def _save(client: TestClient, *, name: str = "Math", rows: list[dict[str, Any]] = _ROWS) -> str:
    """Save a dataset to the caller's library and return its id.

    Args:
        client: Authenticated test client.
        name: Display name for the entry.
        rows: Dataset rows to save.

    Returns:
        The new dataset's id.
    """
    resp = client.post(
        "/datasets/library",
        json={"name": name, "source": "upload", "dataset": rows, "column_schema": _SCHEMA},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset"]["id"]


def _ref_payload(source_dataset_id: str) -> dict[str, Any]:
    """Build a by-reference run payload pointing at a library dataset.

    Args:
        source_dataset_id: Library dataset id to run by reference.

    Returns:
        A run-submission dict carrying ``source_dataset_id`` and no inline rows.
    """
    return {
        "name": "ref-run",
        "module_name": "predict",
        "signature_code": "class Sig(dspy.Signature): q: str = dspy.InputField(); a: str = dspy.OutputField()",
        "metric_code": "def metric(example, pred, trace=None): return 1.0",
        "optimizer_name": "gepa",
        "source_dataset_id": source_dataset_id,
        "column_mapping": _MAPPING,
        "column_order": ["question", "answer"],
        "model_settings": {"name": "gpt-4o-mini"},
    }


def _inline_payload(rows: list[dict[str, Any]] = _ROWS) -> dict[str, Any]:
    """Build an inline run payload carrying its own dataset rows.

    Args:
        rows: Dataset rows to inline.

    Returns:
        A run-submission dict with inline ``dataset`` and a column order.
    """
    return {
        "name": "inline-run",
        "module_name": "predict",
        "signature_code": "class Sig(dspy.Signature): q: str = dspy.InputField(); a: str = dspy.OutputField()",
        "metric_code": "def metric(example, pred, trace=None): return 1.0",
        "optimizer_name": "gepa",
        "dataset": rows,
        "column_mapping": _MAPPING,
        "column_order": ["question", "answer"],
        "dataset_filename": "math.csv",
        "model_settings": {"name": "gpt-4o-mini"},
    }


def test_submit_by_reference_inlines_rows_and_records_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A by-reference submit loads the saved rows and records the dataset link."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    dataset_id = _save(client)

    resp = client.post("/run", json=_ref_payload(dataset_id))
    assert resp.status_code == 201, resp.text
    optimization_id = resp.json()["optimization_id"]

    job = store.get_job(optimization_id)
    assert parse_overview(job)[PAYLOAD_OVERVIEW_SOURCE_DATASET_ID] == dataset_id
    # The persisted payload carries the resolved rows, not the reference.
    assert job["payload"]["dataset"] == _ROWS
    assert job["payload"]["source_dataset_id"] is None


def test_reverse_link_lists_runs_that_used_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dataset→optimizations endpoint lists a run submitted from the dataset."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    dataset_id = _save(client)
    optimization_id = client.post("/run", json=_ref_payload(dataset_id)).json()["optimization_id"]

    listing = client.get(f"/datasets/library/{dataset_id}/optimizations")
    assert listing.status_code == 200, listing.text
    refs = listing.json()["optimizations"]
    assert [r["optimization_id"] for r in refs] == [optimization_id]
    assert refs[0]["name"] == "ref-run"
    assert refs[0]["optimization_type"] == "run"


def test_reverse_link_excludes_runs_that_used_other_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with no source dataset never appears under any dataset's link."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    dataset_id = _save(client)
    client.post("/run", json=_inline_payload())

    listing = client.get(f"/datasets/library/{dataset_id}/optimizations").json()
    assert listing["optimizations"] == []


def test_submit_by_reference_denied_without_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller with no access to the dataset cannot run it by reference (404)."""
    store = _MemStore()
    alice = _client(store, _ALICE, monkeypatch=monkeypatch)
    dataset_id = _save(alice)

    bob = _client(store, _BOB, monkeypatch=monkeypatch)
    resp = bob.post("/run", json=_ref_payload(dataset_id))
    assert resp.status_code == 404
    assert resp.json()["code"] == "dataset.library.not_found"


def test_submit_rejects_dataset_and_source_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supplying both inline rows and a source id fails validation (422)."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    dataset_id = _save(client)
    payload = _ref_payload(dataset_id)
    payload["dataset"] = _ROWS
    assert client.post("/run", json=payload).status_code == 422


def test_save_run_dataset_round_trips_into_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving a run's dataset stores its rows and a schema from the run mapping."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    optimization_id = client.post("/run", json=_inline_payload()).json()["optimization_id"]

    saved = client.post(f"/datasets/library/from-optimization/{optimization_id}")
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["dataset"]["source"] == "optimization"
    assert body["dataset"]["name"] == "math.csv"
    dataset_id = body["dataset"]["id"]

    rows = client.get(f"/datasets/library/{dataset_id}/rows").json()
    assert rows["rows"] == _ROWS
    assert rows["columns"] == ["question", "answer"]
    assert rows["column_schema"]["column_roles"] == {"question": "input", "answer": "output"}
    assert rows["column_schema"]["column_kinds"] == {"question": "text", "answer": "text"}


def test_save_run_dataset_accepts_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit name overrides the run-derived default on the saved entry."""
    store = _MemStore()
    client = _client(store, _ALICE, monkeypatch=monkeypatch)
    optimization_id = client.post("/run", json=_inline_payload()).json()["optimization_id"]

    saved = client.post(
        f"/datasets/library/from-optimization/{optimization_id}", json={"name": "Renamed"}
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["dataset"]["name"] == "Renamed"


def test_save_run_dataset_requires_run_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller who cannot see the run cannot save its dataset (404)."""
    store = _MemStore()
    alice = _client(store, _ALICE, monkeypatch=monkeypatch)
    optimization_id = alice.post("/run", json=_inline_payload()).json()["optimization_id"]

    bob = _client(store, _BOB, monkeypatch=monkeypatch)
    resp = bob.post(f"/datasets/library/from-optimization/{optimization_id}")
    assert resp.status_code == 404


def test_overview_to_base_fields_surfaces_source_dataset_id() -> None:
    """The converter maps the stored source id onto the response field."""
    fields = overview_to_base_fields({PAYLOAD_OVERVIEW_SOURCE_DATASET_ID: "ds-123"})
    assert fields["source_dataset_id"] == "ds-123"
    assert overview_to_base_fields({})["source_dataset_id"] is None
