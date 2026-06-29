"""Tests for the ``/run`` and ``/grid-search`` submission endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...billing.service import GRANT_WINDOW_DAYS
from ...constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    OPTIMIZATION_TYPE_RUN,
    PAYLOAD_OVERVIEW_GENERATION_MODELS,
    PAYLOAD_OVERVIEW_IS_PRIVATE,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_REFLECTION_MODELS,
    PAYLOAD_OVERVIEW_TOTAL_PAIRS,
    PAYLOAD_OVERVIEW_USERNAME,
)
from ...i18n_keys import I18nKey
from ...registry import RegistryError
from ...service_gateway import ServiceError
from ...storage.models import Base, BillingCustomerModel, BillingProviderKeyModel
from ...storage.usage import StorageUsage
from ..model_catalog import CatalogModel, ModelCatalogResponse
from ..routers import submissions as _sub_mod
from ..routers.submissions import create_submissions_router
from .conftest import bypass_auth
from .mocks import fake_background_worker


class _FakeJobStore:
    """Minimal in-memory job store stub for the submissions router tests."""

    # create_job / set_payload_overview are the only write paths exercised here;
    # compute_user_storage + get_effective_user_storage_quota feed the unified
    # byte gate (count_jobs is kept for the legacy count-quota helper's callers).

    def __init__(self, *, job_count: int = 0, storage_used: int = 0, storage_quota: int = 1 << 40) -> None:
        """Initialise with canned counts/quota and an empty job map.

        Args:
            job_count: Value returned from ``count_jobs``.
            storage_used: Canned total returned from ``compute_user_storage``.
            storage_quota: Canned budget returned from
                ``get_effective_user_storage_quota`` (defaults high enough that
                the byte gate is a no-op unless a test lowers it).
        """
        self._count = job_count
        self._storage_used = storage_used
        self._storage_quota = storage_quota
        self._jobs: dict[str, dict] = {}

    def compute_user_storage(self, username: str) -> StorageUsage:
        """Return the canned unified storage usage for any caller.

        Args:
            username: Ignored; the canned total applies to every user.

        Returns:
            A :class:`StorageUsage` with the canned total and empty breakdown
            (the gate reads only ``.total``).
        """
        return StorageUsage(total=self._storage_used, breakdown={})

    def get_effective_user_storage_quota(self, username: str) -> int:
        """Return the canned storage budget for any caller.

        Args:
            username: Ignored; the canned budget applies to every user.

        Returns:
            The canned byte budget from construction.
        """
        return self._storage_quota

    def count_jobs(self, *, username: str | None = None, **_: Any) -> int:
        """Return the canned job count regardless of filter args.

        Args:
            username: Ignored; preserved to match the real signature.
            **_: Ignored extra filters.

        Returns:
            The canned job count from construction.
        """
        return self._count

    def create_job(
        self,
        optimization_id: str,
        estimated_remaining_seconds: float | None = None,
        *,
        username: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        """Record a new job entry under ``optimization_id``.

        Args:
            optimization_id: Identifier of the new job.
            estimated_remaining_seconds: Ignored; matches the real signature.
            username: Owner stored for idempotency lookups.
            idempotency_key: Optional dedup key stored alongside the row.
        """
        self._jobs[optimization_id] = {
            "username": username,
            "idempotency_key": idempotency_key,
            "overview": {},
        }

    def set_payload_overview(self, optimization_id: str, overview: dict) -> None:
        """Persist an overview dict against an existing job id.

        Args:
            optimization_id: Job id to attach the overview to.
            overview: Overview payload dict (deep-copied).
        """
        self._jobs.setdefault(optimization_id, {})["overview"] = dict(overview)

    def find_job_by_idempotency_key(self, username: str, idempotency_key: str) -> str | None:
        """Return the first job id matching ``(username, idempotency_key)``.

        Args:
            username: Submitter scope.
            idempotency_key: Client-supplied dedup key.

        Returns:
            The matching optimization id or ``None`` when not present.
        """
        if not username or not idempotency_key:
            return None
        for job_id, row in self._jobs.items():
            if row.get("username") == username and row.get("idempotency_key") == idempotency_key:
                return job_id
        return None

    def get_job(self, optimization_id: str) -> dict:
        """Return a copy of the stored job row plus its rehydrated overview.

        Args:
            optimization_id: Identifier of the job to fetch.

        Returns:
            A dict shaped like the real ``JobRecord`` consumed by the
            idempotency rehydration helper.

        Raises:
            KeyError: When the job does not exist.
        """
        row = self._jobs[optimization_id]
        return {
            "optimization_id": optimization_id,
            "status": row.get("status", "pending"),
            "created_at": row.get("created_at"),
            "payload_overview": row.get("overview", {}),
            "username": row.get("username"),
        }

    def created_ids(self) -> list[str]:
        """Return all job ids that were created via ``create_job``.

        Returns:
            List of optimization ids in insertion order.
        """
        return list(self._jobs.keys())

    def stage_dataset(self, username: str, dataset_filename: str, rows: list[dict[str, Any]]) -> str:
        """Persist staged rows and return an opaque id.

        Args:
            username: Submitter owner.
            dataset_filename: Original filename (kept for diagnostics).
            rows: Non-empty dataset rows.

        Returns:
            Newly minted staged dataset id.

        Raises:
            ValueError: When ``rows`` is empty.
        """
        if not rows:
            raise ValueError("staged dataset rows must be non-empty")
        staged = getattr(self, "_staged", None) or {}
        staged_id = f"staged-{len(staged) + 1}"
        staged[staged_id] = {"username": username, "filename": dataset_filename, "rows": list(rows)}
        self._staged = staged
        return staged_id

    def get_staged_dataset(self, staged_dataset_id: str, username: str) -> list[dict[str, Any]] | None:
        """Return staged rows when owned by ``username``.

        Args:
            staged_dataset_id: Id previously returned by ``stage_dataset``.
            username: Authenticated caller.

        Returns:
            The staged rows, or ``None`` when the id is unknown or owned by
            another user.
        """
        row = getattr(self, "_staged", {}).get(staged_dataset_id)
        if row is None or row["username"] != username:
            return None
        return list(row["rows"])

    def delete_staged_dataset(self, staged_dataset_id: str, username: str) -> bool:
        """Drop a staged row by id when owned by ``username``.

        Args:
            staged_dataset_id: Id to evict.
            username: Authenticated caller.

        Returns:
            ``True`` when a row was deleted.
        """
        staged = getattr(self, "_staged", {})
        row = staged.get(staged_dataset_id)
        if row is None or row["username"] != username:
            return False
        del staged[staged_dataset_id]
        return True


class _FakeService:
    """Service stub whose validate methods optionally raise a configured error."""

    def __init__(self, *, raise_on_validate: Exception | None = None) -> None:
        """Capture the exception (if any) to raise from validate methods.

        Args:
            raise_on_validate: Exception instance to raise from both validate methods.
        """
        self._exc = raise_on_validate

    def validate_payload(self, payload: Any) -> None:
        """Validate a run payload, raising the configured exception if set.

        Args:
            payload: Run payload (ignored; only the side effect matters).

        Raises:
            Exception: The configured ``raise_on_validate`` error, if any.
        """
        if self._exc:
            raise self._exc

    def validate_grid_search_payload(self, payload: Any) -> None:
        """Validate a grid-search payload, raising the configured exception if set.

        Args:
            payload: Grid-search payload (ignored; only the side effect matters).

        Raises:
            Exception: The configured ``raise_on_validate`` error, if any.
        """
        if self._exc:
            raise self._exc


def _run_payload() -> dict:
    """Build a minimal valid run payload for ``/run`` tests.

    Returns:
        A dict matching the run submission schema.
    """
    return {
        "name": "test-run",
        "username": "alice",
        "module_name": "predict",
        "module_kwargs": {},
        "signature_code": "class Sig(dspy.Signature): q: str = dspy.InputField(); a: str = dspy.OutputField()",
        "metric_code": "def metric(example, pred, trace=None): return 1.0",
        "optimizer_name": "gepa",
        "optimizer_kwargs": {},
        "compile_kwargs": {},
        "dataset": [{"question": "Q?", "answer": "A"}],
        "column_mapping": {"inputs": {"q": "question"}, "outputs": {"a": "answer"}},
        "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "shuffle": True,
        "seed": 42,
        "dataset_filename": "test.csv",
        "model_settings": {"name": "gpt-4o-mini"},
    }


def _grid_payload() -> dict:
    """Build a minimal valid grid-search payload for ``/grid-search`` tests.

    Returns:
        A dict matching the grid-search submission schema.
    """
    return {
        "name": "test-grid",
        "username": "alice",
        "module_name": "predict",
        "module_kwargs": {},
        "signature_code": "class Sig(dspy.Signature): q: str = dspy.InputField(); a: str = dspy.OutputField()",
        "metric_code": "def metric(example, pred, trace=None): return 1.0",
        "optimizer_name": "gepa",
        "optimizer_kwargs": {},
        "compile_kwargs": {},
        "dataset": [{"question": "Q?", "answer": "A"}],
        "column_mapping": {"inputs": {"q": "question"}, "outputs": {"a": "answer"}},
        "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "shuffle": True,
        "seed": None,
        "dataset_filename": "grid.csv",
        "generation_models": [{"name": "gpt-4o-mini"}],
        "reflection_models": [{"name": "gpt-4o"}],
    }


def _make_client(
    service: Any,
    store: _FakeJobStore,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Build a ``TestClient`` exposing the submissions router with stubbed worker.

    Args:
        service: Service stub used by the router.
        store: Fake job store wired into the router factory.
        monkeypatch: Pytest monkeypatch fixture to stub ``get_worker`` and notifier.

    Returns:
        A ``TestClient`` over a minimal FastAPI app.
    """
    worker = fake_background_worker()

    monkeypatch.setattr(_sub_mod, "get_worker", lambda *a, **kw: worker)
    monkeypatch.setattr(_sub_mod, "notify_job_started", lambda **_: None)

    app = FastAPI()
    app.include_router(create_submissions_router(service=service, job_store=store))
    bypass_auth(app)

    @app.exception_handler(HTTPException)
    async def _http_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        """Mirror the production HTTPException handler so ``code``/``params`` round-trip."""
        content: dict[str, object] = {"detail": exc.detail}
        code = getattr(exc, "code", None)
        if code:
            content["code"] = code
            content["params"] = getattr(exc, "params", None) or {}
        return JSONResponse(status_code=exc.status_code, content=content, headers=getattr(exc, "headers", None))

    return TestClient(app, raise_server_exceptions=False)


def test_submit_run_returns_201_with_optimization_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful run submission returns 201 with a fresh optimization id."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["is_private"] = True

    resp = client.post("/run", json=payload)

    assert resp.status_code == 201
    body = resp.json()
    assert "optimization_id" in body
    assert body["status"] == "pending"
    assert body["optimization_type"] == "run"


def test_submit_run_creates_job_in_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful submission creates exactly one matching job in the store."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 201
    created = store.created_ids()
    assert len(created) == 1
    assert created[0] == resp.json()["optimization_id"]


def test_submit_run_echoes_name_and_authenticated_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """The response echoes the payload ``name`` but always uses the auth user's username.

    The submission router overwrites ``payload.username`` from the bearer
    token, so a forged username in the request body is ignored — the
    response carries the authenticated identity.
    """
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["name"] = "my-run"
    payload["username"] = "bob"

    resp = client.post("/run", json=payload)

    body = resp.json()
    assert body["name"] == "my-run"
    assert body["username"] == "alice"


def test_submit_run_returns_400_on_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ServiceError`` from the service layer surfaces as a 400."""
    svc = _FakeService(raise_on_validate=ServiceError("bad module"))
    store = _FakeJobStore()
    client = _make_client(svc, store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 400


def test_submit_run_returns_400_on_registry_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``RegistryError`` from the service layer surfaces as a 400."""
    svc = _FakeService(raise_on_validate=RegistryError("not registered"))
    store = _FakeJobStore()
    client = _make_client(svc, store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 400


def test_submit_run_returns_409_when_over_storage_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submission that would exceed the user's storage budget is rejected with 409."""
    store = _FakeJobStore(storage_quota=0)
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 409
    assert resp.json()["code"] == "user.storage.quota_exceeded"


def test_submit_run_returns_422_on_missing_required_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run payload missing ``signature_code`` returns a 422.

    ``username`` is intentionally optional on the wire — the server overwrites
    it from the authenticated session — so the contract test instead drops
    ``signature_code`` to exercise the schema's still-required surface.
    """
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    del payload["signature_code"]

    resp = client.post("/run", json=payload)

    assert resp.status_code == 422


def test_submit_run_accepts_payload_without_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run payload that omits ``username`` succeeds — auth fills it in."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    del payload["username"]

    resp = client.post("/run", json=payload)

    assert resp.status_code == 201
    assert resp.json()["username"] == "alice"


def test_submit_run_accepts_staged_dataset_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run payload with ``staged_dataset_id`` hydrates the dataset server-side."""
    store = _FakeJobStore()
    rows = [{"q": "q", "a": "a"}]
    staged_id = store.stage_dataset(username="alice", dataset_filename="x.json", rows=rows)
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    del payload["dataset"]
    payload["staged_dataset_id"] = staged_id

    resp = client.post("/run", json=payload)

    assert resp.status_code == 201
    assert store.get_staged_dataset(staged_id, "alice") is None


def test_submit_run_rejects_unknown_staged_dataset_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ``staged_dataset_id`` produces a 400."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    del payload["dataset"]
    payload["staged_dataset_id"] = "does-not-exist"

    resp = client.post("/run", json=payload)

    assert resp.status_code == 400


def test_submit_run_retains_staged_dataset_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged row survives a failed submit so the user can retry without re-uploading.

    Reproduces the chat-flow bug: the agent submits with an invalid payload
    (e.g. missing reflection_model_config). The materialization step used to
    delete the staged row eagerly, so a corrected retry then 400'd with
    staged_dataset_not_found.
    """

    class _RejectingService:
        """Service stub whose ``validate_payload`` always rejects."""

        def validate_payload(self, _payload: Any) -> None:
            """Reject any run payload to simulate a failed submit.

            Raises:
                ServiceError: Always, mimicking a missing reflection model config.
            """
            raise ServiceError("missing reflection_model_config")

    store = _FakeJobStore()
    rows = [{"q": "q", "a": "a"}]
    staged_id = store.stage_dataset(username="alice", dataset_filename="x.json", rows=rows)
    client = _make_client(_RejectingService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    del payload["dataset"]
    payload["staged_dataset_id"] = staged_id

    resp = client.post("/run", json=payload)

    assert resp.status_code == 400
    # The staged row must survive so a corrected retry can reuse it.
    assert store.get_staged_dataset(staged_id, "alice") == rows


def test_submit_run_returns_422_on_invalid_split_fractions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Split fractions that do not sum to 1.0 produce a 422."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["split_fractions"] = {"train": 0.5, "val": 0.5, "test": 0.5}

    resp = client.post("/run", json=payload)

    assert resp.status_code == 422


def test_submit_run_returns_422_on_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty dataset is rejected at the schema layer with a 422."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["dataset"] = []

    resp = client.post("/run", json=payload)

    assert resp.status_code == 422


def test_submit_grid_search_returns_201(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful grid-search submission returns 201 with the right type tag."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/grid-search", json=_grid_payload())

    assert resp.status_code == 201
    body = resp.json()
    assert body["optimization_type"] == "grid_search"
    assert body["status"] == "pending"


def test_submit_grid_search_seed_assigned_when_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``null`` seed at submit time is replaced by a concrete seed in the overview."""
    # Reproducibility contract: when caller sends seed=None, the overview must persist a concrete seed.
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["seed"] = None

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    stored = store._jobs[opt_id]["overview"]
    assert stored.get("seed") is not None


def test_submit_grid_search_returns_400_on_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ServiceError`` during grid validation surfaces as a 400."""
    svc = _FakeService(raise_on_validate=ServiceError("bad optimizer"))
    store = _FakeJobStore()
    client = _make_client(svc, store, monkeypatch=monkeypatch)

    resp = client.post("/grid-search", json=_grid_payload())

    assert resp.status_code == 400


def test_submit_grid_search_returns_422_on_empty_generation_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``generation_models`` list is rejected at the schema layer with 422."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["generation_models"] = []

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 422


def test_submit_run_overview_contains_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted run overview contains the expected canonical keys."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["is_private"] = True

    resp = client.post("/run", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    overview = store._jobs[opt_id]["overview"]

    assert overview[PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE] == OPTIMIZATION_TYPE_RUN
    assert overview[PAYLOAD_OVERVIEW_USERNAME] == "alice"
    assert overview[PAYLOAD_OVERVIEW_MODULE_NAME] == "predict"
    assert overview[PAYLOAD_OVERVIEW_OPTIMIZER_NAME] == "gepa"
    assert overview[PAYLOAD_OVERVIEW_IS_PRIVATE] is True
    assert PAYLOAD_OVERVIEW_MODEL_NAME in overview


def test_submit_grid_search_overview_contains_total_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grid overview's ``total_pairs`` equals ``len(gen) * len(ref)``."""
    # Invariant: total_pairs in overview == len(gen) * len(ref).
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["is_private"] = True
    # 2 generation models × 1 reflection model = 2 pairs
    payload["generation_models"] = [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}]
    payload["reflection_models"] = [{"name": "gpt-4o"}]

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    overview = store._jobs[opt_id]["overview"]

    assert overview[PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE] == OPTIMIZATION_TYPE_GRID_SEARCH
    assert overview[PAYLOAD_OVERVIEW_TOTAL_PAIRS] == 2
    assert overview[PAYLOAD_OVERVIEW_IS_PRIVATE] is True


def test_submit_grid_search_skips_validation_when_service_lacks_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service without ``validate_grid_search_payload`` is tolerated and the job is enqueued."""
    # The route uses a hasattr guard so an older service implementation without
    # validate_grid_search_payload must NOT cause a 500 — the job is enqueued anyway.

    class _ServiceWithoutGridValidation:
        """Service stub that only implements ``validate_payload``."""

        def validate_payload(self, payload: Any) -> None:
            """Accept any run payload without raising."""

        # NOTE: no validate_grid_search_payload

    store = _FakeJobStore()
    client = _make_client(_ServiceWithoutGridValidation(), store, monkeypatch=monkeypatch)

    resp = client.post("/grid-search", json=_grid_payload())

    assert resp.status_code == 201
    assert resp.json()["optimization_type"] == "grid_search"


def _fake_catalog(*values: str) -> ModelCatalogResponse:
    """Build a synthetic catalog from a list of model id strings.

    Args:
        *values: Model id strings to include in the catalog.

    Returns:
        A ``ModelCatalogResponse`` containing one entry per id with provider ``openai``.
    """
    return ModelCatalogResponse(
        providers=[],
        models=[CatalogModel(value=v, label=v, provider="openai", available=True) for v in values],
    )


def test_submit_grid_search_use_all_generation_models_expands_from_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_all_available_generation_models`` expands to the catalog's full list."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _fake_catalog("openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3-5-sonnet"),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload.pop("generation_models")
    payload["use_all_available_generation_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    overview = store._jobs[opt_id]["overview"]
    gen_names = [m["name"] for m in overview[PAYLOAD_OVERVIEW_GENERATION_MODELS]]
    assert gen_names == ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3-5-sonnet"]
    assert overview[PAYLOAD_OVERVIEW_TOTAL_PAIRS] == 3


def test_submit_grid_search_use_all_generation_models_overrides_client_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``use_all`` flag overrides any explicit client-supplied generation models."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _fake_catalog("openai/gpt-4o-mini"),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["generation_models"] = [{"name": "bogus-legacy-model"}]
    payload["use_all_available_generation_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    overview = store._jobs[opt_id]["overview"]
    gen_names = [m["name"] for m in overview[PAYLOAD_OVERVIEW_GENERATION_MODELS]]
    assert gen_names == ["openai/gpt-4o-mini"]


def test_submit_grid_search_use_all_reflection_models_expands_from_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_all_available_reflection_models`` expands to the catalog's full list."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _fake_catalog("openai/gpt-4o-mini", "openai/gpt-4o"),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload.pop("reflection_models")
    payload["use_all_available_reflection_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    opt_id = resp.json()["optimization_id"]
    overview = store._jobs[opt_id]["overview"]
    ref_names = [m["name"] for m in overview[PAYLOAD_OVERVIEW_REFLECTION_MODELS]]
    assert ref_names == ["openai/gpt-4o-mini", "openai/gpt-4o"]
    assert overview[PAYLOAD_OVERVIEW_TOTAL_PAIRS] == len(payload["generation_models"]) * 2


def test_submit_grid_search_use_all_both_sides_multiplies_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting both ``use_all`` flags yields a Cartesian product over the catalog."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _fake_catalog("openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3-5-sonnet"),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload.pop("generation_models")
    payload.pop("reflection_models")
    payload["use_all_available_generation_models"] = True
    payload["use_all_available_reflection_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201
    overview = store._jobs[resp.json()["optimization_id"]]["overview"]
    assert len(overview[PAYLOAD_OVERVIEW_GENERATION_MODELS]) == 3
    assert len(overview[PAYLOAD_OVERVIEW_REFLECTION_MODELS]) == 3
    assert overview[PAYLOAD_OVERVIEW_TOTAL_PAIRS] == 9


def test_submit_grid_search_use_all_generation_models_returns_400_when_catalog_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty catalog plus the generation ``use_all`` flag returns a 400."""
    monkeypatch.setattr(_sub_mod, "get_catalog_cached", _fake_catalog)
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload.pop("generation_models")
    payload["use_all_available_generation_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 400
    assert resp.json()["code"] == I18nKey.SUBMIT_NO_MODELS_AVAILABLE.value


def test_submit_grid_search_use_all_reflection_models_returns_400_when_catalog_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty catalog plus the reflection ``use_all`` flag returns a 400."""
    monkeypatch.setattr(_sub_mod, "get_catalog_cached", _fake_catalog)
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload.pop("reflection_models")
    payload["use_all_available_reflection_models"] = True

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 400
    assert resp.json()["code"] == I18nKey.SUBMIT_NO_MODELS_AVAILABLE.value


_VISION_SIG_CODE = (
    "import dspy\n"
    "class VisionQA(dspy.Signature):\n"
    "    picture: dspy.Image = dspy.InputField()\n"
    "    question: str = dspy.InputField()\n"
    "    answer: str = dspy.OutputField()\n"
)


def _vision_catalog(*, vision_models: list[str], text_only_models: list[str] | None = None) -> ModelCatalogResponse:
    """Build a catalog with explicit vision-capable and text-only entries.

    Args:
        vision_models: Model id strings flagged ``supports_vision=True``.
        text_only_models: Model id strings flagged ``supports_vision=False``.

    Returns:
        A ``ModelCatalogResponse`` mixing the two groups.
    """
    models: list[CatalogModel] = [
        CatalogModel(value=v, label=v, provider="openai", available=True, supports_vision=True)
        for v in vision_models
    ]
    models.extend(
        CatalogModel(value=v, label=v, provider="openai", available=True, supports_vision=False)
        for v in text_only_models or []
    )
    return ModelCatalogResponse(providers=[], models=models)


def test_submit_run_rejects_image_signature_with_non_vision_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with an image signature using a non-vision model is rejected with 400."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _vision_catalog(vision_models=[], text_only_models=["gpt-4o-mini"]),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["signature_code"] = _VISION_SIG_CODE
    payload["column_mapping"] = {
        "inputs": {"picture": "img", "question": "q"},
        "outputs": {"answer": "a"},
    }
    payload["dataset"] = [{"img": "https://example.com/cat.png", "q": "what?", "a": "cat"}]
    payload["model_settings"] = {"name": "gpt-4o-mini"}

    resp = client.post("/run", json=payload)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # Hebrew message includes the field name and the offending model identifier.
    assert "picture" in detail
    assert "gpt-4o-mini" in detail


def test_submit_run_accepts_image_signature_with_vision_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with an image signature using a vision-capable model is accepted."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _vision_catalog(vision_models=["gpt-4o"]),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _run_payload()
    payload["signature_code"] = _VISION_SIG_CODE
    payload["column_mapping"] = {
        "inputs": {"picture": "img", "question": "q"},
        "outputs": {"answer": "a"},
    }
    payload["dataset"] = [{"img": "https://example.com/cat.png", "q": "what?", "a": "cat"}]
    payload["model_settings"] = {"name": "gpt-4o"}

    resp = client.post("/run", json=payload)

    assert resp.status_code == 201


def test_submit_run_text_signature_skips_vision_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pure text signature does not trigger the vision-capable model check."""
    # Empty catalog — would reject every model if the vision gate ran. It must not.
    monkeypatch.setattr(_sub_mod, "get_catalog_cached", lambda: _vision_catalog(vision_models=[]))
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 201


def test_submit_grid_search_rejects_image_signature_with_any_non_vision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grid search rejects image signatures if any selected model lacks vision support."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _vision_catalog(vision_models=["gpt-4o"], text_only_models=["text-only-model"]),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["signature_code"] = _VISION_SIG_CODE
    payload["column_mapping"] = {
        "inputs": {"picture": "img", "question": "q"},
        "outputs": {"answer": "a"},
    }
    payload["dataset"] = [{"img": "https://example.com/cat.png", "q": "what?", "a": "cat"}]
    payload["generation_models"] = [{"name": "gpt-4o"}, {"name": "text-only-model"}]
    payload["reflection_models"] = [{"name": "gpt-4o"}]

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "text-only-model" in detail
    assert "gpt-4o" not in detail.split("text-only-model")[0] or "text-only-model" in detail


def test_submit_grid_search_accepts_image_signature_when_all_models_support_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grid search accepts image signatures when every selected model supports vision."""
    monkeypatch.setattr(
        _sub_mod,
        "get_catalog_cached",
        lambda: _vision_catalog(vision_models=["gpt-4o", "gpt-4o-mini"]),
    )
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    payload = _grid_payload()
    payload["signature_code"] = _VISION_SIG_CODE
    payload["column_mapping"] = {
        "inputs": {"picture": "img", "question": "q"},
        "outputs": {"answer": "a"},
    }
    payload["dataset"] = [{"img": "https://example.com/cat.png", "q": "what?", "a": "cat"}]
    payload["generation_models"] = [{"name": "gpt-4o"}, {"name": "gpt-4o-mini"}]
    payload["reflection_models"] = [{"name": "gpt-4o"}]

    resp = client.post("/grid-search", json=payload)

    assert resp.status_code == 201


def test_submit_run_returns_402_when_credits_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A managed run is blocked at submit when the account has no spendable credits."""
    # StaticPool keeps one shared connection so the in-memory schema is visible
    # from the request threadpool, not just the thread that ran create_all.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="alice",
                stripe_customer_id="cus_alice",
                credit_balance=0,
                grant_remaining=0,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()

    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 402
    assert resp.json()["code"] == "billing.insufficient_credits"
    assert store.created_ids() == []


def test_submit_run_allowed_with_remaining_credits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A managed run with grant left passes the credit gate and is enqueued."""
    # StaticPool keeps one shared connection so the in-memory schema is visible
    # from the request threadpool, not just the thread that ran create_all.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="alice",
                stripe_customer_id="cus_alice",
                credit_balance=0,
                grant_remaining=50,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()

    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 201


def _billing_engine() -> Any:
    """Build a StaticPool in-memory engine with the billing schema."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return engine


def test_submit_run_managed_frontier_model_blocked_without_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed run on a frontier model is refused when the account can't unlock it."""
    engine = _billing_engine()
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="alice",
                stripe_customer_id="cus_alice",
                credit_balance=0,
                grant_remaining=200,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()
    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    payload = {**_run_payload(), "model_settings": {"name": "openai/gpt-4o"}, "token_source": "managed"}
    resp = client.post("/run", json=payload)

    assert resp.status_code == 402
    assert resp.json()["code"] == "billing.frontier_locked"
    assert store.created_ids() == []


def test_submit_run_byok_frontier_model_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BYOK run on a frontier model is never locked (own key), and persists the mode."""
    engine = _billing_engine()
    with Session(engine) as session:
        # BYOK runs now require a saved connection for the model's provider.
        session.add(
            BillingProviderKeyModel(
                username="alice",
                provider="openai",
                secret_ciphertext=b"ciphertext",
                last4="4o20",
                status="verified",
            )
        )
        session.commit()
    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    payload = {**_run_payload(), "model_settings": {"name": "openai/gpt-4o"}, "token_source": "byok"}
    resp = client.post("/run", json=payload)

    assert resp.status_code == 201
    overview = store._jobs[store.created_ids()[0]]["overview"]
    assert overview["token_source"] == "byok"


def test_submit_run_byok_without_connection_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BYOK run is refused at submit when the account saved no key for the provider."""
    engine = _billing_engine()
    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    payload = {**_run_payload(), "model_settings": {"name": "openai/gpt-4o"}, "token_source": "byok"}
    resp = client.post("/run", json=payload)

    assert resp.status_code == 400
    assert resp.json()["code"] == "billing.byok_missing_connection"
    assert store.created_ids() == []


def test_submit_run_managed_frontier_allowed_with_paid_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purchased balance unlocks the frontier catalog for a managed run."""
    engine = _billing_engine()
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="alice", stripe_customer_id="cus_alice", credit_balance=500
            )
        )
        session.commit()
    store = _FakeJobStore()
    store.engine = engine
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    payload = {**_run_payload(), "model_settings": {"name": "openai/gpt-4o"}, "token_source": "managed"}
    resp = client.post("/run", json=payload)

    assert resp.status_code == 201


def test_submit_run_defaults_token_source_to_managed_in_overview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted token_source defaults to managed and is persisted on the overview."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    resp = client.post("/run", json=_run_payload())

    assert resp.status_code == 201
    overview = store._jobs[store.created_ids()[0]]["overview"]
    assert overview["token_source"] == "managed"


def test_submit_run_idempotent_retry_returns_same_optimization_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reposting with the same ``Idempotency-Key`` returns the original id without re-enqueuing."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    headers = {"Idempotency-Key": "run-dedup-1"}

    first = client.post("/run", json=_run_payload(), headers=headers)
    second = client.post("/run", json=_run_payload(), headers=headers)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["optimization_id"] == second.json()["optimization_id"]
    assert len(store.created_ids()) == 1


def test_submit_run_without_idempotency_header_creates_separate_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two posts without an ``Idempotency-Key`` produce two distinct jobs."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)

    first = client.post("/run", json=_run_payload())
    second = client.post("/run", json=_run_payload())

    assert first.json()["optimization_id"] != second.json()["optimization_id"]
    assert len(store.created_ids()) == 2


def test_submit_run_blank_idempotency_header_does_not_dedupe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whitespace-only ``Idempotency-Key`` is treated as absent."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    headers = {"Idempotency-Key": "   "}

    first = client.post("/run", json=_run_payload(), headers=headers)
    second = client.post("/run", json=_run_payload(), headers=headers)

    assert first.json()["optimization_id"] != second.json()["optimization_id"]
    assert len(store.created_ids()) == 2


def test_submit_grid_search_idempotent_retry_returns_same_optimization_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grid-search route honours ``Idempotency-Key`` the same way ``/run`` does."""
    store = _FakeJobStore()
    client = _make_client(_FakeService(), store, monkeypatch=monkeypatch)
    headers = {"Idempotency-Key": "grid-dedup-1"}

    first = client.post("/grid-search", json=_grid_payload(), headers=headers)
    second = client.post("/grid-search", json=_grid_payload(), headers=headers)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["optimization_id"] == second.json()["optimization_id"]
    assert len(store.created_ids()) == 1
