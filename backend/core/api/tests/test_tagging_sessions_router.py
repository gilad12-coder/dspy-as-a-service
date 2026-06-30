"""Tests for the tagger-session persistence router (CRUD + ownership).

Mounts the router on an in-memory SQLite store (the sibling routers' pattern: a
``RemoteDBJobStore`` subclass that skips the pgvector bootstrap so
``Base.metadata.create_all`` stands up the tables). Covers the save/restore
lifecycle — create, list, get, progress-autosave, rename/pin, delete — plus the
ownership guard that a caller cannot touch another user's session.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ...storage.models import Base
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers.tagging_sessions import create_tagging_session_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())

_SESSION_BODY = {
    "name": "Sentiment pass",
    "phase": "annotating",
    "config": {"mode": "binary", "inputColumns": ["text"], "question": "Positive?"},
    "columns": ["text"],
    "data": [
        {"id": "r1", "text": "great"},
        {"id": "r2", "text": "awful"},
        {"id": "r3", "text": "meh"},
    ],
    "annotations": {"r1": "yes"},
    "current_index": 1,
}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for tagger-session tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _client(user: AuthenticatedUser, store: _MemStore | None = None) -> tuple[TestClient, _MemStore]:
    """Mount the router on a store, authed as ``user``.

    Args:
        user: Identity the auth dependency resolves to for every request.
        store: Existing store to reuse (so two users can share one DB); a fresh
            one is created when omitted.

    Returns:
        A ``(client, store)`` pair sharing one in-memory store.
    """
    store = store or _MemStore()
    app = FastAPI()
    app.include_router(create_tagging_session_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: user

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return TestClient(app), store


def _create(client: TestClient) -> str:
    """Create a session via the API and return its new id.

    Args:
        client: The mounted test client.

    Returns:
        The created session's id.
    """
    resp = client.post("/tagging-sessions", json=_SESSION_BODY)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_create_returns_detail_with_derived_counts() -> None:
    """Create echoes the full payload and derives row_count / tagged_count."""
    client, _ = _client(_ALICE)
    resp = client.post("/tagging-sessions", json=_SESSION_BODY)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"]
    assert body["row_count"] == 3
    assert body["tagged_count"] == 1
    assert body["current_index"] == 1
    assert body["data"][0]["text"] == "great"
    assert body["pinned"] is False


def test_list_returns_summary_without_heavy_payload() -> None:
    """The list row carries summary fields only — not data/annotations/config."""
    client, _ = _client(_ALICE)
    _create(client)
    resp = client.get("/tagging-sessions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    (item,) = body["items"]
    assert item["name"] == "Sentiment pass"
    assert item["row_count"] == 3
    assert item["tagged_count"] == 1
    assert "data" not in item
    assert "annotations" not in item
    assert "config" not in item


def test_get_restores_full_state() -> None:
    """Get returns everything needed to resume annotating."""
    client, _ = _client(_ALICE)
    sid = _create(client)
    body = client.get(f"/tagging-sessions/{sid}").json()
    assert body["config"]["mode"] == "binary"
    assert body["columns"] == ["text"]
    assert len(body["data"]) == 3
    assert body["annotations"] == {"r1": "yes"}
    assert body["current_index"] == 1


def test_progress_autosave_updates_mutable_fields() -> None:
    """PUT advances annotations / cursor / phase and recomputes tagged_count."""
    client, _ = _client(_ALICE)
    sid = _create(client)
    resp = client.put(
        f"/tagging-sessions/{sid}",
        json={"annotations": {"r1": "yes", "r2": "no"}, "current_index": 2, "phase": "annotating"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["tagged_count"] == 2
    detail = client.get(f"/tagging-sessions/{sid}").json()
    assert detail["annotations"] == {"r1": "yes", "r2": "no"}
    assert detail["current_index"] == 2


def test_patch_renames_and_pins() -> None:
    """PATCH updates name and pinned; a no-field patch is a 422."""
    client, _ = _client(_ALICE)
    sid = _create(client)
    resp = client.patch(f"/tagging-sessions/{sid}", json={"name": "Renamed", "pinned": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Renamed"
    assert body["pinned"] is True
    empty = client.patch(f"/tagging-sessions/{sid}", json={})
    assert empty.status_code == 422
    assert empty.json()["code"] == "tagger.session.patch_requires_field"


def test_delete_then_get_is_404() -> None:
    """Delete returns the deleted-id body and the row is gone afterwards."""
    client, _ = _client(_ALICE)
    sid = _create(client)
    resp = client.delete(f"/tagging-sessions/{sid}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": sid, "deleted": True}
    missing = client.get(f"/tagging-sessions/{sid}")
    assert missing.status_code == 404
    assert missing.json()["code"] == "tagger.session.not_found"


def test_other_user_cannot_access_session() -> None:
    """Bob gets 403 on Alice's session for read, update, patch and delete."""
    alice_client, store = _client(_ALICE)
    sid = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)

    assert bob_client.get(f"/tagging-sessions/{sid}").status_code == 403
    assert (
        bob_client.put(
            f"/tagging-sessions/{sid}", json={"annotations": {}, "current_index": 0}
        ).status_code
        == 403
    )
    assert bob_client.patch(f"/tagging-sessions/{sid}", json={"pinned": True}).status_code == 403
    assert bob_client.delete(f"/tagging-sessions/{sid}").status_code == 403
    # Bob's own list never sees Alice's session.
    assert bob_client.get("/tagging-sessions").json() == {"items": [], "total": 0}


def test_list_orders_pinned_first() -> None:
    """Pinned sessions sort ahead of more-recently-updated unpinned ones."""
    client, _ = _client(_ALICE)
    first = _create(client)
    second = _create(client)
    client.patch(f"/tagging-sessions/{first}", json={"pinned": True})
    items = client.get("/tagging-sessions").json()["items"]
    assert next(i["id"] for i in items) == first
    assert {i["id"] for i in items} == {first, second}
