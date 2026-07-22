"""Tests for tagger-session sharing (grants, link claim, role enforcement).

Mounts the session CRUD router together with the sharing router on one
in-memory SQLite store (the sibling routers' pattern) and exercises the
Drive-style flows end-to-end: invited viewers read but cannot edit, editors
annotate but cannot manage, an ``anyone`` link claim lists the session in the
claimer's chooser and restricting the link revokes it, and ownership transfer
moves the ``username`` column while demoting the previous owner.
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
from ..routers.tagging_session_share import create_tagging_session_share_router
from ..routers.tagging_sessions import create_tagging_session_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())

_SESSION_BODY = {
    "name": "Sentiment pass",
    "phase": "annotating",
    "config": {"mode": "binary", "inputColumns": ["text"], "question": "Positive?"},
    "columns": ["text"],
    "data": [{"id": "r1", "text": "great"}, {"id": "r2", "text": "awful"}],
    "annotations": {"r1": "yes"},
    "current_index": 1,
}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for session-sharing tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _client(
    user: AuthenticatedUser, store: _MemStore | None = None
) -> tuple[TestClient, _MemStore]:
    """Mount the session CRUD + sharing routers on a store, authed as ``user``.

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
    app.include_router(create_tagging_session_share_router(job_store=store))
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
    """Create a session and return its id."""
    resp = client.post("/tagging-sessions", json=_SESSION_BODY)
    assert resp.status_code == 201
    return resp.json()["id"]


def test_invited_viewer_can_read_but_not_edit_or_manage() -> None:
    """A viewer grant opens the session read-only and hides management."""
    alice_client, store = _client(_ALICE)
    sid = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)

    resp = alice_client.post(
        f"/tagging-sessions/{sid}/sharing/members", json={"username": "bob", "role": "viewer"}
    )
    assert resp.status_code == 200
    assert resp.json()["members"] == [{"username": "bob", "role": "viewer"}]

    detail = bob_client.get(f"/tagging-sessions/{sid}")
    assert detail.status_code == 200
    assert detail.json()["role"] == "viewer"
    assert (
        bob_client.put(
            f"/tagging-sessions/{sid}", json={"annotations": {}, "current_index": 0}
        ).status_code
        == 403
    )
    assert bob_client.patch(f"/tagging-sessions/{sid}", json={"pinned": True}).status_code == 403
    assert bob_client.delete(f"/tagging-sessions/{sid}").status_code == 403
    assert bob_client.get(f"/tagging-sessions/{sid}/sharing").status_code == 403

    listed = bob_client.get("/tagging-sessions").json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == sid
    assert listed["items"][0]["role"] == "viewer"


def test_invited_editor_can_autosave_but_not_manage() -> None:
    """An editor grant allows progress writes but not rename/delete."""
    alice_client, store = _client(_ALICE)
    sid = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)

    alice_client.post(
        f"/tagging-sessions/{sid}/sharing/members", json={"username": "bob", "role": "editor"}
    )
    resp = bob_client.put(
        f"/tagging-sessions/{sid}",
        json={"annotations": {"r1": "yes", "r2": "no"}, "current_index": 2},
    )
    assert resp.status_code == 200
    assert resp.json()["tagged_count"] == 2
    assert resp.json()["role"] == "editor"
    assert bob_client.patch(f"/tagging-sessions/{sid}", json={"name": "x"}).status_code == 403
    assert bob_client.delete(f"/tagging-sessions/{sid}").status_code == 403
    # Editors cannot bulk-delete a shared-in session either.
    bulk = alice_client.post("/tagging-sessions/bulk-delete", json={"ids": [sid]})
    assert bulk.json()["deleted"] == [sid]


def test_anyone_link_claim_lists_session_and_restrict_revokes() -> None:
    """Claiming an ``anyone`` link grants its tier; restricting prunes it."""
    alice_client, store = _client(_ALICE)
    sid = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)

    state = alice_client.put(
        f"/tagging-sessions/{sid}/sharing",
        json={"general_access": "anyone", "general_role": "editor"},
    ).json()
    token = state["token"]
    assert state["share_path"] == f"/tagger/share/{token}"

    claim = bob_client.post(f"/tagging-sessions/share/{token}/claim")
    assert claim.status_code == 200
    assert claim.json() == {"session_id": sid, "role": "editor"}
    assert bob_client.get(f"/tagging-sessions/{sid}").json()["role"] == "editor"
    assert bob_client.get("/tagging-sessions").json()["total"] == 1

    # Link-derived memberships are invisible to the owner's member list.
    assert alice_client.get(f"/tagging-sessions/{sid}/sharing").json()["members"] == []

    alice_client.put(f"/tagging-sessions/{sid}/sharing", json={"general_access": "restricted"})
    assert bob_client.get(f"/tagging-sessions/{sid}").status_code == 404
    assert bob_client.post(f"/tagging-sessions/share/{token}/claim").status_code == 404
    assert bob_client.get("/tagging-sessions").json()["total"] == 0


def test_transfer_ownership_moves_owner_and_demotes_previous() -> None:
    """Transfer hands the ``username`` column over and demotes the old owner."""
    alice_client, store = _client(_ALICE)
    sid = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)

    alice_client.post(
        f"/tagging-sessions/{sid}/sharing/members", json={"username": "bob", "role": "editor"}
    )
    state = alice_client.post(
        f"/tagging-sessions/{sid}/sharing/transfer", json={"username": "bob"}
    ).json()
    assert state["owner"] == "bob"
    assert state["members"] == [{"username": "alice", "role": "editor"}]

    assert bob_client.get(f"/tagging-sessions/{sid}").json()["role"] == "owner"
    assert bob_client.patch(f"/tagging-sessions/{sid}", json={"pinned": True}).status_code == 200
    assert alice_client.patch(f"/tagging-sessions/{sid}", json={"pinned": False}).status_code == 403
    assert (
        alice_client.put(
            f"/tagging-sessions/{sid}", json={"annotations": {}, "current_index": 0}
        ).status_code
        == 200
    )
