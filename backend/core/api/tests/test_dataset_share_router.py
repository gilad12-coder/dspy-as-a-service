"""Tests for dataset sharing — role-gated access, members, links, transfer.

Mounts the library and sharing routers on one in-memory SQLite store (the
``RemoteDBJobStore`` subclass that skips the pgvector bootstrap) and switches the
acting user per client, so a single store backs an owner and the people they
share with. Covers the effective-role gate (viewer reads/clones, editor edits,
owner manages), restricted named-member links, rejection of anyone-link access,
and ownership transfer.
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
from ..routers.dataset_library import create_dataset_library_router
from ..routers.dataset_share import create_dataset_share_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())
_CAROL = AuthenticatedUser(username="carol", role="user", groups=())
_ADMIN = AuthenticatedUser(username="root", role="admin", groups=())

_ROWS = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]
_NEW_ROWS = [{"q": "5+5", "a": "10"}]
_SCHEMA = {
    "column_order": ["q", "a"],
    "column_roles": {"q": "input", "a": "output"},
    "column_kinds": {"q": "text", "a": "text"},
}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for dataset-sharing tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _client(store: _MemStore, user: AuthenticatedUser) -> TestClient:
    """Build a client over the library + share routers, authed as ``user``.

    Args:
        store: Backing store the routers share.
        user: Identity the auth dependency resolves to for every request.

    Returns:
        A FastAPI test client whose ``DomainError``s render the production
        ``code`` envelope.
    """
    app = FastAPI()
    app.include_router(create_dataset_library_router(job_store=store))
    app.include_router(create_dataset_share_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: user

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return TestClient(app)


def _save(client: TestClient, *, name: str = "Math", rows=_ROWS) -> str:
    """Save a dataset and return its id.

    Args:
        client: Authenticated owner client.
        name: Display name for the entry.
        rows: Dataset rows to save.

    Returns:
        The id of the saved dataset.
    """
    resp = client.post(
        "/datasets/library",
        json={"name": name, "source": "upload", "dataset": rows, "column_schema": _SCHEMA},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset"]["id"]


def _add_member(owner: TestClient, dataset_id: str, username: str, role: str) -> None:
    """Invite ``username`` to a dataset at ``role`` via the owner client.

    Args:
        owner: Owner client managing the dataset's sharing.
        dataset_id: Dataset to share.
        username: Grantee to invite.
        role: Tier to grant (``viewer``/``editor``).
    """
    resp = owner.post(
        f"/datasets/library/{dataset_id}/sharing/members",
        json={"username": username, "role": role},
    )
    assert resp.status_code == 200, resp.text


def test_sharing_defaults_restricted_and_rejects_anyone_links() -> None:
    """Keep dataset sharing named-only and reject anyone-with-link access."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    dataset_id = _save(alice)

    initial = alice.get(f"/datasets/library/{dataset_id}/sharing").json()
    assert initial["general_access"] == "restricted"
    assert initial["token"] is None
    assert initial["owner"] == "alice"
    assert initial["members"] == []

    rejected = alice.put(
        f"/datasets/library/{dataset_id}/sharing",
        json={"general_access": "anyone", "general_role": "viewer"},
    )
    assert rejected.status_code == 400


def test_invited_viewer_can_read_and_clone_but_not_edit_or_manage() -> None:
    """A viewer reads metadata/rows and clones, but cannot edit, rename, or manage."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    dataset_id = _save(alice)
    _add_member(alice, dataset_id, "bob", "viewer")

    meta = bob.get(f"/datasets/library/{dataset_id}")
    assert meta.status_code == 200
    assert meta.json()["role"] == "viewer"
    assert meta.json()["owner_username"] == "alice"

    assert bob.get(f"/datasets/library/{dataset_id}/rows").status_code == 200

    cloned = bob.post(f"/datasets/library/{dataset_id}/clone")
    assert cloned.status_code == 200
    assert cloned.json()["dataset"]["owner_username"] == "bob"
    assert cloned.json()["dataset"]["role"] == "owner"

    assert bob.put(f"/datasets/library/{dataset_id}/rows", json={"rows": _NEW_ROWS}).status_code == 403
    assert bob.patch(f"/datasets/library/{dataset_id}", json={"name": "x"}).status_code == 403
    assert bob.delete(f"/datasets/library/{dataset_id}").status_code == 403
    assert bob.get(f"/datasets/library/{dataset_id}/sharing").status_code == 403


def test_editor_can_edit_rows_and_owner_sees_them() -> None:
    """An editor's row replacement re-points the live blob the owner reads."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    dataset_id = _save(alice)
    _add_member(alice, dataset_id, "bob", "editor")

    edited = bob.put(f"/datasets/library/{dataset_id}/rows", json={"rows": _NEW_ROWS})
    assert edited.status_code == 200, edited.text
    assert edited.json()["row_count"] == 1

    owner_view = alice.get(f"/datasets/library/{dataset_id}/rows").json()
    assert owner_view["rows"] == _NEW_ROWS


def test_shared_dataset_appears_in_grantee_library() -> None:
    """A grant lists the dataset in the grantee's library without costing quota."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    dataset_id = _save(alice)
    _add_member(alice, dataset_id, "bob", "viewer")

    listing = bob.get("/datasets/library").json()
    shared = [d for d in listing["datasets"] if d["id"] == dataset_id]
    assert len(shared) == 1
    assert shared[0]["role"] == "viewer"
    assert shared[0]["owner_username"] == "alice"
    assert listing["usage"]["used_bytes"] == 0


def test_cannot_grant_self() -> None:
    """Inviting the owner is rejected with the self-grant code."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    dataset_id = _save(alice)
    resp = alice.post(
        f"/datasets/library/{dataset_id}/sharing/members",
        json={"username": "alice", "role": "viewer"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "dataset.share.cannot_grant_self"


def test_update_then_remove_member_revokes_access() -> None:
    """Promoting then removing a member updates the roster and revokes access."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    dataset_id = _save(alice)
    _add_member(alice, dataset_id, "bob", "viewer")

    promoted = alice.patch(
        f"/datasets/library/{dataset_id}/sharing/members/bob", json={"role": "editor"}
    ).json()
    assert promoted["members"] == [{"username": "bob", "role": "editor"}]

    removed = alice.delete(f"/datasets/library/{dataset_id}/sharing/members/bob").json()
    assert removed["members"] == []
    assert bob.get(f"/datasets/library/{dataset_id}").status_code == 404


def test_member_not_found_and_modify_self_guards() -> None:
    """Targeting a non-member 404s; targeting one's own grant 400s."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    dataset_id = _save(alice)

    missing = alice.patch(
        f"/datasets/library/{dataset_id}/sharing/members/carol", json={"role": "editor"}
    )
    assert missing.status_code == 404
    assert missing.json()["code"] == "dataset.share.member_not_found"

    own = alice.patch(
        f"/datasets/library/{dataset_id}/sharing/members/alice", json={"role": "editor"}
    )
    assert own.status_code == 400
    assert own.json()["code"] == "dataset.share.cannot_modify_self"

    assert alice.delete(f"/datasets/library/{dataset_id}/sharing/members/alice").status_code == 400


def test_non_owner_cannot_manage_sharing() -> None:
    """A stranger 404s on sharing; a viewer 403s on every management route."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    carol = _client(store, _CAROL)
    dataset_id = _save(alice)

    assert carol.get(f"/datasets/library/{dataset_id}/sharing").status_code == 404

    _add_member(alice, dataset_id, "bob", "viewer")
    assert bob.get(f"/datasets/library/{dataset_id}/sharing").status_code == 403
    assert bob.put(
        f"/datasets/library/{dataset_id}/sharing", json={"general_access": "restricted"}
    ).status_code == 403
    assert bob.post(
        f"/datasets/library/{dataset_id}/sharing/members", json={"username": "carol", "role": "viewer"}
    ).status_code == 403


def test_transfer_ownership_moves_owner_and_demotes_previous() -> None:
    """Transfer makes the member the owner and demotes the old owner to editor."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    bob = _client(store, _BOB)
    dataset_id = _save(alice)
    _add_member(alice, dataset_id, "bob", "editor")

    state = alice.post(
        f"/datasets/library/{dataset_id}/sharing/transfer", json={"username": "bob"}
    ).json()
    assert state["owner"] == "bob"
    assert state["members"] == [{"username": "alice", "role": "editor"}]

    assert bob.get(f"/datasets/library/{dataset_id}").json()["role"] == "owner"
    assert alice.get(f"/datasets/library/{dataset_id}").json()["role"] == "editor"
    assert alice.delete(f"/datasets/library/{dataset_id}").status_code == 403
    assert bob.delete(f"/datasets/library/{dataset_id}").status_code == 200


def test_admin_resolves_to_owner_on_any_dataset() -> None:
    """An admin reaches and manages a dataset they did not create, as owner."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    admin = _client(store, _ADMIN)
    dataset_id = _save(alice)

    meta = admin.get(f"/datasets/library/{dataset_id}")
    assert meta.status_code == 200
    assert meta.json()["role"] == "owner"
    assert admin.get(f"/datasets/library/{dataset_id}/sharing").status_code == 200


def test_transfer_to_non_member_and_self_are_rejected() -> None:
    """Transferring to a non-member 404s; transferring to the owner 400s."""
    store = _MemStore()
    alice = _client(store, _ALICE)
    dataset_id = _save(alice)

    to_stranger = alice.post(
        f"/datasets/library/{dataset_id}/sharing/transfer", json={"username": "carol"}
    )
    assert to_stranger.status_code == 404
    assert to_stranger.json()["code"] == "dataset.share.member_not_found"

    to_self = alice.post(
        f"/datasets/library/{dataset_id}/sharing/transfer", json={"username": "alice"}
    )
    assert to_self.status_code == 400
    assert to_self.json()["code"] == "dataset.share.cannot_grant_self"
