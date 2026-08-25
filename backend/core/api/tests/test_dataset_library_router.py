"""Tests for the personal dataset-library CRUD router.

Exercises the owner-scoped save/list/read/rename/delete surface against an
in-memory SQLite store (the sibling routers' pattern: a ``RemoteDBJobStore``
subclass that skips the pgvector bootstrap so ``Base.metadata.create_all``
stands up the ``datasets`` and ``dataset_blobs`` tables). Covers the three save
gates — per-file cap (413), content-hash dedupe, and the unified storage budget
(409) — plus cross-user isolation.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...storage.models import (
    Base,
    DatasetShareGrantModel,
    DatasetShareLinkModel,
    TaggingSessionModel,
    TaggingSessionShareGrantModel,
    TaggingSessionShareLinkModel,
)
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers import dataset_library as dataset_library_module
from ..routers.dataset_library import create_dataset_library_router
from ..sharing_access import LINK_GRANT_MARKER

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())

_ROWS = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]
_SCHEMA = {
    "column_order": ["q", "a"],
    "column_roles": {"q": "input", "a": "output"},
    "column_kinds": {"q": "text", "a": "text"},
}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for dataset-library tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _app_for(store: _MemStore, user: AuthenticatedUser) -> FastAPI:
    """Mount the library router on a store, authed as ``user``, with the error map.

    Args:
        store: Backing store the router reads and writes.
        user: Identity the auth dependency resolves to for every request.

    Returns:
        A FastAPI app whose ``DomainError``s render the production ``code`` envelope.
    """
    app = FastAPI()
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


def _make_client(user: AuthenticatedUser) -> tuple[TestClient, _MemStore]:
    """Build a test client whose library router is authed as ``user``.

    Args:
        user: Identity the auth dependency resolves to for every request.

    Returns:
        A ``(client, store)`` pair sharing one in-memory store.
    """
    store = _MemStore()
    return TestClient(_app_for(store, user)), store


def _save(client: TestClient, *, name: str = "Math", rows=_ROWS) -> dict:
    """Save a dataset and return the decoded JSON envelope.

    Args:
        client: Authenticated test client.
        name: Display name for the entry.
        rows: Dataset rows to save.

    Returns:
        The parsed ``SaveDatasetResponse`` body.
    """
    resp = client.post(
        "/datasets/library",
        json={"name": name, "source": "upload", "dataset": rows, "column_schema": _SCHEMA},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_save_then_read_round_trip() -> None:
    """A saved dataset lists, fetches, and returns its rows with saved columns."""
    client, _ = _make_client(_ALICE)
    saved = _save(client)
    assert saved["deduplicated"] is False
    dataset_id = saved["dataset"]["id"]
    assert saved["dataset"]["row_count"] == 2
    assert saved["dataset"]["column_count"] == 2
    assert saved["dataset"]["owner_username"] == "alice"
    assert saved["dataset"]["role"] == "owner"

    listing = client.get("/datasets/library").json()
    assert [d["id"] for d in listing["datasets"]] == [dataset_id]
    assert listing["usage"]["used_bytes"] > 0
    assert listing["usage"]["quota_bytes"] > 0

    meta = client.get(f"/datasets/library/{dataset_id}").json()
    assert meta["name"] == "Math"

    rows = client.get(f"/datasets/library/{dataset_id}/rows").json()
    assert rows["columns"] == ["q", "a"]
    assert rows["rows"] == _ROWS
    assert rows["row_count"] == 2


def test_rename_updates_name() -> None:
    """PATCH renames the entry and the new name is reflected on read."""
    client, _ = _make_client(_ALICE)
    dataset_id = _save(client)["dataset"]["id"]
    renamed = client.patch(f"/datasets/library/{dataset_id}", json={"name": "Arithmetic"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Arithmetic"
    assert client.get(f"/datasets/library/{dataset_id}").json()["name"] == "Arithmetic"


def test_identical_resave_dedupes() -> None:
    """Re-saving byte-identical rows returns the existing entry, not a copy."""
    client, _ = _make_client(_ALICE)
    first = _save(client)["dataset"]["id"]
    again = _save(client, name="Math copy")
    assert again["deduplicated"] is True
    assert again["dataset"]["id"] == first
    assert len(client.get("/datasets/library").json()["datasets"]) == 1


def test_per_file_cap_rejects_with_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file above the per-file compressed cap is rejected with 413."""
    monkeypatch.setattr(dataset_library_module.settings, "dataset_max_file_bytes", 1)
    client, _ = _make_client(_ALICE)
    resp = client.post(
        "/datasets/library",
        json={"name": "Big", "source": "upload", "dataset": _ROWS, "column_schema": {}},
    )
    assert resp.status_code == 413
    assert resp.json()["code"] == "dataset.library.too_large"


def test_quota_rejects_with_409(monkeypatch: pytest.MonkeyPatch) -> None:
    """A save that would exceed the unified storage budget is rejected with 409."""
    client, _ = _make_client(_ALICE)
    _save(client)
    monkeypatch.setattr(dataset_library_module.settings, "user_storage_quota_bytes", 1)
    resp = client.post(
        "/datasets/library",
        json={"name": "Second", "source": "upload", "dataset": [{"q": "9", "a": "9"}], "column_schema": {}},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "user.storage.quota_exceeded"


def test_delete_removes_entry_and_rows() -> None:
    """Deleting an entry makes both its metadata and rows 404 afterwards."""
    client, _ = _make_client(_ALICE)
    dataset_id = _save(client)["dataset"]["id"]
    deleted = client.delete(f"/datasets/library/{dataset_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/datasets/library/{dataset_id}").status_code == 404
    assert client.get(f"/datasets/library/{dataset_id}/rows").status_code == 404
    assert client.get("/datasets/library").json()["datasets"] == []


def test_owner_scoping_hides_other_users_dataset() -> None:
    """A non-owner gets 404 — never another user's entry — on a shared store."""
    client_a, store = _make_client(_ALICE)
    dataset_id = _save(client_a)["dataset"]["id"]

    client_b = TestClient(_app_for(store, _BOB))

    assert client_b.get(f"/datasets/library/{dataset_id}").status_code == 404
    assert client_b.get("/datasets/library").json()["datasets"] == []


def test_bulk_delete_removes_owned_entries() -> None:
    """Bulk-delete clears every owned id and leaves the library empty."""
    client, _ = _make_client(_ALICE)
    first = _save(client, name="One")["dataset"]["id"]
    second = _save(client, name="Two", rows=[{"q": "9", "a": "9"}])["dataset"]["id"]

    resp = client.post("/datasets/library/bulk-delete", json={"ids": [first, second]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(body["deleted"]) == sorted([first, second])
    assert body["skipped"] == []
    assert client.get("/datasets/library").json()["datasets"] == []


def test_bulk_delete_dedupes_and_skips_unknown() -> None:
    """Duplicate ids collapse to one delete; an unknown id is skipped, not fatal."""
    client, _ = _make_client(_ALICE)
    dataset_id = _save(client)["dataset"]["id"]

    resp = client.post(
        "/datasets/library/bulk-delete",
        json={"ids": [dataset_id, dataset_id, "ghost"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] == [dataset_id]
    assert body["skipped"] == [{"id": "ghost", "reason": "not_found"}]


def test_bulk_delete_skips_other_users_entries() -> None:
    """A caller cannot bulk-delete another user's dataset — it is skipped, not deleted."""
    client_a, store = _make_client(_ALICE)
    alice_id = _save(client_a)["dataset"]["id"]

    client_b = TestClient(_app_for(store, _BOB))
    resp = client_b.post("/datasets/library/bulk-delete", json={"ids": [alice_id]})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"deleted": [], "skipped": [{"id": alice_id, "reason": "not_found"}]}
    assert client_a.get(f"/datasets/library/{alice_id}").status_code == 200


def test_bulk_delete_empty_is_noop() -> None:
    """An empty id list deletes nothing and returns empty result lists."""
    client, _ = _make_client(_ALICE)
    resp = client.post("/datasets/library/bulk-delete", json={"ids": []})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"deleted": [], "skipped": []}


def _insert_session(
    store: _MemStore,
    *,
    session_id: str = "sess-1",
    owner: str = "alice",
    link: tuple[str, str] | None = None,
    grants: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Seed a tagger session (and optional sharing) directly into the store.

    Args:
        store: Backing store to write into.
        session_id: Id for the new session row.
        owner: Username the session is owned by.
        link: Optional ``(general_access, general_role)`` for an active share link.
        grants: Optional ``(grantee, role, created_by)`` triples for member grants.
    """
    with Session(store._engine) as db:
        db.add(
            TaggingSessionModel(
                id=session_id,
                username=owner,
                name="My tagging",
                phase="annotating",
                config={},
                columns=["q", "a"],
                data=_ROWS,
                annotations={},
                row_count=2,
                tagged_count=2,
            )
        )
        if link is not None:
            db.add(
                TaggingSessionShareLinkModel(
                    token="sesslink-tok",
                    session_id=session_id,
                    created_by=owner,
                    general_access=link[0],
                    general_role=link[1],
                )
            )
        for grantee, role, created_by in grants:
            db.add(
                TaggingSessionShareGrantModel(
                    session_id=session_id,
                    grantee_username=grantee,
                    role=role,
                    created_by=created_by,
                )
            )
        db.commit()


def _session_exists(store: _MemStore, session_id: str) -> bool:
    """Report whether a tagger session row is still present in the store."""
    with Session(store._engine) as db:
        return db.get(TaggingSessionModel, session_id) is not None


def _dataset_sharing(
    store: _MemStore, dataset_id: str
) -> tuple[tuple[str, str, str, str] | None, list[tuple[str, str, str]]]:
    """Snapshot a dataset's active link and grants for assertions.

    Returns:
        A ``(link, grants)`` pair — ``link`` is
        ``(token, general_access, general_role, created_by)`` or ``None``, and
        ``grants`` is a sorted list of ``(grantee, role, created_by)`` triples.
    """
    with Session(store._engine) as db:
        link_row = (
            db.query(DatasetShareLinkModel)
            .filter_by(dataset_id=dataset_id, revoked_at=None)
            .one_or_none()
        )
        link = (
            None
            if link_row is None
            else (link_row.token, link_row.general_access, link_row.general_role, link_row.created_by)
        )
        grants = sorted(
            (g.grantee_username, g.role, g.created_by)
            for g in db.query(DatasetShareGrantModel).filter_by(dataset_id=dataset_id).all()
        )
    return link, grants


def _move(client: TestClient, session_id: str, *, name: str = "Moved", rows=_ROWS):
    """POST the move-to-library request for a session and return the raw response."""
    return client.post(
        f"/datasets/library/from-tagging-session/{session_id}",
        json={"name": name, "dataset": rows, "column_schema": _SCHEMA},
    )


def test_move_session_creates_dataset_and_deletes_session() -> None:
    """Moving a session lands a tagger-sourced dataset and removes the session."""
    client, store = _make_client(_ALICE)
    _insert_session(store)

    resp = _move(client, "sess-1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deduplicated"] is False
    assert body["dataset"]["source"] == "tagger"
    assert body["dataset"]["owner_username"] == "alice"
    assert body["dataset"]["role"] == "owner"

    assert _session_exists(store, "sess-1") is False
    listing = client.get("/datasets/library").json()
    assert [d["id"] for d in listing["datasets"]] == [body["dataset"]["id"]]
    # The session is gone, so a repeat move of the same id 404s.
    assert _move(client, "sess-1").status_code == 404


def test_move_transfers_session_sharing_to_dataset() -> None:
    """The moved dataset inherits the session's link policy and every grant."""
    client, store = _make_client(_ALICE)
    _insert_session(
        store,
        link=("anyone", "editor"),
        grants=(
            ("carol", "editor", "alice"),
            ("dave", "viewer", LINK_GRANT_MARKER),
        ),
    )

    body = _move(client, "sess-1").json()
    link, grants = _dataset_sharing(store, body["dataset"]["id"])

    assert link is not None
    token, general_access, general_role, created_by = link
    assert (general_access, general_role) == ("anyone", "editor")
    assert created_by == "alice"
    # A fresh token — the dataset link is a new resource, not the session's.
    assert token != "sesslink-tok"
    assert grants == [
        ("carol", "editor", "alice"),
        ("dave", "viewer", LINK_GRANT_MARKER),
    ]


def test_move_dedupe_leaves_existing_sharing_and_still_deletes_session() -> None:
    """A byte-identical move dedupes without widening the existing entry's sharing."""
    client, store = _make_client(_ALICE)
    existing_id = _save(client)["dataset"]["id"]
    _insert_session(store, link=("anyone", "editor"), grants=(("carol", "editor", "alice"),))

    resp = _move(client, "sess-1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deduplicated"] is True
    assert body["dataset"]["id"] == existing_id

    # The pre-existing dataset had no sharing; the deduped move must not add any.
    link, grants = _dataset_sharing(store, existing_id)
    assert link is None
    assert grants == []
    # The session is still removed — the move never leaves a duplicate behind.
    assert _session_exists(store, "sess-1") is False


def test_move_requires_session_owner() -> None:
    """A non-owner cannot move a session: it 404s, and nothing is created or deleted."""
    _client_a, store = _make_client(_ALICE)
    _insert_session(store)

    client_b = TestClient(_app_for(store, _BOB))
    resp = client_b.post(
        "/datasets/library/from-tagging-session/sess-1",
        json={"name": "Steal", "dataset": _ROWS, "column_schema": _SCHEMA},
    )
    assert resp.status_code == 404
    assert _session_exists(store, "sess-1") is True
    assert client_b.get("/datasets/library").json()["datasets"] == []
