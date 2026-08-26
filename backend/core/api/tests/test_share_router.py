"""Tests for the Google-Drive-style optimization sharing router.

Exercises the owner/editor-gated management surface (general-access policy,
member CRUD, role gating), the access-gated public composite read
(``GET /share/{token}``), and the editor+-only inference path
(``POST /share/{token}/serve``).

The store mirrors the in-memory SQLite pattern of the sibling routers: a
``RemoteDBJobStore`` subclass that skips the pgvector bootstrap and seeds
``JobModel`` rows directly. The serve test monkeypatches the program loader and
language-model builder on the ``share`` module so it never touches a real model.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...storage.models import Base, JobModel
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..routers.share import create_share_router


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for share-router tests (skips pgvector bootstrap)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _seed_job(store: _MemStore, optimization_id: str = "opt-share-1", username: str = "alice") -> None:
    """Insert a successful job owned by ``username`` carrying secrets to scrub.

    Args:
        store: The in-memory store to seed into.
        optimization_id: Optimization id for the seeded job.
        username: Owner username recorded on the job.
    """
    with Session(store.engine) as session:
        session.add(
            JobModel(
                optimization_id=optimization_id,
                status="success",
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                latest_metrics={},
                result=None,
                payload_overview={
                    "optimization_type": "run",
                    "name": "My run",
                    "username": username,
                    "optimizer_name": "gepa",
                    "module_name": "predict",
                },
                payload={
                    "username": username,
                    "signature_code": "class S(dspy.Signature): ...",
                    "metric_code": "def metric(gold, pred, trace=None): return 1.0",
                    "optimizer_name": "gepa",
                    "module_name": "predict",
                    "column_mapping": {"inputs": {"question": "question"}, "outputs": {"answer": "answer"}},
                    "split_fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
                    "shuffle": True,
                    "seed": 42,
                    "dataset": [{"question": f"q{i}", "answer": f"a{i}"} for i in range(40)],
                    "model_config": {
                        "name": "openai/gpt-5.4-nano",
                        "base_url": "https://secret.internal",
                        "extra": {"api_key": "sk-SECRET", "reasoning_effort": "medium"},
                    },
                    "reflection_model_config": {
                        "name": "openai/gpt-5.4-nano",
                        "extra": {"api_key": "sk-SECRET2"},
                    },
                },
                username=username,
            )
        )
        session.commit()


def _client(store: _MemStore, user: str | None = "alice") -> TestClient:
    """Build a TestClient over the share router, optionally authed as ``user``.

    Every route (management and public ``/share``) resolves the caller via
    ``get_authenticated_user``, so a single dependency override sets the
    identity. ``None`` leaves the client anonymous (no override, no bearer) — the
    login-gated routes then 401.

    Args:
        store: Job store wired into the router factory.
        user: Username to authenticate as, or ``None`` for an anonymous client.

    Returns:
        A ``TestClient`` over a minimal app mounting only the share router.
    """
    app = FastAPI()
    app.include_router(create_share_router(job_store=store))
    if user is not None:
        identity = AuthenticatedUser(username=user, role="user", groups=())
        app.dependency_overrides[get_authenticated_user] = lambda: identity
    return TestClient(app, raise_server_exceptions=False)






def test_put_invalid_general_access_400() -> None:
    """An unknown general-access value is rejected with 400."""
    store = _MemStore()
    _seed_job(store)
    owner = _client(store, user="alice")
    assert owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "public"}).status_code == 400


def test_get_sharing_non_owner_404() -> None:
    """A stranger cannot read the sharing config (existence is not leaked)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/opt-share-1/sharing").status_code == 404


def test_put_visibility_toggles_is_private() -> None:
    """The owner can flip explore-corpus visibility, and it round-trips through GET sharing."""
    store = _MemStore()
    _seed_job(store)
    owner = _client(store, user="alice")

    assert owner.get("/optimizations/opt-share-1/sharing").json()["is_private"] is True

    to_private = owner.put("/optimizations/opt-share-1/visibility", json={"is_private": True})
    assert to_private.status_code == 200
    assert to_private.json()["is_private"] is True
    assert owner.get("/optimizations/opt-share-1/sharing").json()["is_private"] is True

    back = owner.put("/optimizations/opt-share-1/visibility", json={"is_private": False})
    assert back.status_code == 200
    assert back.json()["is_private"] is False


def test_put_visibility_non_owner_404() -> None:
    """A stranger cannot change visibility (existence is not leaked)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    stranger = _client(store, user="bob")
    assert (
        stranger.put("/optimizations/opt-share-1/visibility", json={"is_private": True}).status_code
        == 404
    )


































def test_member_crud_add_patch_remove() -> None:
    """A member grant can be added, re-roled, and removed by the owner."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")

    added = owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "Dave", "role": "viewer"}
    )
    assert added.status_code == 200
    assert {"username": "dave", "role": "viewer"} in added.json()["members"]

    patched = owner.patch("/optimizations/opt-share-1/sharing/members/dave", json={"role": "editor"})
    assert patched.status_code == 200
    assert {"username": "dave", "role": "editor"} in patched.json()["members"]

    removed = owner.delete("/optimizations/opt-share-1/sharing/members/dave")
    assert removed.status_code == 200
    assert removed.json()["members"] == []


def test_member_add_invalid_role_400() -> None:
    """An invalid member role is rejected with 400."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    resp = owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "view"}
    )
    assert resp.status_code == 400


def test_member_patch_unknown_member_404() -> None:
    """Patching a non-existent member returns 404."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    assert owner.patch(
        "/optimizations/opt-share-1/sharing/members/ghost", json={"role": "editor"}
    ).status_code == 404


def test_member_management_gated_for_non_owner_404() -> None:
    """A stranger cannot add members (404 — owner existence is not leaked)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    stranger = _client(store, user="bob")
    assert stranger.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "viewer"}
    ).status_code == 404


def test_viewer_member_cannot_manage_sharing_404() -> None:
    """A viewer-tier member lacks manage access and 404s on member endpoints."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "carol", "role": "viewer"})

    viewer = _client(store, user="carol")
    assert viewer.get("/optimizations/opt-share-1/sharing").status_code == 404
    assert viewer.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "viewer"}
    ).status_code == 404


def test_editor_member_cannot_manage_sharing_404() -> None:
    """Management is owner-only: an editor-tier member 404s on the sharing endpoints."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "erin", "role": "editor"})

    editor = _client(store, user="erin")
    assert editor.get("/optimizations/opt-share-1/sharing").status_code == 404
    assert editor.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "viewer"}
    ).status_code == 404


def test_grant_owner_role_400() -> None:
    """Owner is no longer a grantable member tier — granting it is rejected."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    resp = owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "owner"}
    )
    assert resp.status_code == 400


def test_patch_member_to_owner_400() -> None:
    """A member can't be promoted to owner via PATCH — owner isn't a grant tier."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "viewer"})
    resp = owner.patch("/optimizations/opt-share-1/sharing/members/dave", json={"role": "owner"})
    assert resp.status_code == 400


def test_transfer_ownership_demotes_old_owner_and_promotes_member() -> None:
    """Transfer flips the owner, demotes the old owner to editor, drops the new owner's grant."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "viewer"})

    transferred = owner.post("/optimizations/opt-share-1/sharing/transfer", json={"username": "dave"})
    assert transferred.status_code == 200
    body = transferred.json()
    assert body["owner"] == "dave"
    assert {"username": "alice", "role": "editor"} in body["members"]
    assert all(m["username"] != "dave" for m in body["members"])

    # The owner flip took effect: dave manages now, alice (now an editor) does not.
    assert _client(store, user="dave").get("/optimizations/opt-share-1/sharing").status_code == 200
    assert _client(store, user="alice").get("/optimizations/opt-share-1/sharing").status_code == 404


def test_transfer_to_non_member_404() -> None:
    """Transferring to someone who isn't already a member is rejected (Drive parity)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    resp = owner.post("/optimizations/opt-share-1/sharing/transfer", json={"username": "stranger"})
    assert resp.status_code == 404


def test_transfer_to_current_owner_400() -> None:
    """Transferring to the current owner is a no-op error."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    resp = owner.post("/optimizations/opt-share-1/sharing/transfer", json={"username": "alice"})
    assert resp.status_code == 400


def test_transfer_by_non_owner_404() -> None:
    """A non-owner member cannot transfer ownership (404 — existence not leaked)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "dave", "role": "editor"})
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "erin", "role": "viewer"})

    resp = _client(store, user="dave").post(
        "/optimizations/opt-share-1/sharing/transfer", json={"username": "erin"}
    )
    assert resp.status_code == 404




def test_user_search_matches_prefix() -> None:
    """The username autocomplete returns distinct known usernames by prefix."""
    store = _MemStore()
    _seed_job(store, optimization_id="opt-a", username="alice")
    _seed_job(store, optimization_id="opt-al", username="albert")
    _seed_job(store, optimization_id="opt-b", username="bob")
    caller = _client(store, user="alice")

    resp = caller.get("/users/search", params={"q": "al"})
    assert resp.status_code == 200
    names = resp.json()["usernames"]
    assert "alice" in names
    assert "albert" in names
    assert "bob" not in names


def test_user_search_excludes_synthetic_local_accounts() -> None:
    """Synthetic ``.local`` test/load usernames never surface in the picker."""
    store = _MemStore()
    _seed_job(store, optimization_id="opt-real", username="analytics")
    _seed_job(store, optimization_id="opt-fake-1", username="analytics-1-1@s.local")
    _seed_job(store, optimization_id="opt-fake-2", username="probe@sampler.local")
    caller = _client(store, user="analytics")

    names = caller.get("/users/search", params={"q": "analytics"}).json()["usernames"]
    assert "analytics" in names
    assert "analytics-1-1@s.local" not in names

    # A prefix that matches only synthetic accounts returns an empty list.
    assert caller.get("/users/search", params={"q": "probe@"}).json()["usernames"] == []
















def test_public_view_of_public_optimization_is_readable_and_scrubbed() -> None:
    """Any caller can read a public (is_private=false) optimization, secrets stripped.

    Backs the Explore "public" tab: a non-owner with no grant gets the ``viewer``
    tier (read + clone) — owner shown for attribution, ``serve_info`` null (no
    inference), and the payload free of api_key / base_url / username.
    """
    store = _MemStore()
    _seed_job(store, username="alice")
    job_data = store.get_job("opt-share-1")
    overview = {**job_data["payload_overview"], "is_private": False}
    store.update_job("opt-share-1", payload_overview=overview)

    stranger = _client(store, user="bob")
    resp = stranger.get("/optimizations/opt-share-1/public")
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "viewer"
    assert body["owner"] == "alice"
    assert body["serve_info"] is None
    payload = body["payload"]
    assert "username" not in payload
    assert "base_url" not in payload["model_config"]
    assert "api_key" not in payload["model_config"]["extra"]


def test_public_view_of_private_optimization_404() -> None:
    """A private optimization is not publicly viewable."""
    store = _MemStore()
    _seed_job(store, username="alice")
    job_data = store.get_job("opt-share-1")
    overview = {**job_data["payload_overview"], "is_private": True}
    store.update_job("opt-share-1", payload_overview=overview)

    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/opt-share-1/public").status_code == 404


def test_public_view_of_internal_tagger_job_404() -> None:
    """Internal auto-tag jobs are not part of the public optimization corpus."""
    store = _MemStore()
    _seed_job(store, username="alice")
    job_data = store.get_job("opt-share-1")
    overview = {**job_data["payload_overview"], "optimization_type": "tagging_autotag"}
    store.update_job("opt-share-1", payload_overview=overview)

    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/opt-share-1/public").status_code == 404


def test_public_view_of_unknown_internal_job_type_404() -> None:
    """Explore accepts only run and grid-search optimization jobs."""
    store = _MemStore()
    _seed_job(store, username="alice")
    job_data = store.get_job("opt-share-1")
    overview = {**job_data["payload_overview"], "optimization_type": "future_internal_job"}
    store.update_job("opt-share-1", payload_overview=overview)

    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/opt-share-1/public").status_code == 404


def test_public_view_of_non_successful_optimization_404() -> None:
    """Explore exposes only successful optimizations, not in-progress or failed jobs."""
    store = _MemStore()
    _seed_job(store, username="alice")
    store.update_job("opt-share-1", status="failed")

    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/opt-share-1/public").status_code == 404


def test_public_view_unknown_optimization_404() -> None:
    """Reading a public view of an unknown id 404s."""
    store = _MemStore()
    stranger = _client(store, user="bob")
    assert stranger.get("/optimizations/does-not-exist/public").status_code == 404
