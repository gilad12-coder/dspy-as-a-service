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

from contextlib import nullcontext
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...models import RunResponse, ServeInfoResponse
from ...storage.models import Base, JobModel
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..routers import share as share_module
from ..routers.share import create_share_router
from ..sharing_access import get_grant
from .mocks import make_artifact, make_run_result


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


def _enable_anyone(
    store: _MemStore,
    optimization_id: str = "opt-share-1",
    user: str = "alice",
    role: str | None = None,
) -> str:
    """Set ``general_access`` to ``anyone`` and return the minted token.

    Args:
        store: The seeded store.
        optimization_id: Optimization to switch to an anyone-link.
        user: Owner/editor making the change.
        role: Optional link tier (``viewer``/``editor``) to grant signed-in
            visitors; ``None`` leaves the default (``viewer``).

    Returns:
        The active share token.
    """
    owner = _client(store, user=user)
    body: dict[str, str] = {"general_access": "anyone"}
    if role is not None:
        body["general_role"] = role
    resp = owner.put(f"/optimizations/{optimization_id}/sharing", json=body)
    assert resp.status_code == 200
    return resp.json()["token"]


def test_put_general_access_toggles_restricted_and_anyone() -> None:
    """The owner can flip general access between restricted and anyone, keeping the token."""
    store = _MemStore()
    _seed_job(store)
    owner = _client(store, user="alice")

    to_anyone = owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "anyone"})
    assert to_anyone.status_code == 200
    body = to_anyone.json()
    assert body["general_access"] == "anyone"
    token = body["token"]
    assert token
    assert body["share_path"] == f"/share/{token}"
    assert body["owner"] == "alice"

    back = owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "restricted"})
    assert back.status_code == 200
    assert back.json()["general_access"] == "restricted"
    assert back.json()["token"] == token


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

    assert owner.get("/optimizations/opt-share-1/sharing").json()["is_private"] is False

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


def test_anonymous_get_is_unauthorized() -> None:
    """An anonymous caller cannot read a shared run: the app is login-gated (401)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    public = _client(store, user=None)
    assert public.get(f"/share/{token}").status_code == 401


def test_restricted_link_404_for_non_member() -> None:
    """A restricted link grants nothing: a signed-in non-member 404s (no anyone fallback)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    # A token exists but general access stays restricted.
    owner = _client(store, user="alice")
    token = owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "restricted"}).json()["token"]

    stranger = _client(store, user="bob")
    assert stranger.get(f"/share/{token}").status_code == 404


def _serve_info_stub() -> ServeInfoResponse:
    """Return a canned ``ServeInfoResponse`` for share-view serve_info tests."""
    return ServeInfoResponse(
        optimization_id="opt-share-1",
        module_name="predict",
        optimizer_name="gepa",
        model_name="openai/gpt-5.4-nano",
        input_fields=["question"],
        output_fields=["answer"],
        instructions="Be helpful.",
        demo_count=0,
    )


def test_member_viewer_sees_owner_but_not_serve_info() -> None:
    """A viewer member sees the real owner, but serve_info stays null (serve is editor+)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    owner = _client(store, user="alice")
    assert owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "carol", "role": "viewer"}
    ).status_code == 200

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(share_module, "_serve_info", lambda *_args, **_kw: _serve_info_stub())
        viewer = _client(store, user="carol")
        resp = viewer.get(f"/share/{token}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "viewer"
    assert body["owner"] == "alice"
    assert body["serve_info"] is None
    assert body["status"]["username"] == "alice"


def test_member_editor_sees_owner_and_serve_info() -> None:
    """An editor member sees the real owner and serve_info (editor+ may serve)."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    owner = _client(store, user="alice")
    assert owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "erin", "role": "editor"}
    ).status_code == 200

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(share_module, "_serve_info", lambda *_args, **_kw: _serve_info_stub())
        editor = _client(store, user="erin")
        resp = editor.get(f"/share/{token}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "editor"
    assert body["owner"] == "alice"
    assert body["serve_info"]["input_fields"] == ["question"]
    assert body["status"]["username"] == "alice"


def test_claim_anyone_link_persists_grant_and_makes_run_visible() -> None:
    """Redeeming an anyone-link durably grants the link tier and surfaces the run.

    The transient anyone-link tier becomes a stored member grant, so the run
    then shows up in the redeemer's unified table (``list_jobs_visible_to``) —
    Google-Drive "open the link, it lands in your account" semantics.
    """
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    dan = _client(store, user="dan")
    resp = dan.post(f"/share/{token}/claim")
    assert resp.status_code == 200
    body = resp.json()
    assert body["optimization_id"] == "opt-share-1"
    assert body["role"] == "editor"

    with Session(store.engine) as session:
        grant = get_grant(session, "opt-share-1", "dan")
    assert grant is not None
    assert grant.role == "editor"

    visible = {row["optimization_id"] for row in store.list_jobs_visible_to("dan")}
    assert "opt-share-1" in visible


def test_claim_restricted_link_non_member_404_no_grant() -> None:
    """Redeeming a restricted link as a stranger 404s and persists no grant."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    token = owner.put(
        "/optimizations/opt-share-1/sharing", json={"general_access": "restricted"}
    ).json()["token"]

    stranger = _client(store, user="mallory")
    assert stranger.post(f"/share/{token}/claim").status_code == 404

    with Session(store.engine) as session:
        assert get_grant(session, "opt-share-1", "mallory") is None


def test_claim_does_not_downgrade_existing_higher_grant() -> None:
    """A viewer-tier link never lowers an existing editor grant on redeem."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="viewer")

    owner = _client(store, user="alice")
    assert owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "frank", "role": "editor"}
    ).status_code == 200

    frank = _client(store, user="frank")
    resp = frank.post(f"/share/{token}/claim")
    assert resp.status_code == 200
    assert resp.json()["role"] == "editor"

    with Session(store.engine) as session:
        grant = get_grant(session, "opt-share-1", "frank")
    assert grant is not None
    assert grant.role == "editor"


def test_claim_owner_redeems_without_self_grant() -> None:
    """The owner redeeming their own link gets owner access and no grant row."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    owner = _client(store, user="alice")
    resp = owner.post(f"/share/{token}/claim")
    assert resp.status_code == 200
    assert resp.json()["role"] == "owner"

    with Session(store.engine) as session:
        assert get_grant(session, "opt-share-1", "alice") is None


def test_claim_unauthenticated_401() -> None:
    """Redeeming without a bearer is rejected — there is no anonymous claim."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    public = _client(store, user=None)
    assert public.post(f"/share/{token}/claim").status_code == 401


def test_link_member_role_tracks_link_downgrade() -> None:
    """Flipping the link editor→viewer downgrades an existing link member live."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    dan = _client(store, user="dan")
    assert dan.post(f"/share/{token}/claim").json()["role"] == "editor"

    owner = _client(store, user="alice")
    assert owner.put(
        "/optimizations/opt-share-1/sharing",
        json={"general_access": "anyone", "general_role": "viewer"},
    ).status_code == 200

    with Session(store.engine) as session:
        grant = get_grant(session, "opt-share-1", "dan")
    assert grant is not None
    assert grant.role == "viewer"
    visible = {row["optimization_id"] for row in store.list_jobs_visible_to("dan")}
    assert "opt-share-1" in visible


def test_link_member_removed_when_link_restricted() -> None:
    """Restricting the link revokes a link member and drops the run from their table."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    dan = _client(store, user="dan")
    assert dan.post(f"/share/{token}/claim").status_code == 200

    owner = _client(store, user="alice")
    assert owner.put(
        "/optimizations/opt-share-1/sharing", json={"general_access": "restricted"}
    ).status_code == 200

    with Session(store.engine) as session:
        assert get_grant(session, "opt-share-1", "dan") is None
    visible = {row["optimization_id"] for row in store.list_jobs_visible_to("dan")}
    assert "opt-share-1" not in visible
    # The link itself still 404s for the ex-member (restricted = invite-only).
    assert dan.get(f"/share/{token}").status_code == 404


def test_invited_member_unaffected_by_link_changes() -> None:
    """A named invite is authoritative: link role changes/restrict never touch it."""
    store = _MemStore()
    _seed_job(store, username="alice")
    _enable_anyone(store, role="editor")

    owner = _client(store, user="alice")
    assert owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "carol", "role": "viewer"}
    ).status_code == 200
    # Downgrade then restrict the link — carol's named viewer grant must persist.
    owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "anyone", "general_role": "editor"})
    owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "restricted"})

    with Session(store.engine) as session:
        grant = get_grant(session, "opt-share-1", "carol")
    assert grant is not None
    assert grant.role == "viewer"


def test_link_member_excluded_from_members_list() -> None:
    """Link members are covered by the link row, not shown in the people list."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "erin", "role": "editor"})
    _client(store, user="dan").post(f"/share/{token}/claim")

    members = {m["username"] for m in owner.get("/optimizations/opt-share-1/sharing").json()["members"]}
    assert "erin" in members
    assert "dan" not in members


def test_named_invite_supersedes_link_membership() -> None:
    """Inviting a link member by name promotes them to an authoritative invite."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    dan = _client(store, user="dan")
    dan.post(f"/share/{token}/claim")

    owner = _client(store, user="alice")
    assert owner.post(
        "/optimizations/opt-share-1/sharing/members", json={"username": "dan", "role": "editor"}
    ).status_code == 200
    # Restricting the link must NOT drop dan now — his access is a named invite.
    owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "restricted"})

    with Session(store.engine) as session:
        grant = get_grant(session, "opt-share-1", "dan")
    assert grant is not None
    assert grant.role == "editor"
    assert grant.created_by == "alice"
    members = {m["username"] for m in owner.get("/optimizations/opt-share-1/sharing").json()["members"]}
    assert "dan" in members


def test_public_view_payload_is_scrubbed() -> None:
    """The public payload strips username, raw dataset, api_key and base_url."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    reader = _client(store, user="bob")
    body = reader.get(f"/share/{token}").json()
    payload = body["payload"]

    assert payload["signature_code"].startswith("class S")
    assert "username" not in payload
    assert "dataset" not in payload
    assert "base_url" not in payload["model_config"]
    assert "api_key" not in payload["model_config"]["extra"]
    assert payload["model_config"]["extra"]["reasoning_effort"] == "medium"
    assert "api_key" not in payload["reflection_model_config"]["extra"]


def test_public_view_dataset_is_full_not_capped() -> None:
    """The shared dataset returns the FULL split (all 40 rows), not a preview cap."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    reader = _client(store, user="bob")
    body = reader.get(f"/share/{token}").json()
    dataset = body["dataset"]

    assert dataset["total_rows"] == 40
    counts = dataset["split_counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 40
    rows = sum(len(dataset["splits"][s]) for s in ("train", "val", "test"))
    assert rows == 40


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


def test_unknown_token_404() -> None:
    """An unknown share token returns 404 (for a signed-in caller; anonymous 401s)."""
    store = _MemStore()
    reader = _client(store, user="alice")
    assert reader.get("/share/does-not-exist").status_code == 404


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


class _FakePrediction:
    """Stub prediction exposing attribute access for the declared output field."""

    answer: str = "shared-answer"


def _seed_serveable(store: _MemStore, optimization_id: str = "opt-share-1") -> tuple[object, object, dict]:
    """Return the ``(program, RunResponse, overview)`` triple a fake loader yields.

    Builds a single-run result whose artifact declares one input and one output
    field so the serve handler's input validation and output projection both
    exercise the happy path without a real model.

    Args:
        store: The seeded store (unused beyond signature symmetry with loaders).
        optimization_id: Optimization id embedded in the synthesized result.

    Returns:
        A ``(program, RunResponse, overview)`` tuple matching ``load_program``.
    """
    artifact = make_artifact(input_fields=["question"], output_fields=["answer"])
    result: RunResponse = make_run_result(artifact)
    overview = {
        "module_name": "predict",
        "optimizer_name": "gepa",
        "model_name": "openai/gpt-5.4-nano",
    }
    program = lambda **_inputs: _FakePrediction()  # noqa: E731 — terse stub program
    return program, result, overview


def test_serve_allowed_for_editor_member() -> None:
    """An editor member can run inference; the owner key is used server-side."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "erin", "role": "editor"})

    program, result, overview = _seed_serveable(store)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(share_module, "load_program", lambda *_a, **_k: (program, result, overview))
        mp.setattr(share_module, "build_language_model", lambda _cfg: object())
        mp.setattr(share_module.dspy, "context", lambda **_kw: nullcontext())
        editor = _client(store, user="erin")
        resp = editor.post(f"/share/{token}/serve", json={"inputs": {"question": "hi"}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["optimization_id"] == "opt-share-1"
    assert body["outputs"]["answer"] == "shared-answer"
    assert body["input_fields"] == ["question"]
    # The owner's secret key/base_url must never surface in the serve response.
    assert "sk-SECRET" not in resp.text
    assert "secret.internal" not in resp.text


def test_serve_forbidden_for_viewer_member() -> None:
    """A viewer member is forbidden from inference (403) — serving is editor+."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)
    owner = _client(store, user="alice")
    owner.post("/optimizations/opt-share-1/sharing/members", json={"username": "carol", "role": "viewer"})

    viewer = _client(store, user="carol")
    resp = viewer.post(f"/share/{token}/serve", json={"inputs": {"question": "hi"}})
    assert resp.status_code == 403


def test_serve_unauthorized_for_anonymous() -> None:
    """Anonymous inference is rejected at the login gate (401), before any role check."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store)

    public = _client(store, user=None)
    resp = public.post(f"/share/{token}/serve", json={"inputs": {"question": "hi"}})
    assert resp.status_code == 401


def test_put_general_role_sets_link_tier() -> None:
    """The owner can set the anyone-link tier (viewer/editor), echoed in the state."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")

    resp = owner.put(
        "/optimizations/opt-share-1/sharing",
        json={"general_access": "anyone", "general_role": "editor"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["general_access"] == "anyone"
    assert body["general_role"] == "editor"

    # The tier persists even when flipping back to restricted.
    back = owner.put("/optimizations/opt-share-1/sharing", json={"general_access": "restricted"})
    assert back.status_code == 200
    assert back.json()["general_role"] == "editor"


def test_put_invalid_general_role_400() -> None:
    """A general_role outside {viewer, editor} (e.g. owner) is rejected with 400."""
    store = _MemStore()
    _seed_job(store, username="alice")
    owner = _client(store, user="alice")
    resp = owner.put(
        "/optimizations/opt-share-1/sharing",
        json={"general_access": "anyone", "general_role": "owner"},
    )
    assert resp.status_code == 400


def test_signed_in_stranger_gets_editor_link_role_and_can_serve() -> None:
    """An ``anyone -> editor`` link makes a signed-in non-member an editor who can serve."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="editor")

    stranger = _client(store, user="bob")
    read = stranger.get(f"/share/{token}")
    assert read.status_code == 200
    assert read.json()["role"] == "editor"
    assert read.json()["owner"] == "alice"

    program, result, overview = _seed_serveable(store)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(share_module, "load_program", lambda *_a, **_k: (program, result, overview))
        mp.setattr(share_module, "build_language_model", lambda _cfg: object())
        mp.setattr(share_module.dspy, "context", lambda **_kw: nullcontext())
        served = stranger.post(f"/share/{token}/serve", json={"inputs": {"question": "hi"}})
    assert served.status_code == 200


def test_signed_in_stranger_viewer_link_cannot_serve() -> None:
    """An ``anyone -> viewer`` link gives a signed-in stranger viewer: read yes, serve no."""
    store = _MemStore()
    _seed_job(store, username="alice")
    token = _enable_anyone(store, role="viewer")

    stranger = _client(store, user="bob")
    read = stranger.get(f"/share/{token}")
    assert read.status_code == 200
    assert read.json()["role"] == "viewer"

    served = stranger.post(f"/share/{token}/serve", json={"inputs": {"question": "hi"}})
    assert served.status_code == 403


def test_public_view_of_public_optimization_is_readable_and_scrubbed() -> None:
    """Any caller can read a public (is_private=false) optimization, secrets stripped.

    Backs the Explore "public" tab: a non-owner with no grant gets the ``viewer``
    tier (read + clone) — owner shown for attribution, ``serve_info`` null (no
    inference), and the payload free of api_key / base_url / username.
    """
    store = _MemStore()
    _seed_job(store, username="alice")

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
