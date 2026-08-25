"""Wiring tests for the public dashboard search router.

These tests stay DB-free: they monkeypatch the gateway ``search_optimizations``
with a kwargs-capturing spy and the auth resolver with a fixed identity, then
assert how ``POST /dashboard/search`` resolves and forwards the
``owner_username`` (mine) / ``shared_with_username`` (shared-with-me) scopes.
The gateway SQL itself is exercised against a live database elsewhere; here we
only pin the route's scope-resolution contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..auth import AuthenticatedUser
from ..routers import dashboard as dashboard_module
from ..routers.dashboard import create_dashboard_router

_EMPTY_RESULT: dict[str, Any] = {
    "results": [],
    "total": 0,
    "matched_ids": [],
    "search_type": None,
}


def _spy_search(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the gateway ``search_optimizations`` with a kwargs-capturing spy.

    Args:
        monkeypatch: Pytest patcher scoped to the test.

    Returns:
        A dict the spy fills with the forwarded keyword arguments on call.
    """
    captured: dict[str, Any] = {}

    def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return dict(_EMPTY_RESULT)

    monkeypatch.setattr(dashboard_module, "search_optimizations", _fake)
    return captured


def _spy_facets(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the gateway ``fetch_corpus_facets`` with a kwargs-capturing spy.

    Args:
        monkeypatch: Pytest patcher scoped to the test.

    Returns:
        A dict the spy fills with the forwarded keyword arguments on call.
    """
    captured: dict[str, Any] = {}

    def _fake(**kwargs: Any) -> dict[str, list[str]]:
        captured.update(kwargs)
        return {"models": [], "optimizers": [], "modules": []}

    monkeypatch.setattr(dashboard_module, "fetch_corpus_facets", _fake)
    return captured


def _client(monkeypatch: pytest.MonkeyPatch, user: str | None = "adi") -> TestClient:
    """Build a TestClient over the dashboard router with a fixed auth identity.

    Args:
        monkeypatch: Pytest patcher scoped to the test.
        user: Username the auth dependency returns, or ``None`` to leave it
            unauthenticated.

    Returns:
        A ``TestClient`` over a minimal app mounting only the dashboard router.
    """
    app = FastAPI()
    app.include_router(create_dashboard_router(job_store=object()))
    if user is not None:
        identity = AuthenticatedUser(username=user, role="user", groups=())
        app.dependency_overrides[dashboard_module.get_authenticated_user] = lambda: identity
    return TestClient(app, raise_server_exceptions=False)


def test_shared_scope_is_resolved_and_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shared_with_username`` authed as that user forwards it, owner None."""
    captured = _spy_search(monkeypatch)
    client = _client(monkeypatch, user="adi")
    resp = client.post("/dashboard/search", json={"shared_with_username": "Adi"})
    assert resp.status_code == 200
    assert captured["shared_with_username"] == "adi"
    assert captured["owner_username"] is None


def test_owner_scope_takes_precedence_over_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both scopes are sent, owner wins and shared resolves to None."""
    captured = _spy_search(monkeypatch)
    client = _client(monkeypatch, user="adi")
    resp = client.post(
        "/dashboard/search",
        json={"owner_username": "adi", "shared_with_username": "adi"},
    )
    assert resp.status_code == 200
    assert captured["owner_username"] == "adi"
    assert captured["shared_with_username"] is None


def test_public_search_forwards_no_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """An authenticated bare query forwards both scopes as None."""
    captured = _spy_search(monkeypatch)
    client = _client(monkeypatch)
    resp = client.post("/dashboard/search", json={"query": "mipro"})
    assert resp.status_code == 200
    assert captured["owner_username"] is None
    assert captured["shared_with_username"] is None


def test_structured_filters_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every categorical filter, including tasks/modules, reaches the gateway."""
    captured = _spy_search(monkeypatch)
    client = _client(monkeypatch)
    resp = client.post(
        "/dashboard/search",
        json={
            "models": ["openai/gpt-4o"],
            "optimizers": ["MIPROv2"],
            "optimization_types": ["run"],
            "tasks": ["sentiment"],
            "modules": ["Classify"],
        },
    )
    assert resp.status_code == 200
    assert captured["tasks"] == ["sentiment"]
    assert captured["modules"] == ["Classify"]


def test_shared_scope_mismatch_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting another user's shared corpus is rejected with 403."""
    _spy_search(monkeypatch)
    client = _client(monkeypatch, user="adi")
    resp = client.post(
        "/dashboard/search", json={"shared_with_username": "someone-else"}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "auth.owner_mismatch"


def test_facets_public_forwards_no_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticated ``GET /dashboard/facets`` forwards no scope."""
    captured = _spy_facets(monkeypatch)
    client = _client(monkeypatch)
    resp = client.get("/dashboard/facets")
    assert resp.status_code == 200
    assert resp.json() == {"models": [], "optimizers": [], "modules": []}
    assert captured["owner_username"] is None
    assert captured["shared_with_username"] is None


def test_facets_owner_scope_is_resolved_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``owner_username`` authed as that user forwards it, shared None."""
    captured = _spy_facets(monkeypatch)
    client = _client(monkeypatch, user="adi")
    resp = client.get("/dashboard/facets", params={"owner_username": "Adi"})
    assert resp.status_code == 200
    assert captured["owner_username"] == "adi"
    assert captured["shared_with_username"] is None


def test_facets_scope_mismatch_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting another user's facet scope is rejected with 403."""
    _spy_facets(monkeypatch)
    client = _client(monkeypatch, user="adi")
    resp = client.get(
        "/dashboard/facets", params={"shared_with_username": "someone-else"}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "auth.owner_mismatch"
