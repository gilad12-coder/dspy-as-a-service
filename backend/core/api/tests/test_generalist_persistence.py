"""Regression coverage for generalist-agent conversation persistence.

The SSE endpoint persists each turn by resolving the caller via a *direct*
call to :func:`get_authenticated_user` (not a FastAPI ``Depends``, so the
test-suite auth override never reaches it). When the PAT feature added a
required ``request`` positional to that function, the call site still passed
``authorization`` only — raising a ``TypeError`` that the endpoint's
``except Exception: username = None`` swallowed, silently dropping every
conversation. These tests exercise the real auth + persistence path end to
end so that regression can't return unnoticed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import AgentConversationModel, AgentMessageModel, Base
from .. import auth as auth_mod
from .. import model_catalog
from ..routers import generalist_agent as agent_mod
from ..routers.generalist_agent import create_generalist_agent_router

_SECRET = "test-secret"


def _sign(payload: dict[str, Any], secret: str = _SECRET) -> str:
    """Sign a compact HS256 JWT matching the frontend's session token shape.

    Args:
        payload: JWT claims to encode.
        secret: Shared HMAC secret.

    Returns:
        A compact ``header.body.signature`` JWT string.
    """
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8")).decode("ascii").rstrip("=")
    )
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return f"{header}.{body}.{signature}"


def _session_token(name: str = "alice@example.com") -> str:
    """Mint a valid backend session token for ``name``."""
    now = int(time.time())
    return _sign(
        {
            "aud": "skynet-backend",
            "iss": "skynet-frontend",
            "sub": name,
            "name": name,
            "role": "user",
            "iat": now,
            "exp": now + 300,
        }
    )


async def _fake_stream(**_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
    """Stand in for ``run_generalist_agent`` with a minimal one-reply turn."""
    yield {"event": "message_patch", "data": {"chunk": "שלום"}}
    yield {"event": "done", "data": {"assistant_message": "שלום", "model": "test-model"}}


class _StubStore:
    """Job-store double exposing only the engine the persistence path touches."""

    def __init__(self, engine: Engine) -> None:
        """Store the engine the persistence path reads through.

        Args:
            engine: The SQLite engine backing the stub job-store.
        """
        self.engine = engine


@pytest.fixture
def persistence_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Engine]:
    """Mount the generalist router over a real SQLite engine with agent tables.

    Args:
        monkeypatch: Pytest fixture used to stub the agent runtime, the
            embedding hook, and the backend auth secret.

    Returns:
        The bound ``TestClient`` and the SQLite ``Engine`` so tests can assert
        on the persisted rows directly.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[AgentConversationModel.__table__, AgentMessageModel.__table__])
    monkeypatch.setattr(auth_mod.settings, "backend_auth_secret", SecretStr(_SECRET))
    monkeypatch.setattr(agent_mod, "run_generalist_agent", _fake_stream)
    monkeypatch.setattr(agent_mod, "queue_conversation_embed", lambda *a, **k: None)

    store = _StubStore(engine)
    app = FastAPI()
    app.state.job_store = store
    app.include_router(create_generalist_agent_router(job_store=store))
    return TestClient(app), engine


def _post_turn(
    client: TestClient,
    message: str,
    token: str | None,
    *,
    conversation_id: str | None = None,
    regenerate: bool = False,
) -> Any:
    """POST one generalist-agent turn and return the response.

    Args:
        client: Bound test client.
        message: User message text.
        token: Bearer token, or None to omit the Authorization header.
        conversation_id: Existing conversation to continue, if any.
        regenerate: Whether to replace the conversation's final persisted turn.

    Returns:
        The streaming ``Response`` (already fully read by the test client).
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/optimizations/generalist-agent",
        json={
            "user_message": message,
            "chat_history": [],
            "wizard_state": {},
            "trust_mode": "ask",
            "conversation_id": conversation_id,
            "regenerate": regenerate,
        },
        headers=headers,
    )


def test_authenticated_turn_persists_conversation_and_messages(
    persistence_client: tuple[TestClient, Engine],
) -> None:
    """A signed-in turn writes the conversation header and both messages."""
    client, engine = persistence_client

    resp = _post_turn(client, "hi", _session_token())

    assert resp.status_code == 200
    assert "conversation_meta" in resp.text
    with Session(engine) as session:
        convs = session.query(AgentConversationModel).all()
        assert len(convs) == 1
        assert convs[0].title == "hi"
        assert convs[0].username == "alice@example.com"
        msgs = session.query(AgentMessageModel).order_by(AgentMessageModel.id).all()
        assert [(m.role, m.content) for m in msgs] == [
            ("user", "hi"),
            ("assistant", "שלום"),
        ]


def test_turn_forwards_chosen_model(
    persistence_client: tuple[TestClient, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composer menu's model rides the request into the agent engine."""
    client, _ = persistence_client
    seen: dict[str, Any] = {}

    async def capture_stream(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Record the forwarded kwargs and finish in one turn."""
        seen.update(kwargs)
        yield {"event": "done", "data": {"assistant_message": "ok", "model": "m"}}

    monkeypatch.setattr(agent_mod, "run_generalist_agent", capture_stream)
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = client.post(
        "/optimizations/generalist-agent",
        json={
            "user_message": "hi",
            "chat_history": [],
            "wizard_state": {},
            "trust_mode": "ask",
            "model": "openai/gpt-test",
            "reasoning_effort": "high",
        },
        headers={"Authorization": f"Bearer {_session_token()}"},
    )
    assert resp.status_code == 200
    assert seen["model_config"] is not None
    assert seen["model_config"].name == "openai/gpt-test"
    assert seen["model_config"].extra == {"reasoning_effort": "high"}


def test_turn_rejects_unknown_model(
    persistence_client: tuple[TestClient, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-catalog model is refused before the engine ever runs."""
    client, _ = persistence_client
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = client.post(
        "/optimizations/generalist-agent",
        json={
            "user_message": "hi",
            "chat_history": [],
            "wizard_state": {},
            "trust_mode": "ask",
            "model": "openai/not-a-model",
        },
        headers={"Authorization": f"Bearer {_session_token()}"},
    )
    assert resp.status_code == 422


def test_unauthenticated_turn_is_rejected(
    persistence_client: tuple[TestClient, Engine],
) -> None:
    """Without a token the turn is rejected with 401 and nothing is persisted.

    Pins the security contract: an anonymous caller must not drive the
    orchestrator LLM, so the route 401s before streaming instead of degrading
    to an ephemeral anonymous turn.
    """
    client, engine = persistence_client

    resp = _post_turn(client, "hi", token=None)

    assert resp.status_code == 401
    with Session(engine) as session:
        assert session.query(AgentConversationModel).count() == 0
        assert session.query(AgentMessageModel).count() == 0


def test_regenerate_replaces_final_persisted_turn(
    persistence_client: tuple[TestClient, Engine],
) -> None:
    """Regenerating the latest reply keeps one user/assistant pair after reload."""
    client, engine = persistence_client
    token = _session_token()

    first = _post_turn(client, "hi", token)
    assert first.status_code == 200
    with Session(engine) as session:
        conversation_id = session.query(AgentConversationModel.id).scalar()

    regenerated = _post_turn(
        client,
        "hi",
        token,
        conversation_id=conversation_id,
        regenerate=True,
    )

    assert regenerated.status_code == 200
    with Session(engine) as session:
        messages = session.query(AgentMessageModel).order_by(AgentMessageModel.id).all()
        assert [(message.role, message.content) for message in messages] == [
            ("user", "hi"),
            ("assistant", "שלום"),
        ]


_CONV_ID = "conv-1"


@pytest.fixture
def wrapper_engine(monkeypatch: pytest.MonkeyPatch) -> Engine:
    """Create a SQLite engine with agent tables and a seeded conversation row.

    Args:
        monkeypatch: Pytest fixture used to no-op the embedding hook so the
            persistence path can run without a live engine background thread.

    Returns:
        The bound ``Engine`` holding one ``agent_conversations`` row.
    """
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[AgentConversationModel.__table__, AgentMessageModel.__table__])
    monkeypatch.setattr(agent_mod, "queue_conversation_embed", lambda *a, **k: None)
    with Session(engine) as session:
        session.add(AgentConversationModel(id=_CONV_ID, username="alice@example.com", title="hi"))
        session.commit()
    return engine


async def _drain(source: AsyncIterator[dict[str, Any]], engine: Engine) -> list[dict[str, Any]]:
    """Run a turn through ``_wrap_with_persistence`` to natural completion.

    Args:
        source: Upstream event stream to wrap.
        engine: Engine backing the stub job-store.

    Returns:
        Every event the wrapper yielded.
    """
    wrapped = agent_mod._wrap_with_persistence(
        source,
        job_store=_StubStore(engine),
        conversation_id=_CONV_ID,
        title="hi",
        wizard_state_before={},
    )
    return [event async for event in wrapped]


async def _drive_then_close(source: AsyncIterator[dict[str, Any]], engine: Engine, *, until: str) -> None:
    """Pull events until one named ``until`` is seen, then ``aclose()`` early.

    Mirrors the frontend tearing down the SSE stream after a successful submit
    but before the upstream ``done`` event arrives.

    Args:
        source: Upstream event stream to wrap.
        engine: Engine backing the stub job-store.
        until: Event name after which to close the wrapped generator.
    """
    wrapped = agent_mod._wrap_with_persistence(
        source,
        job_store=_StubStore(engine),
        conversation_id=_CONV_ID,
        title="hi",
        wizard_state_before={},
    )
    async for event in wrapped:
        if event.get("event") == until:
            break
    await wrapped.aclose()


def _assistant_rows(engine: Engine) -> list[AgentMessageModel]:
    """Return all persisted assistant-role rows ordered by id.

    Args:
        engine: Engine to query.

    Returns:
        The assistant ``agent_messages`` rows.
    """
    with Session(engine) as session:
        return (
            session.query(AgentMessageModel)
            .filter(AgentMessageModel.role == "assistant")
            .order_by(AgentMessageModel.id)
            .all()
        )


async def test_persist_on_early_close_after_successful_submit(wrapper_engine: Engine) -> None:
    """A successful submit turn persists once even if closed before ``done``.

    Reproduces the lost-turn bug: the source emits ``tool_start`` /
    ``tool_end`` (status ``ok``) for ``submit_job_run_post`` and the wrapper is
    ``aclose()``-d before any ``done`` event, exactly as the frontend does when
    it navigates away on a successful submit.
    """

    async def _submit_then_dangle() -> AsyncIterator[dict[str, Any]]:
        """Emit a successful submit turn that dangles without ever sending ``done``."""
        yield {"event": "message_patch", "data": {"chunk": "מגיש"}}
        yield {
            "event": "tool_start",
            "data": {"id": "t1", "tool": "submit_job_run_post", "arguments": {}},
        }
        yield {
            "event": "tool_end",
            "data": {"id": "t1", "status": "ok", "result": {"job_id": "d94bbebc"}},
        }
        # The frontend has navigated away; ``done`` never arrives.
        while True:
            yield {"event": "message_patch", "data": {"chunk": "..."}}

    await _drive_then_close(_submit_then_dangle(), wrapper_engine, until="tool_end")

    rows = _assistant_rows(wrapper_engine)
    assert len(rows) == 1
    assert rows[0].content == "מגיש"
    assert [c["tool"] for c in (rows[0].tool_calls or [])] == ["submit_job_run_post"]
    assert rows[0].tool_calls[0]["status"] == "done"


async def test_persist_on_done_writes_exactly_once(wrapper_engine: Engine) -> None:
    """When ``done`` fires the turn persists once — the finally must not re-write."""

    async def _normal_turn() -> AsyncIterator[dict[str, Any]]:
        """Emit a normal turn that reaches ``done``."""
        yield {"event": "message_patch", "data": {"chunk": "שלום"}}
        yield {"event": "done", "data": {"assistant_message": "שלום", "model": "test-model"}}

    await _drain(_normal_turn(), wrapper_engine)

    rows = _assistant_rows(wrapper_engine)
    assert len(rows) == 1
    assert rows[0].content == "שלום"
    assert rows[0].model == "test-model"


async def test_empty_greeting_turn_does_not_persist_on_teardown(wrapper_engine: Engine) -> None:
    """A teardown with no text and no settled tool-calls writes no empty row."""

    async def _empty_then_dangle() -> AsyncIterator[dict[str, Any]]:
        """Emit only a ``conversation_meta`` event, then dangle with no settled content."""
        yield {"event": "conversation_meta", "data": {}}
        while True:
            yield {"event": "ping", "data": {}}

    await _drive_then_close(_empty_then_dangle(), wrapper_engine, until="conversation_meta")

    assert _assistant_rows(wrapper_engine) == []
