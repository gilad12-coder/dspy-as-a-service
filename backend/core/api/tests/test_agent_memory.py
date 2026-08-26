"""Tests for the generalist agent's permanent memory (OptMem port).

Runs the agent-memory router against an in-memory SQLite engine: the
note → compress → wake lifecycle end-to-end, block-order enforcement,
recall search, zoom navigation, and the per-user isolation of the log.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import AgentMemoryModel, AgentMemorySettingsModel, AgentMemorySummaryModel
from .. import agent_memory
from ..agent_memory import cover, note, wake_document
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers.agent_memory import create_agent_memory_router

_USER = "user@example.com"


class _Store:
    """Minimal store exposing only the SQLAlchemy engine the routes need."""

    def __init__(self, engine: Any) -> None:
        """Hold the engine the router opens sessions on.

        Args:
            engine: A SQLAlchemy engine with the memory tables created.
        """
        self.engine = engine


@pytest.fixture
def engine() -> Engine:
    """Create an in-memory SQLite engine with the two memory tables."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    for model in (AgentMemoryModel, AgentMemorySummaryModel, AgentMemorySettingsModel):
        model.__table__.create(engine)
    return engine


@pytest.fixture
def client(engine: Engine) -> TestClient:
    """Build a client serving the agent-memory router as a fixed test user.

    Args:
        engine: The shared in-memory engine fixture.

    Returns:
        A ``TestClient`` authenticated as ``_USER``.
    """
    app = FastAPI()

    # The real app's problem-details handler attaches ``code`` + ``params``;
    # mirror just those fields so the assertions see the same envelope.
    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    app.include_router(create_agent_memory_router(job_store=_Store(engine)))
    app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
        username=_USER, role="user", groups=()
    )
    return TestClient(app)


def _note(client: TestClient, text: str) -> dict:
    """POST one memory and return the parsed response body."""
    resp = client.post("/agent/memory/note", json={"text": text})
    assert resp.status_code == 200
    return resp.json()


def _settle_all(client: TestClient, first_request: str | None) -> int:
    """Answer every pending compression request with a stub summary.

    Args:
        client: The fixture client.
        first_request: The ``compression_request`` that started the backlog.

    Returns:
        How many blocks were settled.
    """
    settled = 0
    request = first_request
    while request:
        block = request.split('memory_nap(block="')[1].split('"')[0]
        resp = client.post(
            "/agent/memory/nap", json={"block": block, "summary": f"summary of {block}"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == f"#{block} saved."
        request = body["compression_request"]
        settled += 1
    return settled


def test_cover_tiles_exactly_within_budget() -> None:
    """cover() tiles [0, total) with aligned blocks, finest near the present."""
    for total, budget in ((1, 64), (5, 64), (300, 10), (1000, 16)):
        blocks = cover(total, budget)
        assert blocks[0][0] == 0
        assert blocks[-1][1] == total
        for (_, prev_hi), (lo, _) in pairwise(blocks):
            assert prev_hi == lo
        for lo, hi in blocks:
            n = hi - lo
            assert n >= 1
            assert (n & (n - 1)) == 0
            assert lo % n == 0
    assert cover(5, 64) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert len(cover(1000, 16)) <= 16
    # Detail decays with age: the newest block is raw, the oldest the largest.
    blocks = cover(1000, 16)
    assert blocks[-1][1] - blocks[-1][0] == 1
    assert blocks[0][1] - blocks[0][0] == max(hi - lo for lo, hi in blocks)


def test_first_note_saves_and_wakes(client: TestClient, engine: Engine) -> None:
    """A first memory lands as #0 and the wake document shows it verbatim."""
    body = _note(client, "User prefers Hebrew replies")
    assert body["saved_as"] == 0
    assert body["compression_request"] is None
    with Session(engine) as session:
        doc = wake_document(session, _USER)
    assert "#0" in doc
    assert "User prefers Hebrew replies" in doc
    assert "1 memory" in doc


def test_second_note_requests_first_compression(client: TestClient) -> None:
    """Note #1 completes block 0-1, so its compression request rides back."""
    _note(client, "first")
    body = _note(client, "second")
    assert body["saved_as"] == 1
    request = body["compression_request"]
    assert "Compress memories #0-1" in request
    assert "#0" in request
    assert "#1" in request
    assert 'memory_nap(block="0-1"' in request


def test_nap_settles_blocks_in_order(client: TestClient, engine: Engine) -> None:
    """Four notes build 0-1, 2-3, then 0-3; wake uses the coarse summary."""
    request = None
    for i in range(4):
        request = _note(client, f"memory number {i}")["compression_request"] or request
    assert _settle_all(client, request) == 3
    with Session(engine) as session:
        assert session.get(AgentMemorySummaryModel, (_USER, 4, 0)) is not None
        doc = wake_document(session, _USER)
    # All four fit the wake budget raw, so the summaries stay in reserve.
    assert "memory number 3" in doc
    assert "Compress" not in doc


def test_wake_falls_back_to_halves_when_unsummarized(
    client: TestClient, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget-forced block with no summary renders its halves, not a hole."""
    monkeypatch.setattr(agent_memory, "WAKE_LINES", 4)
    for i in range(8):
        _note(client, f"memory number {i}")
    with Session(engine) as session:
        doc = wake_document(session, _USER)
    # Every memory is reachable raw (no summaries exist yet) and the pending
    # compression is requested at the tail instead of blocking the wake.
    for i in range(8):
        assert f"memory number {i}" in doc
    assert 'memory_nap(block="0-1"' in doc


def test_wake_uses_summaries_once_settled(
    client: TestClient, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once settled, a coarse tiling renders one summary line per block."""
    request = None
    for i in range(8):
        request = _note(client, f"memory number {i}")["compression_request"] or request
    _settle_all(client, request)
    monkeypatch.setattr(agent_memory, "WAKE_LINES", 4)
    with Session(engine) as session:
        doc = wake_document(session, _USER)
    assert "#0-3 summary of 0-3" in doc
    assert "memory number 7" in doc


def test_nap_wrong_block_names_the_next_one(client: TestClient) -> None:
    """Settling out of order 422s with the block that should come first."""
    _note(client, "first")
    _note(client, "second")
    _note(client, "third")
    _note(client, "fourth")
    resp = client.post("/agent/memory/nap", json={"block": "2-3", "summary": "too early"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "agent_memory.wrong_block"
    assert body["params"]["next_block"] == "0-1"


def test_nap_already_settled_and_exhausted(client: TestClient) -> None:
    """Re-settling a done block reports it; an empty backlog says so."""
    _note(client, "first")
    request = _note(client, "second")["compression_request"]
    _settle_all(client, request)
    resp = client.post("/agent/memory/nap", json={"block": "0-1", "summary": "again"})
    assert resp.json()["status"] == "Nothing left to compress."


def test_note_validation(client: TestClient) -> None:
    """Empty, multi-line, and oversized notes are rejected with their codes."""
    resp = client.post("/agent/memory/note", json={"text": "   "})
    assert resp.status_code == 422
    assert resp.json()["code"] == "agent_memory.note_empty"
    resp = client.post("/agent/memory/note", json={"text": "one\ntwo"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "agent_memory.note_multiline"
    resp = client.post("/agent/memory/note", json={"text": "x" * 281})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "agent_memory.note_too_long"
    assert body["params"] == {"length": 281, "limit": 280}


def test_recall_matches_case_insensitively(client: TestClient) -> None:
    """recall returns matching lines with a count, or a definitive no-match."""
    _note(client, "User works at ACME Corp")
    _note(client, "Dataset has 5k rows of sentiment labels")
    _note(client, "acme project uses GEPA")
    resp = client.get("/agent/memory/recall", params={"pattern": "acme"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "ACME Corp" in result
    assert "acme project" in result
    assert "2 matches." in result
    assert "sentiment" not in result
    resp = client.get("/agent/memory/recall", params={"pattern": "nonexistent"})
    assert resp.json()["result"] == "No match."
    resp = client.get("/agent/memory/recall", params={"pattern": "(unclosed"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "agent_memory.bad_pattern"


def test_zoom_opens_a_node_into_halves(client: TestClient) -> None:
    """zoom renders each half as a raw memory or its summary."""
    request = None
    for i in range(4):
        request = _note(client, f"memory number {i}")["compression_request"] or request
    _settle_all(client, request)
    resp = client.get("/agent/memory/zoom", params={"block": "0-3"})
    assert resp.status_code == 200
    assert resp.json()["result"] == "#0-1 summary of 0-1\n#2-3 summary of 2-3"
    resp = client.get("/agent/memory/zoom", params={"block": "0-1"})
    lines = resp.json()["result"].splitlines()
    assert lines[0].startswith("#0 ")
    assert lines[0].endswith("memory number 0")
    assert lines[1].startswith("#1 ")
    assert lines[1].endswith("memory number 1")


def test_zoom_rejects_bad_and_beyond_blocks(client: TestClient) -> None:
    """Misaligned ids and blocks past the log end are 422s with their codes."""
    _note(client, "only one")
    resp = client.get("/agent/memory/zoom", params={"block": "5-6"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "agent_memory.invalid_block"
    resp = client.get("/agent/memory/zoom", params={"block": "4-5"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "agent_memory.block_beyond_log"


def test_settings_defaults_and_roundtrip(client: TestClient) -> None:
    """Knobs start at defaults; overrides persist and null resets them."""
    body = client.get("/agent/memory/settings").json()
    assert body["wake_lines"] == {"value": 64, "override": None, "default": 64, "min": 16, "max": 160}
    assert body["entry_chars"]["value"] == 280
    assert body["recall_chars"]["value"] == 4000
    body = client.put("/agent/memory/settings", json={"wake_lines": 32}).json()
    assert body["wake_lines"]["value"] == 32
    assert body["wake_lines"]["override"] == 32
    assert body["entry_chars"]["override"] is None
    body = client.put("/agent/memory/settings", json={"wake_lines": None}).json()
    assert body["wake_lines"] == {"value": 64, "override": None, "default": 64, "min": 16, "max": 160}


def test_settings_out_of_range(client: TestClient) -> None:
    """A value outside its knob's range 422s naming the knob and bounds."""
    resp = client.put("/agent/memory/settings", json={"wake_lines": 4})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "agent_memory.setting_out_of_range"
    assert body["params"] == {"name": "wake_lines", "min": 16, "max": 160}
    resp = client.put("/agent/memory/settings", json={"entry_chars": 500})
    assert resp.status_code == 422


def test_entry_chars_knob_gates_notes(client: TestClient) -> None:
    """Lowering entry_chars tightens note validation to the override."""
    client.put("/agent/memory/settings", json={"entry_chars": 100})
    resp = client.post("/agent/memory/note", json={"text": "x" * 150})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "agent_memory.note_too_long"
    assert body["params"] == {"length": 150, "limit": 100}
    assert _note(client, "x" * 100)["saved_as"] == 0


def test_wake_lines_knob_bounds_the_wake_document(client: TestClient, engine: Engine) -> None:
    """A lowered wake budget renders fewer, coarser lines once settled."""
    request = None
    for i in range(32):
        request = _note(client, f"memory number {i}")["compression_request"] or request
    _settle_all(client, request)
    with Session(engine) as session:
        full = wake_document(session, _USER)
    client.put("/agent/memory/settings", json={"wake_lines": 16})
    with Session(engine) as session:
        small = wake_document(session, _USER)
    assert len(full.splitlines()) > len(small.splitlines())
    # Header + at most 16 cover lines; everything is settled, so no fallback.
    assert len(small.splitlines()) <= 17
    assert "#0-3 summary of 0-3" in small


def test_recall_chars_knob_caps_output(client: TestClient) -> None:
    """A lowered recall budget keeps only the newest matches and says so."""
    for i in range(30):
        _note(client, f"acme note {i} " + "y" * 80)
    client.put("/agent/memory/settings", json={"recall_chars": 1000})
    result = client.get("/agent/memory/recall", params={"pattern": "acme"}).json()["result"]
    assert "Newest" in result
    assert "of 30 matches. Narrow the regex." in result
    assert "acme note 29" in result
    assert "acme note 0 " not in result


def test_memory_is_per_user(client: TestClient, engine: Engine) -> None:
    """One user's notes never appear in another user's wake or recall."""
    _note(client, "belongs to the fixture user")
    with Session(engine) as session:
        note(session, "other@example.com", "belongs to someone else")
        doc = wake_document(session, _USER)
        other_doc = wake_document(session, "other@example.com")
    assert "someone else" not in doc
    assert "fixture user" not in other_doc
