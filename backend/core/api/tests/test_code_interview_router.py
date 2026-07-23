"""Tests for the Signature & Metric interview SSE endpoint.

Mounts the code-agent router with the engine's stream monkeypatched, so the
wire contract (event framing, argument forwarding, sample-row cap, error
translation) is covered without an LLM. The ``interview_brief`` pass-through
on the seed endpoint is covered the same way against ``run_code_agent``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from .. import model_catalog, model_router
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers import code_agent as code_agent_router
from ..routers.code_agent import create_code_agent_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())

_INTERVIEW_BODY = {
    "dataset_columns": ["text", "label"],
    "column_roles": {"text": "input", "label": "output"},
    "column_kinds": {"text": "text"},
    "sample_rows": [{"text": f"row {i}", "label": "x"} for i in range(7)],
    "turns": [{"role": "assistant", "content": "Q1?"}, {"role": "user", "content": "A1"}],
    "job_model": "openai/gpt-4o-mini",
    "locale": "en",
}


def _client() -> TestClient:
    """Mount the code-agent router authed as Alice."""
    app = FastAPI()
    app.include_router(create_code_agent_router())
    app.dependency_overrides[get_authenticated_user] = lambda: _ALICE

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return TestClient(app)


def test_interview_streams_events_and_forwards_args(monkeypatch) -> None:
    """The route relays engine events as SSE and forwards the request fields."""
    seen: dict[str, Any] = {}

    async def fake_stream(**kwargs: Any) -> Any:
        """Record the forwarded kwargs and yield a two-event stream."""
        seen.update(kwargs)
        yield {"event": "message_patch", "data": {"chunk": "שאלה"}}
        yield {
            "event": "interview_done",
            "data": {"message": "שאלה", "options": [], "brief": [], "done": False, "model": "m"},
        }

    monkeypatch.setattr(code_agent_router, "interview_turn_stream", fake_stream)
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = _client().post(
        "/optimizations/code-interview",
        json={**_INTERVIEW_BODY, "model": "openai/gpt-test", "reasoning_effort": "high"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: message_patch" in resp.text
    assert "event: interview_done" in resp.text
    assert "שאלה" in resp.text

    assert seen["dataset_columns"] == ["text", "label"]
    assert seen["job_model"] == "openai/gpt-4o-mini"
    assert seen["locale"] == "en"
    assert seen["turns"] == _INTERVIEW_BODY["turns"]
    assert len(seen["sample_rows"]) == 5
    assert seen["model"] == "openai/gpt-test"
    assert seen["reasoning_effort"] == "high"
    assert seen["lm_extra_body"] is None


def test_interview_auto_tiers_ride_the_auto_router(monkeypatch) -> None:
    """Absent and 'auto:intelligent' models route through OpenRouter's router."""
    seen: list[dict[str, Any]] = []

    async def fake_stream(**kwargs: Any) -> Any:
        """Record the forwarded kwargs and finish immediately."""
        seen.append(kwargs)
        yield {
            "event": "interview_done",
            "data": {"message": "q", "options": [], "brief": [], "done": False, "model": "m"},
        }

    monkeypatch.setattr(code_agent_router, "interview_turn_stream", fake_stream)
    monkeypatch.setattr(
        model_router,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openrouter/anthropic/claude-sonnet-5")]),
    )
    client = _client()
    assert client.post("/optimizations/code-interview", json=_INTERVIEW_BODY).status_code == 200
    assert (
        client.post(
            "/optimizations/code-interview",
            json={**_INTERVIEW_BODY, "model": "auto:intelligent"},
        ).status_code
        == 200
    )
    assert [k["model"] for k in seen] == [model_router.OPENROUTER_AUTO_ID] * 2
    assert [k["lm_extra_body"] for k in seen] == [
        {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 5}]},
        {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 0}]},
    ]


def test_interview_rejects_unknown_model(monkeypatch) -> None:
    """A non-catalog interview model is refused before any LLM spend."""
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = _client().post(
        "/optimizations/code-interview", json={**_INTERVIEW_BODY, "model": "openai/not-a-model"}
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "models.unknown_model"


def test_interview_translates_engine_failure_to_error_event(monkeypatch) -> None:
    """An engine exception becomes a terminal error event, not a broken stream."""

    async def failing_stream(**_: Any) -> Any:
        """Blow up before yielding anything."""
        raise RuntimeError("provider down")
        yield  # pragma: no cover - marks this as a generator

    monkeypatch.setattr(code_agent_router, "interview_turn_stream", failing_stream)
    # The model-less request auto-routes, which consults the catalog — keep
    # the test hermetic instead of probing live providers.
    monkeypatch.setattr(
        model_router,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = _client().post("/optimizations/code-interview", json=_INTERVIEW_BODY)
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "submit.code.interview.llm_failed" in resp.text


def test_seed_endpoint_forwards_interview_brief(monkeypatch) -> None:
    """The confirmed brief rides the ai-generate-code request into the engine."""
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> Any:
        """Record kwargs and return an immediately-done stream."""
        seen.update(kwargs)

        async def gen() -> Any:
            yield {"event": "done", "data": {"signature_code": "", "metric_code": ""}}

        return gen()

    monkeypatch.setattr(code_agent_router, "run_code_agent", fake_run)
    body = {
        "dataset_columns": ["text", "label"],
        "column_roles": {"text": "input", "label": "output"},
        "sample_rows": [],
        "user_message": "",
        "interview_brief": ["Outputs must be lowercase.", "Penalize hedging."],
    }
    resp = _client().post("/optimizations/ai-generate-code", json=body)
    assert resp.status_code == 200
    assert seen["interview_brief"] == ["Outputs must be lowercase.", "Penalize hedging."]


_SEED_BODY = {
    "dataset_columns": ["text", "label"],
    "column_roles": {"text": "input", "label": "output"},
    "sample_rows": [],
    "user_message": "",
}


def _fake_run(seen: dict[str, Any]):
    """Build a ``run_code_agent`` stub that records kwargs and finishes at once."""

    def fake_run(**kwargs: Any) -> Any:
        """Record kwargs and return an immediately-done stream."""
        seen.update(kwargs)

        async def gen() -> Any:
            yield {"event": "done", "data": {"signature_code": "", "metric_code": ""}}

        return gen()

    return fake_run


def test_seed_endpoint_forwards_model_and_effort(monkeypatch) -> None:
    """An explicit catalog model + effort ride ai-generate-code into the engine."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(code_agent_router, "run_code_agent", _fake_run(seen))
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = _client().post(
        "/optimizations/ai-generate-code",
        json={**_SEED_BODY, "model": "openai/gpt-test", "reasoning_effort": "high"},
    )
    assert resp.status_code == 200
    assert seen["model"] == "openai/gpt-test"
    assert seen["reasoning_effort"] == "high"
    assert seen["lm_extra_body"] is None


def test_seed_endpoint_auto_tiers_ride_the_auto_router(monkeypatch) -> None:
    """Absent and 'auto:intelligent' models route the code author through OpenRouter."""
    seen: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> Any:
        """Record kwargs and return an immediately-done stream."""
        seen.append(kwargs)

        async def gen() -> Any:
            yield {"event": "done", "data": {"signature_code": "", "metric_code": ""}}

        return gen()

    monkeypatch.setattr(code_agent_router, "run_code_agent", fake_run)
    monkeypatch.setattr(
        model_router,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openrouter/anthropic/claude-sonnet-5")]),
    )
    client = _client()
    assert client.post("/optimizations/ai-generate-code", json=_SEED_BODY).status_code == 200
    assert (
        client.post(
            "/optimizations/ai-generate-code",
            json={**_SEED_BODY, "model": "auto:intelligent"},
        ).status_code
        == 200
    )
    assert [k["model"] for k in seen] == [model_router.OPENROUTER_AUTO_ID] * 2
    assert [k["lm_extra_body"] for k in seen] == [
        {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 5}]},
        {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 0}]},
    ]


def test_seed_endpoint_rejects_unknown_model(monkeypatch) -> None:
    """A non-catalog code-author model is refused before any LLM spend."""
    monkeypatch.setattr(
        model_catalog,
        "get_catalog_cached",
        lambda: SimpleNamespace(models=[SimpleNamespace(value="openai/gpt-test")]),
    )
    resp = _client().post(
        "/optimizations/ai-generate-code", json={**_SEED_BODY, "model": "openai/not-a-model"}
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "models.unknown_model"
