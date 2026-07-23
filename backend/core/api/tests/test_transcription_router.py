"""Tests for the dictation transcription route (Groq-only + guards).

Mounts the router with the auth dependency overridden and the Groq leg
monkeypatched — no network. Covers the unconfigured 503, the size-cap 413,
the happy path, and the provider-failure 502.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ...config import settings
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers import transcription
from ..routers.transcription import create_transcription_router

_USER = AuthenticatedUser(username="alice", role="user", groups=())


def _client() -> TestClient:
    """Mount the transcription router authed as a fixed user."""
    app = FastAPI()
    app.include_router(create_transcription_router())
    app.dependency_overrides[get_authenticated_user] = lambda: _USER

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return TestClient(app)


def _post_audio(client: TestClient, payload: bytes = b"RIFFxxxx") -> httpx.Response:
    """POST a small multipart clip to /transcribe."""
    return client.post(
        "/transcribe",
        files={"audio": ("take.webm", payload, "audio/webm")},
        data={"language": "he-IL"},
    )


def test_unconfigured_returns_typed_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a Groq key, the route answers an honest 503."""
    monkeypatch.setattr(settings, "groq_api_key", None)
    resp = _post_audio(_client())
    assert resp.status_code == 503
    assert resp.json()["code"] == "transcription.unconfigured"


def test_oversized_clip_rejected_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip over the cap is rejected before the provider is contacted."""
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("gsk-test"))
    monkeypatch.setattr(transcription, "_MAX_AUDIO_BYTES", 4)
    resp = _post_audio(_client(), payload=b"12345")
    assert resp.status_code == 413
    assert resp.json()["code"] == "transcription.too_large"


def test_groq_leg_returns_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a Groq key set, the leg serves the transcript."""
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("gsk-test"))

    seen: dict[str, str] = {}

    async def _fake_groq(_client, _audio, _filename, key) -> str:
        seen["key"] = key
        return "מהיר מאוד"

    monkeypatch.setattr(transcription, "_groq_transcribe", _fake_groq)
    resp = _post_audio(_client())
    assert resp.status_code == 200
    assert resp.json() == {"text": "מהיר מאוד", "provider": "groq"}
    assert seen == {"key": "gsk-test"}


def test_provider_failure_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing Groq call answers a typed 502."""
    monkeypatch.setattr(settings, "groq_api_key", SecretStr("gsk-test"))

    async def _boom(_client, _audio, _filename, _key) -> str:
        raise RuntimeError("groq transcribe: 500")

    monkeypatch.setattr(transcription, "_groq_transcribe", _boom)
    resp = _post_audio(_client())
    assert resp.status_code == 502
    assert resp.json()["code"] == "transcription.failed"
