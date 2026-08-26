"""Tests for the internal dictation transcription route and guards.

Mounts the router with the auth dependency overridden and the configured leg
monkeypatched — no network. Covers the unconfigured 503, the size-cap 413,
the happy path, and the provider-failure 502.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

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
    """Without an internal endpoint, the route answers an honest 503."""
    monkeypatch.setattr(settings, "transcription_base_url", "")
    resp = _post_audio(_client())
    assert resp.status_code == 503
    assert resp.json()["code"] == "transcription.unconfigured"


def test_oversized_clip_rejected_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip over the cap is rejected before the provider is contacted."""
    monkeypatch.setattr(settings, "transcription_base_url", "https://speech.internal/v1")
    monkeypatch.setattr(transcription, "_MAX_AUDIO_BYTES", 4)
    resp = _post_audio(_client(), payload=b"12345")
    assert resp.status_code == 413
    assert resp.json()["code"] == "transcription.too_large"


def test_configured_leg_returns_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward audio to the configured private endpoint."""
    monkeypatch.setattr(settings, "transcription_base_url", "https://speech.internal/v1")

    seen: dict[str, str] = {}

    async def _fake_transcribe(_client, _audio, _filename, base_url, model, api_key) -> str:
        seen.update(base_url=base_url, model=model, api_key=str(api_key))
        return "מהיר מאוד"

    monkeypatch.setattr(transcription, "_transcribe_audio", _fake_transcribe)
    resp = _post_audio(_client())
    assert resp.status_code == 200
    assert resp.json() == {"text": "מהיר מאוד", "provider": "configured"}
    assert seen["base_url"] == "https://speech.internal/v1"


def test_provider_failure_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing private endpoint answers a typed 502."""
    monkeypatch.setattr(settings, "transcription_base_url", "https://speech.internal/v1")

    async def _boom(_client, _audio, _filename, _base_url, _model, _api_key) -> str:
        raise RuntimeError("transcription endpoint: 500")

    monkeypatch.setattr(transcription, "_transcribe_audio", _boom)
    resp = _post_audio(_client())
    assert resp.status_code == 502
    assert resp.json()["code"] == "transcription.failed"
