"""Tests for the dictation transcription route (provider chain + guards).

Mounts the router with the auth dependency overridden and the provider legs
monkeypatched — no network. Covers the unconfigured 503, the size-cap 413,
the happy path, and the fall-through to the next leg when the best one fails.
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
    """With no provider key at all, the route answers an honest 503."""
    monkeypatch.setattr(settings, "soniox_api_key", None)
    monkeypatch.setattr(settings, "elevenlabs_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    resp = _post_audio(_client())
    assert resp.status_code == 503
    assert resp.json()["code"] == "transcription.unconfigured"


def test_oversized_clip_rejected_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip over the cap is rejected before any provider is contacted."""
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test"))
    monkeypatch.setattr(transcription, "_MAX_AUDIO_BYTES", 4)
    resp = _post_audio(_client(), payload=b"12345")
    assert resp.status_code == 413
    assert resp.json()["code"] == "transcription.too_large"


def test_whisper_leg_returns_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only the gateway key set, the whisper leg serves the transcript."""
    monkeypatch.setattr(settings, "soniox_api_key", None)
    monkeypatch.setattr(settings, "elevenlabs_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test"))

    async def _fake_whisper(_client, _audio, _filename, _key, _base) -> str:
        return "שלום עולם"

    monkeypatch.setattr(transcription, "_whisper_transcribe", _fake_whisper)
    resp = _post_audio(_client())
    assert resp.status_code == 200
    assert resp.json() == {"text": "שלום עולם", "provider": "openai"}


def test_chain_falls_through_to_next_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing best leg falls through; all legs failing answers a 502."""
    monkeypatch.setattr(settings, "soniox_api_key", None)
    monkeypatch.setattr(settings, "elevenlabs_api_key", SecretStr("xi-test"))
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test"))

    async def _boom(_client, _audio, _filename, _key) -> str:
        raise RuntimeError("elevenlabs transcribe: 500")

    async def _fake_whisper(_client, _audio, _filename, _key, _base) -> str:
        return "fallback text"

    monkeypatch.setattr(transcription, "_elevenlabs_transcribe", _boom)
    monkeypatch.setattr(transcription, "_whisper_transcribe", _fake_whisper)
    resp = _post_audio(_client())
    assert resp.status_code == 200
    assert resp.json() == {"text": "fallback text", "provider": "openai"}

    async def _boom_whisper(_client, _audio, _filename, _key, _base) -> str:
        raise RuntimeError("whisper transcribe: 500")

    monkeypatch.setattr(transcription, "_whisper_transcribe", _boom_whisper)
    resp = _post_audio(_client())
    assert resp.status_code == 502
    assert resp.json()["code"] == "transcription.failed"
