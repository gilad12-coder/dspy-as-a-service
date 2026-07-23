"""Speech-to-text for composer dictation. [INTERNAL]

``POST /transcribe`` — accept one recorded clip (multipart ``audio`` plus an
optional ``language`` hint from the UI locale) and return its transcript.

Groq's LPU-served Whisper large-v3-turbo is the platform's sole dictation
provider: a clip returns in hundreds of milliseconds, and running one leg
keeps the failure surface honest — no key configured answers a typed 503 the
composer turns into a transient failure notice, a provider error answers 502.
The ``language`` field is accepted for wire compatibility but never
forwarded: Whisper treats the param as a directive, and the UI locale isn't
necessarily the spoken language.
"""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from ...config import settings
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError

logger = logging.getLogger("skynet.api.transcription")

_GROQ_BASE = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "whisper-large-v3-turbo"
# whisper's documented upload cap; dictation takes are far below it.
_MAX_AUDIO_MB = 25
_MAX_AUDIO_BYTES = _MAX_AUDIO_MB * 1024 * 1024
_HTTP_TIMEOUT_S = 120.0


class TranscriptionResponse(BaseModel):
    """Response body for ``POST /transcribe``: transcript plus provider used."""

    text: str
    provider: str


async def _groq_transcribe(
    client: httpx.AsyncClient, audio: bytes, filename: str, api_key: str
) -> str:
    """Transcribe one clip via Whisper large-v3-turbo on Groq.

    Args:
        client: Shared HTTP client.
        audio: Raw audio bytes.
        filename: Client filename, used for container detection.
        api_key: Groq bearer token.

    Returns:
        The transcript text.

    Raises:
        RuntimeError: On a non-OK provider response.
    """
    res = await client.post(
        f"{_GROQ_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (filename, audio)},
        data={"model": _GROQ_MODEL},
    )
    if res.status_code >= 400:
        raise RuntimeError(f"groq transcribe: {res.status_code}")
    return str(res.json().get("text") or "")


def create_transcription_router() -> APIRouter:
    """Build the dictation transcription router.

    Returns:
        A configured :class:`APIRouter` exposing ``POST /transcribe``.
    """
    router = APIRouter()

    @router.post(
        "/transcribe",
        response_model=TranscriptionResponse,
        summary="Transcribe one recorded audio clip to text",
    )
    async def transcribe(
        _user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
        audio: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = None,
    ) -> TranscriptionResponse:
        """Run the clip through Groq Whisper and return the transcript.

        Args:
            _user: Authenticated caller (dictation is login-gated like the
                composers that host it).
            audio: The recorded clip (webm/opus everywhere, AAC-in-MP4 on
                Safari).
            language: Optional BCP-47 tag from the UI locale; accepted but
                unused — Whisper auto-detects the spoken language.

        Returns:
            The transcript and which provider produced it.

        Raises:
            DomainError: 413 when the clip exceeds the size cap, 503 when no
                Groq key is configured, 502 when the provider call failed.
        """
        del language
        data = await audio.read()
        if len(data) > _MAX_AUDIO_BYTES:
            raise DomainError("transcription.too_large", status=413, max_mb=_MAX_AUDIO_MB)
        if not settings.groq_api_key:
            raise DomainError("transcription.unconfigured", status=503)
        filename = audio.filename or "take.webm"

        async with httpx.AsyncClient(timeout=httpx.Timeout(_HTTP_TIMEOUT_S)) as client:
            try:
                text = await _groq_transcribe(
                    client, data, filename, settings.groq_api_key.get_secret_value()
                )
            except (RuntimeError, httpx.HTTPError, KeyError, ValueError) as err:
                logger.warning("transcription failed (groq): %s", err)
                raise DomainError("transcription.failed", status=502) from err
        return TranscriptionResponse(text=text, provider="groq")

    return router
