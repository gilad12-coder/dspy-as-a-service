"""Speech-to-text for composer dictation. [INTERNAL]

``POST /transcribe`` accepts one recorded clip and forwards it to an explicitly
configured OpenAI-compatible endpoint inside the private network. Unconfigured
deployments expose no dictation control in the UI and return a typed 503 if the
endpoint is called directly.
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

# whisper's documented upload cap; dictation takes are far below it.
_MAX_AUDIO_MB = 25
_MAX_AUDIO_BYTES = _MAX_AUDIO_MB * 1024 * 1024
_HTTP_TIMEOUT_S = 120.0


class TranscriptionResponse(BaseModel):
    """Response body for ``POST /transcribe``: transcript plus provider used."""

    text: str
    provider: str


async def _transcribe_audio(
    client: httpx.AsyncClient,
    audio: bytes,
    filename: str,
    base_url: str,
    model: str,
    api_key: str | None,
) -> str:
    """Transcribe one clip through an OpenAI-compatible endpoint.

    Args:
        client: Shared HTTP client.
        audio: Raw audio bytes.
        filename: Client filename, used for container detection.
        base_url: Configured API base URL.
        model: Transcription model identifier.
        api_key: Optional bearer token.

    Returns:
        The transcript text.

    Raises:
        RuntimeError: On a non-OK provider response.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    res = await client.post(
        f"{base_url.rstrip('/')}/audio/transcriptions",
        headers=headers,
        files={"file": (filename, audio)},
        data={"model": model},
    )
    if res.status_code >= 400:
        raise RuntimeError(f"transcription endpoint: {res.status_code}")
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
        if not settings.transcription_base_url:
            raise DomainError("transcription.unconfigured", status=503)
        filename = audio.filename or "take.webm"
        api_key = (
            settings.transcription_api_key.get_secret_value()
            if settings.transcription_api_key is not None
            else None
        )

        async with httpx.AsyncClient(timeout=httpx.Timeout(_HTTP_TIMEOUT_S)) as client:
            try:
                text = await _transcribe_audio(
                    client,
                    data,
                    filename,
                    settings.transcription_base_url,
                    settings.transcription_model,
                    api_key,
                )
            except (RuntimeError, httpx.HTTPError, KeyError, ValueError) as err:
                logger.warning("transcription failed: %s", err)
                raise DomainError("transcription.failed", status=502) from err
        return TranscriptionResponse(text=text, provider="configured")

    return router
