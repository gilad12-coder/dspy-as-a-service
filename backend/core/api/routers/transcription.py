"""Speech-to-text for composer dictation. [INTERNAL]

``POST /transcribe`` — accept one recorded clip (multipart ``audio`` plus an
optional ``language`` hint from the UI locale) and return its transcript.

Provider chain, best-first (July-2026 multilingual STT survey, ported from the
knowledge-system capture flow): Soniox stt-async-v5 has the best published
Hebrew WER and 60+ languages; ElevenLabs Scribe v2 is the strongest
single-call REST fallback; whisper-1 runs on the OpenAI-compatible gateway the
app already talks to. Every provider auto-detects language — the hint is soft.
No provider configured → an honest, typed 503 the composer turns into a
transient failure notice.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from ...config import settings
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError

logger = logging.getLogger("skynet.api.transcription")

_SONIOX_BASE = "https://api.soniox.com/v1"
# whisper's documented upload cap; dictation takes are far below it.
_MAX_AUDIO_MB = 25
_MAX_AUDIO_BYTES = _MAX_AUDIO_MB * 1024 * 1024
# Soniox is async (upload → create → poll); give the poll loop headroom.
_SONIOX_POLL_ATTEMPTS = 90
_HTTP_TIMEOUT_S = 120.0


class TranscriptionResponse(BaseModel):
    """Response body for ``POST /transcribe``: transcript plus provider used."""

    text: str
    provider: str


async def _soniox_transcribe(
    client: httpx.AsyncClient,
    audio: bytes,
    filename: str,
    api_key: str,
    lang_hint: str | None,
) -> str:
    """Transcribe via Soniox's async pipeline: upload, create, poll, fetch.

    Args:
        client: Shared HTTP client for the request chain.
        audio: Raw audio bytes.
        filename: Client filename, used for container detection.
        api_key: Soniox bearer token.
        lang_hint: Base language tag from the UI locale, or ``None``.

    Returns:
        The transcript text.

    Raises:
        RuntimeError: On any non-OK provider response, poll timeout, or error
            status — the chain treats it as "try the next provider".
    """
    auth = {"Authorization": f"Bearer {api_key}"}

    file_res = await client.post(
        f"{_SONIOX_BASE}/files",
        headers=auth,
        files={"file": (filename, audio)},
    )
    if file_res.status_code >= 400:
        raise RuntimeError(f"soniox file upload: {file_res.status_code}")
    file_id = file_res.json()["id"]

    create_res = await client.post(
        f"{_SONIOX_BASE}/transcriptions",
        headers=auth,
        json={
            "file_id": file_id,
            "model": "stt-async-v5",
            "enable_language_identification": True,
            # Soft hints only — identification stays on, so other spoken
            # languages still transcribe.
            "language_hints": [lang_hint, "he", "en"]
            if lang_hint and lang_hint not in ("he", "en")
            else ["he", "en"],
        },
    )
    if create_res.status_code >= 400:
        raise RuntimeError(f"soniox create: {create_res.status_code}")
    job_id = create_res.json()["id"]

    try:
        for _ in range(_SONIOX_POLL_ATTEMPTS):
            await asyncio.sleep(1)
            poll = await client.get(f"{_SONIOX_BASE}/transcriptions/{job_id}", headers=auth)
            if poll.status_code >= 400:
                raise RuntimeError(f"soniox poll: {poll.status_code}")
            status = poll.json()
            if status.get("status") == "completed":
                text_res = await client.get(
                    f"{_SONIOX_BASE}/transcriptions/{job_id}/transcript", headers=auth
                )
                if text_res.status_code >= 400:
                    raise RuntimeError(f"soniox transcript: {text_res.status_code}")
                body = text_res.json()
                if body.get("text"):
                    return str(body["text"])
                tokens = body.get("tokens")
                if isinstance(tokens, list):
                    return "".join(str(tok.get("text", "")) for tok in tokens)
                return ""
            if status.get("status") == "error":
                raise RuntimeError(f"soniox: {status.get('error_message')}")
        raise RuntimeError("soniox: timed out")
    finally:
        # Uploaded takes shouldn't accumulate in Soniox storage.
        with contextlib.suppress(httpx.HTTPError):
            await client.delete(f"{_SONIOX_BASE}/files/{file_id}", headers=auth)


async def _elevenlabs_transcribe(
    client: httpx.AsyncClient, audio: bytes, filename: str, api_key: str
) -> str:
    """Transcribe via ElevenLabs Scribe v2 in one REST call.

    Args:
        client: Shared HTTP client.
        audio: Raw audio bytes.
        filename: Client filename, used for container detection.
        api_key: ElevenLabs API key.

    Returns:
        The transcript text.

    Raises:
        RuntimeError: On a non-OK provider response.
    """
    res = await client.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": api_key},
        files={"file": (filename, audio)},
        data={"model_id": "scribe_v2"},
    )
    if res.status_code >= 400:
        raise RuntimeError(f"elevenlabs transcribe: {res.status_code}")
    return str(res.json().get("text") or "")


async def _whisper_transcribe(
    client: httpx.AsyncClient, audio: bytes, filename: str, api_key: str, api_base: str
) -> str:
    """Transcribe via whisper-1 on the configured OpenAI-compatible gateway.

    whisper-1, not gpt-4o-transcribe, and no ``language`` param: whisper treats
    the param as a directive, and the UI locale isn't necessarily the spoken
    language.

    Args:
        client: Shared HTTP client.
        audio: Raw audio bytes.
        filename: Client filename, used for container detection.
        api_key: Gateway bearer token.
        api_base: Gateway base URL (e.g. ``https://api.openai.com/v1``).

    Returns:
        The transcript text.

    Raises:
        RuntimeError: On a non-OK gateway response.
    """
    res = await client.post(
        f"{api_base.rstrip('/')}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (filename, audio)},
        data={"model": "whisper-1"},
    )
    if res.status_code >= 400:
        raise RuntimeError(f"whisper transcribe: {res.status_code}")
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
        """Run the clip through the provider chain and return the transcript.

        Args:
            _user: Authenticated caller (dictation is login-gated like the
                composers that host it).
            audio: The recorded clip (webm/opus everywhere, AAC-in-MP4 on
                Safari).
            language: Optional BCP-47 tag from the UI locale; reduced to its
                base language as a soft STT hint.

        Returns:
            The transcript and which provider produced it.

        Raises:
            DomainError: 413 when the clip exceeds the size cap, 503 when no
                provider is configured, 502 when every configured leg failed.
        """
        data = await audio.read()
        if len(data) > _MAX_AUDIO_BYTES:
            raise DomainError("transcription.too_large", status=413, max_mb=_MAX_AUDIO_MB)
        filename = audio.filename or "take.webm"
        lang_hint = language.split("-")[0] if language else None

        chain: list[tuple[str, Callable[[httpx.AsyncClient], Awaitable[str]]]] = []
        if settings.soniox_api_key:
            soniox_key = settings.soniox_api_key.get_secret_value()
            chain.append(
                (
                    "soniox",
                    lambda c: _soniox_transcribe(c, data, filename, soniox_key, lang_hint),
                )
            )
        if settings.elevenlabs_api_key:
            eleven_key = settings.elevenlabs_api_key.get_secret_value()
            chain.append(
                ("elevenlabs", lambda c: _elevenlabs_transcribe(c, data, filename, eleven_key))
            )
        if settings.openai_api_key:
            openai_key = settings.openai_api_key.get_secret_value()
            openai_base = settings.openai_api_base or "https://api.openai.com/v1"
            chain.append(
                (
                    "openai",
                    lambda c: _whisper_transcribe(c, data, filename, openai_key, openai_base),
                )
            )
        if not chain:
            raise DomainError("transcription.unconfigured", status=503)

        async with httpx.AsyncClient(timeout=httpx.Timeout(_HTTP_TIMEOUT_S)) as client:
            for provider, run in chain:
                try:
                    text = await run(client)
                    return TranscriptionResponse(text=text, provider=provider)
                except (RuntimeError, httpx.HTTPError, KeyError, ValueError) as err:
                    logger.warning("transcription leg failed (%s): %s", provider, err)
        raise DomainError("transcription.failed", status=502)

    return router
