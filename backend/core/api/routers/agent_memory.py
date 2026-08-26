"""Agent-facing routes for the generalist agent's permanent memory. [INTERNAL]

The four OptMem operations the agent drives itself: ``memory_note`` appends
one memory (and hands back the next pending compression), ``memory_nap``
settles a compression the agent authored, ``memory_recall`` regex-searches
the whole log, and ``memory_zoom`` opens one summary node into its halves.
The fifth operation, ``wake``, is not a tool — the router serving the agent
turn injects :func:`core.api.agent_memory.wake_document` as the
``memory_context`` signature input on every turn.

The tool routes are ``tags=["agent"]`` (projected as MCP tools); the
settings pair (GET/PUT ``/agent/memory/settings`` — OptMem's ``config``
surface, driving the knobs in the settings modal's Agent tab) stays
REST-only. Everything is keyed by the authenticated caller — memory is
strictly per-user and never shared.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...storage.models import AgentMemorySettingsModel
from ..agent_memory import KNOBS, MEMO_CHARS, note, recall, save_nap, zoom
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


class MemoryNoteRequest(BaseModel):
    """Body for ``memory_note``."""

    text: str = Field(
        ...,
        description=f"The memory: one line, at most {MEMO_CHARS} characters, in English.",
    )


class MemoryNoteResponse(BaseModel):
    """Result of ``memory_note``."""

    saved_as: int
    compression_request: str | None


class MemoryNapRequest(BaseModel):
    """Body for ``memory_nap``."""

    block: str = Field(..., description='The block id from the compression request, like "16-31".')
    summary: str = Field(
        ...,
        description=f"Your one-line compression of the block, at most {MEMO_CHARS} characters.",
    )


class MemoryNapResponse(BaseModel):
    """Result of ``memory_nap``."""

    status: str
    compression_request: str | None


class MemoryTextResponse(BaseModel):
    """Plain-text result of ``memory_recall`` / ``memory_zoom``."""

    result: str


class MemoryKnobInfo(BaseModel):
    """One memory size knob: its effective value, override state, and range."""

    value: int
    override: int | None
    default: int
    min: int
    max: int


class MemorySettingsResponse(BaseModel):
    """The caller's memory size knobs (OptMem's ``config`` surface)."""

    wake_lines: MemoryKnobInfo
    entry_chars: MemoryKnobInfo
    recall_chars: MemoryKnobInfo


class MemorySettingsUpdate(BaseModel):
    """Partial knob update: omitted = unchanged, null = back to default."""

    wake_lines: int | None = None
    entry_chars: int | None = None
    recall_chars: int | None = None


def create_agent_memory_router(*, job_store) -> APIRouter:
    """Build the agent-memory router.

    Args:
        job_store: Job-store instance whose ORM engine backs the memory tables.

    Returns:
        A FastAPI ``APIRouter`` exposing the four memory tools.
    """
    router = APIRouter()

    def _settings_response(session: Session, username: str) -> MemorySettingsResponse:
        """Assemble the knob snapshot for ``username`` from row + defaults."""
        row = session.get(AgentMemorySettingsModel, username)
        knobs = {}
        for name, (default, lo, hi) in KNOBS.items():
            override = getattr(row, name) if row is not None else None
            knobs[name] = MemoryKnobInfo(
                value=override or default,
                override=override,
                default=default,
                min=lo,
                max=hi,
            )
        return MemorySettingsResponse(**knobs)

    @router.get(
        "/agent/memory/settings",
        response_model=MemorySettingsResponse,
        summary="Return the caller's memory size knobs",
    )
    def get_memory_settings(user: AuthenticatedUserDep) -> MemorySettingsResponse:
        """Return each knob's effective value, override, default, and range.

        Args:
            user: Authenticated caller whose settings are read.

        Returns:
            The knob snapshot; pure defaults when nothing was ever changed.
        """
        with Session(job_store.engine) as session:
            return _settings_response(session, user.username)

    @router.put(
        "/agent/memory/settings",
        response_model=MemorySettingsResponse,
        summary="Update the caller's memory size knobs",
    )
    def update_memory_settings(
        req: MemorySettingsUpdate, user: AuthenticatedUserDep
    ) -> MemorySettingsResponse:
        """Apply a partial knob update and return the resulting snapshot.

        A field omitted from the body is left unchanged; an explicit null
        clears the override so the knob follows the tool default again —
        OptMem's ``memo config NAME=`` semantics. Changing a knob only
        selects what is rendered or accepted from now on; no stored memory
        is touched or recomputed.

        Args:
            req: The partial update (any subset of the three knobs).
            user: Authenticated caller whose settings are written.

        Returns:
            The full knob snapshot after the update.

        Raises:
            DomainError: 422 when a supplied value falls outside its range.
        """
        supplied = req.model_dump(exclude_unset=True)
        with Session(job_store.engine) as session:
            row = session.get(AgentMemorySettingsModel, user.username)
            if row is None:
                row = AgentMemorySettingsModel(username=user.username)
                session.add(row)
            for name, value in supplied.items():
                _, lo, hi = KNOBS[name]
                if value is not None and not lo <= value <= hi:
                    raise DomainError(
                        "agent_memory.setting_out_of_range", status=422, name=name, min=lo, max=hi
                    )
                setattr(row, name, value)
            session.commit()
            return _settings_response(session, user.username)

    @router.post(
        "/agent/memory/note",
        response_model=MemoryNoteResponse,
        operation_id="memory_note",
        summary="Record one permanent memory",
        tags=["agent"],
    )
    def memory_note(req: MemoryNoteRequest, user: AuthenticatedUserDep) -> MemoryNoteResponse:
        """Append one line to the caller's permanent memory log.

        When the note unlocks a pending compression, the response carries a
        ``compression_request`` — author the one-line summary it asks for and
        call ``memory_nap`` before ending the turn.

        Args:
            req: The memory text.
            user: Authenticated caller who owns the memory.

        Returns:
            The id the memory saved as, plus the next pending compression
            request (or null).

        Raises:
            DomainError: 422 when the text is empty, multi-line, or too long.
        """
        with Session(job_store.engine) as session:
            seq, nap = note(session, user.username, req.text)
        return MemoryNoteResponse(saved_as=seq, compression_request=nap)

    @router.post(
        "/agent/memory/nap",
        response_model=MemoryNapResponse,
        operation_id="memory_nap",
        summary="Settle one pending memory compression",
        tags=["agent"],
    )
    def memory_nap(req: MemoryNapRequest, user: AuthenticatedUserDep) -> MemoryNapResponse:
        """Save the agent-authored summary of the next pending block.

        Blocks compress strictly in order; the request must name the block
        from the latest ``compression_request``. Any compression still
        pending afterwards rides back on the response.

        Args:
            req: The block id and its one-line summary.
            user: Authenticated caller who owns the memory.

        Returns:
            What happened to the block, plus the next pending compression
            request (or null).

        Raises:
            DomainError: 422 on a malformed block id, invalid summary text,
                or a block that is not the next pending one.
        """
        with Session(job_store.engine) as session:
            status, nap = save_nap(session, user.username, req.block, req.summary)
        return MemoryNapResponse(status=status, compression_request=nap)

    @router.get(
        "/agent/memory/recall",
        response_model=MemoryTextResponse,
        operation_id="memory_recall",
        summary="Search every memory ever recorded",
        tags=["agent"],
    )
    def memory_recall(
        user: AuthenticatedUserDep,
        pattern: str = Query(..., description="Case-insensitive regular expression."),
    ) -> MemoryTextResponse:
        """Regex-search the caller's whole memory log, word for word.

        Args:
            user: Authenticated caller who owns the memory.
            pattern: Case-insensitive regular expression.

        Returns:
            The newest matching lines within the output cap, with a match
            count.

        Raises:
            DomainError: 422 on an invalid or catastrophic pattern.
        """
        with Session(job_store.engine) as session:
            return MemoryTextResponse(result=recall(session, user.username, pattern))

    @router.get(
        "/agent/memory/zoom",
        response_model=MemoryTextResponse,
        operation_id="memory_zoom",
        summary="Open one memory-tree node into its two halves",
        tags=["agent"],
    )
    def memory_zoom(
        user: AuthenticatedUserDep,
        block: str = Query(..., description='A block id as the memory context prints them, like "16-31".'),
    ) -> MemoryTextResponse:
        """Expand a ``#lo-hi`` summary node into its halves, toward raw memories.

        Args:
            user: Authenticated caller who owns the memory.
            block: Block id string as the wake context prints them.

        Returns:
            One line per half — a raw memory once a half is single, its
            summary otherwise.

        Raises:
            DomainError: 422 on a malformed id or a block beyond the log.
        """
        with Session(job_store.engine) as session:
            return MemoryTextResponse(result=zoom(session, user.username, block))

    return router
