"""Agent-facing routes for the generalist agent's permanent memory. [INTERNAL]

The four OptMem operations the agent drives itself: ``memory_note`` appends
one memory (and hands back the next pending compression), ``memory_nap``
settles a compression the agent authored, ``memory_recall`` regex-searches
the whole log, and ``memory_zoom`` opens one summary node into its halves.
The fifth operation, ``wake``, is not a tool — the router serving the agent
turn injects :func:`core.api.agent_memory.wake_document` as the
``memory_context`` signature input on every turn.

All routes are ``tags=["agent"]`` (projected as MCP tools) and keyed by the
authenticated caller — memory is strictly per-user and never shared.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..agent_memory import MEMO_CHARS, note, recall, save_nap, zoom
from ..auth import AuthenticatedUser, get_authenticated_user

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


def create_agent_memory_router(*, job_store) -> APIRouter:
    """Build the agent-memory router.

    Args:
        job_store: Job-store instance whose ORM engine backs the memory tables.

    Returns:
        A FastAPI ``APIRouter`` exposing the four memory tools.
    """
    router = APIRouter()

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
