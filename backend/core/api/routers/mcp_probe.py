"""Route for probing a live MCP server from the submit wizard. [INTERNAL]

``POST /mcp/probe`` — open one short-lived MCP session against the given
URL, list its tools, and report reachability. The wizard calls this as the
user fills in an MCP URL so tool-using runs (react, workflows with tool
nodes) get immediate connected/unreachable feedback instead of failing at
submit or mid-optimization. Hidden from the public Scalar reference.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

from ..auth import AuthenticatedUser, get_authenticated_user
from ..response_limits import AGENT_MAX_ERROR, truncate_text

# Generous enough for a cold server spin-up, short enough that the wizard's
# inline status never looks hung.
_PROBE_TIMEOUT_S = 8.0


class McpProbeRequest(BaseModel):
    """Request body for ``POST /mcp/probe``."""

    mcp_url: str = Field(min_length=1, max_length=2000)
    auth_header: str | None = Field(default=None, max_length=4000)


class McpProbeResponse(BaseModel):
    """Response body for ``POST /mcp/probe``: reachability plus tool roster."""

    ok: bool
    tool_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    error: str | None = None


async def _list_tool_names(mcp_url: str, auth_header: str | None) -> list[str]:
    """Open one MCP session and return the server's tool names.

    Mirrors the session bootstrap used by the react training path
    (``run_react._list_live_tools``) minus the dspy wrapping — the probe only
    needs to prove the server answers and enumerate what it exposes.

    Args:
        mcp_url: MCP server URL.
        auth_header: Optional ``Authorization`` header to forward.

    Returns:
        The names of every tool the server exposes.
    """
    headers = {"Authorization": auth_header} if auth_header else None
    async with (
        streamablehttp_client(mcp_url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listing = await session.list_tools()
        return [tool.name for tool in listing.tools]


def create_mcp_probe_router() -> APIRouter:
    """Build the MCP probe router.

    Returns:
        A configured :class:`APIRouter` exposing ``POST /mcp/probe``.
    """
    router = APIRouter()

    @router.post(
        "/mcp/probe",
        response_model=McpProbeResponse,
        summary="Check that a live MCP server is reachable and list its tools",
    )
    async def probe_mcp(
        payload: McpProbeRequest,
        _user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    ) -> McpProbeResponse:
        """Connect to the MCP server and list its tools, bounded by a timeout.

        Never raises 5xx — connection failures come back as ``ok=False`` with
        the error message so the wizard can render the status inline.

        Args:
            payload: The MCP URL and optional Authorization header.
            _user: Authenticated caller; the probe issues outbound requests,
                so it is not exposed anonymously.

        Returns:
            A :class:`McpProbeResponse` with ``ok``, the tool roster, and any
            connection error.
        """
        try:
            names = await asyncio.wait_for(
                _list_tool_names(payload.mcp_url.strip(), payload.auth_header or None),
                timeout=_PROBE_TIMEOUT_S,
            )
        except TimeoutError:
            return McpProbeResponse(ok=False, error=f"Connection timed out after {int(_PROBE_TIMEOUT_S)}s")
        # Catch-all: transport and protocol failures vary wildly by server
        # (DNS, TLS, HTTP status, malformed MCP handshake) — all of them mean
        # "not connected", which is exactly what the wizard needs to show.
        except Exception as exc:
            detail = truncate_text(str(exc), AGENT_MAX_ERROR) or type(exc).__name__
            return McpProbeResponse(ok=False, error=detail)
        return McpProbeResponse(ok=True, tool_count=len(names), tool_names=names)

    return router
