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

# The wizard shows tool descriptions in full; the cap only guards the payload
# against pathological servers, not normal docstrings.
_TOOL_DESCRIPTION_MAX = 4000


class McpProbeRequest(BaseModel):
    """Request body for ``POST /mcp/probe``."""

    mcp_url: str = Field(min_length=1, max_length=2000)
    auth_header: str | None = Field(default=None, max_length=4000)


class McpProbeTool(BaseModel):
    """One tool exposed by the probed server: name plus display summary."""

    name: str
    description: str | None = None


class McpProbeResponse(BaseModel):
    """Response body for ``POST /mcp/probe``: reachability plus tool roster."""

    ok: bool
    tool_count: int = 0
    tools: list[McpProbeTool] = Field(default_factory=list)
    error: str | None = None


def _clean_description(description: str | None) -> str | None:
    """Bound a raw tool description for transport, keeping its text intact.

    Args:
        description: Raw description from the MCP tool listing.

    Returns:
        The full description, stripped and capped at the payload guard, or
        ``None`` when the tool carries no description.
    """
    if not description:
        return None
    return truncate_text(description.strip(), _TOOL_DESCRIPTION_MAX) or None


async def _list_tools(mcp_url: str, auth_header: str | None) -> list[McpProbeTool]:
    """Open one MCP session and return the server's tools.

    Mirrors the session bootstrap used by the react training path
    (``run_react._list_live_tools``) minus the dspy wrapping — the probe only
    needs to prove the server answers and enumerate what it exposes.

    Args:
        mcp_url: MCP server URL.
        auth_header: Optional ``Authorization`` header to forward.

    Returns:
        Every tool the server exposes, descriptions trimmed for display.
    """
    headers = {"Authorization": auth_header} if auth_header else None
    async with (
        streamablehttp_client(mcp_url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listing = await session.list_tools()
        return [
            McpProbeTool(name=tool.name, description=_clean_description(tool.description))
            for tool in listing.tools
        ]


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
            tools = await asyncio.wait_for(
                _list_tools(payload.mcp_url.strip(), payload.auth_header or None),
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
        return McpProbeResponse(ok=True, tool_count=len(tools), tools=tools)

    return router
