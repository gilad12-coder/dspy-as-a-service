"""Streaming endpoint for the submit-wizard AI code agent. [INTERNAL]

All endpoints are hidden from the public Scalar reference (none are in
``_SCALAR_PUBLIC_PATHS``). Used by the wizard UI to author DSPy code
interactively — not part of the dev integration surface.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ...service_gateway.agents.code import run_code_agent
from ...service_gateway.agents.code_interview import interview_turn_stream
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..model_router import route_menu_model
from ._helpers import enforce_llm_credits, sse_from_events, stream_with_llm_metering

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


class ChatTurn(BaseModel):
    """A single prior turn in the agent conversation."""

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message text.")


class CodeAgentRequest(BaseModel):
    """Input for streaming code generation.

    The server only needs the dataset's columns + roles + a small sample
    (not the full dataset); payloads stay under ~20 KB even for wide schemas.
    ``user_message`` toggles mode: empty triggers the non-agentic seed,
    non-empty invokes the ReAct chat agent (which also sees ``chat_history``
    and the current editor contents in ``prior_signature`` / ``prior_metric``).
    """

    dataset_columns: list[str] = Field(..., min_length=1, description="All column names in the dataset.")
    column_roles: dict[str, str] = Field(..., description="Column → 'input'|'output'|'ignore'.")
    column_kinds: dict[str, str] = Field(
        default_factory=dict,
        description="Input column → 'text'|'image'. Image columns get a dspy.Image typed InputField.",
    )
    sample_rows: list[dict] = Field(default_factory=list, description="Up to 5 sample rows.")
    user_message: str = Field(default="", description="User's latest message. Empty triggers seed mode.")
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior {role, content} turns; seen by the chat agent only.",
    )
    prior_signature: str = Field(default="", description="Current signature code in the editor.")
    prior_metric: str = Field(default="", description="Current metric code in the editor.")
    prior_signature_validation: str = Field(
        default="",
        description=(
            "Short summary of the latest validation result for the current "
            "signature ('OK' / 'errors: ...' / empty). Surfaced to the chat "
            "agent so follow-up edits can target real errors."
        ),
    )
    prior_metric_validation: str = Field(
        default="",
        description=("Short summary of the latest validation result for the current metric."),
    )
    initial_signature: str = Field(
        default="",
        description=(
            "The original signature code from the very first version — used by "
            "the chat agent to honor revert requests. May equal prior_signature "
            "when no edits have happened yet."
        ),
    )
    initial_metric: str = Field(
        default="",
        description=(
            "The original metric code from the very first version — used by "
            "the chat agent to honor revert requests. May equal prior_metric "
            "when no edits have happened yet."
        ),
    )
    prior_workflow: dict | None = Field(
        default=None,
        description=(
            "The workflow graph currently on the canvas (WorkflowSpec wire "
            "shape). Non-None switches both modes to their graph-aware "
            "paths: seed drafts the full DAG, chat gets graph tools."
        ),
    )
    initial_workflow: dict | None = Field(
        default=None,
        description="The original graph from the very first version, for revert support.",
    )
    locale: str | None = Field(
        default=None,
        description=(
            "UI locale code of the client (e.g. 'he', 'en', 'fr-CA'). Sets "
            "the language of the agent's replies and edit rationales; "
            "unknown or missing falls back to Hebrew."
        ),
    )
    interview_brief: list[str] = Field(
        default_factory=list,
        description=(
            "Directives confirmed at the end of the Signature & Metric "
            "interview. The seed authors honor every directive; empty when "
            "no interview happened."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "LiteLLM id of the catalog model that authors the code (the "
            "composer's model menu). Absent routes automatically (balanced "
            "tier); the sentinel 'auto:intelligent' routes to a frontier-"
            "quality model."
        ),
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Explicit reasoning-effort level for the chosen model; absent "
            "keeps the model's default."
        ),
    )


class CodeInterviewRequest(BaseModel):
    """Input for one streamed Signature & Metric interview turn.

    Stateless like the tagger interview: the client owns the transcript and
    re-sends it on every turn. The dataset context mirrors
    :class:`CodeAgentRequest` (columns + roles + a small sample, never the
    full dataset).
    """

    dataset_columns: list[str] = Field(..., min_length=1, description="All column names in the dataset.")
    column_roles: dict[str, str] = Field(..., description="Column → 'input'|'output'|'ignore'.")
    column_kinds: dict[str, str] = Field(
        default_factory=dict,
        description="Input column → 'text'|'image'.",
    )
    sample_rows: list[dict] = Field(default_factory=list, description="Up to 5 sample rows.")
    turns: list[ChatTurn] = Field(
        default_factory=list,
        max_length=40,
        description="Prior {role, content} interview turns, oldest first.",
    )
    job_model: str = Field(
        default="",
        description="LiteLLM id of the model the optimized program will run on; empty when not chosen yet.",
    )
    locale: str | None = Field(
        default=None,
        description=(
            "UI locale code of the client (e.g. 'he', 'en', 'fr-CA'). Sets "
            "the interview's language; unknown or missing falls back to Hebrew."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "LiteLLM id of the catalog model conducting the interview (the "
            "composer's model menu). Absent routes automatically (balanced "
            "tier); the sentinel 'auto:intelligent' routes to a frontier-"
            "quality model."
        ),
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Explicit reasoning-effort level for the chosen model; absent "
            "keeps the model's default."
        ),
    )


class EditCodeRequest(BaseModel):
    """Compact input for the MCP-exposed ``edit_code`` tool.

    This is a narrower, MCP-friendly projection of :class:`CodeAgentRequest`.
    The generalist agent only needs to state its goal, the current editor
    contents, and enough dataset context to drive the code agent — not the
    full chat history or validation-state scaffolding used by the wizard.
    """

    goal: str = Field(
        ...,
        min_length=1,
        description=("What to change. Non-empty; empty triggers seed mode on the SSE endpoint."),
    )
    current_signature: str = Field(default="", description="Current signature code (may be empty on first call).")
    current_metric: str = Field(default="", description="Current metric code (may be empty on first call).")
    dataset_columns: list[str] = Field(..., min_length=1, description="All column names in the dataset.")
    column_roles: dict[str, str] = Field(..., description="Column → 'input'|'output'|'ignore'.")
    column_kinds: dict[str, str] = Field(
        default_factory=dict,
        description="Input column → 'text'|'image'.",
    )
    sample_rows: list[dict] = Field(default_factory=list, description="Up to 5 sample rows.")


class EditCodeResponse(BaseModel):
    """Final output from a blocking ``edit_code`` call."""

    signature_code: str
    metric_code: str
    assistant_message: str = ""


class RequestCodeAuthoringRequest(BaseModel):
    """Request body for ``POST /optimizations/request-code-authoring``."""

    goal: str = Field(
        default="",
        max_length=400,
        description=(
            "Optional plain-language goal for the Signature/Metric (empty seeds "
            "both from the dataset). Echoed back so the card can show it."
        ),
    )


class RequestCodeAuthoringResponse(BaseModel):
    """Envelope for ``POST /optimizations/request-code-authoring`` — UI-trigger marker."""

    awaiting_code: bool
    goal: str


def create_code_agent_router(*, job_store=None) -> APIRouter:
    """Mount the ``POST /optimizations/ai-generate-code`` SSE endpoint.

    Args:
        job_store: Optional job-store whose engine backs the credit gate and
            per-turn usage metering. When ``None``, turns stream unmetered
            (legacy behavior).

    Returns:
        A configured :class:`APIRouter` with the code-agent endpoints attached.
    """
    router = APIRouter()

    @router.post(
        "/optimizations/ai-generate-code",
        summary="Stream AI-generated signature + metric code",
    )
    async def ai_generate_code(req: CodeAgentRequest, current_user: AuthenticatedUserDep) -> StreamingResponse:
        """Stream DSPy code-agent events as SSE.

        Event types:

        * ``signature_patch`` / ``metric_patch`` — ``{"chunk": "<token>"}``
          (seed mode only)
        * ``reasoning_patch`` — ``{"chunk": "<token>", "source": "<stream>"}``
          (both modes; ``source`` separates the parallel seed authors:
          ``signature`` / ``metric`` / ``workflow`` / ``agent``)
        * ``tool_start`` — ``{"id", "tool", "reason"}`` (chat mode)
        * ``signature_replace`` / ``metric_replace`` — ``{"code"}``
        * ``workflow_replace`` — ``{"workflow", "changed_node_id"}``
          (workflow mode: seed snapshot + one per graph tool op)
        * ``tool_end`` — ``{"id", "tool", "status"}``
        * ``message_patch`` — ``{"chunk": "<token>"}`` (chat mode reply stream)
        * ``done`` — ``{"signature_code", "metric_code", "assistant_message"}``
          (workflow mode carries ``workflow`` + ``workflow_valid`` instead
          of ``signature_code``)
        * ``error`` — ``{"error": "<message>"}``

        Args:
            req: Request body controlling code-agent inputs and chat history.
            current_user: The authenticated caller (required; gates LLM spend).

        Returns:
            A :class:`StreamingResponse` of Server-Sent Events.
        """
        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        model, lm_extra_body = route_menu_model(req.model)
        usage_sink: list = []
        source = run_code_agent(
            dataset_columns=req.dataset_columns,
            column_roles=req.column_roles,
            column_kinds=req.column_kinds,
            sample_rows=req.sample_rows,
            user_message=req.user_message,
            chat_history=[t.model_dump() for t in req.chat_history],
            prior_signature=req.prior_signature,
            prior_metric=req.prior_metric,
            prior_signature_validation=req.prior_signature_validation,
            prior_metric_validation=req.prior_metric_validation,
            initial_signature=req.initial_signature,
            initial_metric=req.initial_metric,
            prior_workflow=req.prior_workflow,
            initial_workflow=req.initial_workflow,
            interview_brief=req.interview_brief,
            locale=req.locale,
            model=model,
            reasoning_effort=req.reasoning_effort,
            lm_extra_body=lm_extra_body,
            usage_sink=usage_sink,
        )
        metered = stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Code authoring",
            usage_sink=usage_sink,
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/optimizations/code-interview",
        summary="Stream one Signature & Metric interview turn",
    )
    async def code_interview(req: CodeInterviewRequest, current_user: AuthenticatedUserDep) -> StreamingResponse:
        """Stream one interview turn as SSE, mirroring the tagger interview.

        Event types:

        * ``reasoning_patch`` — ``{"chunk": "<token>"}`` (provider thinking)
        * ``message_patch`` — ``{"chunk": "<token>"}`` (reply stream)
        * ``message_end`` — ``{}`` (the reply is fully streamed; options and
          brief are still generating)
        * ``turn_hint`` — ``{"final": bool}`` (the streamed ``done`` field
          settled; the client picks the matching placeholder)
        * ``message_reset`` — ``{}`` (a failed attempt is being retried; the
          client drops any partial reply streamed so far)
        * ``interview_done`` — ``{"message", "options", "brief", "done",
          "model"}`` (terminal; ``options`` is a list of ``{label,
          description}`` picks, ``brief`` is empty until ``done``)
        * ``error`` — ``{"error": "<message>"}``

        Args:
            req: Dataset context, the client-owned transcript, and locale.
            current_user: The authenticated caller (required; gates LLM spend).

        Returns:
            A :class:`StreamingResponse` of Server-Sent Events.
        """
        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        model, lm_extra_body = route_menu_model(req.model)
        usage_sink: list = []

        async def source() -> AsyncIterator[dict]:
            """Relay engine events, translating failures into an error event."""
            try:
                async for event in interview_turn_stream(
                    dataset_columns=req.dataset_columns,
                    column_roles=req.column_roles,
                    column_kinds=req.column_kinds,
                    sample_rows=req.sample_rows[:5],
                    job_model=req.job_model,
                    turns=[t.model_dump() for t in req.turns],
                    locale=req.locale,
                    model=model,
                    reasoning_effort=req.reasoning_effort,
                    lm_extra_body=lm_extra_body,
                    usage_sink=usage_sink,
                ):
                    yield event
            except Exception:
                logger.exception("code interview stream failed")
                yield {"event": "error", "data": {"code": "submit.code.interview.llm_failed"}}

        metered = stream_with_llm_metering(
            source(),
            job_store=job_store,
            username=current_user.username,
            description="Code interview",
            usage_sink=usage_sink,
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/optimizations/edit-code",
        response_model=EditCodeResponse,
        summary="Delegate signature + metric editing to the code agent",
        tags=["agent"],
    )
    async def edit_code(req: EditCodeRequest, current_user: AuthenticatedUserDep) -> EditCodeResponse:
        """Run the code agent to completion and return the final code.

        Consumes the same event stream as the SSE endpoint but blocks until
        the ``done`` event, so a ReAct tool call sees a single request /
        response. Streaming updates remain available to human-driven UIs via
        ``POST /optimizations/ai-generate-code``.

        Args:
            req: Compact MCP-friendly input: goal plus current editor contents.
            current_user: The authenticated caller (required; gates LLM spend).

        Returns:
            An :class:`EditCodeResponse` with the final signature, metric, and
            optional assistant message.

        Raises:
            DomainError: 502 when the code agent emits an ``error`` event.
        """
        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        final_signature = req.current_signature
        final_metric = req.current_metric
        assistant_message = ""

        usage_sink: list = []
        source = run_code_agent(
            dataset_columns=req.dataset_columns,
            column_roles=req.column_roles,
            column_kinds=req.column_kinds,
            sample_rows=req.sample_rows,
            user_message=req.goal,
            chat_history=[],
            prior_signature=req.current_signature,
            prior_metric=req.current_metric,
            usage_sink=usage_sink,
        )
        async for event in stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Code authoring",
            usage_sink=usage_sink,
        ):
            name = event["event"]
            data = event["data"]
            if name == "done":
                final_signature = data.get("signature_code", final_signature)
                final_metric = data.get("metric_code", final_metric)
                assistant_message = data.get("assistant_message", "")
            elif name == "error":
                raise DomainError(
                    "code_agent.upstream_failed",
                    status=502,
                    error=str(data.get("error", "code agent failed")),
                )

        return EditCodeResponse(
            signature_code=final_signature,
            metric_code=final_metric,
            assistant_message=assistant_message,
        )

    @router.post(
        "/optimizations/request-code-authoring",
        response_model=RequestCodeAuthoringResponse,
        operation_id="request_code_authoring",
        summary="Ask the chat panel to author the Signature + Metric via the code agent",
        tags=["agent"],
    )
    def request_code_authoring(
        req: RequestCodeAuthoringRequest,
        current_user: AuthenticatedUserDep,
    ) -> RequestCodeAuthoringResponse:
        """Signal the chat UI to render an inline code-authoring card.

        Stateless: the endpoint exists only so the generalist agent can call a
        named tool the frontend recognizes via its ``tool_start`` SSE event.
        The card streams the dedicated code agent (the same ``run_code_agent``
        the wizard uses) over ``POST /optimizations/ai-generate-code``, renders
        the reading→signature→metric timeline, and writes the finished code
        back into the shared wizard state — so the generalist never hand-writes
        ``signature_code`` / ``metric_code``. The agent calls this once, ends
        its turn, and reads the authored code on the next turn.

        Args:
            req: Optional plain-language goal for the Signature/Metric.
            current_user: The authenticated caller (required; gates LLM spend).

        Returns:
            A :class:`RequestCodeAuthoringResponse` marker carrying the goal
            back so the card can display it.
        """
        return RequestCodeAuthoringResponse(awaiting_code=True, goal=req.goal.strip())

    return router
