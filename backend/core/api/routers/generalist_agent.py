"""SSE endpoint for the generalist agent that drives the Skynet wizard. [INTERNAL]

Mirrors the ``code_agent`` router's shape: one streaming POST that emits
reasoning / tool / message events, plus a companion confirm POST so the
client can respond to ``pending_approval`` events (the SSE channel is
server → client only).

Persistence: when the caller is authenticated and ``job_store`` was wired
in at router construction, every turn is mirrored into the
``agent_conversations`` / ``agent_messages`` tables. The first emitted SSE
event is ``conversation_meta`` carrying the canonical conversation id so
new threads materialise without a separate round-trip.

All endpoints are hidden from the public Scalar reference (none are in
``_SCALAR_PUBLIC_PATHS``) — wizard-internal flow.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from ...models import ModelConfig
from ...service_gateway.agents.generalist import (
    TrustMode,
    WizardState,
    get_approval_registry,
    run_generalist_agent,
)
from ...service_gateway.embedding_pipeline import queue_conversation_embed
from ...storage.models import AgentConversationModel, AgentMessageModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..model_catalog import require_known_model
from ..model_router import resolve_auto_tier, route_auto_model
from ._helpers import enforce_llm_credits, sse_from_events, stream_with_llm_metering

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

TITLE_MAX_CHARS = 40


class ChatTurn(BaseModel):
    """A single prior turn in the agent conversation."""

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message text.")


class GeneralistAgentRequest(BaseModel):
    """Input for a single generalist-agent turn."""

    user_message: str = Field(..., description="The user's latest message.")
    chat_history: list[ChatTurn] = Field(default_factory=list, description="Prior {role, content} turns.")
    wizard_state: dict = Field(
        default_factory=dict,
        description=(
            "Snapshot of the wizard: ``{dataset_ready, columns_configured, "
            "signature_code, metric_code, model_configured, staged_dataset_id}``."
        ),
    )
    trust_mode: TrustMode = Field(
        default="ask",
        description="'ask' (confirm every mutation), 'auto_safe' (confirm destructive only), 'yolo' (never confirm).",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional id of an existing thread to append to. When absent the "
            "server creates a new conversation and emits its id via the "
            "``conversation_meta`` SSE event before any other event."
        ),
    )
    regenerate: bool = Field(
        default=False,
        description=(
            "Replace the existing conversation's final user/assistant turn "
            "instead of appending a duplicate turn."
        ),
    )
    locale: str | None = Field(
        default=None,
        description=(
            "UI locale code of the client (e.g. 'he', 'en', 'fr-CA'). Sets "
            "the language of the agent's replies; unknown or missing falls "
            "back to Hebrew."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "LiteLLM id of the catalog model to run this turn on (the "
            "composer's model menu). Absent routes automatically per turn "
            "(balanced tier); the sentinel 'auto:intelligent' routes to a "
            "frontier model on every turn."
        ),
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Explicit reasoning-effort level for the chosen model; absent "
            "keeps the model's default."
        ),
    )


class ConfirmApprovalRequest(BaseModel):
    """Client → server reply to a ``pending_approval`` SSE event."""

    call_id: str = Field(..., description="The id carried by the pending_approval event.")
    approved: bool = Field(..., description="True to proceed with the tool, False to decline.")


class ConfirmApprovalResponse(BaseModel):
    """Ack for an approval confirm call."""

    resolved: bool


def _derive_title(user_message: str) -> str:
    """Auto-title a fresh conversation from the user's opening message.

    Truncates whitespace-collapsed text to ``TITLE_MAX_CHARS`` and appends an
    ellipsis when truncation actually loses characters.

    Args:
        user_message: The first user turn's text.

    Returns:
        A short, single-line title suitable for the conversation header.
    """
    collapsed = " ".join(user_message.split())
    if len(collapsed) <= TITLE_MAX_CHARS:
        return collapsed
    return collapsed[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def _ensure_conversation(job_store, conversation_id: str | None, username: str, user_message: str) -> tuple[str, str]:
    """Create a new conversation row when one isn't supplied; touch existing rows.

    A fresh conversation gets an auto-derived title from the user's first
    message; an existing one keeps its (possibly user-renamed) title and only
    has ``updated_at`` bumped.

    Args:
        job_store: Job-store instance whose engine backs the DB session.
        conversation_id: Optional caller-supplied conversation id.
        username: Authenticated principal that owns the row.
        user_message: First user message (used for auto-title on creation).

    Returns:
        ``(conversation_id, title)`` — the id is freshly generated when input
        was ``None``; ``title`` is the post-state value.

    Raises:
        DomainError: 403 when ``conversation_id`` exists but is owned by a
            different user; 404 when an explicit id is unknown.
    """
    now = datetime.now(UTC)
    with Session(job_store.engine) as session:
        if conversation_id:
            row = session.get(AgentConversationModel, conversation_id)
            if row is None:
                raise DomainError("agent.conversation.not_found", status=404)
            if row.username != username:
                raise DomainError("agent.conversation.forbidden", status=403)
            row.updated_at = cast(Any, now)
            session.commit()
            return cast(str, row.id), cast(str, row.title)
        new_id = str(uuid4())
        title = _derive_title(user_message)
        row = AgentConversationModel(
            id=new_id,
            username=username,
            title=title,
            pinned=False,
            archived_at=None,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.commit()
        return new_id, title


def _persist_user_turn(job_store, conversation_id: str, content: str) -> None:
    """Insert the user's message into ``agent_messages``.

    Args:
        job_store: Job-store instance whose engine backs the DB session.
        conversation_id: Owning conversation id.
        content: User message text exactly as received from the client.
    """
    with Session(job_store.engine) as session:
        session.add(
            AgentMessageModel(
                conversation_id=conversation_id,
                role="user",
                content=content,
                tool_calls=None,
                model=None,
                created_at=datetime.now(UTC),
            )
        )
        session.commit()


def _discard_latest_turn(job_store, conversation_id: str) -> None:
    """Delete the final persisted user turn and every response after it.

    The regenerate UI replaces the latest assistant response in memory. Remove
    the matching persisted suffix before saving its replacement so reopening
    the conversation does not reveal duplicate copies of the same turn. A
    user-only suffix is also removed, which covers retrying a failed stream
    before an assistant row was written.

    Args:
        job_store: Job-store instance whose engine backs the DB session.
        conversation_id: Conversation whose latest turn should be replaced.
    """
    with Session(job_store.engine) as session:
        latest_user = (
            session.query(AgentMessageModel)
            .filter(
                AgentMessageModel.conversation_id == conversation_id,
                AgentMessageModel.role == "user",
            )
            .order_by(AgentMessageModel.id.desc())
            .first()
        )
        if latest_user is None:
            return
        session.query(AgentMessageModel).filter(
            AgentMessageModel.conversation_id == conversation_id,
            AgentMessageModel.id >= latest_user.id,
        ).delete(synchronize_session=False)
        session.commit()


def _persist_assistant_turn(
    job_store,
    conversation_id: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    model: str | None,
    *,
    wizard_state_before: dict[str, Any] | None = None,
    wizard_state_after: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    tool_schema_hashes: dict[str, str] | None = None,
    router_metadata: dict[str, Any] | None = None,
) -> None:
    """Insert the assistant's completed turn (with tool-call payloads).

    Called once, from the SSE wrapper, when the upstream ``done`` event fires.
    ``tool_calls`` carries the fully-resolved tool history accumulated from
    the ``tool_start`` / ``tool_end`` SSE events so the frontend can rehydrate
    the panel from this row alone.

    Args:
        job_store: Job-store instance whose engine backs the DB session.
        conversation_id: Owning conversation id.
        content: Final assistant message text.
        tool_calls: Accumulated tool-call records (matching ``AgentToolCall``).
        model: Model identifier reported by the agent runtime, when known.
        wizard_state_before: Wizard snapshot at turn start (training metadata).
        wizard_state_after: Wizard snapshot at turn end (training metadata).
        allowed_tools: Tool names exposed to the agent this turn.
        tool_schema_hashes: ``{tool_name: sha256(schema_json)}`` snapshot.
        router_metadata: OpenRouter upstream id + served-by host + latency.
            ``None`` until the runtime captures it (see spec §4).
    """
    now = datetime.now(UTC)
    with Session(job_store.engine) as session:
        session.add(
            AgentMessageModel(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
                model=model,
                wizard_state_before=wizard_state_before,
                wizard_state_after=wizard_state_after,
                allowed_tools=allowed_tools,
                tool_schema_hashes=tool_schema_hashes,
                router_metadata=router_metadata,
                created_at=now,
            )
        )
        row = session.get(AgentConversationModel, conversation_id)
        if row is not None:
            row.updated_at = cast(Any, now)
        session.commit()
    # Refresh the haystack embedding so the next search hit reflects the
    # turn we just persisted. Runs on a daemon thread and is best-effort —
    # the startup backfill heals any failures on the next deploy.
    queue_conversation_embed(conversation_id, engine=job_store.engine)


async def _wrap_with_persistence(
    source: AsyncIterator[dict[str, Any]],
    *,
    job_store,
    conversation_id: str | None,
    title: str | None,
    wizard_state_before: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Mirror upstream SSE events, accumulate state, persist on ``done``.

    Prepended with one ``conversation_meta`` event when persistence is on so
    the client can stash the canonical id (especially for newly-created
    threads). Accumulates the assistant text and every settled tool-call so a
    single ``agent_messages`` row captures the whole turn.

    Also captures the four training-metadata fields the optimize CLI needs
    (training_ground_SPEC.md §4): ``wizard_state_before`` from the caller,
    ``allowed_tools`` and ``tool_schema_hashes`` from the runtime's
    ``turn_metadata`` event, and ``wizard_state_after`` by merging each
    tool's ``result.wizard_state`` patch onto the running snapshot. Drops
    the ``turn_metadata`` event before forwarding so the frontend doesn't
    have to learn a new SSE schema.

    Args:
        source: Upstream async event stream from ``run_generalist_agent``.
        job_store: Job-store, or ``None`` when persistence is disabled.
        conversation_id: Persisted conversation id, or ``None`` to passthrough.
        title: Current conversation title (emitted in ``conversation_meta``).
        wizard_state_before: Wizard snapshot the caller handed to the agent
            this turn — persisted verbatim alongside the row.

    Yields:
        The same ``{event, data}`` mappings the SSE serializer expects, with
        the leading ``conversation_meta`` envelope when persistence is active.
    """
    if conversation_id and job_store is not None:
        yield {
            "event": "conversation_meta",
            "data": {"conversation_id": conversation_id, "title": title or ""},
        }

    assistant_buf: list[str] = []
    tool_calls: dict[str, dict[str, Any]] = {}
    tool_order: list[str] = []
    model_used: str | None = None
    served_model_used: str | None = None
    allowed_tools: list[str] | None = None
    tool_schema_hashes: dict[str, str] | None = None
    wizard_state_after: dict[str, Any] = dict(wizard_state_before) if wizard_state_before else {}
    persisted = False

    async def _do_persist(content: str) -> None:
        """Write the accumulated turn exactly once; never raise.

        The synchronous psycopg2 write is offloaded to a worker thread so it
        does not block the event loop while the SSE stream is still draining.

        Args:
            content: Final assistant message text for the row.
        """
        nonlocal persisted
        if persisted or not (conversation_id and job_store is not None):
            return
        ordered_tools = [tool_calls[tid] for tid in tool_order if tid in tool_calls]
        try:
            await asyncio.to_thread(
                _persist_assistant_turn,
                job_store,
                conversation_id,
                content,
                ordered_tools,
                model_used,
                wizard_state_before=wizard_state_before,
                wizard_state_after=wizard_state_after or None,
                allowed_tools=allowed_tools,
                tool_schema_hashes=tool_schema_hashes,
                router_metadata={"served_model": served_model_used} if served_model_used else None,
            )
        except Exception:
            logger.exception("Failed to persist assistant turn")
        persisted = True

    try:
        async for event in source:
            name = event.get("event")
            data = event.get("data") or {}
            if name == "message_patch":
                chunk = data.get("chunk")
                if isinstance(chunk, str):
                    assistant_buf.append(chunk)
            elif name == "turn_metadata":
                raw_allowed = data.get("allowed_tools")
                if isinstance(raw_allowed, list):
                    allowed_tools = [str(t) for t in raw_allowed]
                raw_hashes = data.get("tool_schema_hashes")
                if isinstance(raw_hashes, dict):
                    tool_schema_hashes = {str(k): str(v) for k, v in raw_hashes.items()}
                # Internal envelope — never forward to the frontend.
                continue
            elif name == "tool_start":
                tid = str(data.get("id", ""))
                if tid:
                    tool_calls[tid] = {
                        "id": tid,
                        "tool": data.get("tool", ""),
                        "reason": data.get("reason", ""),
                        "status": "running",
                        "startedAt": int(datetime.now(UTC).timestamp() * 1000),
                        "endedAt": None,
                        "payload": {"arguments": data.get("arguments", {})},
                    }
                    tool_order.append(tid)
            elif name == "tool_end":
                tid = str(data.get("id", ""))
                existing = tool_calls.get(tid)
                if existing is not None:
                    existing["status"] = "done" if data.get("status") == "ok" else "error"
                    existing["endedAt"] = int(datetime.now(UTC).timestamp() * 1000)
                    payload = existing.get("payload") or {}
                    result = data.get("result")
                    payload["result"] = result
                    existing["payload"] = payload
                    _merge_wizard_patch(wizard_state_after, result)
            elif name == "done":
                final_text = data.get("assistant_message")
                content = final_text if isinstance(final_text, str) and final_text else "".join(assistant_buf)
                raw_model = data.get("model")
                model_used = raw_model if isinstance(raw_model, str) and raw_model else None
                raw_served = data.get("served_model")
                served_model_used = raw_served if isinstance(raw_served, str) and raw_served else None
                await _do_persist(content)
            yield event
    finally:
        # The frontend tears down the SSE stream the moment ``submit_job_run_post``
        # succeeds, so ``done`` often never arrives and the wrapped generator is
        # closed (GeneratorExit / CancelledError). Salvage the turn here when it
        # carries real content; an empty greeting turn writes nothing.
        if not persisted and (assistant_buf or any(c["status"] in ("done", "error") for c in tool_calls.values())):
            await _do_persist("".join(assistant_buf))


def _merge_wizard_patch(state: dict[str, Any], result: Any) -> None:
    """Merge a tool result's ``wizard_state`` patch into the running state.

    Tools that mutate the wizard (``update_wizard_state``,
    ``set_column_roles``, …) echo the validated patch under
    ``result.wizard_state``. We treat the patch as a shallow overlay so the
    persisted ``wizard_state_after`` reflects every change the agent
    successfully made this turn. Tool results that don't carry the field
    leave the state untouched.

    Args:
        state: Running ``wizard_state_after`` buffer; mutated in place.
        result: Raw tool result payload from the ``tool_end`` SSE event.
    """
    if not isinstance(result, dict):
        return
    patch = result.get("wizard_state")
    if not isinstance(patch, dict):
        return
    for key, value in patch.items():
        state[str(key)] = value


def create_generalist_agent_router(*, job_store=None) -> APIRouter:
    """Mount the ``/optimizations/generalist-agent`` SSE + confirm endpoints.

    Args:
        job_store: Optional job-store whose engine backs conversation
            persistence. When ``None``, the SSE endpoint streams without
            writing anything (legacy behavior).

    Returns:
        A configured :class:`APIRouter` with the generalist-agent endpoints attached.
    """
    router = APIRouter()

    # The shared DB engine backs cross-replica approval handoff: a confirm
    # that lands on a replica that doesn't hold the stream's in-process future
    # persists the decision for the owning replica's poll loop.
    engine = getattr(job_store, "engine", None) if job_store is not None else None
    if engine is not None:
        get_approval_registry().bind_engine(engine)

    @router.post(
        "/optimizations/generalist-agent",
        summary="Stream generalist-agent events for one user turn",
    )
    async def generalist_agent(
        req: GeneralistAgentRequest,
        current_user: AuthenticatedUserDep,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        """Stream the generalist agent's reasoning, tool calls, and reply as SSE.

        Event types: ``conversation_meta`` (only when persistence is on),
        ``reasoning_patch``, ``tool_start``, ``tool_end``, ``status_patch``,
        ``pending_approval``, ``approval_resolved``, ``message_patch``,
        ``done``, ``error``.

        Args:
            req: Request body with user message, chat history, wizard
                snapshot, trust mode, and optional ``conversation_id``.
            current_user: The authenticated caller. Required — the route 401s
                before any streaming or LLM work when auth is missing/invalid,
                and persisted conversations are attributed to this user.
            authorization: Caller's bearer token, forwarded into the agent's
                MCP session so its tool calls authenticate against
                ``get_authenticated_user`` on the same FastAPI app.

        Returns:
            A :class:`StreamingResponse` of Server-Sent Events.
        """

        def _setup_turn() -> tuple[str | None, str | None]:
            """Persist the user turn off the event loop.

            Runs the blocking conversation upsert and user-message insert in a
            worker thread so the single event loop is not stalled before the SSE
            stream starts. Auth is already enforced by the route dependency, so
            the turn is always attributed to ``current_user``; a conversation
            ``DomainError`` (403/404) is allowed to propagate.

            Returns:
                ``(conversation_id, title)`` — both ``None`` when persistence is
                off (no ``job_store``) or failed non-fatally.
            """
            if job_store is None:
                return None, None
            try:
                cid, ttl = _ensure_conversation(job_store, req.conversation_id, current_user.username, req.user_message)
                if req.regenerate:
                    _discard_latest_turn(job_store, cid)
                _persist_user_turn(job_store, cid, req.user_message)
            except DomainError:
                raise
            except Exception:
                logger.exception("Failed to persist user turn")
                return None, None
            return cid, ttl

        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        conversation_id, title = await asyncio.to_thread(_setup_turn)

        wizard_state: WizardState = {**req.wizard_state}  # type: ignore[typeddict-item]
        requested_model, auto_tier = resolve_auto_tier(req.model)
        require_known_model(requested_model)
        model_config = (
            ModelConfig(
                name=requested_model,
                extra=({"reasoning_effort": req.reasoning_effort} if req.reasoning_effort else {}),
            )
            if requested_model
            else route_auto_model(auto_tier or "balanced", conversation_id)
        )
        usage_sink: list = []
        source = run_generalist_agent(
            wizard_state=wizard_state,
            chat_history=[t.model_dump() for t in req.chat_history],
            user_message=req.user_message,
            trust_mode=req.trust_mode,
            auth_header=authorization,
            locale=req.locale,
            model_config=model_config,
            usage_sink=usage_sink,
        )
        metered = stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Agent chat",
            usage_sink=usage_sink,
        )
        wrapped = _wrap_with_persistence(
            metered,
            job_store=job_store,
            conversation_id=conversation_id,
            title=title,
            wizard_state_before=dict(wizard_state),
        )
        return StreamingResponse(
            sse_from_events(wrapped),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/optimizations/generalist-agent/confirm",
        response_model=ConfirmApprovalResponse,
        summary="Resolve a pending generalist-agent approval",
    )
    def confirm_approval(req: ConfirmApprovalRequest, current_user: AuthenticatedUserDep) -> ConfirmApprovalResponse:
        """Resolve an outstanding approval from the client.

        Args:
            req: Confirm payload with the ``call_id`` and approval boolean.
            current_user: The authenticated caller (required).

        Returns:
            A :class:`ConfirmApprovalResponse` with ``resolved=True`` on success.

        Raises:
            DomainError: 404 when the call id is unknown locally and no durable
                store is bound (store-less deployments) — the client surfaces
                it as a UI warning. With a store, an unmatched confirm is
                persisted for the replica that owns the stream to pick up.
        """
        resolved = get_approval_registry().resolve_or_persist(req.call_id, req.approved)
        if not resolved:
            raise DomainError("agent.approval.unknown_call_id", status=404)
        return ConfirmApprovalResponse(resolved=True)

    return router
