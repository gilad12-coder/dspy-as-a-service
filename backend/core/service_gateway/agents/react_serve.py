"""Live chat driver for a served, GEPA-optimized ReActV2 program.

The optimization detail view exposes a chat playground for ``module=react``
runs. Unlike the scalar serve path (which only ever lists tool *schemas*),
this driver opens a **live** MCP session that stays open for the turn, binds
the program's tool roster to it so calls actually execute, and streams the
agent's reasoning, tool lifecycle, and final answer over the same SSE envelope
the generalist agent uses — so the frontend chat primitives render identically.

Tool calls are gated by the same approval machinery as the generalist
(:class:`~.generalist.ApprovalRegistry`, the ``pending_approval`` /
``approval_resolved`` events, and the companion confirm endpoint). The roster
here is arbitrary (whatever MCP the user pointed the run at), so the gating
policy is simpler than the wizard's destructive/safe classifier: confirm every
tool except in ``yolo`` mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from functools import partial
from typing import Any

import dspy

from ...exceptions import ServiceError
from ..optimization.retrying_react import RetryingReActV2
from ..optimization.tool_overlay import (
    ToolSchemaDriftError,
    _apply_bundle_tool_overrides,
    _apply_tool_name_overrides,
    _assert_tool_set_matches,
)
from .code import ReactReplyStream, ReasoningStreamListener, _format_agent_error
from .constants import REASONING_FIELD
from .generalist import (
    ApprovalRegistry,
    TrustMode,
    _emit_to_queue_threadsafe,
    _mcp_session,
    _wrap_tool_with_approval,
    get_approval_registry,
)

logger = logging.getLogger(__name__)


def _react_needs_approval(_tool_name: str, trust_mode: TrustMode) -> bool:
    """Gate every served tool call unless the caller chose ``yolo``.

    The served roster is whatever MCP the run was optimized against, so there
    is no destructive/safe classification to lean on (the generalist's
    wizard-tool sets don't apply). Treating every tool as approval-worthy in
    both ``ask`` and ``auto_safe`` is the safe default; ``yolo`` opts out.

    Args:
        _tool_name: The MCP tool's registered name (unused — policy is uniform).
        trust_mode: The caller's selected trust level.

    Returns:
        ``True`` for every tool unless ``trust_mode`` is ``yolo``.
    """
    return trust_mode != "yolo"


def _filter_tools(
    tools: list[dspy.Tool], tool_filter: list[str] | None
) -> list[dspy.Tool]:
    """Keep and order ``tools`` by ``tool_filter`` (names absent are skipped).

    Args:
        tools: The live MCP roster.
        tool_filter: Optional ordered allow-list of tool names; ``None`` keeps
            the roster unchanged.

    Returns:
        The filtered, filter-ordered roster.
    """
    if not tool_filter:
        return tools
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in tool_filter if name in by_name]


def _format_react_outputs(prediction: Any, output_fields: list[str]) -> str:
    """Render a ReActV2 prediction's output fields into one chat reply string.

    A single output field is returned verbatim; multiple fields are labelled
    so the chat bubble shows each. Non-string values are JSON-encoded.

    Args:
        prediction: The terminal :class:`dspy.Prediction` from the program.
        output_fields: The signature's declared output field names, in order.

    Returns:
        The assembled reply text (empty string when nothing was produced).
    """
    parts: list[str] = []
    multi = len(output_fields) > 1
    for name in output_fields:
        value = getattr(prediction, name, None)
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        parts.append(f"{name}: {text}" if multi else text)
    return "\n\n".join(part for part in parts if part)


async def _drive_react_chat(
    *,
    signature_cls: type,
    program_state_json: str,
    react_overlay: Any,
    user_message: str,
    trust_mode: TrustMode,
    registry: ApprovalRegistry,
    emit: Callable[[dict], None],
    lm: Any,
    mcp_url: str,
    auth_header: str | None,
) -> str:
    """Build a fresh live ReActV2 for this turn, run it, and return the reply.

    The serve materializer's cached program binds its roster to a dead session
    and is shared across requests, so we build a throwaway program here: source
    the roster from a live MCP session, drift-check it against the training-time
    snapshot, re-apply the GEPA tool-wording overlay + renames, wrap each tool
    for approval, then construct ``ReActV2`` (which adds its own ``submit``) and
    load the optimized state.

    Args:
        signature_cls: The program's signature (supplies input/output fields).
        program_state_json: Serialized optimized state to load onto the program.
        react_overlay: The artifact's ``ReactOverlay`` (schema hashes, tool
            wording, renames, ``tool_source``).
        user_message: The user's latest chat message.
        trust_mode: Caller's trust level for tool gating.
        registry: Approval registry coordinating tool confirmations.
        emit: Thread-safe SSE event emitter.
        lm: Language model bound to the program for this turn.
        mcp_url: Live MCP endpoint to bind the roster to.
        auth_header: Verbatim ``Authorization`` header forwarded to the MCP
            session so tool calls authenticate as the chatting owner.

    Returns:
        The assistant reply assembled from the program's output fields.

    Raises:
        ToolSchemaDriftError: When the live roster no longer matches the
            snapshot recorded at training time.
    """
    async with _mcp_session(mcp_url, auth_header=auth_header) as session:
        listing = await session.list_tools()
        tool_source = react_overlay.tool_source or {}
        tool_filter = tool_source.get("tool_filter") if isinstance(tool_source, dict) else None
        tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in listing.tools]
        tools = _filter_tools(tools, tool_filter)

        # Strict: the live chat must run against the exact tool surface the run
        # was optimised against — added/removed tools are drift too, not just
        # hash mismatches. Matches the serve-info gate so the tab's visibility
        # and the chat's executability share one verdict.
        _assert_tool_set_matches(react_overlay.tool_schema_hashes, tools, strict=True)
        _apply_bundle_tool_overrides(
            tools,
            tool_descriptions=react_overlay.tool_descriptions,
            tool_arg_descriptions=react_overlay.tool_arg_descriptions,
        )
        _apply_tool_name_overrides(tools, react_overlay.tool_names)

        outer_loop = asyncio.get_running_loop()
        wrapped = [
            _wrap_tool_with_approval(
                tool,
                trust_mode=trust_mode,
                registry=registry,
                emit=emit,
                outer_loop=outer_loop,
                needs_approval=_react_needs_approval,
            )
            for tool in tools
        ]
        # Tool descriptions are read from this roster at call time, so the
        # overlay wording above is what the LM sees; the renamed names match
        # the names GEPA baked into the loaded instructions. ReActV2 adds its
        # own reserved ``submit`` tool (final answer carrier, never gated).
        program = RetryingReActV2(signature_cls, tools=wrapped, max_iters=react_overlay.max_iters)
        program.load_state(program_state_json)

        output_fields = list(signature_cls.output_fields)
        input_fields = list(signature_cls.input_fields)
        primary_out = output_fields[0] if output_fields else None

        reply_stream = ReactReplyStream(program, primary_out) if primary_out else None
        listeners = (
            reply_stream.listeners()
            if reply_stream is not None
            else [ReasoningStreamListener(predict=program.react, allow_reuse=True)]
        )
        stream_program = dspy.streamify(
            program, stream_listeners=listeners, async_streaming=True
        )

        # Single-composer chat: the user's message drives the primary input
        # field; any further declared inputs start empty (the optimized
        # signature is rarely multi-input for a tool-using agent).
        inputs: dict[str, Any] = dict.fromkeys(input_fields, "")
        if input_fields:
            inputs[input_fields[0]] = user_message

        reply_text = ""
        with dspy.context(lm=lm):
            async for chunk in stream_program(**inputs):
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    if chunk.signature_field_name == REASONING_FIELD:
                        emit({"event": "reasoning_patch", "data": {"chunk": chunk.chunk}})
                    elif reply_stream is not None:
                        delta = reply_stream.reply_delta(chunk)
                        if delta:
                            reply_text += delta
                            emit({"event": "message_patch", "data": {"chunk": delta}})
                elif isinstance(chunk, dspy.Prediction):
                    final = _format_react_outputs(chunk, output_fields)
                    if final:
                        reply_text = final
        return reply_text


async def run_react_chat(
    *,
    signature_cls: type,
    program_state_json: str,
    react_overlay: Any,
    user_message: str,
    trust_mode: TrustMode,
    lm: Any,
    model_name: str,
    mcp_url: str,
    auth_header: str | None = None,
    approval_registry: ApprovalRegistry | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream one chat turn against a served, optimized ReActV2 program.

    Emits the same SSE envelope as :func:`~.generalist.run_generalist_agent`
    (``reasoning_patch``, ``tool_start`` / ``tool_end``, ``pending_approval`` /
    ``approval_resolved``, ``message_patch``, ``done``, ``error``) so the
    frontend agent primitives work unchanged.

    Args:
        signature_cls: The program's signature class.
        program_state_json: Serialized optimized state loaded onto the program.
        react_overlay: The artifact's ``ReactOverlay``.
        user_message: The user's latest chat message.
        trust_mode: Trust level controlling which tool calls require approval.
        lm: Language model bound to the program.
        model_name: Identifier surfaced in the terminal ``done`` event.
        mcp_url: Live MCP endpoint to bind the tool roster to.
        auth_header: Verbatim ``Authorization`` header forwarded to the MCP
            session.
        approval_registry: Registry used for tool-approval coordination;
            defaults to the process-wide singleton.

    Yields:
        SSE event dicts of shape ``{"event": str, "data": dict}``.

    Raises:
        asyncio.CancelledError: Re-raised when the stream is cancelled.
    """
    registry = approval_registry or get_approval_registry()
    out_queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    emit: Callable[[dict], None] = partial(_emit_to_queue_threadsafe, loop, out_queue)

    drive_task = asyncio.create_task(
        _drive_react_chat(
            signature_cls=signature_cls,
            program_state_json=program_state_json,
            react_overlay=react_overlay,
            user_message=user_message,
            trust_mode=trust_mode,
            registry=registry,
            emit=emit,
            lm=lm,
            mcp_url=mcp_url,
            auth_header=auth_header,
        )
    )
    try:
        while not drive_task.done() or not out_queue.empty():
            getter = asyncio.create_task(out_queue.get())
            done, _pending = await asyncio.wait({drive_task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
            else:
                getter.cancel()
            if drive_task in done and out_queue.empty():
                break
        reply = await drive_task
        yield {"event": "done", "data": {"assistant_message": reply, "model": model_name}}
    except asyncio.CancelledError:
        drive_task.cancel()
        raise
    except ToolSchemaDriftError as exc:
        logger.warning("react chat tool-schema drift: %s", exc)
        yield {"event": "error", "data": {"error": _format_agent_error(exc)}}
    except ServiceError as exc:
        yield {"event": "error", "data": {"error": str(exc)}}
    except Exception as exc:
        logger.exception("react chat failed")
        yield {"event": "error", "data": {"error": _format_agent_error(exc)}}
